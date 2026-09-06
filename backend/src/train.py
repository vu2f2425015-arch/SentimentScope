import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    log_loss
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D

from src.data_loader import load_and_split_data, LABEL_MAP, REVERSE_LABEL_MAP
from src.features import TFIDFExtractor, SequentialExtractor

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes average Brier score loss across all 3 one-hot encoded classes.
    Lower is better (0.0 = perfect calibration).
    """
    n_classes = y_prob.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Computes accuracy, macro precision, recall, macro F1, confusion matrix, and class-level metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    brier = compute_multiclass_brier_score(y_true, y_prob)
    
    # Class-level metrics
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    class_metrics = {}
    for idx, label_name in REVERSE_LABEL_MAP.items():
        class_metrics[label_name] = {
            "precision": round(float(p_class[idx]), 4),
            "recall": round(float(r_class[idx]), 4),
            "f1": round(float(f1_class[idx]), 4)
        }

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "class_metrics": class_metrics
    }


def build_bilstm_model(vocab_size: int, embed_dim: int = 128, max_len: int = 150) -> Sequential:
    """
    Constructs a Keras Bi-LSTM Neural Network for 3-class sentiment classification.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, dropout=0.2, return_sequences=False)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate_all():
    """
    Executes comprehensive model training pipeline on CardiffNLP TweetEval Benchmark:
    1. Class balance verification & class weight calculation.
    2. Grid search for TF-IDF features (max_features, ngram_range) on validation set.
    3. Grid search for Logistic Regression (C) on train CV.
    4. Probability calibration via CalibratedClassifierCV (Platt sigmoid vs Isotonic).
    5. 5-Fold Stratified Cross-Validation reporting mean ± std Macro F1.
    6. Held-out test split evaluation & deployment.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=== Step 1: Loading Data & Class Balance Check ===", flush=True)
    train_df, val_df, test_df = load_and_split_data()

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    train_counts = train_df["sentiment"].value_counts().to_dict()
    print(f"[Class Balance] Train split distribution: {train_counts}", flush=True)

    classes = np.unique(y_train)
    cw_array = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_dict = {int(cls): float(weight) for cls, weight in zip(classes, cw_array)}
    print(f"[Class Balance] Calculated class weights: {class_weights_dict}", flush=True)

    print("\n=== Step 2: TF-IDF Feature Extraction & Grid Search ===", flush=True)
    max_features_options = [5000, 10000, 20000]
    ngram_range_options = [(1, 1), (1, 2), (1, 3)]

    best_tfidf_score = -1.0
    best_tfidf_params = (10000, (1, 2))

    for mf in max_features_options:
        for ng in ngram_range_options:
            ext = TFIDFExtractor(max_features=mf, ngram_range=ng)
            X_tr = ext.fit_transform(train_df["clean_text"].tolist())
            X_va = ext.transform(val_df["clean_text"].tolist())
            
            clf = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_tr, y_train)
            val_preds = clf.predict(X_va)
            _, _, f1_val, _ = precision_recall_fscore_support(y_val, val_preds, average="macro", zero_division=0)
            print(f"TF-IDF max_features={mf:5d}, ngram_range={ng} -> Val Macro F1: {f1_val:.4f}", flush=True)
            
            if f1_val > best_tfidf_score:
                best_tfidf_score = f1_val
                best_tfidf_params = (mf, ng)

    print(f"--> WINNING TF-IDF PARAMS: max_features={best_tfidf_params[0]}, ngram_range={best_tfidf_params[1]} (Val Macro F1 = {best_tfidf_score:.4f})", flush=True)

    # Fit final winning TF-IDF extractor
    tfidf_extractor = TFIDFExtractor(max_features=best_tfidf_params[0], ngram_range=best_tfidf_params[1])
    X_train_tfidf = tfidf_extractor.fit_transform(train_df["clean_text"].tolist())
    X_val_tfidf = tfidf_extractor.transform(val_df["clean_text"].tolist())
    X_test_tfidf = tfidf_extractor.transform(test_df["clean_text"].tolist())
    tfidf_extractor.save(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))

    # Sequential Features for Bi-LSTM
    seq_extractor = SequentialExtractor(num_words=10000, max_len=150)
    X_train_seq = seq_extractor.fit_transform(train_df["clean_text"].tolist())
    X_val_seq = seq_extractor.transform(val_df["clean_text"].tolist())
    X_test_seq = seq_extractor.transform(test_df["clean_text"].tolist())
    seq_extractor.save(os.path.join(MODELS_DIR, "lstm_tokenizer.joblib"))

    print("\n=== Step 3: Logistic Regression Hyperparameter Search ===", flush=True)
    lr_param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    lr_base = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
    grid_search = GridSearchCV(lr_base, lr_param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid_search.fit(X_train_tfidf, y_train)

    best_lr_params = grid_search.best_params_
    print(f"--> WINNING LOGISTIC REGRESSION PARAMS: {best_lr_params} (5-fold Train Macro F1 = {grid_search.best_score_:.4f})", flush=True)

    print("\n=== Step 4: 5-Fold Stratified Cross-Validation on Candidate Models ===", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}

    # 1. Multinomial Naive Bayes 5-Fold CV
    mnb_cv_scores = []
    for train_idx, val_idx in skf.split(X_train_tfidf, y_train):
        m = MultinomialNB(alpha=0.5)
        m.fit(X_train_tfidf[train_idx], y_train[train_idx])
        preds = m.predict(X_train_tfidf[val_idx])
        _, _, f1, _ = precision_recall_fscore_support(y_train[val_idx], preds, average="macro", zero_division=0)
        mnb_cv_scores.append(f1)
    cv_summary["Multinomial Naive Bayes"] = {
        "mean_f1": round(float(np.mean(mnb_cv_scores)), 4),
        "std_f1": round(float(np.std(mnb_cv_scores)), 4)
    }
    print(f"MNB 5-Fold CV Macro F1: {np.mean(mnb_cv_scores):.4f} ± {np.std(mnb_cv_scores):.4f}", flush=True)

    # 2. Logistic Regression 5-Fold CV
    lr_cv_scores = []
    for train_idx, val_idx in skf.split(X_train_tfidf, y_train):
        m = LogisticRegression(**best_lr_params, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
        m.fit(X_train_tfidf[train_idx], y_train[train_idx])
        preds = m.predict(X_train_tfidf[val_idx])
        _, _, f1, _ = precision_recall_fscore_support(y_train[val_idx], preds, average="macro", zero_division=0)
        lr_cv_scores.append(f1)
    cv_summary["Logistic Regression"] = {
        "mean_f1": round(float(np.mean(lr_cv_scores)), 4),
        "std_f1": round(float(np.std(lr_cv_scores)), 4)
    }
    print(f"LR 5-Fold CV Macro F1:  {np.mean(lr_cv_scores):.4f} ± {np.std(lr_cv_scores):.4f}", flush=True)

    # 3. Bi-LSTM 5-Fold CV
    lstm_cv_scores = []
    vocab_size = len(seq_extractor.tokenizer.word_index) + 1
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_seq, y_train)):
        m = build_bilstm_model(vocab_size=vocab_size, max_len=150)
        m.fit(X_train_seq[train_idx], y_train[train_idx], epochs=3, batch_size=256, class_weight=class_weights_dict, verbose=0)
        probs = m.predict(X_train_seq[val_idx], verbose=0)
        preds = np.argmax(probs, axis=1)
        _, _, f1, _ = precision_recall_fscore_support(y_train[val_idx], preds, average="macro", zero_division=0)
        lstm_cv_scores.append(f1)
    cv_summary["Bi-LSTM"] = {
        "mean_f1": round(float(np.mean(lstm_cv_scores)), 4),
        "std_f1": round(float(np.std(lstm_cv_scores)), 4)
    }
    print(f"Bi-LSTM 5-Fold CV Macro F1: {np.mean(lstm_cv_scores):.4f} ± {np.std(lstm_cv_scores):.4f}", flush=True)

    print("\n=== Step 5: Probability Calibration Evaluation ===", flush=True)
    # Train uncalibrated base MNB & LR models on full train split
    mnb_base = MultinomialNB(alpha=0.5)
    mnb_base.fit(X_train_tfidf, y_train)

    lr_base = LogisticRegression(**best_lr_params, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
    lr_base.fit(X_train_tfidf, y_train)

    # Pick top classical base model based on CV
    if cv_summary["Multinomial Naive Bayes"]["mean_f1"] >= cv_summary["Logistic Regression"]["mean_f1"]:
        best_base_model = mnb_base
        best_base_name = "Multinomial Naive Bayes"
    else:
        best_base_model = lr_base
        best_base_name = "Logistic Regression"

    # Calibration methods evaluation on validation set
    uncal_probs_val = best_base_model.predict_proba(X_val_tfidf)
    uncal_brier = compute_multiclass_brier_score(y_val, uncal_probs_val)
    uncal_loss = float(log_loss(y_val, uncal_probs_val))

    # Sigmoid calibration (Platt scaling) using 5-fold CV
    calib_sig = CalibratedClassifierCV(estimator=best_base_model, method="sigmoid", cv=5)
    calib_sig.fit(X_train_tfidf, y_train)
    sig_probs_val = calib_sig.predict_proba(X_val_tfidf)
    sig_brier = compute_multiclass_brier_score(y_val, sig_probs_val)
    sig_loss = float(log_loss(y_val, sig_probs_val))

    # Isotonic calibration using 5-fold CV
    calib_iso = CalibratedClassifierCV(estimator=best_base_model, method="isotonic", cv=5)
    calib_iso.fit(X_train_tfidf, y_train)
    iso_probs_val = calib_iso.predict_proba(X_val_tfidf)
    iso_brier = compute_multiclass_brier_score(y_val, iso_probs_val)
    iso_loss = float(log_loss(y_val, iso_probs_val))

    print(f"Validation Calibration Results for '{best_base_name}':", flush=True)
    print(f"  - Uncalibrated: Brier Score = {uncal_brier:.4f}, Log Loss = {uncal_loss:.4f}", flush=True)
    print(f"  - Sigmoid:      Brier Score = {sig_brier:.4f}, Log Loss = {sig_loss:.4f}", flush=True)
    print(f"  - Isotonic:     Brier Score = {iso_brier:.4f}, Log Loss = {iso_loss:.4f}", flush=True)

    calib_scores = {"uncalibrated": uncal_brier, "sigmoid": sig_brier, "isotonic": iso_brier}
    winning_calib_method = min(calib_scores, key=calib_scores.get)
    print(f"--> WINNING CALIBRATION METHOD: '{winning_calib_method}' (Brier Score = {calib_scores[winning_calib_method]:.4f})", flush=True)

    # Final trained artifacts save
    if winning_calib_method == "sigmoid":
        final_classical_model = calib_sig
    elif winning_calib_method == "isotonic":
        final_classical_model = calib_iso
    else:
        final_classical_model = best_base_model

    mnb_path = os.path.join(MODELS_DIR, "multinomial_nb.joblib")
    joblib.dump(mnb_base if best_base_name != "Multinomial Naive Bayes" else final_classical_model, mnb_path)

    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    joblib.dump(lr_base if best_base_name != "Logistic Regression" else final_classical_model, lr_path)

    # Train & Save full Bi-LSTM model
    print("\n=== Step 6: Fitting Full Bi-LSTM Model ===", flush=True)
    lstm_model = build_bilstm_model(vocab_size=vocab_size, max_len=150)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    lstm_model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=6,
        batch_size=256,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=1
    )
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    lstm_model.save(lstm_path)

    print("\n=== Step 7: Final Held-Out Test Evaluation ===", flush=True)
    metrics_report = {}

    # MNB Test
    mnb_eval_model = final_classical_model if best_base_name == "Multinomial Naive Bayes" else mnb_base
    mnb_test_pred = mnb_eval_model.predict(X_test_tfidf)
    mnb_test_prob = mnb_eval_model.predict_proba(X_test_tfidf)
    mnb_metrics = evaluate_model_performance(y_test, mnb_test_pred, mnb_test_prob)
    mnb_metrics["cv_f1_mean"] = cv_summary["Multinomial Naive Bayes"]["mean_f1"]
    mnb_metrics["cv_f1_std"] = cv_summary["Multinomial Naive Bayes"]["std_f1"]
    metrics_report["Multinomial Naive Bayes"] = mnb_metrics
    print(f"MNB Test Accuracy: {mnb_metrics['accuracy']} | Macro F1: {mnb_metrics['macro_f1']} | Brier: {mnb_metrics['brier_score']}", flush=True)

    # LR Test
    lr_eval_model = final_classical_model if best_base_name == "Logistic Regression" else lr_base
    lr_test_pred = lr_eval_model.predict(X_test_tfidf)
    lr_test_prob = lr_eval_model.predict_proba(X_test_tfidf)
    lr_metrics = evaluate_model_performance(y_test, lr_test_pred, lr_test_prob)
    lr_metrics["cv_f1_mean"] = cv_summary["Logistic Regression"]["mean_f1"]
    lr_metrics["cv_f1_std"] = cv_summary["Logistic Regression"]["std_f1"]
    metrics_report["Logistic Regression"] = lr_metrics
    print(f"LR Test Accuracy:  {lr_metrics['accuracy']} | Macro F1: {lr_metrics['macro_f1']} | Brier: {lr_metrics['brier_score']}", flush=True)

    # Bi-LSTM Test
    lstm_test_prob = lstm_model.predict(X_test_seq)
    lstm_test_pred = np.argmax(lstm_test_prob, axis=1)
    lstm_metrics = evaluate_model_performance(y_test, lstm_test_pred, lstm_test_prob)
    lstm_metrics["cv_f1_mean"] = cv_summary["Bi-LSTM"]["mean_f1"]
    lstm_metrics["cv_f1_std"] = cv_summary["Bi-LSTM"]["std_f1"]
    metrics_report["Bi-LSTM"] = lstm_metrics
    print(f"Bi-LSTM Test Accuracy: {lstm_metrics['accuracy']} | Macro F1: {lstm_metrics['macro_f1']} | Brier: {lstm_metrics['brier_score']}", flush=True)

    best_model_name = max(metrics_report, key=lambda k: metrics_report[k]["macro_f1"])
    best_metrics = metrics_report[best_model_name]
    print(f"\n--> BEST PERFORMING MODEL: '{best_model_name}' with Macro F1 = {best_metrics['macro_f1']} and Accuracy = {best_metrics['accuracy']}", flush=True)

    # Save report JSON
    comparison_report_path = os.path.join(REPORTS_DIR, "model_comparison.json")
    full_report = {
        "best_model": best_model_name,
        "dataset": "CardiffNLP TweetEval 3-Class Sentiment Benchmark",
        "target_met": (best_metrics["accuracy"] >= 0.60 and best_metrics["macro_f1"] >= 0.60),
        "train_class_distribution": train_counts,
        "winning_tfidf_params": {"max_features": best_tfidf_params[0], "ngram_range": list(best_tfidf_params[1])},
        "winning_lr_params": best_lr_params,
        "calibration_eval": {
            "uncalibrated_brier": round(uncal_brier, 4),
            "sigmoid_brier": round(sig_brier, 4),
            "isotonic_brier": round(iso_brier, 4),
            "winning_calibration": winning_calib_method
        },
        "cross_validation": cv_summary,
        "models": metrics_report
    }
    with open(comparison_report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"[Train] Full report saved to '{comparison_report_path}'.", flush=True)

    model_type_map = {
        "Multinomial Naive Bayes": {"type": "sklearn", "artifact": "multinomial_nb.joblib", "vectorizer": "tfidf_vectorizer.joblib"},
        "Logistic Regression": {"type": "sklearn", "artifact": "logistic_regression.joblib", "vectorizer": "tfidf_vectorizer.joblib"},
        "Bi-LSTM": {"type": "keras", "artifact": "lstm_model.keras", "tokenizer": "lstm_tokenizer.joblib"}
    }
    
    meta = {
        "best_model_name": best_model_name,
        "details": model_type_map[best_model_name],
        "metrics": best_metrics
    }
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Train] Deployed model metadata saved to '{meta_path}'.", flush=True)

    # Automatically generate visualization plots for evaluation reports
    try:
        from src.visualize import generate_all_visualizations
        generate_all_visualizations()
    except Exception as e:
        print(f"[Train Warning] Could not auto-generate visualization plots: {e}", flush=True)


if __name__ == "__main__":
    train_and_evaluate_all()

