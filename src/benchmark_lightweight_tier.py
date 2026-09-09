import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import sys
sys.path.insert(0, ".")
from src.preprocessing import clean_text

def evaluate_preds(y_true, y_pred, y_prob):
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    n_classes = y_prob.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    brier = float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_precision": round(float(p_macro), 4),
        "brier_score": round(brier, 4),
        "negative_recall": round(float(r_cls[0]), 4),
        "negative_f1": round(float(f1_cls[0]), 4),
        "neutral_f1": round(float(f1_cls[1]), 4),
        "positive_f1": round(float(f1_cls[2]), 4),
        "confusion_matrix": cm
    }

def main():
    print("=== Benchmarking Modernized Lightweight Cloud Tier ===", flush=True)

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train_raw = train_df["clean_text"].fillna("").tolist()
    y_train = train_df["label"].values

    X_val_raw = val_df["clean_text"].fillna("").tolist()
    y_val = val_df["label"].values

    X_test_raw = test_df["clean_text"].fillna("").tolist()
    y_test = test_df["label"].values

    print(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}", flush=True)

    # Define Feature Extractors
    # 1. Word-only unigram (baseline)
    vec_unigram = TfidfVectorizer(max_features=10000, ngram_range=(1, 1), sublinear_tf=True)
    # 2. Word unigram + bigram (widen n-gram)
    vec_bigram = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    # 3. Hybrid: Word (1,2) + Char_wb (3,5) subwords
    vec_hybrid = FeatureUnion([
        ("word", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)),
        ("char_wb", TfidfVectorizer(analyzer="char_wb", max_features=10000, ngram_range=(3, 5), sublinear_tf=True))
    ])

    extractors = {
        "Word (1,1)": vec_unigram,
        "Word (1,2)": vec_bigram,
        "Hybrid Word(1,2)+Char(3,5)": vec_hybrid
    }

    models_to_test = {
        "LogisticRegression (C=1.0)": lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42),
        "Calibrated LinearSVC": lambda: CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=42), method="sigmoid", cv=3)
    }

    results = {}

    for ext_name, ext in extractors.items():
        print(f"\n--- Extracting features with: {ext_name} ---", flush=True)
        t0 = time.time()
        X_tr = ext.fit_transform(X_train_raw)
        X_va = ext.transform(X_val_raw)
        X_te = ext.transform(X_test_raw)
        print(f"Feature shape: {X_tr.shape} in {time.time()-t0:.2f}s", flush=True)

        for m_name, model_fn in models_to_test.items():
            run_key = f"{ext_name} + {m_name}"
            print(f"Training: {run_key}...", flush=True)
            clf = model_fn()
            clf.fit(X_tr, y_train)

            # Evaluate on Val
            va_probs = clf.predict_proba(X_va)
            va_preds = np.argmax(va_probs, axis=1)
            val_eval = evaluate_preds(y_val, va_preds, va_probs)

            # Evaluate on Test
            te_probs = clf.predict_proba(X_te)
            te_preds = np.argmax(te_probs, axis=1)
            test_eval = evaluate_preds(y_test, te_preds, te_probs)

            # Test OOV sample: "SentimentScope is working amazingly well!"
            oov_text = clean_text("SentimentScope is working amazingly well!")
            oov_vec = ext.transform([oov_text])
            oov_probs = clf.predict_proba(oov_vec)[0]

            # Test Negation sample: "The service was not good at all."
            neg_text = clean_text("The service was not good at all.")
            neg_vec = ext.transform([neg_text])
            neg_probs = clf.predict_proba(neg_vec)[0]

            results[run_key] = {
                "val": val_eval,
                "test": test_eval,
                "oov_test_probs": {
                    "negative": round(float(oov_probs[0]), 4),
                    "neutral": round(float(oov_probs[1]), 4),
                    "positive": round(float(oov_probs[2]), 4)
                },
                "negation_test_probs": {
                    "negative": round(float(neg_probs[0]), 4),
                    "neutral": round(float(neg_probs[1]), 4),
                    "positive": round(float(neg_probs[2]), 4)
                }
            }

            print(f"  -> Test Acc: {test_eval['accuracy']*100:.2f}% | Macro F1: {test_eval['macro_f1']:.4f} | Neg Recall: {test_eval['negative_recall']*100:.2f}% | OOV Pos Prob: {oov_probs[2]*100:.1f}% | Negation Neg Prob: {neg_probs[0]*100:.1f}%", flush=True)

    os.makedirs("reports", exist_ok=True)
    with open("reports/lightweight_tier_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[Lightweight Benchmark] Saved detailed results to reports/lightweight_tier_benchmark.json")

if __name__ == "__main__":
    main()
