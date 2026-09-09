import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, ".")
from src.tune_decision_thresholds import find_optimal_threshold, evaluate_predictions_with_threshold
from src.benchmark_lightweight_tier import evaluate_preds

def main():
    print("=== Phase 4: Negative Recall Threshold Optimization ===", flush=True)

    # Let's retrain Hybrid Word(1,2)+Char(3,5) + LogisticRegression to get full validation & test probabilities
    from sklearn.pipeline import FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    vec_hybrid = FeatureUnion([
        ("word", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)),
        ("char_wb", TfidfVectorizer(analyzer="char_wb", max_features=10000, ngram_range=(3, 5), sublinear_tf=True))
    ])

    print("Fitting hybrid extractor on train set...", flush=True)
    X_tr = vec_hybrid.fit_transform(train_df["clean_text"].fillna("").tolist())
    X_va = vec_hybrid.transform(val_df["clean_text"].fillna("").tolist())
    X_te = vec_hybrid.transform(test_df["clean_text"].fillna("").tolist())

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_tr, y_train)

    val_probs = clf.predict_proba(X_va)
    test_probs = clf.predict_proba(X_te)

    # Search for optimal theta_neg on Validation split
    tuning_val = find_optimal_threshold(y_val, val_probs, threshold_range=(0.22, 0.40), step=0.01)
    best_cfg = tuning_val["optimal_threshold_config"]
    optimal_theta = best_cfg["theta_neg"]
    print(f"\n[Threshold Tuning] Selected optimal theta_neg on validation set: {optimal_theta}", flush=True)
    print(f"  Validation Baseline Argmax: Acc={tuning_val['baseline_argmax']['accuracy']*100:.2f}%, Macro F1={tuning_val['baseline_argmax']['macro_f1']:.4f}, Neg Recall={tuning_val['baseline_argmax']['negative_recall']*100:.2f}%", flush=True)
    print(f"  Validation Tuned Threshold: Acc={best_cfg['accuracy']*100:.2f}%, Macro F1={best_cfg['macro_f1']:.4f}, Neg Recall={best_cfg['negative_recall']*100:.2f}%", flush=True)

    # Evaluate on held-out test split
    # 1. Baseline argmax on test
    base_test_preds = np.argmax(test_probs, axis=1)
    base_test_metrics = evaluate_preds(y_test, base_test_preds, test_probs)

    # 2. Tuned threshold on test
    tuned_test_metrics = evaluate_predictions_with_threshold(y_test, test_probs, theta_neg=optimal_theta)

    # True Negative to Predicted Neutral breakdown
    base_cm = np.array(base_test_metrics["confusion_matrix"])
    tuned_cm = np.array(tuned_test_metrics["confusion_matrix"])

    # True Negative class is row 0: [TrueNeg_PredNeg, TrueNeg_PredNeu, TrueNeg_PredPos]
    base_tn_pneu = int(base_cm[0, 1])
    tuned_tn_pneu = int(tuned_cm[0, 1])
    recovered_negatives = base_tn_pneu - tuned_tn_pneu

    tuning_report = {
        "dataset": "CardiffNLP TweetEval Held-Out Test Set (8,947 samples)",
        "model": "Hybrid TF-IDF + Logistic Regression",
        "optimal_theta_neg": optimal_theta,
        "baseline_argmax_test": base_test_metrics,
        "tuned_threshold_test": tuned_test_metrics,
        "true_neg_pred_neu_reduction": {
            "baseline_misclassified_as_neutral": base_tn_pneu,
            "tuned_misclassified_as_neutral": tuned_tn_pneu,
            "false_neutrals_rescued_to_negative": recovered_negatives,
            "reduction_percentage": round((recovered_negatives / base_tn_pneu) * 100, 2)
        }
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/threshold_tuning_report.json", "w") as f:
        json.dump(tuning_report, f, indent=2)

    print("\n" + "="*70)
    print("THRESHOLD TUNING IMPACT ON TEST SET:")
    print(f"  Threshold theta_neg:           {optimal_theta}")
    print(f"  Accuracy:                      {base_test_metrics['accuracy']*100:.2f}% -> {tuned_test_metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1:                      {base_test_metrics['macro_f1']:.4f} -> {tuned_test_metrics['macro_f1']:.4f}")
    print(f"  Negative Recall:               {base_test_metrics['negative_recall']*100:.2f}% -> {tuned_test_metrics['negative_recall']*100:.2f}% (+{(tuned_test_metrics['negative_recall'] - base_test_metrics['negative_recall'])*100:.2f}%)")
    print(f"  Negative F1:                   {base_test_metrics['negative_f1']:.4f} -> {tuned_test_metrics['negative_f1']:.4f}")
    print(f"  True Neg -> Pred Neu Misses:   {base_tn_pneu} -> {tuned_tn_pneu} ({recovered_negatives} false neutrals rescued!)")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
