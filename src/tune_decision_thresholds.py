import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_predictions_with_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    theta_neg: float = 0.33,
    min_pos_diff: float = 0.05
) -> Dict[str, Any]:
    """
    Cost-sensitive sentiment decision rule:
    Prioritize negative classification when negative probability exceeds theta_neg
    and clearly dominates positive probability (probs[:, 0] > probs[:, 2] + min_pos_diff).
    Otherwise, fall back to standard argmax.
    """
    preds = np.argmax(probs, axis=1)
    
    # Identify indices eligible for negative promotion
    neg_promoted = (probs[:, 0] >= theta_neg) & (probs[:, 0] > (probs[:, 2] + min_pos_diff))
    preds[neg_promoted] = 0

    acc = float(accuracy_score(y_true, preds))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, preds, average="macro", zero_division=0)
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_true, preds, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2]).tolist()

    return {
        "theta_neg": round(theta_neg, 3),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_precision": round(float(macro_prec), 4),
        "negative_recall": round(float(r_class[0]), 4),
        "negative_precision": round(float(p_class[0]), 4),
        "negative_f1": round(float(f1_class[0]), 4),
        "neutral_f1": round(float(f1_class[1]), 4),
        "positive_f1": round(float(f1_class[2]), 4),
        "confusion_matrix": cm
    }

def find_optimal_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    threshold_range: Tuple[float, float] = (0.22, 0.45),
    step: float = 0.01
) -> Dict[str, Any]:
    """
    Performs grid search over negative threshold values on validation split.
    Selects threshold maximizing Negative Recall subject to Macro F1 retaining >= 98.5% of peak.
    """
    # Baseline standard argmax metrics
    baseline_preds = np.argmax(val_probs, axis=1)
    base_acc = accuracy_score(y_val, baseline_preds)
    _, _, base_macro_f1, _ = precision_recall_fscore_support(y_val, baseline_preds, average="macro", zero_division=0)
    _, base_rec_class, base_f1_class, _ = precision_recall_fscore_support(y_val, baseline_preds, average=None, labels=[0, 1, 2], zero_division=0)

    best_candidate = None
    best_neg_recall = -1.0
    all_grid_results = []

    current_theta = threshold_range[0]
    while current_theta <= threshold_range[1]:
        res = evaluate_predictions_with_threshold(y_val, val_probs, theta_neg=current_theta)
        all_grid_results.append(res)
        
        # Criterion: Macro F1 must not degrade more than 1.5% from baseline, while maximizing Negative Recall
        if res["macro_f1"] >= (base_macro_f1 * 0.985):
            if res["negative_recall"] > best_neg_recall:
                best_neg_recall = res["negative_recall"]
                best_candidate = res
                
        current_theta += step

    return {
        "baseline_argmax": {
            "accuracy": round(float(base_acc), 4),
            "macro_f1": round(float(base_macro_f1), 4),
            "negative_recall": round(float(base_rec_class[0]), 4),
            "negative_f1": round(float(base_f1_class[0]), 4)
        },
        "optimal_threshold_config": best_candidate,
        "grid_sweep_count": len(all_grid_results)
    }
