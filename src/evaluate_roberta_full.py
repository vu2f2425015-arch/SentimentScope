import os
import sys
import time
import json
import psutil
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_classes = y_prob.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))

def main():
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    TEST_CSV = "data/processed/test.csv"
    OUTPUT_JSON = "reports/full_dataset_roberta_benchmark.json"

    process = psutil.Process(os.getpid())
    base_mem = process.memory_info().rss / (1024 * 1024)
    print(f"[Benchmark] Initial Python RAM: {base_mem:.2f} MB", flush=True)

    if not os.path.exists(TEST_CSV):
        print(f"[Error] {TEST_CSV} not found!", flush=True)
        sys.exit(1)

    df_test = pd.read_csv(TEST_CSV)
    print(f"[Benchmark] Loaded held-out test set: {len(df_test)} samples.", flush=True)

    print(f"[Benchmark] Loading {MODEL_NAME} tokenizer and model...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    load_time = time.time() - t0
    load_mem = process.memory_info().rss / (1024 * 1024)
    print(f"[Benchmark] Model loaded in {load_time:.2f}s. RAM after model load: {load_mem:.2f} MB (Delta: {load_mem - base_mem:.2f} MB)", flush=True)

    # Warm-up single sample and measure single-sample latency
    warmup_text = "SentimentScope is working amazingly well!"
    warmup_inputs = tokenizer(warmup_text, return_tensors="pt")
    single_latencies = []
    with torch.no_grad():
        for _ in range(20):
            t_s = time.perf_counter()
            _ = model(**warmup_inputs)
            single_latencies.append((time.perf_counter() - t_s) * 1000)

    p50_latency = float(np.percentile(single_latencies, 50))
    p95_latency = float(np.percentile(single_latencies, 95))
    mean_latency = float(np.mean(single_latencies))
    print(f"[Benchmark] Single-sample CPU latency: p50={p50_latency:.2f}ms, p95={p95_latency:.2f}ms, mean={mean_latency:.2f}ms", flush=True)

    # Batched evaluation over all 8,947 samples
    texts = df_test["text"].fillna("").tolist()
    labels = df_test["label"].tolist()

    batch_size = 64
    all_preds = []
    all_probs = []

    print(f"[Benchmark] Starting batched evaluation across {len(texts)} samples (batch_size={batch_size})...", flush=True)
    t_eval_start = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds)
            all_probs.extend(probs)

            if (i // batch_size) % 25 == 0 or (i + len(batch_texts)) == len(texts):
                pct = (i + len(batch_texts)) / len(texts) * 100
                elapsed = time.time() - t_eval_start
                print(f"  Processed {i + len(batch_texts):5d}/{len(texts)} ({pct:5.1f}%) in {elapsed:5.1f}s", flush=True)

    eval_total_time = time.time() - t_eval_start
    peak_mem = process.memory_info().rss / (1024 * 1024)
    print(f"[Benchmark] Evaluation finished in {eval_total_time:.2f}s ({eval_total_time / len(texts) * 1000:.2f} ms/sample). Peak RAM: {peak_mem:.2f} MB", flush=True)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    labels = np.array(labels)

    acc = float(accuracy_score(labels, all_preds))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(labels, all_preds, average="macro", zero_division=0)
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(labels, all_preds, labels=[0, 1, 2]).tolist()
    brier = compute_multiclass_brier_score(labels, all_probs)

    report_data = {
        "model_name": MODEL_NAME,
        "dataset": "CardiffNLP TweetEval 3-Class Held-Out Test Set (Full 60k Split)",
        "test_samples": len(labels),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_precision": round(float(macro_prec), 4),
        "brier_score": round(brier, 4),
        "latency_ms": {
            "p50_ms": round(p50_latency, 2),
            "p95_ms": round(p95_latency, 2),
            "mean_ms": round(mean_latency, 2)
        },
        "memory_profile_mb": {
            "base_python_ram": round(base_mem, 2),
            "ram_after_model_load": round(load_mem, 2),
            "peak_inference_ram": round(peak_mem, 2)
        },
        "confusion_matrix": cm,
        "class_metrics": {
            "negative": {
                "precision": round(float(p_class[0]), 4),
                "recall": round(float(r_class[0]), 4),
                "f1": round(float(f1_class[0]), 4)
            },
            "neutral": {
                "precision": round(float(p_class[1]), 4),
                "recall": round(float(r_class[1]), 4),
                "f1": round(float(f1_class[1]), 4)
            },
            "positive": {
                "precision": round(float(p_class[2]), 4),
                "recall": round(float(r_class[2]), 4),
                "f1": round(float(f1_class[2]), 4)
            }
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[Benchmark] Successfully written results to {OUTPUT_JSON}")

    # Formatted comparison against DistilBERT baseline
    print("\n" + "="*70)
    print(f"APPLES-TO-APPLES COMPARISON (Full 8,947 Held-Out Test Split):")
    print(f"{'Metric':<25} | {'DistilBERT (Current)':<20} | {'Twitter-RoBERTa (New)':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {'72.57%':<20} | {acc*100:.2f}%")
    print(f"{'Macro F1':<25} | {'0.7221':<20} | {macro_f1:.4f}")
    print(f"{'Negative Recall':<25} | {'74.22%':<20} | {r_class[0]*100:.2f}%")
    print(f"{'Negative F1':<25} | {'0.6939':<20} | {f1_class[0]:.4f}")
    print(f"{'Neutral F1':<25} | {'0.7176':<20} | {f1_class[1]:.4f}")
    print(f"{'Positive F1':<25} | {'0.7548':<20} | {f1_class[2]:.4f}")
    print(f"{'Brier Score':<25} | {'0.3746':<20} | {brier:.4f}")
    print(f"{'CPU Latency (p50)':<25} | {'~14.0 ms (raw)':<20} | {p50_latency:.2f} ms")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
