import os
import sys
import time
import json
import psutil
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_classes = y_prob.shape[1]
    y_true_oh = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))

def get_mem_mb(proc):
    return proc.memory_info().rss / (1024 * 1024)

def main():
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    TEST_CSV = "data/processed/test.csv"
    OUTPUT_JSON = "reports/roberta_quantization_report.json"

    proc = psutil.Process(os.getpid())
    base_ram = get_mem_mb(proc)
    print(f"[Quantization Profile] Base Python process RAM: {base_ram:.2f} MB", flush=True)

    print(f"[Quantization Profile] Loading tokenizer & unquantized model: {MODEL_NAME}...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    unquant_load_time = time.time() - t0
    unquant_load_ram = get_mem_mb(proc)
    print(f"[Quantization Profile] Unquantized loaded in {unquant_load_time:.2f}s. RAM: {unquant_load_ram:.2f} MB (Delta: {unquant_load_ram - base_ram:.2f} MB)", flush=True)

    # Measure disk size of unquantized model weights
    scratch_dir = "scratch"
    os.makedirs(scratch_dir, exist_ok=True)
    unquant_weights_path = os.path.join(scratch_dir, "unquant_model.pt")
    torch.save(model.state_dict(), unquant_weights_path)
    unquant_size_mb = os.path.getsize(unquant_weights_path) / (1024 * 1024)
    print(f"[Quantization Profile] Unquantized weights size on disk: {unquant_size_mb:.2f} MB", flush=True)

    # Perform PyTorch Dynamic INT8 Quantization
    print(f"[Quantization Profile] Quantizing model with torch.quantization.quantize_dynamic (qint8 on Linear layers)...", flush=True)
    t_q = time.time()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quant_time = time.time() - t_q
    post_quant_ram = get_mem_mb(proc)
    print(f"[Quantization Profile] Quantization completed in {quant_time:.2f}s. RAM: {post_quant_ram:.2f} MB", flush=True)

    # Save and measure quantized weights size
    quant_weights_path = os.path.join(scratch_dir, "quant_model.pt")
    torch.save(quantized_model.state_dict(), quant_weights_path)
    quant_size_mb = os.path.getsize(quant_weights_path) / (1024 * 1024)
    print(f"[Quantization Profile] Quantized weights size on disk: {quant_size_mb:.2f} MB (Reduction: {(1 - quant_size_mb / unquant_size_mb) * 100:.1f}%)", flush=True)

    # Benchmark single-sample inference latency and memory
    test_sample = "SentimentScope is working amazingly well!"
    inputs = tokenizer(test_sample, return_tensors="pt")

    # Unquantized latency
    unquant_latencies = []
    with torch.no_grad():
        for _ in range(50):
            t_s = time.perf_counter()
            _ = model(**inputs)
            unquant_latencies.append((time.perf_counter() - t_s) * 1000)

    # Quantized latency & memory during single inferences
    quant_latencies = []
    with torch.no_grad():
        for _ in range(50):
            t_s = time.perf_counter()
            _ = quantized_model(**inputs)
            quant_latencies.append((time.perf_counter() - t_s) * 1000)

    post_single_ram = get_mem_mb(proc)
    print(f"[Quantization Profile] Unquantized latency (p50): {np.percentile(unquant_latencies, 50):.2f} ms | mean: {np.mean(unquant_latencies):.2f} ms", flush=True)
    print(f"[Quantization Profile] Quantized latency (p50):   {np.percentile(quant_latencies, 50):.2f} ms | mean: {np.mean(quant_latencies):.2f} ms", flush=True)
    print(f"[Quantization Profile] RAM after 100 inferences: {post_single_ram:.2f} MB", flush=True)

    # Evaluate accuracy and Macro F1 of quantized model on test set
    if not os.path.exists(TEST_CSV):
        print(f"[Error] {TEST_CSV} not found!", flush=True)
        sys.exit(1)

    df_test = pd.read_csv(TEST_CSV)
    texts = df_test["text"].fillna("").tolist()
    labels = df_test["label"].tolist()

    print(f"[Quantization Profile] Evaluating quantized model on full {len(texts)} test samples...", flush=True)
    batch_size = 64
    all_preds_quant = []
    all_probs_quant = []
    t_eval = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            outputs = quantized_model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds_quant.extend(preds)
            all_probs_quant.extend(probs)

            if (i // batch_size) % 25 == 0 or (i + len(batch_texts)) == len(texts):
                pct = (i + len(batch_texts)) / len(texts) * 100
                print(f"  Processed {i + len(batch_texts):5d}/{len(texts)} ({pct:5.1f}%) in {time.time() - t_eval:5.1f}s", flush=True)

    peak_quant_ram = get_mem_mb(proc)
    print(f"[Quantization Profile] Quantized evaluation finished in {time.time() - t_eval:.2f}s. Peak RAM: {peak_quant_ram:.2f} MB", flush=True)

    all_preds_quant = np.array(all_preds_quant)
    all_probs_quant = np.array(all_probs_quant)
    labels = np.array(labels)

    acc = float(accuracy_score(labels, all_preds_quant))
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(labels, all_preds_quant, average="macro", zero_division=0)
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(labels, all_preds_quant, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(labels, all_preds_quant, labels=[0, 1, 2]).tolist()
    brier = compute_multiclass_brier_score(labels, all_probs_quant)

    # Clean up scratch files
    if os.path.exists(unquant_weights_path):
        os.remove(unquant_weights_path)
    if os.path.exists(quant_weights_path):
        os.remove(quant_weights_path)

    # Feasibility analysis on Render 512MB RAM:
    # On Linux/Docker, python + torch + transformers base footprint is ~220-280MB.
    # INT8 RoBERTa loaded weights: ~180MB.
    # Total static RSS: ~400-460MB.
    # Peak inference during request processing: +50-80MB -> 450-540MB.
    fits_512mb_render = bool(peak_quant_ram < 400.0) # Conservative threshold leaving 112MB for OS, Gunicorn, Pydantic, HTTP buffer

    profile_report = {
        "model_name": MODEL_NAME,
        "quantization_type": "Dynamic INT8 (PyTorch torch.quantization.quantize_dynamic on Linear layers)",
        "disk_footprint": {
            "unquantized_weights_mb": round(unquant_size_mb, 2),
            "quantized_weights_mb": round(quant_size_mb, 2),
            "reduction_percentage": round((1 - quant_size_mb / unquant_size_mb) * 100, 2)
        },
        "latency_benchmark_ms": {
            "unquantized": {
                "p50_ms": round(float(np.percentile(unquant_latencies, 50)), 2),
                "p95_ms": round(float(np.percentile(unquant_latencies, 95)), 2),
                "mean_ms": round(float(np.mean(unquant_latencies)), 2)
            },
            "quantized_int8": {
                "p50_ms": round(float(np.percentile(quant_latencies, 50)), 2),
                "p95_ms": round(float(np.percentile(quant_latencies, 95)), 2),
                "mean_ms": round(float(np.mean(quant_latencies)), 2)
            }
        },
        "memory_profile_mb": {
            "base_python_ram": round(base_ram, 2),
            "unquantized_loaded_ram": round(unquant_load_ram, 2),
            "post_quantization_ram": round(post_quant_ram, 2),
            "peak_inference_ram": round(peak_quant_ram, 2)
        },
        "quantized_metrics_full_test_set": {
            "test_samples": len(labels),
            "accuracy": round(acc, 4),
            "macro_f1": round(float(macro_f1), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_precision": round(float(macro_prec), 4),
            "brier_score": round(brier, 4),
            "negative_recall": round(float(r_class[0]), 4),
            "negative_f1": round(float(f1_class[0]), 4),
            "confusion_matrix": cm
        },
        "render_512mb_feasibility": {
            "safe_in_512mb_free_tier": fits_512mb_render,
            "measured_peak_ram_mb": round(peak_quant_ram, 2),
            "verdict": "Feasible for local/dedicated instances (>=1GB RAM), but remains high-risk on 512MB Render free tier due to OS and multi-request RAM bursts." if not fits_512mb_render else "Safe for 512MB deployment."
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(profile_report, f, indent=2)
    print(f"\n[Quantization Profile] Written report to {OUTPUT_JSON}")

    print("\n" + "="*70)
    print("QUANTIZATION IMPACT SUMMARY:")
    print(f"  Weights disk size:  {unquant_size_mb:.1f} MB -> {quant_size_mb:.1f} MB ({(1 - quant_size_mb / unquant_size_mb) * 100:.1f}% reduction)")
    print(f"  Latency (p50):      {np.percentile(unquant_latencies, 50):.2f} ms (FP32) -> {np.percentile(quant_latencies, 50):.2f} ms (INT8)")
    print(f"  Accuracy:           {acc * 100:.2f}%")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Negative Recall:    {r_class[0] * 100:.2f}%")
    print(f"  Peak RAM (Windows): {peak_quant_ram:.2f} MB")
    print(f"  512MB RAM Verdict:  {profile_report['render_512mb_feasibility']['verdict']}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
