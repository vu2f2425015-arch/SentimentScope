"""
SentimentScope - High-Throughput Offline Batch Inference Engine
Processes large CSV datasets (10,000 to 1,000,000+ rows) with streaming chunked execution,
vectorized TF-IDF inference, and low memory footprint (<150MB RAM).
"""

import os
import sys
import time
import argparse
import json
import joblib
import pandas as pd
import numpy as np

# Ensure root workspace is in sys.path
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.preprocessing import clean_text
from src.data_loader import REVERSE_LABEL_MAP

MODELS_DIR = os.getenv("MODELS_DIR", "models")


def load_model_artifacts():
    """Loads best model and vectorizer artifacts from models directory."""
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Model metadata not found at '{meta_path}'. Please run training first.")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    details = meta["details"]
    artifact_path = os.path.join(MODELS_DIR, details["artifact"])
    vectorizer_path = os.path.join(MODELS_DIR, details["vectorizer"])
    
    print(f"[*] Loading active model: '{meta.get('best_model_name')}' from '{artifact_path}'...")
    model = joblib.load(artifact_path)
    print(f"[*] Loading vectorizer from '{vectorizer_path}'...")
    vectorizer = joblib.load(vectorizer_path)
    
    return model, vectorizer, meta.get("best_model_name", "Logistic Regression")


def run_batch_pipeline(
    input_csv: str,
    output_csv: str,
    text_column: str = "text",
    chunk_size: int = 10000
):
    """
    Streams and processes a CSV file in chunks, writing predictions to output_csv.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file '{input_csv}' not found.")
    
    model, vectorizer, model_name = load_model_artifacts()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    start_time = time.time()
    total_processed = 0
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    print(f"\n" + "=" * 60)
    print(f"  SentimentScope Batch Processing Engine")
    print(f"  Input:      {input_csv}")
    print(f"  Output:     {output_csv}")
    print(f"  Chunk Size: {chunk_size:,} rows")
    print(f"=" * 60 + "\n")
    
    chunk_idx = 0
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        chunk_idx += 1
        c_start = time.time()
        
        if text_column not in chunk.columns:
            # Fallback: check case-insensitive or take first column
            candidates = [c for c in chunk.columns if c.lower() in ("text", "review", "sentence", "comment")]
            if candidates:
                actual_col = candidates[0]
            else:
                actual_col = chunk.columns[0]
        else:
            actual_col = text_column
        
        raw_texts = chunk[actual_col].fillna("").astype(str).tolist()
        cleaned_texts = [clean_text(t) for t in raw_texts]
        
        # Vectorized feature extraction & prediction
        features = vectorizer.transform(cleaned_texts)
        probs = model.predict_proba(features)
        pred_labels = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        sentiments = [REVERSE_LABEL_MAP[int(l)] for l in pred_labels]
        
        chunk["predicted_sentiment"] = sentiments
        chunk["confidence"] = np.round(confidences, 4)
        chunk["negative_prob"] = np.round(probs[:, 0], 4)
        chunk["neutral_prob"] = np.round(probs[:, 1], 4)
        chunk["positive_prob"] = np.round(probs[:, 2], 4)
        
        # Write to disk incrementally
        header = (chunk_idx == 1)
        chunk.to_csv(output_csv, mode="a", index=False, header=header)
        
        # Accumulate metrics
        n_rows = len(chunk)
        total_processed += n_rows
        pos_count += sentiments.count("positive")
        neg_count += sentiments.count("negative")
        neu_count += sentiments.count("neutral")
        
        elapsed = time.time() - start_time
        speed = int(total_processed / elapsed) if elapsed > 0 else 0
        c_elapsed = time.time() - c_start
        print(f"  [Chunk {chunk_idx:3d}] Processed {total_processed:>9,} rows ({speed:>6,} rows/sec) - Chunk time: {c_elapsed:.2f}s")
    
    total_elapsed = time.time() - start_time
    pos_pct = round((pos_count / total_processed) * 100, 2) if total_processed else 0.0
    neg_pct = round((neg_count / total_processed) * 100, 2) if total_processed else 0.0
    neu_pct = round((neu_count / total_processed) * 100, 2) if total_processed else 0.0
    
    print("\n" + "=" * 60)
    print("  BATCH PROCESSING COMPLETED SUCCESSFULLY")
    print(f"  Total Processed: {total_processed:,} rows in {total_elapsed:.2f}s ({int(total_processed/total_elapsed):,} rows/sec)")
    print(f"  Positive:        {pos_count:,} ({pos_pct}%)")
    print(f"  Neutral:         {neu_count:,} ({neu_pct}%)")
    print(f"  Negative:        {neg_count:,} ({neg_pct}%)")
    print(f"  Predictions:     {output_csv}")
    print("=" * 60 + "\n")
    
    return {
        "total_rows": total_processed,
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "elapsed_seconds": round(total_elapsed, 2),
        "output_file": output_csv
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentimentScope High-Throughput Batch Inference")
    parser.add_argument("--input", "-i", type=str, default="data/processed/test.csv", help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str, default="reports/batch_predictions_output.csv", help="Output CSV file path")
    parser.add_argument("--text-col", "-t", type=str, default="text", help="Name of text column")
    parser.add_argument("--chunksize", "-c", type=int, default=10000, help="Streaming chunk size (default: 10000)")
    
    args = parser.parse_args()
    run_batch_pipeline(
        input_csv=args.input,
        output_csv=args.output,
        text_column=args.text_col,
        chunk_size=args.chunksize
    )
