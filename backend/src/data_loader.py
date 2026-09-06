import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text

RAW_DATA_PATH = os.path.join("data", "raw", "sentiment_dataset.csv")
PROCESSED_DIR = os.path.join("data", "processed")

# Label mappings
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
REVERSE_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
TWEETEVAL_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def fetch_tweeteval_benchmark_dataset() -> pd.DataFrame:
    """
    Downloads the official CardiffNLP TweetEval 3-class sentiment dataset from GitHub.
    Combines train, val, and test splits into a unified authentic dataset containing real-world human text.
    """
    base_url = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    print("[DataLoader] Fetching official CardiffNLP TweetEval 3-class sentiment benchmark dataset...", flush=True)
    
    records = []
    splits = [("train_text.txt", "train_labels.txt"), ("val_text.txt", "val_labels.txt"), ("test_text.txt", "test_labels.txt")]
    
    for text_file, label_file in splits:
        try:
            r_text = requests.get(base_url + text_file, headers=headers, timeout=30)
            r_label = requests.get(base_url + label_file, headers=headers, timeout=30)
            
            texts = r_text.text.splitlines()
            labels = [int(x.strip()) for x in r_label.text.splitlines() if x.strip() != ""]
            
            for t, l in zip(texts, labels):
                if t.strip() and l in TWEETEVAL_LABEL_MAP:
                    records.append({
                        "text": t.strip(),
                        "sentiment": TWEETEVAL_LABEL_MAP[l],
                        "label": l
                    })
        except Exception as e:
            print(f"[DataLoader] Warning fetching {text_file}: {e}", flush=True)

    df = pd.DataFrame(records)
    print(f"[DataLoader] Downloaded authentic TweetEval dataset: {len(df)} rows.", flush=True)
    return df


def load_and_split_data(raw_csv_path: str = RAW_DATA_PATH, save_processed: bool = True, force_download: bool = False, cap_15k: bool = False):
    """
    Loads raw CSV dataset, handles missing/malformed rows, preprocesses text,
    enforces strict deduplication on clean_text, and performs 70/15/15 stratified train/validation/test split.
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if force_download or not os.path.exists(raw_csv_path):
        print(f"[DataLoader] Downloading authentic TweetEval dataset to '{raw_csv_path}'...", flush=True)
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        df = fetch_tweeteval_benchmark_dataset()
        df.to_csv(raw_csv_path, index=False)
        print(f"[DataLoader] Saved authentic raw dataset with {len(df)} rows to '{raw_csv_path}'.", flush=True)
    else:
        print(f"[DataLoader] Loading raw dataset from '{raw_csv_path}'...", flush=True)
        df = pd.read_csv(raw_csv_path)

    # 1. Clean malformed / null / empty rows
    initial_len = len(df)
    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy()
    
    # Map sentiment string to numeric label if needed
    if "label" not in df.columns:
        df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
        df = df[df["sentiment"].isin(LABEL_MAP.keys())].copy()
        df["label"] = df["sentiment"].map(LABEL_MAP)
    
    print(f"[DataLoader] Cleaned raw dataset: {len(df)} valid rows (dropped {initial_len - len(df)} malformed rows).", flush=True)
    
    # 2. Preprocess text
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"] != ""].copy()

    # 3. STRICT DEDUPLICATION: Drop exact duplicates on clean_text
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    dedup_dropped = before_dedup - len(df)
    print(f"[DataLoader] Deduplication: {len(df)} unique clean texts remaining (dropped {dedup_dropped} duplicates).", flush=True)

    # 4. Optional 15k capping
    if cap_15k and len(df) > 15000:
        df, _ = train_test_split(df, train_size=15000, stratify=df["label"], random_state=42)
        df = df.reset_index(drop=True)
        print(f"[DataLoader] Sampled stratified subset of {len(df)} rows for balanced benchmark training.", flush=True)

    # 5. Stratified Train / Validation / Test Split (70 / 15 / 15)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )

    # Verify zero text overlap between splits
    train_set = set(train_df["clean_text"])
    val_set = set(val_df["clean_text"])
    test_set = set(test_df["clean_text"])
    
    tv_overlap = len(train_set.intersection(val_set))
    tt_overlap = len(train_set.intersection(test_set))
    vt_overlap = len(val_set.intersection(test_set))
    
    print(f"[DataLoader] Overlap Check -> Train/Val: {tv_overlap}, Train/Test: {tt_overlap}, Val/Test: {vt_overlap}.", flush=True)
    assert tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0, "Data leakage detected between splits!"

    print(f"[DataLoader] Stratified splits -> Train: {len(train_df)} (70%), Val: {len(val_df)} (15%), Test: {len(test_df)} (15%).", flush=True)

    if save_processed:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
        val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
        print(f"[DataLoader] Saved processed split CSVs to '{PROCESSED_DIR}'.", flush=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    load_and_split_data()
