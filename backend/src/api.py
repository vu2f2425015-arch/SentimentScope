import os
import io
import json
import joblib
import time
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.preprocessing import clean_text, minimal_clean_text
from src.features import TFIDFExtractor, SequentialExtractor
import asyncio
from src.data_loader import REVERSE_LABEL_MAP
from src.db import init_db, insert_batch_predictions, get_recent_predictions, get_rolling_stats
from src.live_ingestion import (
    process_live_ingestion,
    background_ingestion_loop,
    stop_background_scheduler,
    validate_keyword
)

MODELS_DIR = os.getenv("MODELS_DIR", "models")
REPORTS_DIR = os.getenv("REPORTS_DIR", "reports")
ENABLE_LIVE_INGESTION = os.getenv("ENABLE_LIVE_INGESTION", "true").lower() in ("true", "1", "yes")
PUBLIC_DIR = "public"

# Global model state loaded on startup
model_state: Dict[str, Any] = {}
ingestion_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler to load best trained model, initialize SQLite, and launch background ingestion.
    """
    global ingestion_task
    
    # 1. Initialize SQLite rolling results store
    init_db()
    
    # 2. Load trained model & vectorizer
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    if not os.path.exists(meta_path):
        print(f"[API Warning] Best model metadata not found at '{meta_path}'. Please run training first.")
    else:
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        best_name = meta["best_model_name"]
        details = meta["details"]
        model_type = details["type"]
        
        artifact_path = os.path.join(MODELS_DIR, details["artifact"])
        if model_type == "sklearn":
            model = joblib.load(artifact_path)
            vectorizer_path = os.path.join(MODELS_DIR, details["vectorizer"])
            vectorizer = joblib.load(vectorizer_path)
            model_state["type"] = "sklearn"
            model_state["model"] = model
            model_state["vectorizer"] = vectorizer
        elif model_type == "keras":
            import tensorflow as tf
            model = tf.keras.models.load_model(artifact_path)
            tokenizer_path = os.path.join(MODELS_DIR, details["tokenizer"])
            seq_extractor = SequentialExtractor.load(tokenizer_path, max_len=150)
            model_state["type"] = "keras"
            model_state["model"] = model
            model_state["tokenizer"] = seq_extractor
        elif model_type == "transformer":
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer_path = os.path.join(MODELS_DIR, details["tokenizer"])
            if not os.path.exists(artifact_path):
                print(f"[API Warning] Transformer artifact directory '{artifact_path}' not found. Falling back to default pretrained checkpoint.")
                artifact_path = "distilbert-base-uncased"
                tokenizer_path = "distilbert-base-uncased"
            
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForSequenceClassification.from_pretrained(artifact_path, num_labels=3)
            model.eval()
            model_state["type"] = "transformer"
            model_state["model"] = model
            model_state["tokenizer"] = tokenizer
            model_state["max_len"] = 64  # Matches training max_len = 64 exactly

        model_state["name"] = best_name
        model_state["metrics"] = meta.get("metrics", {})
        
        # Warmup NLTK and model inference to avoid first-call cold-start overhead
        _ = clean_text("Warmup initial text load")
        if model_state["type"] == "sklearn":
            _ = model_state["model"].predict_proba(model_state["vectorizer"].transform(["warmup"]).toarray())
        elif model_state["type"] == "transformer":
            import torch
            inputs = model_state["tokenizer"]("Warmup initial text load", max_length=64, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                _ = model_state["model"](**inputs)
        
        print(f"[API] Successfully loaded best model '{best_name}' ({model_type}).")

    # 3. Start background live ingestion scheduler if enabled
    if ENABLE_LIVE_INGESTION:
        ingestion_task = asyncio.create_task(background_ingestion_loop())
        print("[API] Background live ingestion scheduler started.")
        
    yield
    
    if ingestion_task:
        stop_background_scheduler()
        ingestion_task.cancel()
        
    model_state.clear()



app = FastAPI(
    title="SentimentScope API",
    description="Production REST API for real-time and batch 3-class sentiment analysis.",
    version="1.0.0",
    lifespan=lifespan
)

# Configurable CORS via environment variable ALLOWED_ORIGINS
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000,*")
is_wildcard = "*" in [o.strip() for o in raw_origins.split(",")]

if is_wildcard:
    origins = ["*"]
else:
    origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False if is_wildcard else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for same-origin frontend serving
if os.path.exists(PUBLIC_DIR):
    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


class PredictRequest(BaseModel):
    text: str = Field(..., example="This product works exceptionally well! Highly recommended.")


class PredictResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    pipeline_model: str
    tokenizer_type: str


class BatchSummary(BaseModel):
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    total_rows: int
    predictions: List[Dict[str, Any]]


def run_inference(raw_text: str) -> Dict[str, Any]:
    """
    Executes pre-processing and model inference for a single string.
    Includes zero-feature OOV detection and tie-breaking fallback to neutral.
    """
    if "model" not in model_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model has not been loaded. Please ensure training has completed."
        )

    start_time = time.time()
    
    # Select text preprocessing appropriate for the active model family
    if model_state["type"] == "transformer":
        cleaned = minimal_clean_text(raw_text)
        tokenizer_name = "WordPiece Tokenizer (DistilBERT)"
    else:
        cleaned = clean_text(raw_text)
        tokenizer_name = "TF-IDF Vectorizer" if model_state["type"] == "sklearn" else "Sequential Tokenizer"
    
    nnz_count = 0
    feat_sum = 0.0
    is_oov = False
    
    # If text becomes empty after cleaning, handle gracefully as neutral fallback
    if not cleaned:
        probs = np.array([0.3333, 0.3334, 0.3333])
        pred_label = 1 # Neutral fallback
        is_oov = True
    else:
        if model_state["type"] == "sklearn":
            features = model_state["vectorizer"].transform([cleaned]).toarray()
            nnz_count = int(np.count_nonzero(features))
            feat_sum = round(float(np.sum(features)), 4)
            
            if nnz_count == 0:
                # All-zero feature vector (Out-Of-Vocabulary input)
                probs = np.array([0.3333, 0.3334, 0.3333])
                pred_label = 1  # Neutral fallback for uninformative text
                is_oov = True
            else:
                probs = model_state["model"].predict_proba(features)[0]
                if (np.max(probs) - np.min(probs)) < 0.005:
                    pred_label = 1  # Neutral fallback for tie / non-discriminating
                else:
                    pred_label = int(np.argmax(probs))
        elif model_state["type"] == "keras":
            seq = model_state["tokenizer"].transform([cleaned])
            nnz_count = int(np.count_nonzero(seq))
            feat_sum = float(np.sum(seq))
            if nnz_count == 0:
                probs = np.array([0.3333, 0.3334, 0.3333])
                pred_label = 1
                is_oov = True
            else:
                probs = model_state["model"].predict(seq, verbose=0)[0]
                if (np.max(probs) - np.min(probs)) < 0.005:
                    pred_label = 1
                else:
                    pred_label = int(np.argmax(probs))
        elif model_state["type"] == "transformer":
            tokenizer = model_state["tokenizer"]
            model = model_state["model"]
            max_len = model_state.get("max_len", 64)
            inputs = tokenizer(cleaned, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                probs_tensor = torch.softmax(outputs.logits, dim=-1).squeeze(0)
                probs = probs_tensor.cpu().numpy()
            
            nnz_count = int(inputs["input_ids"].shape[1])
            feat_sum = float(torch.sum(inputs["input_ids"]).item())
            
            if (np.max(probs) - np.min(probs)) < 0.005:
                pred_label = 1  # Neutral fallback
            else:
                pred_label = int(np.argmax(probs))

    confidence = float(np.max(probs))
    sentiment_str = REVERSE_LABEL_MAP[pred_label]
    model_name = model_state.get("name", "Unknown Model")
    
    probabilities_dict = {
        "negative": round(float(probs[0]), 4),
        "neutral": round(float(probs[1]), 4),
        "positive": round(float(probs[2]), 4)
    }
    
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    # Diagnostic print for debugging feature extraction & probabilities
    print(f"[Inference Debug] Input: '{raw_text}' | Cleaned: '{cleaned}' | Model: {model_name} | Probs: {probabilities_dict} -> Sentiment: {sentiment_str}")
    
    return {
        "text": raw_text,
        "cleaned_text": cleaned,
        "sentiment": sentiment_str,
        "confidence": round(confidence, 4),
        "probabilities": probabilities_dict,
        "latency_ms": latency_ms,
        "pipeline_model": model_name,
        "tokenizer_type": tokenizer_name
    }




@app.get("/health")
@app.get("/api/health")
def health():
    """
    Health check endpoint returning service status, model state, and version.
    """
    is_loaded = "model" in model_state
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_name": model_state.get("name", None),
        "version": "1.0.0"
    }


@app.get("/", include_in_schema=False)
def root():
    """
    Serves the frontend SPA index.html or API status fallback.
    """
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(
            index_path,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return {
        "status": "online",
        "app": "SentimentScope API",
        "version": "1.0.0",
        "loaded_model": model_state.get("name", "None loaded"),
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict"
    }



@app.post("/predict", response_model=PredictResponse)
@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """
    Accepts text input and returns 3-class sentiment prediction, confidence score, and class probabilities.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty or whitespace only."
        )
        
    return run_inference(payload.text)


@app.post("/predict/batch", response_model=BatchSummary)
@app.post("/api/predict/batch", response_model=BatchSummary)
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts a CSV file upload, processes each text row, and returns per-row predictions and aggregate summary.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a CSV file."
        )
        
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file is empty."
        )

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}"
        )

    if "text" not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file must contain a 'text' column."
        )

    results = []
    sentiments = []
    
    for idx, row in df.iterrows():
        raw_text = str(row["text"]) if pd.notna(row["text"]) else ""
        res = run_inference(raw_text)
        results.append(res)
        sentiments.append(res["sentiment"])

    total = len(sentiments)
    if total == 0:
        return {
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "total_rows": 0,
            "predictions": []
        }

    # Store batch predictions in rolling SQLite database tagged as batch_upload
    insert_batch_predictions(source="batch_upload", predictions_list=results)

    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")
    neu_count = sentiments.count("neutral")

    return {
        "positive_pct": round((pos_count / total) * 100, 2),
        "negative_pct": round((neg_count / total) * 100, 2),
        "neutral_pct": round((neu_count / total) * 100, 2),
        "total_rows": total,
        "predictions": results
    }


@app.get("/live/feed")
@app.get("/api/live/feed")
def get_live_feed(limit: int = 50, source: Optional[str] = "live"):
    """
    Returns recent predictions from the rolling SQLite store.
    Defaults to source='live' so batch uploads don't clutter the live feed.
    """
    return get_recent_predictions(limit=min(limit, 200), source=source)


@app.get("/live/stats")
@app.get("/api/live/stats")
def get_live_stats(source: Optional[str] = "live"):
    """
    Returns rolling sentiment statistics and retention pruning telemetry (oldest record timestamp & rows pruned last cycle).
    Defaults to source='live'.
    """
    return get_rolling_stats(source=source)


@app.post("/live/trigger")
@app.post("/api/live/trigger")
def trigger_live_ingestion(keyword: Optional[str] = "ai"):
    """
    Manually triggers live post ingestion for a given keyword.
    Enforces validation and per-keyword 30-second cooldown.
    """
    try:
        clean_kw = validate_keyword(keyword or "ai")
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )

    res = process_live_ingestion(keyword=clean_kw)
    if res.get("status") == "cooldown":
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=res["message"]
        )
    return res


@app.get("/model/metrics")
@app.get("/api/model/metrics")
def get_model_metrics():
    """
    Returns stored evaluation metrics for the currently loaded model and model comparisons.
    """
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    trans_report_path = os.path.join(REPORTS_DIR, "full_dataset_transformer_comparison.json")
    report_path = os.path.join(REPORTS_DIR, "model_comparison.json")
    
    meta_data = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta_data = json.load(f)

    active_name = model_state.get("name") or meta_data.get("best_model_name", "DistilBERT (Full Dataset)")
    active_metrics = meta_data.get("metrics", {})

    models_dict = {}
    if os.path.exists(trans_report_path):
        with open(trans_report_path, "r") as f:
            t_data = json.load(f)
            models_dict.update(t_data.get("all_models_held_out_test_comparison", {}))
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            c_data = json.load(f)
            for k, v in c_data.get("models", {}).items():
                if k not in models_dict:
                    models_dict[k] = v

    if active_name not in models_dict and active_metrics:
        models_dict[active_name] = active_metrics

    response = {
        "best_model": active_name,
        "best_model_name": active_name,
        "metrics": active_metrics,
        "models": models_dict,
        "details": meta_data.get("details", {})
    }

    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            c_data = json.load(f)
            for key in ["winning_tfidf_params", "winning_lr_params", "calibration_eval"]:
                if key in c_data:
                    response[key] = c_data[key]

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api:app", host="127.0.0.1", port=port, reload=True)

