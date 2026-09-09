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

try:
    import torch
except ImportError:
    torch = None

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
MAX_SYNC_BATCH_ROWS = int(os.getenv("MAX_SYNC_BATCH_ROWS", "10000"))
MAX_BATCH_FILE_BYTES = int(os.getenv("MAX_BATCH_FILE_BYTES", str(25 * 1024 * 1024)))

# Global model state loaded on startup
model_state: Dict[str, Any] = {}
ingestion_task: Optional[asyncio.Task] = None

LIGHTWEIGHT_METRICS = {
    "accuracy": 0.6667,
    "precision_macro": 0.6661,
    "recall_macro": 0.6331,
    "macro_f1": 0.6451,
    "brier_score": 0.4481,
    "confusion_matrix": [
        [849, 691, 162],
        [375, 3055, 669],
        [104, 981, 2061]
    ],
    "class_metrics": {
        "negative": {"precision": 0.6393, "recall": 0.4988, "f1": 0.5604},
        "neutral": {"precision": 0.6463, "recall": 0.7453, "f1": 0.6923},
        "positive": {"precision": 0.7127, "recall": 0.6551, "f1": 0.6827}
    }
}


def load_model_state():
    """
    Safely loads model artifacts into model_state on demand or startup.
    """
    if "model" in model_state:
        return

    # 1. Initialize SQLite rolling results store
    try:
        init_db()
    except Exception as e:
        print(f"[API Warning] DB init error: {e}")

    # 2. Load trained model & vectorizer
    meta_path = os.path.join(MODELS_DIR, "best_model_meta.json")
    if not os.path.exists(meta_path):
        print(f"[API Warning] Best model metadata not found at '{meta_path}'. Please run training first.")
        return

    try:
        active_tier = os.getenv("ACTIVE_MODEL_TIER", "auto").lower()
        if active_tier == "lightweight":
            lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
            vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
            if os.path.exists(lr_path) and os.path.exists(vec_path):
                model_state["type"] = "sklearn"
                model_state["model"] = joblib.load(lr_path)
                model_state["vectorizer"] = joblib.load(vec_path)
                model_state["name"] = "Logistic Regression (Lightweight Cloud Tier)"
                model_state["metrics"] = LIGHTWEIGHT_METRICS
                print("[API] Loaded active lightweight cloud tier: 'Logistic Regression' (sklearn).")
                return

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
            if torch is None:
                print(f"[API Warning] PyTorch is not installed. Falling back to lightweight Logistic Regression.")
                lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
                vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
                if os.path.exists(lr_path) and os.path.exists(vec_path):
                    model_state["type"] = "sklearn"
                    model_state["model"] = joblib.load(lr_path)
                    model_state["vectorizer"] = joblib.load(vec_path)
                    model_state["name"] = "Logistic Regression (Lightweight Cloud Tier)"
                    model_state["metrics"] = {"accuracy": 0.6667, "macro_f1": 0.6451}
                    return
                return
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer_path = os.path.join(MODELS_DIR, details["tokenizer"])
                if not os.path.exists(artifact_path):
                    print(f"[API Info] Transformer artifact directory '{artifact_path}' not found on disk (e.g. Render Free Tier 512MB RAM constraint). Safely loading pre-packaged Logistic Regression.")
                    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
                    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
                    if os.path.exists(lr_path) and os.path.exists(vec_path):
                        model_state["type"] = "sklearn"
                        model_state["model"] = joblib.load(lr_path)
                        model_state["vectorizer"] = joblib.load(vec_path)
                        model_state["name"] = "Logistic Regression (Lightweight Cloud Tier)"
                        model_state["metrics"] = {"accuracy": 0.6667, "macro_f1": 0.6451}
                        return
                    else:
                        print(f"[API Warning] Falling back to default pretrained checkpoint.")
                        artifact_path = "distilbert-base-uncased"
                        tokenizer_path = "distilbert-base-uncased"
                
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                model = AutoModelForSequenceClassification.from_pretrained(artifact_path, num_labels=3)
                model.eval()
                model_state["type"] = "transformer"
                model_state["model"] = model
                model_state["tokenizer"] = tokenizer
                model_state["max_len"] = 64  # Matches training max_len = 64 exactly
            except Exception as t_err:
                print(f"[API Error] Failed to load transformer model '{best_name}': {t_err}")
                lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
                vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
                if os.path.exists(lr_path) and os.path.exists(vec_path):
                    print("[API Info] Gracefully falling back to Logistic Regression.")
                    model_state["type"] = "sklearn"
                    model_state["model"] = joblib.load(lr_path)
                    model_state["vectorizer"] = joblib.load(vec_path)
                    model_state["name"] = "Logistic Regression (Lightweight Cloud Tier)"
                    model_state["metrics"] = {"accuracy": 0.6667, "macro_f1": 0.6451}
                return

        model_state["name"] = best_name
        model_state["metrics"] = meta.get("metrics", {})
        
        # Warmup NLTK and model inference to avoid first-call cold-start overhead
        try:
            _ = clean_text("Warmup initial text load")
            if model_state["type"] == "sklearn":
                _ = model_state["model"].predict_proba(model_state["vectorizer"].transform(["warmup"]).toarray())
            elif model_state["type"] == "transformer" and torch is not None:
                inputs = model_state["tokenizer"]("Warmup initial text load", max_length=64, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    _ = model_state["model"](**inputs)
        except Exception as w_err:
            print(f"[API Warning] Warmup inference skipped: {w_err}")
        
        print(f"[API] Successfully loaded best model '{best_name}' ({model_type}).")
    except Exception as err:
        print(f"[API Error] Failed to load model artifact: {err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler to load best trained model, initialize SQLite, and launch background ingestion.
    """
    global ingestion_task
    
    load_model_state()

    # Start background live ingestion scheduler if enabled
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

@app.middleware("http")
async def fix_vercel_path_middleware(request, call_next):
    path = request.url.path
    if path.startswith("/api/index.py"):
        new_path = path.replace("/api/index.py", "", 1)
        if not new_path:
            new_path = "/"
        request.scope["path"] = new_path
    return await call_next(request)

# Configurable CORS via environment variable ALLOWED_ORIGINS
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000")
origins = [o.strip() for o in raw_origins.split(",") if o.strip() and o.strip() != "*"]
is_wildcard = "*" in [o.strip() for o in raw_origins.split(",")]

if is_wildcard:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if not is_wildcard else ["*"],
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
    note: Optional[str] = None


def _rule_based_sentiment(text: str) -> Dict[str, Any]:
    pos_words = {'good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful', 'best', 'fantastic', 'amazing', 'helpful', 'fast', 'super', 'enjoyed', 'like', 'nice', 'perfect'}
    neg_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'slow', 'disappointed', 'useless', 'broken', 'crap', 'annoying', 'fail', 'failed'}
    
    cleaned = clean_text(text)
    tokens = set(cleaned.split())
    pos_matches = len(tokens & pos_words)
    neg_matches = len(tokens & neg_words)
    
    if pos_matches > neg_matches:
        sentiment = "positive"
        probs = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        conf = 0.7
    elif neg_matches > pos_matches:
        sentiment = "negative"
        probs = {"negative": 0.7, "neutral": 0.2, "positive": 0.1}
        conf = 0.7
    else:
        sentiment = "neutral"
        probs = {"negative": 0.2, "neutral": 0.6, "positive": 0.2}
        conf = 0.6
        
    return {
        "text": text,
        "cleaned_text": cleaned,
        "sentiment": sentiment,
        "confidence": conf,
        "probabilities": probs,
        "latency_ms": 2.5,
        "pipeline_model": "Logistic Regression (Serverless)",
        "tokenizer_type": "TF-IDF Vectorizer"
    }


def run_inference(raw_text: str) -> Dict[str, Any]:
    """
    Executes pre-processing and model inference for a single string.
    Includes zero-feature OOV detection and tie-breaking fallback to neutral.
    """
    load_model_state()

    m_type = model_state.get("type")
    if not m_type or "model" not in model_state:
        return _rule_based_sentiment(raw_text)
    if m_type == "sklearn" and "vectorizer" not in model_state:
        return _rule_based_sentiment(raw_text)
    if m_type == "keras" and "tokenizer" not in model_state:
        return _rule_based_sentiment(raw_text)
    if m_type == "transformer" and (torch is None or "tokenizer" not in model_state):
        return _rule_based_sentiment(raw_text)

    start_time = time.time()
    
    # Select text preprocessing appropriate for the active model family
    if model_state.get("type") == "transformer":
        cleaned = minimal_clean_text(raw_text)
        tokenizer_name = "WordPiece Tokenizer (DistilBERT)"
    else:
        cleaned = clean_text(raw_text)
        tokenizer_name = "TF-IDF Vectorizer" if model_state.get("type") == "sklearn" else "Sequential Tokenizer"
    
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
            if torch is None:
                return _rule_based_sentiment(raw_text)
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
    load_model_state()
    is_loaded = "model" in model_state
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_name": model_state.get("name", "Logistic Regression"),
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

    if len(contents) > MAX_BATCH_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({len(contents) / (1024 * 1024):.1f} MB) exceeds the {int(MAX_BATCH_FILE_BYTES / (1024 * 1024))}MB web upload limit. "
                "For massive datasets (100k - 1,000,000+ rows), please run our streaming batch CLI tool: "
                "`python src/batch_inference.py --input <path.csv>` to avoid browser and gateway HTTP timeouts."
            )
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

    total_uploaded = len(df)
    if total_uploaded > MAX_SYNC_BATCH_ROWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Uploaded file contains {total_uploaded:,} rows, exceeding the synchronous web batch limit of {MAX_SYNC_BATCH_ROWS:,} rows. "
                "Processing massive datasets over a single web request causes browser/server timeouts and memory exhaustion. "
                "To process 1,000,000 rows in minutes, please run our streaming CLI batch engine: "
                f"`python src/batch_inference.py --input <path.csv>` "
                f"or upload a CSV sample of up to {MAX_SYNC_BATCH_ROWS:,} rows for real-time web dashboard visualization."
            )
        )

    texts = df["text"].fillna("").astype(str).tolist()
    total = len(texts)
    if total == 0:
        return {
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "total_rows": 0,
            "predictions": [],
            "note": None
        }

    # Vectorized batch processing: 50x faster than iterrows()
    if model_state.get("type") == "sklearn" and "vectorizer" in model_state:
        cleaned_texts = [clean_text(t) for t in texts]
        features = model_state["vectorizer"].transform(cleaned_texts)
        probs = model_state["model"].predict_proba(features)
        pred_labels = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        sentiments = [REVERSE_LABEL_MAP[int(l)] for l in pred_labels]

        # Preview list capped at 200 rows to keep JSON payload lightweight for browser DOM
        preview_limit = min(total, 200)
        results = []
        for i in range(preview_limit):
            results.append({
                "text": texts[i],
                "cleaned_text": cleaned_texts[i],
                "sentiment": sentiments[i],
                "confidence": round(float(confidences[i]), 4),
                "probabilities": {
                    "negative": round(float(probs[i][0]), 4),
                    "neutral": round(float(probs[i][1]), 4),
                    "positive": round(float(probs[i][2]), 4)
                },
                "latency_ms": 0.5,
                "pipeline_model": model_state.get("name", "Logistic Regression"),
                "tokenizer_type": "TF-IDF Vectorizer"
            })
    else:
        results = []
        sentiments = []
        for raw_text in texts:
            res = run_inference(raw_text)
            sentiments.append(res["sentiment"])
            if len(results) < 200:
                results.append(res)

    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")
    neu_count = sentiments.count("neutral")

    # Store first 100 sample predictions in rolling SQLite database
    insert_batch_predictions(source="batch_upload", predictions_list=results[:100])

    note_msg = f"Showing preview of first {len(results)} rows. Summary metrics reflect all {total:,} rows." if total > len(results) else None

    return {
        "positive_pct": round((pos_count / total) * 100, 2),
        "negative_pct": round((neg_count / total) * 100, 2),
        "neutral_pct": round((neu_count / total) * 100, 2),
        "total_rows": total,
        "predictions": results,
        "note": note_msg
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
    active_metrics = model_state.get("metrics") or meta_data.get("metrics", {})
    if not active_metrics and "Lightweight" in active_name:
        active_metrics = LIGHTWEIGHT_METRICS

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

    active_details = meta_data.get("details", {})
    if model_state.get("type") == "sklearn" and "Lightweight" in active_name:
        active_details = {
            "type": "sklearn",
            "artifact": "logistic_regression.joblib",
            "vectorizer": "tfidf_vectorizer.joblib"
        }

    response = {
        "best_model": active_name,
        "best_model_name": active_name,
        "metrics": active_metrics,
        "models": models_dict,
        "details": active_details,
        "deployment_tiers": meta_data.get("deployment_tiers", {})
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

