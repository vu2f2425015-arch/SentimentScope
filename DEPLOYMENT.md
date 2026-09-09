# SentimentScope Production Deployment Guide (Render.com)

This document details the deployment guide for **SentimentScope** hosted on **Render** (100% Free Web Service with Docker).

---

## 🏗 Architecture Overview

SentimentScope runs as a unified, full-stack application inside a single Docker container on Render:
- **Interactive Web SPA**: Served directly at the root URL (`/`) from `public/index.html`.
- **FastAPI REST Server**: High-performance asynchronous backend running via Uvicorn.
- **Machine Learning Inference**: Pre-loaded scikit-learn models (Logistic Regression, MNB, Bi-LSTM) and Hugging Face Transformers.
- **Interactive Documentation**: Swagger UI at `/docs` and ReDoc at `/redoc`.

```
+-------------------------------------------------------------------+
|                        User Browser / SPA                         |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Render Cloud Web Service                      |
|                  https://<app-name>.onrender.com                  |
|                                                                   |
| - Root (GET /)            -> Serves public/index.html SPA         |
| - Health (GET /health)    -> Service uptime & model telemetry     |
| - Predict (POST /predict) -> Real-time & batch sentiment analysis |
| - Docs (GET /docs)        -> Interactive OpenAPI Swagger UI       |
+-------------------------------------------------------------------+
```

---

## 📁 Repository Layout & Source of Truth

The repository maintains a clean, single-source-of-truth structure:

| Component | Canonical Location | Deployment / Serving Target | Description |
| :--- | :--- | :--- | :--- |
| **Backend API & ML Engine** | `src/api.py` | Render Web Service (`Dockerfile`) | FastAPI REST API containerized with Uvicorn, scikit-learn, PyTorch, and live ingestion. |
| **Frontend Web SPA** | `public/index.html` | Render Container (`/`) | Single-page reactive dashboard with dynamic charts and live particle telemetry. |
| **UI Reference Components** | `references/DotField/` | Source Archive | Reference React/TSX components preserved for future Next.js/React migrations. |
| **Reports & Evaluation** | `reports/` | Local & Artifacts | Evaluation metrics, plots, JSON logs, and `PROJECT_REPORT.docx`. |

---

## 🚀 Render Deployment Instructions

### Step 1: Connect Repository to Render
1. Log in to [Render.com](https://render.com) (free tier, no credit card required).
2. Click **New +** -> **Blueprint**.
3. Connect your `SentimentScope` GitHub repository.
4. Render will automatically detect [`render.yaml`](render.yaml):

```yaml
services:
  - type: web
    name: sentimentscope-api
    env: docker
    plan: free
    region: oregon
    healthCheckPath: /health
    envVars:
      - key: ALLOWED_ORIGINS
        value: http://localhost:3000,http://localhost:8000
      - key: MODELS_DIR
        value: models
      - key: REPORTS_DIR
        value: reports
      - key: ENABLE_LIVE_INGESTION
        value: "true"
      - key: INGESTION_DB_PATH
        value: data/rolling_store.db
      - key: MAX_SYNC_BATCH_ROWS
        value: "50000"
      - key: MAX_BATCH_FILE_BYTES
        value: "36700160"
      - key: ACTIVE_MODEL_TIER
        value: "lightweight"
```

5. Click **Apply**. Render will build the Docker container and deploy the service.
6. Once deployed, your web application and API are live at your Render URL (e.g., `https://sentimentscope-api-nj7l.onrender.com/`).

---

## 🧠 Model Artifact Management & Memory Budget

- **Dual-Tier Production Architecture (Option B)**:
  - **Tier 1 (High-Accuracy Engine)**: `DistilBERT (Full Dataset)` (72.57% Acc / 0.7221 Macro F1) is selected by default in `models/best_model_meta.json` for local runs, Docker Compose, or dedicated compute (`>=1GB RAM`).
  - **Tier 2 (Zero-OOM Cloud Edge Tier)**: `Logistic Regression` (66.67% Acc / 0.6451 Macro F1, <1ms latency, ~50MB RAM) is pinned on Render Free Tier via `ACTIVE_MODEL_TIER="lightweight"` to prevent OOM termination (`Exit 137`).
- **Baseline Models (`Logistic Regression`, `Multinomial NB`, `LSTM`)**: Pre-trained scikit-learn/Keras model weights are committed directly in Git under `models/` (< 65MB total). They are automatically built into the Docker container image on Render.
- **Transformer Models (`DistilBERT` / `RoBERTa`)**: Large `.safetensors` files (> 260MB) are excluded by `.gitignore`. For dedicated/local environments running DistilBERT, artifacts reside in `models/distilbert_transformer/`.
- **Render Free Tier Memory Safety (512MB RAM Limit)**:
  - Render's Free Web Service plan allocates **512MB RAM**.
  - With `ACTIVE_MODEL_TIER="lightweight"`, the container operates at **~50MB RSS**, guaranteeing 100% crash-free uptime even under heavy concurrent traffic.
  - To run DistilBERT directly in cloud production, upgrade Render to **Starter (2GB RAM)** and set `ACTIVE_MODEL_TIER="auto"`.

---

## ✅ Verification Checklist

### 1. Web Application Verification
Open your deployed Render URL in any browser:
```
https://sentimentscope-api-nj7l.onrender.com/
```
**Expected**: The interactive dark-mode dashboard loads immediately with live particle mesh animation, test analyzer, batch upload, and model telemetry.

### 2. Service Health Check
Execute `curl` against your Render deployment:
```bash
curl -X GET "https://sentimentscope-api-nj7l.onrender.com/health"
```
**Expected Response (Render Free Edge Tier)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Logistic Regression (Lightweight Cloud Tier)",
  "version": "1.0.0"
}
```

*(Note: Local / Dedicated instances with `ACTIVE_MODEL_TIER="auto"` will return `"model_name": "DistilBERT (Full Dataset)"`.)*
```

### 3. Interactive API Documentation
```
https://sentimentscope-api-nj7l.onrender.com/docs
```
OpenAPI / Swagger test console for running predictions directly in the browser.
