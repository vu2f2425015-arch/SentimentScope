# SentimentScope Dual-Cloud Deployment Guide

This document outlines the step-by-step process to deploy **SentimentScope** across a dual-cloud production setup:
- **Frontend SPA**: Hosted as a high-performance static web application on **Vercel**.
- **Backend API & ML Inference Engine**: Hosted inside a Docker container on **Railway**.

---

## 🏗 Architecture Overview

```
+-------------------------------------------------------------------+
|                        User Browser / SPA                         |
+-------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| Vercel Global Edge CDN          | Railway PaaS Cloud Container    |
| - Hosts static public/index.html| - FastAPI REST Server (Uvicorn) |
| - Dynamic ENV_API_URL binding   | - Scikit-Learn / ML Inference   |
| - Zero serverless cold-starts   | - SQLite Volume Persistent Store|
+---------------------------------+---------------------------------+
```

### Key Technical Advantages
1. **Decoupled Architecture**: Frontend static assets load instantly via Vercel's global CDN; heavy ML inference runs in a dedicated Python container on Railway.
2. **Dynamic CORS Protection**: `CORSMiddleware` in `src/api.py` uses `allow_origin_regex=r"https://.*\.vercel\.app"` to dynamically authorize Vercel production and preview deployment URLs while respecting environment-configured `ALLOWED_ORIGINS`.
3. **Data Persistence**: Railway persistent storage volume attached at `/app/data` guarantees SQLite rolling predictions survive container restarts and redeployments.

---

## 🚆 1. Backend Deployment (Railway)

### Step 1: Create Service on Railway
1. Log in to [Railway.app](https://railway.app).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select your `SentimentScope` repository.

### Step 2: Explicit Dockerfile Builder Config
Railway automatically detects `railway.json` in the root directory:
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```
> **Note**: Railway assigns a dynamic `$PORT` variable at runtime. The root `Dockerfile` automatically executes `uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}`.

### Step 3: Configure Environment Variables
In Railway -> **Settings** -> **Variables**, add:
| Variable | Value | Description |
| :--- | :--- | :--- |
| `ALLOWED_ORIGINS` | `https://sentimentscope.vercel.app,http://localhost:3000` | Allowed CORS origins list |
| `INGESTION_DB_PATH` | `/app/data/rolling_store.db` | Path to persistent SQLite database |
| `ENABLE_LIVE_INGESTION` | `true` | Enables background telemetry ingestion loop |

### Step 4: Attach Persistent Volume
1. In the Railway dashboard service view, click **+ Add Volume**.
2. Mount path: `/app/data`.
3. This ensures `/app/data/rolling_store.db` persists across service restarts, container re-images, and code redeployments.

### Step 5: Configure Health Check Path
In Railway -> **Settings** -> **Deployments**:
- Set **Healthcheck Path** to `/health`.
- Railway will verify HTTP 200 OK from `/health` before marking deployments healthy.

---

## ⚡ 2. Frontend Deployment (Vercel)

### Step 1: Import Project to Vercel
1. Log in to [Vercel.com](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your `SentimentScope` GitHub repository.

### Step 2: Configure Vercel Project Settings
- **Framework Preset**: `Other`
- **Root Directory**: `./` (Root directory)
- **Output Directory**: `public` (Vercel serves `public/index.html` statically)
- **Serverless API Routes**: *None* (Vercel `vercel.json` contains static single-page rewrites only; all REST calls route directly to Railway).

### Step 3: Set Environment Variables
In Vercel Project Settings -> **Environment Variables**, add:
| Key | Value |
| :--- | :--- |
| `ENV_API_URL` | `https://your-railway-app.up.railway.app` |

---

## 🧠 3. Model Artifact Management

- **Baseline Models (`Logistic Regression`, `Multinomial NB`, `LSTM`)**: Pre-trained scikit-learn/Keras model weights are committed directly in Git under `models/` (< 65MB total). They are automatically built into the Docker container image on Railway.
- **Transformer Models (`DistilBERT` / `RoBERTa`)**: `.safetensors` files (> 260MB) are excluded by `.gitignore`. If a Transformer model is selected, `src/api.py` automatically downloads `distilbert-base-uncased` from Hugging Face Hub during server startup.

---

## ✅ 4. Comprehensive Verification Checklist

### 1. Railway API & Model Loading Verification
Execute `curl` against your Railway deployment domain:
```bash
curl -X GET "https://your-railway-app.up.railway.app/health"
```
**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Logistic Regression",
  "version": "1.0.0"
}
```

Verify interactive API documentation:
```
https://your-railway-app.up.railway.app/docs
```

### 2. CORS Verification from Vercel Frontend
Open your deployed Vercel frontend (`https://your-app.vercel.app`) in Chrome/Firefox Developer Tools:
1. Navigate to the **Network** tab.
2. Trigger a single prediction or test ping in the UI.
3. Check the response headers for the `/predict` or `/health` request:
   - `Access-Control-Allow-Origin: https://your-app.vercel.app`
   - `Access-Control-Allow-Credentials: true`
4. Confirm zero CORS policy blocking errors in the browser Console.

### 3. Persistent Storage Verification
1. Run a prediction or trigger live post ingestion on the deployed frontend.
2. In Railway Dashboard, click **Restart Service**.
3. After restart completes, load the frontend page.
4. Verify that historical rolling telemetry and live feed items remain present.
