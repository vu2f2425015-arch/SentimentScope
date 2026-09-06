# SentimentScope Dual-Cloud Deployment Guide (Vercel + Render)

This document details the complete step-by-step deployment guide for **SentimentScope**:
- **Frontend SPA**: Hosted on **Vercel** (Global Static Edge CDN).
- **Backend API & ML Inference Engine**: Hosted inside a Docker container on **Render** (100% Free Web Service, no credit card required).

---

## 🏗 Architecture Overview

```
+-------------------------------------------------------------------+
|                        User Browser / SPA                         |
+-------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------+---------------------------------+
| Vercel Global Edge CDN          | Render Free Web Service         |
| - Static public/index.html SPA  | - FastAPI REST Server (Uvicorn) |
| - Dynamic ENV_API_URL binding   | - Scikit-Learn / ML Inference   |
| - Zero serverless cold-starts   | - Auto-detected render.yaml     |
+---------------------------------+---------------------------------+
```

### Key Technical Advantages
1. **100% Free Tier**: Render provides free Docker container web services without requiring a credit card or trial expiration.
2. **Decoupled Architecture**: Frontend static assets load instantly via Vercel's global CDN; heavy ML inference runs in a dedicated Python container on Render.
3. **Dynamic CORS Protection**: `CORSMiddleware` in `src/api.py` uses `allow_origin_regex=r"https://.*\.vercel\.app"` to dynamically authorize Vercel production and preview deployment URLs while respecting environment-configured `ALLOWED_ORIGINS`.

---

## 🚀 1. Backend Deployment (Render.com)

### Step 1: Create Blueprint Service on Render
1. Log in to [Render.com](https://render.com) (or sign up for free, no credit card needed).
2. Click **New +** -> **Blueprint**.
3. Connect your `SentimentScope` GitHub repository.
4. Render will automatically read [`render.yaml`](file:///c:/MY%20PROJECTS/SentimentScope/render.yaml):
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
        value: https://sentimentscope.vercel.app,http://localhost:3000
      - key: MODELS_DIR
        value: models
      - key: REPORTS_DIR
        value: reports
      - key: ENABLE_LIVE_INGESTION
        value: "true"
      - key: INGESTION_DB_PATH
        value: data/rolling_store.db
```
5. Click **Apply**. Render will automatically build the Docker container and deploy your FastAPI service!
6. Once deployed, copy your Render web service URL (e.g., `https://sentimentscope-api.onrender.com`).

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
- **Serverless API Routes**: *None* (Vercel `vercel.json` contains static single-page rewrites only; all REST calls route directly to Render).

### Step 3: Set Environment Variables
In Vercel Project Settings -> **Environment Variables**, add:
| Key | Value |
| :--- | :--- |
| `ENV_API_URL` | `https://sentimentscope-api.onrender.com` *(your Render web service URL)* |

---

## 🧠 3. Model Artifact Management

- **Baseline Models (`Logistic Regression`, `Multinomial NB`, `LSTM`)**: Pre-trained scikit-learn/Keras model weights are committed directly in Git under `models/` (< 65MB total). They are automatically built into the Docker container image on Render.
- **Transformer Models (`DistilBERT` / `RoBERTa`)**: `.safetensors` files (> 260MB) are excluded by `.gitignore`. If a Transformer model is selected, `src/api.py` automatically downloads `distilbert-base-uncased` from Hugging Face Hub during server startup.

---

## ✅ 4. Verification Checklist

### 1. Render API & Model Loading Verification
Execute `curl` against your Render deployment domain:
```bash
curl -X GET "https://sentimentscope-api.onrender.com/health"
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
https://sentimentscope-api.onrender.com/docs
```

### 2. CORS Verification from Vercel Frontend
Open your deployed Vercel frontend (`https://your-app.vercel.app`) in your browser:
1. Navigate to Developer Tools -> **Network** tab.
2. Trigger a single prediction or test ping.
3. Verify response header: `Access-Control-Allow-Origin: https://your-app.vercel.app`.
