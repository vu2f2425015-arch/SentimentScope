# Complete Deployment Guide: Vercel Full-Stack Serverless Platform

This guide walks you through deploying **SentimentScope** as a unified full-stack application on **Vercel** (Frontend UI + Python FastAPI Serverless API).

---

## Vercel Full-Stack Serverless Deployment

### Architecture Overview
```
                         +---------------------------------------------+
                         |           User Browser / Mobile             |
                         +---------------------------------------------+
                                        |
                                        v
                         +---------------------------------------------+
                         |               Vercel Edge                   |
                         |  +--------------------+ +-----------------+  |
                         |  | Static Frontend UI | | Python FastAPI  |  |
                         |  | (HTML, JS, CSS)    | | Serverless API  |  |
                         |  +--------------------+ +-----------------+  |
                         +---------------------------------------------+
```

### Key Advantages
- **Single Domain**: Zero CORS issues; API calls run on the same domain (`/api/predict`, `/health`).
- **Instant Responses**: Fast serverless response powered by lightweight Scikit-Learn inference.
- **Zero Configuration**: Vercel automatically detects `api/index.py` and routes `/api/*`, `/health`, `/predict` serverless functions.

### Step-by-Step Vercel Deployment
1. Log in to [vercel.com](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository (`SentimentScope`).
4. In **Configure Project**:
   - **Framework Preset**: `Other` (default)
   - **Root Directory**: `./` *(Leave root directory empty / root folder)*
   - **Build & Output Settings**: Leave default
5. Click **Deploy**.

Vercel will deploy both the static frontend and Python serverless API. Your app will be live at:
```
https://sentimentscope.vercel.app
```

Verify status by opening the health endpoint:
```
https://sentimentscope.vercel.app/health
```
Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Logistic Regression",
  "version": "1.0.0"
}
```

---

## Connecting Frontend to Backend

The frontend features dynamic API resolution:
1. **Auto-Detection (Vercel Serverless)**:
   When deployed on Vercel, the frontend automatically connects to `window.location.origin` (the Vercel Serverless API).
2. **Interactive UI Configuration Modal**:
   - Click the **Health Status Badge** or **"Change Backend URL"** on the offline banner in the frontend dashboard.
   - Presets provided:
     - **Vercel API (Same Origin)**: `window.location.origin`
     - **Localhost**: `http://127.0.0.1:8000`
   - Click **Test Ping** -> **Save & Connect**. Settings persist in `localStorage`.

---

## Local Development

### Terminal 1: Backend API
```bash
cd backend
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: Frontend
```bash
cd frontend
npx serve -l 3000 .
# Or: python -m http.server 3000
```
Open `http://localhost:3000` in your browser. The frontend auto-connects to `http://127.0.0.1:8000`.

---

## Troubleshooting Common Issues

### Issue 1: Vercel Serverless Function Exceeds Size Limit (250 MB)
- **Fix Applied**: `api/requirements.txt` is pre-configured with lightweight ML dependencies (`scikit-learn`, `numpy`, `pandas`, `fastapi`, `joblib`), keeping function size under ~40 MB.
