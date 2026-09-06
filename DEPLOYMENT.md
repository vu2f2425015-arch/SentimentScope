# Complete Deployment Guide: Vercel (Full-Stack / Serverless) + Render (Standalone Backend)

This guide walks you through deploying **SentimentScope** with either:
1. **Option A: Full-Stack Vercel Deployment (Recommended)** — Deploy both the **Frontend UI** and **Python FastAPI Backend** as Vercel Serverless Functions in 1 click under the same domain.
2. **Option B: Divided Architecture** — Frontend on **Vercel** + Containerized Backend on **Render**.

---

## Option A: Full-Stack Vercel Deployment (Frontend + Python Serverless API)

### Key Advantages
- **Single Domain**: Zero CORS issues; API calls run on the same domain (`/api/predict`, `/health`).
- **Instant Cold Starts**: Fast serverless response powered by lightweight Scikit-Learn inference.
- **Zero Configuration**: Vercel automatically detects `api/index.py` and routes `/api/*`, `/health`, `/predict` serverless functions.

### Step-by-Step Vercel Deployment
1. Log in to [vercel.com](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository (`SentimentScope`).
4. In **Configure Project**:
   - **Framework Preset**: `Other` (leave as default)
   - **Root Directory**: `./` *(Leave root directory empty / root folder so Vercel picks up `api/` and `frontend/`)*
   - **Build & Output Settings**: Leave default
5. Click **Deploy**.

Vercel will deploy both the static frontend and Python serverless API. Your app will be live at:
```
https://sentimentscope.vercel.app
```
Health Check:
```
https://sentimentscope.vercel.app/health
```

---

## Option B: Deploy Backend to Render (Standalone Containerized Service)

If you prefer to run a dedicated backend process with Docker on Render:

### Step-by-Step Deployment
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository (`SentimentScope`).
4. Fill in settings:
   - **Name**: `sentimentscope`
   - **Branch**: `main`
   - **Root Directory**: `backend` *(Tells Render to build from backend/)*
   - **Runtime**: `Docker`
   - **Instance Type**: `Free`
5. Under **Advanced Settings**:
   - **Health Check Path**: `/health`
6. Click **Create Web Service**.

Backend URL:
```
https://sentimentscope-iz3a.onrender.com
```

---

## Connecting Frontend to Backend

The frontend features dynamic API resolution:
1. **Auto-Detection (Vercel Serverless)**:
   When deployed on Vercel, the frontend automatically connects to `window.location.origin` (the Vercel Serverless API).
2. **Interactive UI Configuration Modal**:
   - Click the **Health Status Badge** or **"Change Backend URL"** on the offline banner in the frontend dashboard.
   - Use quick preset buttons:
     - **Vercel API (Same Origin)**: `window.location.origin`
     - **Render Server**: `https://sentimentscope-iz3a.onrender.com`
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

### Issue 2: Render Service Offline / Sleeping
- Render free tier instances sleep after 15 minutes of inactivity.
- The first request wakes the service up (~30-50s).
- Alternatively, use the **Vercel Serverless API** option for instant responses without sleep timers.
