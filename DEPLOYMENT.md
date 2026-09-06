# Complete Deployment Guide: Vercel (Frontend) + Render (Backend)

This guide walks you through deploying **SentimentScope** with a divided architecture:
- **Frontend** deployed on **Vercel** (Global Edge CDN, zero cold-starts, high-speed static delivery).
- **Backend API** deployed on **Render** (Containerized FastAPI service with RAM-optimized ML inference).

---

## Architecture Overview

```
                        +---------------------------------------------+
                        |           User Browser / Mobile             |
                        +---------------------------------------------+
                                       |              |
                    Frontend Assets    |              | REST API Requests
                    (HTML, JS, CSS)    |              | (CORS Enabled)
                                       v              v
                        +-------------------+  +----------------------+
                        |      Vercel       |  |        Render        |
                        | (frontend/ folder)|  |   (backend/ folder)  |
                        |   Global Edge     |  |   FastAPI + Scikit   |
                        +-------------------+  +----------------------+
```

---

## Part 1: Deploy Backend to Render

### Prerequisites
- A free account on [Render.com](https://render.com).
- Your GitHub repository pushed with the latest code.

### Step-by-Step Deployment
1. Log in to [dashboard.render.com](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository (`SentimentScope`).
4. Fill in the settings:
   - **Name**: `sentimentscope` (or any custom name)
   - **Region**: Choose closest to you (e.g., Oregon, Frankfurt, Singapore)
   - **Branch**: `main`
   - **Root Directory**: `backend` *(Important: this tells Render to build from the backend folder)*
   - **Runtime / Environment**: `Docker`
   - **Instance Type**: `Free` (0.1 CPU, 512 MB RAM)
5. Under **Advanced Settings**:
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: `Yes`
6. Click **Create Web Service**.

Render will build the Docker container and deploy the service. Once deployed, Render will display your public backend URL:
```
https://<your-service-name>.onrender.com
```
*(Example: `https://sentimentscope-iz3a.onrender.com`)*

Verify deployment by opening `https://<your-service-name>.onrender.com/health` in your browser. You should receive:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Logistic Regression",
  "version": "1.0.0"
}
```

---

## Part 2: Deploy Frontend to Vercel

### Prerequisites
- A free account on [Vercel.com](https://vercel.com).

### Step-by-Step Deployment
1. Log in to [vercel.com](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository (`SentimentScope`).
4. In the **Configure Project** screen:
   - **Project Name**: `sentimentscope` (or your choice)
   - **Framework Preset**: `Other`
   - **Root Directory**: Click **Edit** and select **`frontend`**
   - **Build and Output Settings**: Leave default (no build command needed)
5. Click **Deploy**.

Vercel will deploy your frontend in seconds and give you a public URL:
```
https://sentimentscope.vercel.app
```

---

## Part 3: Connecting Frontend to Backend

The frontend is already configured with a dynamic resolver:
1. **Zero-Configuration Default**:
   If deployed on the web, the frontend automatically points to your production Render URL (`https://sentimentscope-iz3a.onrender.com`).
2. **Interactive UI Configuration Modal**:
   - In the frontend navigation bar, click the **Health Status Badge** (or the **"Change Backend URL"** button on the offline banner).
   - Enter your Render backend URL (e.g. `https://your-service.onrender.com`).
   - Click **Test Ping** to confirm connectivity and latency.
   - Click **Save & Connect**. The setting is persisted in your browser's `localStorage`!

---

## Local Development (Full-Stack)

To run both frontend and backend locally on your machine:

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
Open `http://localhost:3000` in your browser. The frontend detects `localhost` and automatically connects to `http://127.0.0.1:8000`.

---

## Troubleshooting Common Issues

### Issue 1: "Server Offline" in Frontend
- Check if your Render instance has spun down due to inactivity (free instances sleep after 15 mins).
- The first request can take 30-50 seconds to wake up the free instance.
- Click the Health Badge -> **Test Ping** to view live connectivity status.

### Issue 2: CORS Policy Error
- The backend in [`backend/src/api.py`](file:///c:/MY%20PROJECTS/SentimentScope/backend/src/api.py) is pre-configured to allow:
  - All `https://*.vercel.app` domains automatically.
  - `*` wildcard when credentials are not required.
  - Localhost ports (8000, 3000, 5173).
- If using a custom domain (e.g. `https://mycustomdomain.com`), add it to the `ALLOWED_ORIGINS` environment variable in your Render dashboard settings.
