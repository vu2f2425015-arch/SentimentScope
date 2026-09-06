# SentimentScope Backend API (Render)

This directory contains the standalone production REST API for **SentimentScope**, designed for deployment on [Render](https://render.com).

## Key Features
- **FastAPI Core**: Ultra-low latency (<50ms inference time).
- **RAM-Optimized**: Startup memory under 200MB (safely fits within Render Free Tier's 512MB RAM cap).
- **Lazy Loading**: TensorFlow and heavy deep learning modules are lazy-loaded only when requested.
- **Dynamic Port Binding**: Synchronized with Render's `$PORT` environment variable.
- **Permissive CORS**: Automatically accepts requests from all `https://*.vercel.app` domains, custom origins, and localhost.
- **Rolling SQLite Telemetry**: Background Hacker News live ingestion with automated retention pruning.

---

## Deploy to Render (2 Options)

### Option A: Using Render Blueprint (1-Click)
1. Push your repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com), navigate to **Blueprints** -> **New Blueprint Instance**.
3. Connect your `SentimentScope` repository.
4. Render detects `render.yaml` and deploys the web service automatically with health checks at `/health`.

### Option B: Manual Web Service
1. In the [Render Dashboard](https://dashboard.render.com), click **New** -> **Web Service**.
2. Connect your `SentimentScope` repository.
3. Configure the service:
   - **Name**: `sentimentscope-api`
   - **Language / Environment**: `Docker`
   - **Root Directory**: `backend` (or leave root if deploying from a dedicated backend repo)
   - **Instance Type**: Free (512 MB)
   - **Health Check Path**: `/health`
4. Click **Create Web Service**.

---

## Local Development & Testing

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run pytest test suite
pytest tests/ -v

# Start local development server
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

---

## Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status banner & documentation links |
| `GET` | `/health` | Healthcheck (returns service & model state) |
| `POST` | `/predict` | Single text sentiment analysis with class probabilities |
| `POST` | `/predict/batch` | Multi-row CSV sentiment batch processing |
| `GET` | `/live/feed` | Rolling telemetry feed from live ingestion |
| `GET` | `/live/stats` | Rolling sentiment statistics and pruning counters |
| `POST` | `/live/trigger` | Manually triggers live post ingestion for a keyword |
| `GET` | `/model/metrics` | Model comparison benchmark and confusion matrix |
| `GET` | `/docs` | Interactive Swagger OpenAPI UI |
