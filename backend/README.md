# SentimentScope Backend API

This directory contains the production REST API engine for **SentimentScope**, designed for deployment as a Vercel Serverless Function or containerized FastAPI service.

## Key Features
- **FastAPI Core**: Ultra-low latency (<50ms inference time).
- **RAM-Optimized**: Startup memory under 200MB.
- **Lazy Loading**: Heavy deep learning modules are lazy-loaded only when requested.
- **Permissive CORS**: Automatically accepts requests from all `https://*.vercel.app` domains, custom origins, and localhost.
- **Rolling SQLite Telemetry**: Live ingestion with automated retention pruning.

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
