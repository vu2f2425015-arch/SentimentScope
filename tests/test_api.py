import io
import pytest
from fastapi.testclient import TestClient
from src.api import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"] or "json" in response.headers["content-type"]


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "model_loaded" in data
        assert data["version"] == "1.0.0"



def test_predict_happy_path():
    with TestClient(app) as client:
        # Warmup prediction call to avoid initial CPU cold-start thread allocation skew
        _ = client.post("/predict", json={"text": "Warmup text snippet."})
        
        response = client.post(
            "/predict",
            json={"text": "This product is fantastic! Works perfectly."}
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert data["sentiment"] in ["positive", "negative", "neutral"]
        assert "confidence" in data
        assert "probabilities" in data
        assert "positive" in data["probabilities"]
        assert "negative" in data["probabilities"]
        assert "neutral" in data["probabilities"]
        assert data["latency_ms"] < 2000.0 # Inference latency under 2 seconds


def test_predict_empty_text_error():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 400
        data = response.json()
        assert "Input text cannot be empty" in data["detail"]


def test_predict_batch_happy_path():
    csv_content = "text\nGreat product love it\nTerrible item broke on arrival\nAverage quality okay\n"
    file_tuple = ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            files={"file": file_tuple}
        )
        assert response.status_code == 200
        data = response.json()
        assert "positive_pct" in data
        assert "negative_pct" in data
        assert "neutral_pct" in data
        assert data["total_rows"] == 3
        assert len(data["predictions"]) == 3


def test_predict_batch_invalid_csv_header():
    csv_content = "invalid_header\nsome content\n"
    file_tuple = ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            files={"file": file_tuple}
        )
        assert response.status_code == 400
        data = response.json()
        assert "CSV file must contain a 'text' column" in data["detail"]


def test_get_metrics():
    with TestClient(app) as client:
        response = client.get("/model/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or "best_model_name" in data


def test_predict_batch_oversized_rows():
    from unittest.mock import patch
    with patch("src.api.MAX_SYNC_BATCH_ROWS", 5):
        csv_content = "text\n" + "\n".join([f"Review text number {i}" for i in range(10)])
        file_tuple = ("test_large.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
        with TestClient(app) as client:
            response = client.post(
                "/predict/batch",
                files={"file": file_tuple}
            )
            assert response.status_code == 400
            assert "exceeding the synchronous web batch limit" in response.json()["detail"]

