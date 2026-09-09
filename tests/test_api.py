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


def test_transformer_inference_not_rule_based():
    """
    Regression Test:
    Ensures that when a transformer model is active in model_state, the inference
    pipeline executes model-based inference and DOES NOT silently fall back to
    _rule_based_sentiment() due to checking for 'vectorizer' instead of 'tokenizer'.

    Note on why the legacy test suite missed this bug:
    The existing tests ran exclusively against Logistic Regression (type: sklearn)
    which populated 'vectorizer' in model_state. The transformer inference branch
    was never exercised in tests/test_api.py.
    """
    import os
    from unittest.mock import patch, MagicMock
    import numpy as np

    # Test with mock transformer if weights not present, or verify directly
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    
    import torch
    mock_inputs = {
        "input_ids": torch.tensor([[101, 2054, 102]])
    }
    mock_tokenizer.return_value = mock_inputs
    
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[-2.0, 0.5, 4.0]])
    mock_model.return_value = mock_output

    transformer_state = {
        "type": "transformer",
        "model": mock_model,
        "tokenizer": mock_tokenizer,
        "name": "DistilBERT (Full Dataset)",
        "max_len": 64,
        "metrics": {"accuracy": 0.7257}
    }

    with patch.dict("src.api.model_state", transformer_state, clear=True):
        with TestClient(app) as client:
            resp = client.post("/predict", json={"text": "The battery life is phenomenal!"})
            assert resp.status_code == 200
            data = resp.json()

            # Crucial assertion: Must NOT be the rule-based fallback signature
            assert data["pipeline_model"] != "Logistic Regression (Serverless)"
            assert data["pipeline_model"] == "DistilBERT (Full Dataset)"
            assert data["tokenizer_type"] == "WordPiece Tokenizer (DistilBERT)"
            assert data["sentiment"] == "positive"
            assert "probabilities" in data
            assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01

