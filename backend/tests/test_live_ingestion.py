import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api import app
from src.db import (
    init_db,
    insert_prediction,
    insert_batch_predictions,
    get_recent_predictions,
    get_rolling_stats,
    prune_old_records
)
from src.live_ingestion import (
    validate_keyword,
    check_cooldown,
    record_cooldown,
    process_live_ingestion,
    fetch_hn_posts
)


def test_db_init_and_crud(tmp_path):
    test_db = str(tmp_path / "test_store.db")
    init_db(test_db)

    # Insert single prediction
    success = insert_prediction(
        source="live",
        text="This is a test live post.",
        cleaned_text="test live post",
        sentiment="positive",
        confidence=0.95,
        probabilities={"negative": 0.02, "neutral": 0.03, "positive": 0.95},
        latency_ms=12.5,
        external_id="hn_99999",
        db_path=test_db
    )
    assert success is True

    # Test duplicate ignore
    dup_success = insert_prediction(
        source="live",
        text="This is a test live post.",
        cleaned_text="test live post",
        sentiment="positive",
        confidence=0.95,
        probabilities={"negative": 0.02, "neutral": 0.03, "positive": 0.95},
        latency_ms=12.5,
        external_id="hn_99999",
        db_path=test_db
    )
    assert dup_success is False

    # Fetch items
    items = get_recent_predictions(limit=10, source="live", db_path=test_db)
    assert len(items) == 1
    assert items[0]["external_id"] == "hn_99999"
    assert items[0]["sentiment"] == "positive"


def test_db_source_filtering(tmp_path):
    test_db = str(tmp_path / "test_filter.db")
    init_db(test_db)

    insert_prediction("live", "Live text", "live text", "positive", 0.9, {"positive": 0.9}, 5.0, "hn_1", test_db)
    insert_prediction("batch_upload", "Batch text", "batch text", "negative", 0.8, {"negative": 0.8}, 5.0, "csv_1", test_db)

    # Filter source=live
    live_items = get_recent_predictions(limit=10, source="live", db_path=test_db)
    assert len(live_items) == 1
    assert live_items[0]["source"] == "live"

    # Filter source=batch_upload
    batch_items = get_recent_predictions(limit=10, source="batch_upload", db_path=test_db)
    assert len(batch_items) == 1
    assert batch_items[0]["source"] == "batch_upload"

    # Live stats filtering
    live_stats = get_rolling_stats(source="live", db_path=test_db)
    assert live_stats["total_count"] == 1
    assert live_stats["positive_pct"] == 100.0


def test_db_retention_pruning(tmp_path):
    test_db = str(tmp_path / "test_prune.db")
    init_db(test_db)

    # Insert 15 records
    batch_items = []
    for i in range(15):
        batch_items.append({
            "external_id": f"ext_{i}",
            "text": f"Test item {i}",
            "cleaned_text": f"test item {i}",
            "sentiment": "positive" if i % 2 == 0 else "negative",
            "confidence": 0.9,
            "probabilities": {"positive": 0.9},
            "latency_ms": 10.0
        })
    insert_batch_predictions("live", batch_items, db_path=test_db)

    # Prune to max 10
    pruned = prune_old_records(max_rows=10, db_path=test_db)
    assert pruned == 5

    stats = get_rolling_stats(source="live", db_path=test_db)
    assert stats["total_count"] == 10
    assert stats["rows_pruned_last_cycle"] == 5
    assert stats["oldest_record_timestamp"] is not None


def test_keyword_validation_and_cooldown():
    # Empty / invalid keyword testing
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_keyword("")
    with pytest.raises(ValueError, match="cannot be whitespace"):
        validate_keyword("   ")
    with pytest.raises(ValueError, match="cannot exceed 100 characters"):
        validate_keyword("a" * 101)

    assert validate_keyword("  deep learning  ") == "deep learning"

    # Cooldown testing
    record_cooldown("cooldown_test_kw")
    rem = check_cooldown("cooldown_test_kw")
    assert rem is not None
    assert rem > 0.0


@patch("src.live_ingestion.httpx.Client.get")
def test_mock_live_ingestion(mock_get, tmp_path):
    test_db = str(tmp_path / "test_ingest.db")
    
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "hits": [
            {
                "objectID": "300001",
                "title": "New AI breakthrough released today",
                "story_text": "Model reaches state of the art performance.",
                "created_at": "2026-08-28T10:00:00Z"
            }
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    with TestClient(app) as client:
        # Mocking model inference in test
        res = process_live_ingestion(keyword="unittest_kw_unique", db_path=test_db)
        assert res["status"] == "success"
        assert res["ingested_count"] == 1

        stats = get_rolling_stats(source="live", db_path=test_db)
        assert stats["total_count"] == 1


def test_live_api_endpoints():
    with TestClient(app) as client:
        # Test stats
        res = client.get("/live/stats")
        assert res.status_code == 200
        data = res.json()
        assert "total_count" in data
        assert "rows_pruned_last_cycle" in data

        # Test feed
        res_feed = client.get("/live/feed")
        assert res_feed.status_code == 200
        assert isinstance(res_feed.json(), list)

        # Test trigger validation error
        res_err = client.post("/live/trigger?keyword=" + "x" * 105)
        assert res_err.status_code == 400

        # Test trigger happy path
        res_trig = client.post("/live/trigger?keyword=unique_test_kw_99")
        assert res_trig.status_code in [200, 429]
