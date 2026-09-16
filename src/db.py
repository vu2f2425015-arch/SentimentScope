import os
import json
import sqlite3
import time
import threading
from functools import lru_cache
from typing import Dict, Any, List, Optional

DEFAULT_DB_PATH = os.getenv("INGESTION_DB_PATH", os.path.join("data", "rolling_store.db"))
DEFAULT_MAX_ROWS = int(os.getenv("INGESTION_MAX_ROWS", "5000"))

# Global telemetry counter for last pruning operation
_rows_pruned_last_cycle = 0

# Reusable shared connection and thread lock
_shared_connection: Optional[sqlite3.Connection] = None
_db_lock = threading.RLock()


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Returns a thread-safe reused SQLite connection configured with WAL mode and busy timeout.
    If a custom db_path is provided (e.g. during isolated unit tests), opens a dedicated connection.
    """
    global _shared_connection
    target_path = db_path or DEFAULT_DB_PATH

    # For custom paths (e.g. unit tests with temp directories), return dedicated connection
    if db_path is not None and os.path.abspath(db_path) != os.path.abspath(DEFAULT_DB_PATH):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        except Exception:
            pass
        conn = sqlite3.connect(target_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return conn

    with _db_lock:
        if _shared_connection is None:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            except Exception:
                pass
            conn = sqlite3.connect(target_path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except Exception:
                pass
            try:
                conn.execute("PRAGMA busy_timeout=5000;")
            except Exception:
                pass
            _shared_connection = conn
        return _shared_connection


def close_db_connection():
    """
    Closes the shared SQLite connection during application shutdown.
    """
    global _shared_connection
    with _db_lock:
        if _shared_connection is not None:
            try:
                _shared_connection.close()
            except Exception:
                pass
            _shared_connection = None


def init_db(db_path: Optional[str] = None):
    """
    Initializes the SQLite schema for storing predictions.
    """
    try:
        with _db_lock, get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    external_id TEXT UNIQUE,
                    text TEXT NOT NULL,
                    cleaned_text TEXT,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON predictions_store(source);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON predictions_store(created_at);")
            conn.commit()
    except Exception as e:
        print(f"[DB Warning] init_db error: {e}")


def prune_old_records(max_rows: Optional[int] = None, db_path: Optional[str] = None) -> int:
    """
    Prunes oldest records exceeding max_rows limit to enforce memory retention policies.
    Returns number of rows pruned in this cycle.
    """
    global _rows_pruned_last_cycle
    limit = max_rows or DEFAULT_MAX_ROWS
    pruned_count = 0
    
    with _db_lock, get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM predictions_store;")
        total = cursor.fetchone()["cnt"]
        
        if total > limit:
            excess = total - limit
            cursor.execute("""
                DELETE FROM predictions_store
                WHERE id IN (
                    SELECT id FROM predictions_store
                    ORDER BY id ASC
                    LIMIT ?
                );
            """, (excess,))
            pruned_count = cursor.rowcount
            conn.commit()
            
    _rows_pruned_last_cycle = pruned_count
    if pruned_count > 0:
        clear_rolling_stats_cache()
    return pruned_count


def insert_prediction(
    source: str,
    text: str,
    cleaned_text: str,
    sentiment: str,
    confidence: float,
    probabilities: Dict[str, float],
    latency_ms: float,
    external_id: Optional[str] = None,
    db_path: Optional[str] = None
) -> bool:
    """
    Inserts a single sentiment prediction into the SQLite database.
    Ignores duplicates based on external_id.
    """
    probs_str = json.dumps(probabilities)
    with _db_lock, get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO predictions_store
                (source, external_id, text, cleaned_text, sentiment, confidence, probabilities_json, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (source, external_id, text, cleaned_text, sentiment, confidence, probs_str, latency_ms))
            conn.commit()
            if cursor.rowcount > 0:
                clear_rolling_stats_cache()
                return True
            return False
        except sqlite3.Error as e:
            print(f"[DB Error] Insert failed: {e}")
            return False


def insert_batch_predictions(
    source: str,
    predictions_list: List[Dict[str, Any]],
    db_path: Optional[str] = None
) -> int:
    """
    Bulk inserts a list of predictions.
    """
    inserted = 0
    with _db_lock, get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for item in predictions_list:
            probs_str = json.dumps(item.get("probabilities", {}))
            ext_id = item.get("external_id")
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO predictions_store
                    (source, external_id, text, cleaned_text, sentiment, confidence, probabilities_json, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    source,
                    ext_id,
                    item.get("text", ""),
                    item.get("cleaned_text", ""),
                    item.get("sentiment", "neutral"),
                    float(item.get("confidence", 0.0)),
                    probs_str,
                    float(item.get("latency_ms", 0.0))
                ))
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.Error:
                pass
        conn.commit()
    if inserted > 0:
        clear_rolling_stats_cache()
    return inserted


def get_recent_predictions(
    limit: int = 50,
    offset: int = 0,
    source: Optional[str] = "live",
    db_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetches recent predictions ordered by created_at descending with pagination offset.
    If source is specified (e.g. 'live'), filters strictly by source.
    """
    with _db_lock:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        if source:
            cursor.execute("""
                SELECT id, source, external_id, text, cleaned_text, sentiment, confidence, probabilities_json, latency_ms, created_at
                FROM predictions_store
                WHERE source = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?;
            """, (source, limit, offset))
        else:
            cursor.execute("""
                SELECT id, source, external_id, text, cleaned_text, sentiment, confidence, probabilities_json, latency_ms, created_at
                FROM predictions_store
                ORDER BY id DESC
                LIMIT ? OFFSET ?;
            """, (limit, offset))
            
        rows = cursor.fetchall()
        results = []
        for r in rows:
            probs = {}
            try:
                probs = json.loads(r["probabilities_json"])
            except Exception:
                pass
            results.append({
                "id": r["id"],
                "source": r["source"],
                "external_id": r["external_id"],
                "text": r["text"],
                "cleaned_text": r["cleaned_text"],
                "sentiment": r["sentiment"],
                "confidence": r["confidence"],
                "probabilities": probs,
                "latency_ms": r["latency_ms"],
                "created_at": r["created_at"]
            })
        return results


@lru_cache(maxsize=32)
def _cached_rolling_stats_query(
    source: Optional[str],
    db_path: Optional[str],
    bucket: int
) -> tuple:
    """
    Internal cached helper for get_rolling_stats using time bucket.
    """
    with _db_lock:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        where_clause = "WHERE source = ?" if source else ""
        params = (source,) if source else ()
        
        cursor.execute(f"SELECT COUNT(*) as total, AVG(confidence) as avg_conf FROM predictions_store {where_clause};", params)
        summary = cursor.fetchone()
        total = summary["total"] if summary else 0
        avg_conf = round(float(summary["avg_conf"]), 4) if (summary and summary["avg_conf"] is not None) else 0.0
        
        pos_count, neg_count, neu_count = 0, 0, 0
        latest_timestamp = None
        oldest_timestamp = None
        
        if total > 0:
            cursor.execute(f"""
                SELECT 
                    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as pos,
                    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as neg,
                    SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neu
                FROM predictions_store {where_clause};
            """, params)
            counts = cursor.fetchone()
            if counts:
                pos_count = counts["pos"] or 0
                neg_count = counts["neg"] or 0
                neu_count = counts["neu"] or 0

            cursor.execute(f"SELECT created_at FROM predictions_store {where_clause} ORDER BY id DESC LIMIT 1;", params)
            latest_row = cursor.fetchone()
            if latest_row:
                latest_timestamp = latest_row["created_at"]

            cursor.execute(f"SELECT created_at FROM predictions_store {where_clause} ORDER BY id ASC LIMIT 1;", params)
            oldest_row = cursor.fetchone()
            if oldest_row:
                oldest_timestamp = oldest_row["created_at"]

        pos_pct = round((pos_count / total) * 100, 2) if total > 0 else 0.0
        neg_pct = round((neg_count / total) * 100, 2) if total > 0 else 0.0
        neu_pct = round((neu_count / total) * 100, 2) if total > 0 else 0.0

        res = {
            "total_count": total,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "avg_confidence": avg_conf,
            "latest_record_timestamp": latest_timestamp,
            "oldest_record_timestamp": oldest_timestamp,
            "filtered_source": source or "all"
        }
        return tuple(res.items())


def clear_rolling_stats_cache():
    """
    Explicitly clears the in-memory rolling stats cache upon new record insertions or pruning.
    """
    _cached_rolling_stats_query.cache_clear()


def get_rolling_stats(
    source: Optional[str] = "live",
    db_path: Optional[str] = None,
    ttl_seconds: int = 8
) -> Dict[str, Any]:
    """
    Calculates aggregated statistics for stored predictions with TTL caching (5-10 seconds).
    Defaults to source='live' so batch CSV uploads don't distort live ingestion metrics.
    """
    global _rows_pruned_last_cycle
    bucket = int(time.time() // max(1, ttl_seconds))
    stats_data = dict(_cached_rolling_stats_query(source, db_path, bucket))
    stats_data["rows_pruned_last_cycle"] = _rows_pruned_last_cycle
    return stats_data
