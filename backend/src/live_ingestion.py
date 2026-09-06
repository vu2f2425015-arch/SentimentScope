import os
import time
import asyncio
import logging
import httpx
from typing import Dict, Any, List, Optional
from src.db import init_db, insert_prediction, prune_old_records, get_rolling_stats

logger = logging.getLogger("SentimentScope.Ingestion")
logger.setLevel(logging.INFO)

HN_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_KEYWORD = os.getenv("INGESTION_KEYWORD", "ai")
DEFAULT_INTERVAL_MINUTES = int(os.getenv("INGESTION_INTERVAL_MINUTES", "15"))
COOLDOWN_SECONDS = int(os.getenv("INGESTION_COOLDOWN_SECONDS", "30"))

# In-memory keyword cooldown tracker: keyword -> timestamp of last fetch
_cooldown_tracker: Dict[str, float] = {}
_scheduler_running = False


def validate_keyword(keyword: str) -> str:
    """
    Validates user or task supplied keyword string.
    Raises ValueError if empty or exceeding 100 characters.
    """
    if not keyword or not isinstance(keyword, str):
        raise ValueError("Keyword cannot be empty.")
    kw = keyword.strip()
    if not kw:
        raise ValueError("Keyword cannot be whitespace only.")
    if len(kw) > 100:
        raise ValueError("Keyword length cannot exceed 100 characters.")
    return kw


def check_cooldown(keyword: str) -> Optional[float]:
    """
    Checks whether keyword is in cooldown period.
    Returns remaining seconds if in cooldown, or None if clear.
    """
    clean_kw = keyword.lower().strip()
    now = time.time()
    last_time = _cooldown_tracker.get(clean_kw, 0.0)
    elapsed = now - last_time
    if elapsed < COOLDOWN_SECONDS:
        return round(COOLDOWN_SECONDS - elapsed, 1)
    return None


def record_cooldown(keyword: str):
    """
    Records timestamp for keyword fetch.
    """
    _cooldown_tracker[keyword.lower().strip()] = time.time()


def fetch_hn_posts(keyword: str, max_items: int = 15) -> List[Dict[str, Any]]:
    """
    Fetches recent posts from Hacker News Search API matching keyword.
    """
    clean_kw = validate_keyword(keyword)
    params = {
        "query": clean_kw,
        "tags": "story",
        "hitsPerPage": max_items
    }
    
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(HN_ALGOLIA_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", [])
            
            posts = []
            for hit in hits:
                story_id = str(hit.get("objectID", ""))
                title = hit.get("title", "")
                text = hit.get("story_text", "")
                
                # Combine title and story text if available
                full_text = f"{title}. {text}".strip() if text else title.strip()
                if full_text and story_id:
                    posts.append({
                        "external_id": f"hn_{story_id}",
                        "text": full_text,
                        "url": hit.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                        "created_at": hit.get("created_at")
                    })
            return posts
    except Exception as e:
        logger.warning(f"[Live Ingestion] Network/API fetch error for '{keyword}': {e}")
        return []


def process_live_ingestion(keyword: str = DEFAULT_KEYWORD, max_items: int = 15, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrates live ingestion: fetches posts, executes model inference, stores in SQLite, and prunes old records.
    Returns status summary dictionary. Fully fail-safe.
    """
    from src.api import run_inference # Deferred import to avoid circular dependency
    
    clean_kw = validate_keyword(keyword)
    
    # Verify cooldown
    rem = check_cooldown(clean_kw)
    if rem is not None:
        return {
            "status": "cooldown",
            "message": f"Keyword '{clean_kw}' is in cooldown. Please wait {rem} seconds.",
            "remaining_seconds": rem,
            "ingested_count": 0
        }

    # Record cooldown timestamp
    record_cooldown(clean_kw)
    
    init_db(db_path)
    posts = fetch_hn_posts(clean_kw, max_items=max_items)
    
    if not posts:
        return {
            "status": "warning",
            "message": f"No new posts retrieved for keyword '{clean_kw}'.",
            "ingested_count": 0
        }

    ingested = 0
    for p in posts:
        try:
            inf_res = run_inference(p["text"])
            success = insert_prediction(
                source="live",
                text=inf_res["text"],
                cleaned_text=inf_res["cleaned_text"],
                sentiment=inf_res["sentiment"],
                confidence=inf_res["confidence"],
                probabilities=inf_res["probabilities"],
                latency_ms=inf_res["latency_ms"],
                external_id=p["external_id"],
                db_path=db_path
            )
            if success:
                ingested += 1
        except Exception as err:
            logger.error(f"[Live Ingestion] Failed processing item '{p.get('external_id')}': {err}")

    # Enforce database retention pruning policy
    pruned_count = prune_old_records(db_path=db_path)
    
    logger.info(f"[Live Ingestion] Completed cycle for '{clean_kw}': Ingested {ingested} items, Pruned {pruned_count} old items.")
    
    return {
        "status": "success",
        "keyword": clean_kw,
        "fetched_count": len(posts),
        "ingested_count": ingested,
        "pruned_count": pruned_count
    }


async def background_ingestion_loop():
    """
    Asynchronous background task loop running ingestion periodically.
    """
    global _scheduler_running
    _scheduler_running = True
    logger.info(f"[Live Ingestion Scheduler] Started background loop for keyword '{DEFAULT_KEYWORD}' every {DEFAULT_INTERVAL_MINUTES} minutes.")
    
    while _scheduler_running:
        try:
            _ = process_live_ingestion(keyword=DEFAULT_KEYWORD)
        except Exception as e:
            logger.error(f"[Live Ingestion Scheduler] Unexpected loop exception: {e}")
        
        # Sleep for configured interval
        await asyncio.sleep(DEFAULT_INTERVAL_MINUTES * 60)


def stop_background_scheduler():
    global _scheduler_running
    _scheduler_running = False
