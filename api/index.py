import os
import sys

# Ensure root directory and backend directory are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
backend_dir = os.path.join(root_dir, "backend")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Environment overrides for Vercel serverless environment
os.environ.setdefault("MODELS_DIR", os.path.join(root_dir, "models"))
os.environ.setdefault("REPORTS_DIR", os.path.join(root_dir, "reports"))
os.environ.setdefault("INGESTION_DB_PATH", "/tmp/rolling_store.db")
os.environ.setdefault("ENABLE_LIVE_INGESTION", "false")

try:
    from backend.src.api import app
except ImportError:
    from src.api import app
