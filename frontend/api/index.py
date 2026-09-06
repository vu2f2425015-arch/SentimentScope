import os
import sys

# Dynamic root & backend path resolution for Vercel serverless function
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if os.path.exists(os.path.join(parent_dir, "backend")):
    root_dir = parent_dir
else:
    root_dir = os.path.dirname(parent_dir)

backend_dir = os.path.join(root_dir, "backend")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Environment overrides for Vercel serverless environment
os.environ["MODELS_DIR"] = os.path.join(root_dir, "models")
os.environ["REPORTS_DIR"] = os.path.join(root_dir, "reports")
os.environ["INGESTION_DB_PATH"] = "/tmp/rolling_store.db"
os.environ["ENABLE_LIVE_INGESTION"] = "false"

# Top-level ASGI app import for Vercel static AST analysis
from backend.src.api import app
