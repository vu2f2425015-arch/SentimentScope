import os
import sys

# Dynamic root & backend path resolution for Vercel serverless function
curr = os.path.dirname(os.path.abspath(__file__))
root_dir = None

for _ in range(5):
    if os.path.exists(os.path.join(curr, "models", "best_model_meta.json")) or os.path.exists(os.path.join(curr, "backend", "src", "api.py")):
        root_dir = curr
        break
    parent = os.path.dirname(curr)
    if parent == curr:
        break
    curr = parent

if not root_dir:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
from backend.src.api import app, load_model_state

# Pre-warm model load during serverless function container initialization
try:
    load_model_state()
except Exception as _err:
    print(f"[Vercel Init Warning] Pre-warm model load error: {_err}")
