import os
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Safe defaults for tests (avoid accidental external calls)
os.environ.setdefault("ENABLE_TRACING", "false")
os.environ.setdefault("USE_LOCAL_EMBEDDINGS", "true")
os.environ.setdefault("ALLOWED_ORIGINS", "*")
