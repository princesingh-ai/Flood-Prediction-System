import sys
from pathlib import Path

# Ensure Backend module is discoverable in Vercel serverless environment
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "Backend"))

from Backend.main import app
