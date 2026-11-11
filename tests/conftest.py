import os
import sys
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")   # CPU-only tests: the Triton interpreter, never the local GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
