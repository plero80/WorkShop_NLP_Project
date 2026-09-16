"""Run the saved-vector HH-RLHF comparisons from this checkout."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT / "code/experiments"), str(PROJECT / "code/core")]

if __name__ == "__main__":
    from hh_offline.run import main
    raise SystemExit(main(project=PROJECT))
