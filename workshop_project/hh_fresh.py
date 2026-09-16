"""Fresh HH-RLHF training, two memory refreshes, and matched ridge PPO."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT / 'code/experiments'), str(PROJECT / 'code/core')]

if __name__ == '__main__':
    from hh_fresh.run import main
    raise SystemExit(main(project=PROJECT))
