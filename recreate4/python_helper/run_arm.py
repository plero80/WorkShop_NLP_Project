"""Resolve the original experiment CLI, then train one arm without opening final data."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def command_for(runtime, recipe, output, seed, stage, arm):
    sys.path.insert(0, str(Path(runtime).resolve()))
    from experiment_cli.cli import command, materialize, resolve
    plan = resolve(str(recipe), stage="pilot", output=str(output),
                   overrides=(f"seed={seed}", "data_seed=42"))
    if plan["seeds"] is not None or plan["settings"]["arms"] != ["proxy", "judge", "knn_static"]:
        raise ValueError("An arm worker requires the same complete three-arm single-seed configuration.")
    target = plan["settings"]["ppo"]["full_updates" if stage == "full" else "pilot_updates"]
    materialize(plan)
    # The existing pilot stage permits --updates. This runs the requested PPO
    # budget while leaving all final evaluations to the original full finalizer.
    return command(plan, "run") + ["--updates", str(target), "--arms", arm]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", required=True, choices=("pilot", "full"))
    parser.add_argument("--arm", required=True, choices=("proxy", "judge", "knn_static"))
    args = parser.parse_args(argv)
    cmd = command_for(args.runtime, args.recipe, args.output, args.seed, args.stage, args.arm)
    print(f"Training seed {args.seed}, arm {args.arm}; final data remains unopened in this worker.", flush=True)
    return subprocess.call(cmd, cwd=args.runtime)


if __name__ == "__main__":
    raise SystemExit(main())
