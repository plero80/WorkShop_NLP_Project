"""Run the original single-seed CLI with an explicit memory execution profile."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

from memory_profiles import validate_settings
from windows_io import replacement_retries


ARMS = ["proxy", "judge", "knn_static"]


def load_cli(runtime):
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime))
    cli = importlib.import_module("experiment_cli.cli")
    if Path(cli.__file__).resolve() != runtime / "experiment_cli" / "cli.py":
        raise ValueError("The original experiment CLI was imported from a different runtime.")
    return cli


def command_for(runtime, recipe, output, seed, stage, vram_gb, arm=None, *, cli=None):
    if stage not in ("prepare", "pilot", "full"):
        raise ValueError("Unsupported profile worker stage.")
    if arm is not None and (arm not in ARMS or stage == "prepare"):
        raise ValueError("An arm worker requires pilot/full stage and one canonical arm.")
    cli = cli or load_cli(runtime)
    plan = cli.resolve(str(recipe), stage="pilot" if arm else stage, output=str(output),
                       overrides=(f"seed={seed}", "data_seed=42"))
    if plan["seeds"] is not None or plan["settings"]["arms"] != ARMS:
        raise ValueError("A profile worker requires the complete three-arm single-seed configuration.")
    validate_settings(plan["settings"], vram_gb)
    cmd = cli.command(plan, "run")
    if cmd[:4] != [sys.executable, "-u", "-m", "gsm8k_experiment.run"]:
        raise ValueError("The original CLI did not resolve to the expected single-seed runner.")
    if arm:
        target = plan["settings"]["ppo"]["full_updates" if stage == "full" else "pilot_updates"]
        cmd += ["--updates", str(target), "--arms", arm]
    cli.materialize(plan)
    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage", required=True, choices=("prepare", "pilot", "full"))
    parser.add_argument("--vram-gb", required=True, type=int, choices=(12, 24, 48))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(argv)
    runtime, recipe, output = (path.resolve() for path in (args.runtime, args.recipe, args.output))
    with replacement_retries():
        return run(args, runtime, recipe, output)


def run(args, runtime, recipe, output):
    # Materialization also writes atomic JSON files, so the Windows policy must
    # already be installed when the original CLI first touches the run directory.
    command = command_for(runtime, recipe, output, args.seed, args.stage, args.vram_gb, args.arm)
    previous_directory = Path.cwd()
    controller = None
    try:
        os.chdir(runtime)
        if args.vram_gb == 12:
            from gpu_residency import install
            controller = install()
        # Calling the original main in this process retains the residency hooks;
        # its config, run lock, fingerprints, checkpoints and final protocol remain
        # owned by the original runtime. Arm workers still use pilot + --updates.
        runner = importlib.import_module("gsm8k_experiment.run")
        print(f"Seed {args.seed}: {args.vram_gb} GB execution profile" +
              (f", arm {args.arm}; final data stays unopened in this worker." if args.arm else "."),
              flush=True)
        return runner.main(command[4:])
    finally:
        try:
            if controller is not None:
                controller.close()
        finally:
            os.chdir(previous_directory)


if __name__ == "__main__":
    raise SystemExit(main())
