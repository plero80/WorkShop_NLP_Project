"""Launch GSM8K directly from project source, without making code copies.

    python gsm8k.py run gsm8k-b200 --dry-run
    python gsm8k.py run gsm8k-b200 --stage full

Existing runs in ../run/gsm8k keep using their original runtime and checkpoints.
New runs use code/core and code/experiments, with outputs in gsm8k_outputs/.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
LEGACY = PROJECT.parent / "run" / "gsm8k"


def load_cli():
    spec = importlib.util.spec_from_file_location("project_cli", PROJECT / "code/experiments/experiment_cli/cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_run(output):
    folder = Path(output)
    return any((folder / name).is_file() for name in ("manifest.json", "suite_protocol.json", "config.json"))


def invocation(args):
    """Extract routing options; the existing CLI validates all user arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("action", nargs="?")
    parser.add_argument("recipe", nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--stage")
    parser.add_argument("--set", action="append", default=[])
    parsed, _ = parser.parse_known_args(args)
    return parsed


def select_layout(cli, args):
    options = invocation(args)
    native = {"root": PROJECT, "settings": PROJECT / "configs/gsm8k/settings.json",
              "presets": PROJECT / "configs/experiments"}
    if options.recipe and not any(a in ("--help", "-h") for a in args):
        plan = cli.resolve(options.recipe, options.stage, options.output, options.set, layout=native)
        if (LEGACY / "gsm8k_experiment/settings.json").is_file():
            legacy = {"root": LEGACY, "settings": LEGACY / "gsm8k_experiment/settings.json",
                      "presets": PROJECT / "configs/experiments"}
            previous = cli.resolve(options.recipe, options.stage, options.output, options.set, layout=legacy)
            old_output = Path(previous["output"])
            if old_output.is_relative_to(LEGACY.resolve()) and has_run(old_output):
                if Path(plan["output"]) != old_output and has_run(plan["output"]):
                    raise ValueError("Both project and legacy runs exist. Choose one with an absolute --output path.")
                return legacy
    env = os.environ.copy()
    paths = [str(PROJECT / "code/experiments"), str(PROJECT / "code/core")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    native["env"] = env
    return native


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["setup"]:
        print("GSM8K runs directly from workshop_project/code; no runtime copy is created.")
        print(f'python -m pip install -r "{PROJECT / "requirements-gsm8k.txt"}"')
        return 0
    cli = load_cli()
    try:
        layout = select_layout(cli, args)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if args and args[0] == "validate":
        # Historical runs keep their training source; CPU validation always uses
        # the current project implementation while reading the resolved old output.
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(PROJECT / "code/experiments"),
                                           str(PROJECT / "code/core"), env.get("PYTHONPATH", "")])
        layout = {**layout, "env": env}
    elif layout["root"] == LEGACY and not any(a in ("--dry-run", "--help", "-h") for a in args):
        print(f"Using existing run at {LEGACY}; its code and checkpoints stay in place.", flush=True)
    return cli.main(args, layout=layout)


if __name__ == "__main__":
    raise SystemExit(main())
