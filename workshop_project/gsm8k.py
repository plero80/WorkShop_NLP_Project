"""Run GSM8K from workshop_project, reusing the existing Runpod runtime and outputs.

    python gsm8k.py run gsm8k-b200 --dry-run
    python gsm8k.py run gsm8k-b200 --stage full

`setup` prepares the runtime and prints the dependency installation command.
No experiment starts during setup, show or a dry run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parent
RUNTIME = PROJECT.parent / "run" / "gsm8k"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_files():
    """Only the shared core, GSM8K, CLI and their configuration are required."""
    shared = json.loads((PROJECT / "configs/gsm8k/shared_sources.json").read_text(encoding="utf-8"))
    records = []
    for name in ("file_map.json", "additions.json"):
        records.extend(json.loads((PROJECT / "reproducibility" / name).read_text(encoding="utf-8"))["files"])
    rows = [r for r in records if r["original"] in {*shared, "requirements.txt"}
            or r["original"].startswith(("gsm8k_experiment/", "experiment_cli/"))]
    if len({r["original"] for r in rows}) != len(rows):
        raise ValueError("Duplicate GSM8K runtime paths in the file maps.")
    return rows


def ensure_runtime():
    manager = load_module("gsm8k_restore", PROJECT / "reproducibility/manage.py")
    rows = runtime_files()
    for row in rows:
        source = manager.safe(PROJECT, row["organized"])
        if not source.is_file() or manager.digest(source) != row["sha256"]:
            raise ValueError(f"Project file does not match its manifest: {row['organized']}")
    if RUNTIME.exists():
        # Checkpoint identity covers these files. Never replace them underneath a run.
        for row in rows:
            if row["original"].startswith("experiment_cli/"):
                continue  # This launcher uses the CLI from workshop_project.
            path = manager.safe(RUNTIME, row["original"])
            if not path.is_file() or manager.digest(path) != row["sha256"]:
                raise ValueError(
                    f"Existing runtime differs at {row['original']}. Nothing was overwritten. "
                    "Use the documented GSM8K source upgrade before resuming this older runtime.")
        expected = {Path(r["original"]).name for r in rows
                    if Path(r["original"]).parent.as_posix() == "gsm8k_experiment" and r["original"].endswith(".py")}
        if {p.name for p in (RUNTIME / "gsm8k_experiment").glob("*.py")} != expected:
            raise ValueError("Unexpected GSM8K source files in the existing runtime. Nothing was overwritten.")
        return
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete runtime at once; interrupted setup leaves no half-built destination.
    with tempfile.TemporaryDirectory(prefix=".gsm8k-setup-", dir=RUNTIME.parent) as temporary:
        staged = Path(temporary) / "runtime"
        staged.mkdir()
        for row in rows:
            manager.copy_checked(manager.safe(PROJECT, row["organized"]), manager.safe(staged, row["original"]), row)
        staged.rename(RUNTIME)
    print(f"Prepared GSM8K runtime: {RUNTIME}", flush=True)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["setup"]:
        try:
            ensure_runtime()
        except (ValueError, OSError) as error:
            print(f"Error: {error}", file=sys.stderr)
            return 2
        print("Install dependencies in your existing CUDA environment:")
        print(f'python -m pip install -r "{RUNTIME / "experiment_cli/requirements.txt"}"')
        return 0
    cli = load_module("gsm8k_project_cli", PROJECT / "code/experiments/experiment_cli/cli.py")
    layout = {"root": RUNTIME, "settings": PROJECT / "configs/gsm8k/settings.json",
              "presets": PROJECT / "configs/experiments"}
    return cli.main(args, layout=layout, prepare_runtime=ensure_runtime)


if __name__ == "__main__":
    raise SystemExit(main())
