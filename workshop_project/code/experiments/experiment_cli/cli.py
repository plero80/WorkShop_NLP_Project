"""Resolve YAML recipes into calls to existing runners; no training code here."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
STAGES = ("prepare", "pilot", "full", "report")


def yaml_value(text):
    try:
        import yaml
    except ImportError as error:
        raise ValueError('Install the YAML dependency: python -m pip install "PyYAML>=6.0.3,<7"') from error

    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings.")
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        return yaml.load(text, Loader=UniqueLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error


def merge_settings(base, changes, prefix="settings"):
    if not isinstance(changes, dict):
        raise ValueError(f"{prefix} must be a mapping.")
    result = copy.deepcopy(base)
    for key, value in changes.items():
        name = f"{prefix}.{key}"
        if key not in base:
            raise ValueError(f"Unknown setting: {name}")
        old = base[key]
        if isinstance(old, dict):
            result[key] = merge_settings(old, value, name)
        elif isinstance(old, list):
            if not isinstance(value, list) or not value:
                raise ValueError(f"{name} must be a nonempty list.")
            for item in value:
                if old:
                    check_type(old[0], item, name)
            result[key] = value
        else:
            check_type(old, value, name)
            result[key] = value
    return result


def check_type(old, value, name):
    if type(old) is float:
        valid = type(value) in (int, float) and math.isfinite(value)
    else:
        valid = type(value) is type(old)
    if not valid:
        raise ValueError(f"{name} expects {type(old).__name__}, got {type(value).__name__}. Use 1.0e-5 for YAML scientific notation.")


def recipe_path(name, presets=None):
    path = Path(name).expanduser()
    if path.is_file():
        return path.resolve()
    if path.name == name and path.suffix == "":
        preset = (Path(presets) if presets is not None else PACKAGE / "presets") / f"{name}.yaml"
        if preset.is_file():
            return preset
    raise ValueError(f"Recipe not found: {name}")


def resolve(name, stage=None, output=None, overrides=(), layout=None):
    root = Path(layout["root"]) if layout is not None else ROOT
    source = recipe_path(name, layout["presets"] if layout is not None else None)
    recipe = yaml_value(source.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError("The recipe must be a YAML mapping.")
    unknown = set(recipe) - {"version", "experiment", "stage", "output", "seeds", "settings"}
    if unknown:
        raise ValueError(f"Unknown recipe keys: {', '.join(sorted(unknown))}")
    if type(recipe.get("version")) is not int or recipe["version"] != 1:
        raise ValueError("Recipe version must be 1.")
    if recipe.get("experiment") != "gsm8k":
        raise ValueError("Supported experiment: gsm8k. Other retained experiments still use their existing launchers.")
    stage = stage or recipe.get("stage", "pilot")
    if stage not in STAGES:
        raise ValueError(f"Stage must be one of {STAGES}.")
    configured_output = output if output is not None else recipe.get("output", "gsm8k_outputs/main")
    if not isinstance(configured_output, str) or not configured_output.strip():
        raise ValueError("output must be a nonempty path string.")
    destination = Path(configured_output).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    seeds = recipe.get("seeds")
    if seeds is not None:
        if (not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("seeds must be a nonempty list of distinct integers in [0, 2**32).")
        if stage == "report":
            raise ValueError("For a suite, use status/export; report is a single-run stage.")
    default = Path(layout["settings"]) if layout is not None else root / "gsm8k_experiment/settings.json"
    if not default.is_file():
        raise ValueError("Run this CLI from a restored project; see the runtime restore instructions.")
    base = json.loads(default.read_text(encoding="utf-8"))
    settings = merge_settings(base, recipe.get("settings", {}))
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError("--set expects a setting path and YAML value, e.g. ppo.learning_rate=1.0e-5")
        change = yaml_value(value)
        for part in reversed(key.split(".")):
            change = {part: change}
        settings = merge_settings(settings, change)
    # Cheap checks before spawning a GPU runner; the existing runner performs its full validation.
    if settings["generation"]["temperature"] != 1.0:
        raise ValueError("The shared PPO engine requires generation.temperature=1.0.")
    if not 0 <= settings["seed"] < 2**32:
        raise ValueError("seed must be in [0, 2**32).")
    arms = settings["arms"]
    if len(set(arms)) != len(arms) or not set(arms) <= {"proxy", "judge", "knn_static", "knn_static_30b", "knn_refresh", "oracle"}:
        raise ValueError("Unknown or duplicate reward arms.")
    ppo = settings["ppo"]
    if min(ppo[k] for k in ("pilot_updates", "full_updates", "prompts_per_update", "responses_per_prompt", "minibatch_size", "epochs", "checkpoint_every", "monitor_every")) < 1:
        raise ValueError("PPO counts and intervals must be positive.")
    if ppo["minibatch_size"] > ppo["prompts_per_update"] * ppo["responses_per_prompt"]:
        raise ValueError("PPO minibatch exceeds rollout size.")
    encoded = json.dumps(settings, sort_keys=True, indent=2, allow_nan=False) + "\n"
    config = root / ".experiment_cli/configs" / (hashlib.sha256(encoded.encode()).hexdigest() + ".json")
    return {"recipe": str(source), "experiment": "gsm8k", "stage": stage,
            "output": str(destination.resolve()), "seeds": seeds,
            "config": str(config), "settings": settings, "config_text": encoded}


def command(plan, action, destination=None):
    prefix = [sys.executable, "-u", "-m"]
    output = ["--output", plan["output"]]
    if action == "status":
        return prefix + ["gsm8k_experiment.status"] + output
    if action == "export":
        target = Path(destination).expanduser().resolve() if destination else Path(plan["output"] + ".zip")
        return prefix + ["gsm8k_experiment.export"] + output + ["--destination", str(target)]
    module = "gsm8k_experiment.suite" if plan["seeds"] is not None else "gsm8k_experiment.run"
    args = prefix + [module, "--config", plan["config"], "--stage", plan["stage"]] + output
    if plan["seeds"] is not None:
        args += ["--seeds", *map(str, plan["seeds"])]
    return args


def materialize(plan):
    """Never overwrite a config used by an earlier invocation."""
    path = Path(plan["config"])
    if path.exists():
        if path.read_text(encoding="utf-8") != plan["config_text"]:
            raise ValueError(f"Saved resolved configuration changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(plan["config_text"])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None, *, layout=None, prepare_runtime=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("show", "run", "status", "export"):
        child = sub.add_parser(action)
        child.add_argument("recipe", help="Preset name (gsm8k, gsm8k-three-seeds) or YAML file")
        child.add_argument("--output", help="Output directory; relative paths use the restored project root")
        if action in ("run", "show"):
            child.add_argument("--stage", choices=STAGES)
            child.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a nested setting; repeatable")
        if action == "run":
            child.add_argument("--dry-run", action="store_true", help="Print resolved settings and command without writing files or starting a runner")
        if action == "export":
            child.add_argument("--destination", help="New ZIP path")
    args = parser.parse_args(argv)
    try:
        plan = resolve(args.recipe, getattr(args, "stage", None), args.output, getattr(args, "set", ()), layout=layout)
        cmd = command(plan, "run" if args.action == "show" else args.action, getattr(args, "destination", None))
        if args.action == "show" or getattr(args, "dry_run", False):
            print(json.dumps({k: v for k, v in plan.items() if k != "config_text"} | {"command": cmd}, indent=2))
            return 0
        if prepare_runtime is not None:
            prepare_runtime()
        if args.action == "run":
            materialize(plan)
            print(f"{plan['experiment']} / {plan['stage']} -> {plan['output']}", flush=True)
        return subprocess.call(cmd, cwd=Path(layout["root"]) if layout is not None else ROOT)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
