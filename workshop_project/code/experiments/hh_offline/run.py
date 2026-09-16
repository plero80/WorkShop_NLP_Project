"""Matched offline ridge baseline and conversation-level memory-size ablation."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from common import canonical_hash, file_hash, read_json, write_json
from evaluation import save_npz
from experiment_cli.cli import yaml_value
from knn_core import top_neighbors
from . import data
from .metrics import summarize, paired_intervals

VERSION = "hh_offline_v1"
METHODS = ("proxy", "mean_gap", "knn_fixed", "knn_tuned", "ridge")


def resolve(project, config_path, output=None, skip_transfer=False):
    project = Path(project).resolve()
    config = yaml_value(Path(config_path).read_text(encoding="utf-8"))
    expected = {"version", "experiment", "output", "candidate_bank", "memory", "validation", "test", "transfer", "settings"}
    if not isinstance(config, dict) or set(config) != expected or config["version"] != 1 or config["experiment"] != "hh_offline":
        raise ValueError("Expected a version-1 hh_offline recipe with the documented keys")
    s = config["settings"]
    expected_s = {"memory_fractions", "subset_seeds", "ridge_alphas", "knn_k", "knn_temperature", "knn_k_grid", "knn_temperature_grid", "cpu_threads", "bootstrap_samples", "bootstrap_seed"}
    if not isinstance(s, dict) or set(s) != expected_s:
        raise ValueError("Unknown or missing HH offline setting")
    for name in ("memory_fractions", "ridge_alphas", "knn_temperature_grid"):
        v = s[name]
        if not isinstance(v, list) or not v or any(type(x) not in (float, int) or not np.isfinite(x) or x <= 0 for x in v) or len(set(v)) != len(v):
            raise ValueError("Expected distinct positive finite values: " + name)
    if sorted(s["memory_fractions"]) != s["memory_fractions"] or max(s["memory_fractions"]) != 1:
        raise ValueError("Memory fractions must increase and finish at 1.0")
    for name in ("subset_seeds", "knn_k_grid"):
        v = s[name]
        if not isinstance(v, list) or not v or len(set(v)) != len(v) or any(type(x) is not int or x < (1 if name == "knn_k_grid" else 0) for x in v):
            raise ValueError("Invalid integer list: " + name)
    for name in ("knn_k", "cpu_threads", "bootstrap_samples", "bootstrap_seed"):
        if type(s[name]) is not int or s[name] < (1 if name in ("knn_k", "cpu_threads") else 0):
            raise ValueError("Invalid integer setting: " + name)
    if type(s["knn_temperature"]) not in (float, int) or not np.isfinite(s["knn_temperature"]) or s["knn_temperature"] <= 0:
        raise ValueError("Invalid kNN temperature")
    if skip_transfer:
        config["transfer"] = {}
    if not isinstance(config["transfer"], dict):
        raise ValueError("transfer must map cohort names to completed evaluation folders")
    def path(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Input/output paths must be nonempty strings")
        p = Path(value).expanduser()
        return (p if p.is_absolute() else project / p).resolve()
    cohorts = {"validation": config["validation"], "test": config["test"], **config["transfer"]}
    if set(config["transfer"]) & {"validation", "test"}:
        raise ValueError("Reserved transfer cohort name")
    if any(not isinstance(v, list) or not v for v in cohorts.values()):
        raise ValueError("Each cohort must have at least one evaluation folder")
    cohort_paths = {name: [path(p) for p in v] for name, v in cohorts.items()}
    bank, memory = path(config["candidate_bank"]), path(config["memory"])
    inputs = [bank, *[memory / n for n in ("complete.json", "protocol.json", "embedding_information.json", "detectors/gap_knn.npz")]]
    inputs += [p / n for paths in cohort_paths.values() for p in paths for n in ("complete.json", "predictions.csv", "features.npz")]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise ValueError("Missing saved inputs (no models will be downloaded):\n" + "\n".join(missing))
    sources = [*Path(__file__).parent.glob("*.py"), project / "hh_offline.py",
               *[project / "code/core" / n for n in ("knn_core.py", "common.py", "evaluation.py", "chat_format.py")],
               *[project / "code/experiments/knn_distillation" / n for n in ("data.py", "io.py")],
               project / "code/experiments/experiment_cli/cli.py"]
    identity = {"version": VERSION, "recipe": config,
                "inputs": {str(p.relative_to(project)) if p.is_relative_to(project) else str(p): file_hash(p) for p in sorted(set(inputs))},
                "sources": {str(p.relative_to(project)): file_hash(p) for p in sorted(sources)},
                "versions": {n: metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")}}
    run_id = canonical_hash(identity)
    destination = path(str(output)) if output is not None else path(config["output"]) / f"comparison_{run_id[:16]}"
    if any(p.is_relative_to(destination) for p in inputs):
        raise ValueError("Output must be a separate directory outside the saved input trees")
    return {"identity": identity, "run_id": run_id, "output": destination, "bank": bank,
            "memory": memory, "cohorts": cohort_paths, "settings": s}


def knn_from_neighbors(similarity, neighbors, gaps, k, temperature):
    weights = np.exp((similarity[:, :k] - similarity[:, :1]) / temperature)
    return np.sum(weights * gaps[neighbors[:, :k]], axis=1) / weights.sum(axis=1)


def fit(train_x, train_y, val_x, val_frame, settings):
    """Selection receives memory + validation only. No test object is passed."""
    y = np.asarray(val_frame.gap, float)
    sizes = val_frame.groupby("group").group.transform("size").to_numpy()
    weights = 1 / sizes
    loss = lambda pred: float(np.average((y-pred)**2, weights=weights))
    candidates, best = [], None
    for alpha in settings["ridge_alphas"]:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
        model.fit(np.asarray(train_x, float), train_y)
        mse = loss(model.predict(val_x))
        candidates.append({"method": "ridge", "alpha": alpha, "validation_mse": mse})
        if best is None or (mse, -alpha) < best:
            best = mse, -alpha
            ridge = {"coef": model.coef_.copy(), "intercept": float(model.intercept_), "alpha": alpha}
    ks = [k for k in settings["knn_k_grid"] if k <= len(train_x)]
    if len(train_x) < settings["knn_k"] or not ks:
        raise ValueError("A conversation subset is too small for the specified kNN settings")
    similarity, neighbors = top_neighbors(val_x, train_x, max(ks + [settings["knn_k"]]))
    best, tuned = None, None
    for k in ks:
        for temperature in settings["knn_temperature_grid"]:
            mse = loss(knn_from_neighbors(similarity, neighbors, train_y, k, temperature))
            candidates.append({"method": "knn_tuned", "k": k, "temperature": temperature, "validation_mse": mse})
            if best is None or (mse, k, temperature) < best:
                best = mse, k, temperature
                tuned = {"k": k, "temperature": temperature}
    fixed = {"k": settings["knn_k"], "temperature": settings["knn_temperature"]}
    return {"ridge": ridge, "knn_fixed": fixed, "knn_tuned": tuned,
            "mean_gap": float(np.mean(train_y)), "candidates": candidates}


def predict(fitted, train_x, train_y, query):
    similarity, neighbors = top_neighbors(query, train_x, max(fitted[m]["k"] for m in ("knn_fixed", "knn_tuned")))
    return {"proxy": np.zeros(len(query)), "mean_gap": np.full(len(query), fitted["mean_gap"]),
            "ridge": query @ fitted["ridge"]["coef"] + fitted["ridge"]["intercept"],
            **{m: knn_from_neighbors(similarity, neighbors, train_y, **fitted[m]) for m in ("knn_fixed", "knn_tuned")}}


def run(plan):
    out, s = plan["output"], plan["settings"]
    manifest_path = out / "manifest.json"
    if out.exists() and any(out.iterdir()) and not manifest_path.exists():
        raise ValueError("Output is not an HH offline run; choose a new directory")
    if manifest_path.exists() and read_json(manifest_path)["identity"] != plan["identity"]:
        raise ValueError("Inputs, settings or source changed; choose a new output")
    if (out / "complete.json").exists():
        for name, expected in read_json(out / "complete.json")["artifacts"].items():
            if file_hash(out / name) != expected:
                raise ValueError("Completed analysis artifact changed: " + name)
        print("Completed report: " + str(out / "report.md"), flush=True)
        return read_json(out / "summary.json")
    started = time.monotonic()
    train, train_x, protocol = data.memory(plan["bank"], plan["memory"])
    cal = protocol["calibration"]
    validation, val_x = data.evaluation(plan["cohorts"]["validation"], cal)
    data.disjoint(train, validation)
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Memory and validation embedding dimensions differ")
    subsets = data.nested_subsets(train.group, s["memory_fractions"], s["subset_seeds"])
    write_json(manifest_path, {"identity": plan["identity"], "run_id": plan["run_id"],
               "retrospective_analysis": True, "test_used_for_tuning": False,
               "scope": "Offline gap/reward prediction; no PPO or new grader calls",
               "calibration": cal, "embedding_identity": protocol["embedding_identity"],
               "training_rows": len(train), "training_groups": int(train.group.nunique()),
               "validation_rows": len(validation), "validation_groups": int(validation.group.nunique()),
               "tuning_objective": "Validation conversation-weighted gap MSE; no refit on validation",
               "subset_unit": "Normalized first-human-turn group; retain all its memory answers",
               "subset_seeds_are_ppo_seeds": False})
    jobs, selection_rows = [], []
    with threadpool_limits(limits=s["cpu_threads"]):
        # All models and choices are frozen before any final CSV/NPZ is parsed.
        for n, subset in enumerate(subsets):
            ix = subset["indices"]
            model_id = f"fraction_{subset['fraction']:g}_sample_{subset['subset_seed']}"
            print(f"Fit {n+1}/{len(subsets)}: {model_id}; {len(ix)} labels from {subset['memory_groups']} groups", flush=True)
            fitted = fit(train_x[ix], train.gap.to_numpy()[ix], val_x, validation, s)
            meta = {k: v for k, v in subset.items() if k != "indices"}
            meta.update(model_id=model_id, memory_rows=len(ix))
            lock = {**meta, "mean_gap": fitted["mean_gap"], "ridge_alpha": fitted["ridge"]["alpha"],
                    "knn_fixed": fitted["knn_fixed"], "knn_tuned": fitted["knn_tuned"],
                    "selected_memory_indices": ix.tolist(), "test_used": False,
                    "selected_bank_ids": train.iloc[ix].bank_id.tolist()}
            write_json(out / "models" / f"{model_id}.json", lock)
            save_npz(out / "models" / f"{model_id}.npz", coef=fitted["ridge"]["coef"], intercept=np.array(fitted["ridge"]["intercept"]))
            selection_rows += [{**meta, **r} for r in fitted["candidates"]]
            jobs.append((meta, ix, fitted))
        pd.DataFrame(selection_rows).to_csv(out / "validation_candidates.csv", index=False)
        write_json(out / "selection_complete.json", {"run_id": plan["run_id"], "models": len(jobs),
                   "test_answers_opened": False, "fitted_artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted((out / "models").glob("*"))}})
        result_rows, predictions, intervals, cohorts, replay = [], [], {}, {}, {}
        for cohort, folders in plan["cohorts"].items():
            if cohort == "validation":
                continue
            print("Evaluate frozen models: " + cohort, flush=True)
            frame, query = data.evaluation(folders, cal)
            data.disjoint(train, validation, frame)
            if cohort == "test":
                test_groups = set(frame.group)
            elif test_groups & set(frame.group):
                raise ValueError("Primary test and transfer conversation groups overlap")
            cohorts[cohort] = {"answers": len(frame), "groups": int(frame.group.nunique()), "policy_ids": sorted(set(frame.policy_id))}
            for meta, ix, fitted in jobs:
                if cohort != "test" and meta["fraction"] != 1:
                    continue
                pred = predict(fitted, train_x[ix], train.gap.to_numpy()[ix], query)
                for method in METHODS:
                    result_rows.append({"cohort": cohort, **meta, "method": method,
                                        "training_labels_used": 0 if method == "proxy" else len(ix),
                                        **summarize(frame, pred[method], cal["theta"])})
                columns = frame[["group", "prompt_id", "policy_id", "proxy_z", "judge_z", "gap"]].copy()
                for method in METHODS:
                    columns[method + "_predicted_gap"] = pred[method]
                columns["cohort"], columns["model_id"] = cohort, meta["model_id"]
                predictions.append(columns)
                if meta["fraction"] == 1:
                    intervals[cohort] = paired_intervals(frame, pred["knn_tuned"], pred["ridge"], cal["theta"], s["bootstrap_samples"], s["bootstrap_seed"])
                    if cohort == "test" and "original_gap_hat" in frame:
                        error = abs(pred["knn_fixed"] - frame.original_gap_hat)
                        replay[cohort] = {"max_abs_error": float(np.max(error)), "answers_over_1e_5": int((error > 1e-5).sum()),
                                          "historical_mse": float(np.mean((frame.original_gap_hat-frame.gap)**2)),
                                          "recomputed_mse": float(np.mean((pred["knn_fixed"]-frame.gap)**2)),
                                          "reference": "Saved original_gap_hat, with original k=31 and temperature=0.05"}
        result = {"version": VERSION, "run_id": plan["run_id"], "retrospective": True,
                  "cohorts": cohorts, "results": result_rows, "paired_full_memory_intervals": intervals,
                  "interval_scope": "Paired conversation bootstrap, conditional on selected predictors and fixed saved policies; not PPO-seed uncertainty",
                  "original_knn_replay": replay, "settings": s, "seconds": time.monotonic()-started}
        pd.DataFrame(result_rows).to_csv(out / "metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(out / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        write_json(out / "summary.json", result)
        from .report import report
        report(out, result, read_json(manifest_path))
    write_json(out / "complete.json", {"run_id": plan["run_id"], "artifacts": {
        str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "complete.json"}})
    print("Report: " + str(out / "report.md"), flush=True)
    return result


def main(argv=None, project=None):
    project = Path(project) if project is not None else Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"])
    parser.add_argument("--config", type=Path, default=project / "configs/experiments/hh-offline.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-transfer", action="store_true", help="Run only the primary follow-up comparison")
    parser.add_argument("--dry-run", action="store_true", help="Check input availability and print the plan; write nothing")
    args = parser.parse_args(argv)
    try:
        plan = resolve(project, args.config, args.output, args.skip_transfer)
        if args.dry_run:
            print(json.dumps({"output": str(plan["output"]), "settings": plan["settings"],
                              "cohorts": {k: len(v) for k, v in plan["cohorts"].items()},
                              "gpu_required": False, "new_grader_calls": 0, "starts_ppo": False}, indent=2))
        else:
            run(plan)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0
