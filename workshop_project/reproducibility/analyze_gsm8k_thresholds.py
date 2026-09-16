"""Exploratory diagnostics from saved GSM8K scores; never changes run artifacts.

Run from the repository root. Requires NumPy, but no model/GPU dependencies.
The candidate label definitions are ranked on selection data before final data
is loaded. This is a post hoc analysis, not a new confirmatory experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code/experiments"))
from gsm8k_experiment.metrics import auroc

QUANTILES = [i / 100 for i in range(5, 100, 5)]


def labels(gaps, threshold, comparator):
    return gaps >= threshold if comparator == ">=" else gaps > threshold


def support(y, ids):
    return {"n": len(y), "positive": int(y.sum()), "negative": int((~y).sum()),
            "positive_questions": len(set(ids[y])),
            "negative_questions": len(set(ids[~y]))}


def bootstrap_auc(y, scores, ids, samples, seed):
    """Resample question IDs, retaining all answers belonging to each question."""
    groups = [np.flatnonzero(ids == key) for key in sorted(set(ids))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        value = auroc(y[index], scores[index])
        values.append(np.nan if value is None else value)
    return np.asarray(values)


def interval(values):
    valid = values[np.isfinite(values)]
    return np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None


def operating_point(y, scores):
    """Choose predicted-gap >= t to maximize selection balanced accuracy.

    This changes a confusion matrix, not the continuous-score AUROC.
    Ties prefer the higher threshold (fewer flagged answers).
    """
    candidates = []
    for t in np.unique(scores):
        detected = scores >= t
        tp, fp = int((y & detected).sum()), int((~y & detected).sum())
        fn, tn = int((y & ~detected).sum()), int((~y & ~detected).sum())
        recall, specificity = tp / (tp + fn), tn / (tn + fp)
        candidates.append({"threshold": float(t), "comparator": ">=",
                           "balanced_accuracy": (recall + specificity) / 2,
                           "recall": recall, "specificity": specificity,
                           "precision": tp / (tp + fp),
                           "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    return max(candidates, key=lambda x: (x["balanced_accuracy"], x["threshold"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "results/gsm8k/b200")
    parser.add_argument("--output", type=Path, default=PROJECT / "docs/gsm8k/b200_seed42")
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()
    root, out = args.input.resolve(), args.output.resolve()
    if root == out or root in out.parents:
        parser.error("Write the exploratory analysis outside the original run directory.")
    if args.samples < 1:
        parser.error("--samples must be positive")
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(relative):
        path = root / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".jsonl":
            # File iteration splits physical lines, not Unicode separators in answers.
            with path.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        return json.loads(path.read_text(encoding="utf-8"))

    def write_json(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    selection = {}
    excluded_ids = set()
    candidate_table = []
    for teacher, folder in [("4B", "prepared"), ("30B", "prepared_30b")]:
        norm = read(f"{folder}/normalization.json")
        cal = read(f"{folder}/calibration_raw.jsonl")
        memory = read(f"{folder}/memory_raw.jsonl")
        rows = read(f"{folder}/selection_scored.jsonl")
        cohorts = [set(r["id"] for r in cohort) for cohort in (cal, memory, rows)]
        assert all(not cohorts[i] & cohorts[j] for i in range(3) for j in range(i)), "Cohort overlap"
        excluded_ids.update(set.union(*cohorts))

        def gap(r):
            return ((r["proxy_score"] - norm["proxy_mean"]) / norm["proxy_std"]
                    - (r["judge_score"] - norm["judge_mean"]) / norm["judge_std"])

        cal_gaps = np.array([gap(r) for r in cal])
        gaps, scores = (np.array([r[key] for r in rows]) for key in ("gap", "predicted_gap"))
        ids = np.array([r["id"] for r in rows])
        assert np.isfinite(gaps).all() and np.isfinite(scores).all()
        assert np.allclose(gaps, [gap(r) for r in rows], rtol=0, atol=1e-12)
        assert float(np.quantile(cal_gaps, 0.95)) == norm["threshold"]
        candidates = [(q, ">", float(np.quantile(cal_gaps, q))) for q in QUANTILES]
        candidates.append((0.95, ">=", norm["threshold"]))
        # Include every observed calibration cutoff, even those between grid
        # quantiles, and the interpretable zero-gap boundary.
        candidates.extend((None, ">", float(t)) for t in sorted(set(cal_gaps) | {0.0}))
        cache, records = {}, []
        for q, op, threshold in candidates:
            y = labels(gaps, threshold, op)
            key = y.tobytes()
            counts = support(y, ids)
            if key not in cache:
                cache[key] = (bootstrap_auc(y, scores, ids, args.samples, 42)
                              if y.any() and not y.all() else np.full(args.samples, np.nan))
            ci = interval(cache[key])
            cal_positive = int(labels(cal_gaps, threshold, op).sum())
            record = {"teacher": teacher, "quantile": q, "comparator": op,
                      "threshold": threshold, "calibration_positive": cal_positive,
                      **counts, "auroc": auroc(y, scores), "ci95_low": ci[0] if ci else None,
                      "ci95_high": ci[1] if ci else None,
                      "valid_bootstraps": int(np.isfinite(cache[key]).sum()),
                      "eligible": min(counts[k] for k in ("positive", "negative", "positive_questions", "negative_questions")) >= 20,
                      "upper_tail": threshold >= 0 and 0 < cal_positive <= len(cal) / 2}
            records.append(record)

        # If several cutoffs give the same ranking score, prefer strict > and
        # then the largest quantile. All aliases remain visible in the CSV.
        ranked = sorted((r for r in records if r["eligible"]),
                        key=lambda r: (r["auroc"], r["comparator"] == ">", r["quantile"] or -1), reverse=True)
        if not ranked:
            raise ValueError(f"No supported threshold for {teacher}")
        unrestricted_winner = ranked[0]
        ranked = [r for r in ranked if r["upper_tail"]]
        if not ranked:
            raise ValueError(f"No supported upper-tail threshold for {teacher}")
        winner = ranked[0]
        winner_y = labels(gaps, winner["threshold"], winner["comparator"])
        runner = next((r for r in ranked if not np.array_equal(
            winner_y, labels(gaps, r["threshold"], r["comparator"]))), None)
        difference = None
        if runner:
            runner_y = labels(gaps, runner["threshold"], runner["comparator"])
            difference = {"runner_up": runner, "auroc_difference": winner["auroc"] - runner["auroc"],
                          "paired_question_bootstrap_ci95": interval(cache[winner_y.tobytes()] - cache[runner_y.tobytes()])}
        all_possible_gaps = np.array([gap({"proxy_score": p, "judge_score": j})
                                     for p in range(1, 6) for j in range(1, 6)])
        selection[teacher] = {"normalization": norm, "calibration_questions": len(cohorts[0]),
                              "selection_questions": len(cohorts[2]), "winner": winner,
                              "unrestricted_winner": unrestricted_winner,
                              "winner_vs_next_distinct_definition": difference,
                              "selection_operating_point": operating_point(winner_y, scores),
                              "q90_strict_equals_q95_inclusive_for_all_25_grade_pairs": bool(np.array_equal(
                                  all_possible_gaps > np.quantile(cal_gaps, 0.9), all_possible_gaps >= norm["threshold"]))}
        candidate_table.extend(records)

    # Persist selection choices before opening final response files. This records
    # analysis order; it does not claim a preregistration before the original run.
    settings = {"analysis": "exploratory, post hoc; seed 42 only", "quantiles": QUANTILES,
                "candidate_definition": "strict > at calibration Q05..Q95 in steps of 5 percentiles, every distinct observed calibration gap, and zero; inclusive >= Q95",
                "selection_rule": "maximum selection AUROC among upper-tail definitions (nonnegative cutoff; <=50% calibration positives); >=20 selection answers and >=20 selection questions in each class; ties prefer > then highest quantile",
                "unrestricted_rule": "same selection ranking without the upper-tail constraint; reported separately, not used for final diagnostic labels",
                "bootstrap": {"samples": args.samples, "seed": 42, "unit": "question with all its answers",
                              "scope": "fixed fitted memories, normalization and label definitions; no selection or seed-variability adjustment"},
                "selection": selection}
    write_json("threshold_selection.json", settings)

    winner = selection["4B"]["winner"]
    final = []
    for arm in ["base", "proxy", "judge", "knn_static", "knn_static_30b"]:
        step = 0 if arm == "base" else 400
        rows = read(f"evaluations/final/{arm}/step_{step:06d}/responses.jsonl")
        ids = np.array([r["id"] for r in rows])
        assert len(set(ids)) == len(ids) and not set(ids) & excluded_ids
        assert all(r["diagnostic_teacher"] == r["diagnostic_memory_teacher"] == "judge" for r in rows)
        y = labels(np.array([r["gap"] for r in rows]), winner["threshold"], winner["comparator"])
        scores = np.array([r["predicted_gap"] for r in rows])
        ci = interval(bootstrap_auc(y, scores, ids, args.samples, 42))
        final.append({"policy": arm, "diagnostic_teacher": "4B", "threshold": winner["threshold"],
                      "comparator": winner["comparator"], **support(y, ids),
                      "auroc": auroc(y, scores), "ci95_low": ci[0] if ci else None, "ci95_high": ci[1] if ci else None})

    for filename, rows in [("threshold_candidates.csv", candidate_table), ("threshold_final_metrics.csv", final)]:
        with (out / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for relative, sha in hashes.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == sha, "Input changed during analysis"
    write_json("threshold_audit.json", {"input_sha256": hashes, "numpy_version": np.__version__,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "metrics_source_sha256": hashlib.sha256((PROJECT / "code/experiments/gsm8k_experiment/metrics.py").read_bytes()).hexdigest(),
               "checks": {"calibration_memory_selection_disjoint": True, "final_disjoint_from_preparation": True,
                          "saved_gaps_recomputed": True, "original_q95_recomputed": True, "inputs_unchanged": True},
               "final": final})
    print(json.dumps({"selection": selection, "final": final}, indent=2))


if __name__ == "__main__":
    main()
