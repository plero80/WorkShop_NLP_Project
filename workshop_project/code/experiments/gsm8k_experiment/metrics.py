from __future__ import annotations

import numpy as np


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and min(a.std(), b.std()) > 0 else None


def auroc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if not npos or not nneg:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def summarize_rows(rows, threshold):
    def avg(key):
        values = [x[key] for x in rows if x.get(key) is not None]
        return float(np.mean(values)) if values else None
    paired = [x for x in rows if x.get("gap") is not None and x.get("predicted_gap") is not None]
    graded = [x for x in rows if x.get("proxy_score") is not None and x.get("judge_score") is not None]
    true = np.asarray([x["gap"] > threshold for x in paired], dtype=bool)
    detected = np.asarray([x["predicted_gap"] > threshold for x in paired], dtype=bool)
    tp, fp = int((true & detected).sum()), int((~true & detected).sum())
    fn, tn = int((true & ~detected).sum()), int((~true & ~detected).sum())
    gap = np.asarray([x["gap"] for x in paired], dtype=float)
    pred = np.asarray([x["predicted_gap"] for x in paired], dtype=float)
    def mean(values):
        return float(np.mean(values)) if len(values) else None
    numeric_metrics = {}
    if rows and all("numeric_match" in row for row in rows):
        numeric_metrics = {"numeric_accuracy": avg("numeric_match"),
                           "numeric_matches": sum(int(x["numeric_match"]) for x in rows),
                           "numeric_unresolved": sum(int(x["numeric_unresolved"]) for x in rows),
                           "numeric_unresolved_rate": avg("numeric_unresolved"),
                           "numeric_mismatches": sum(not x["numeric_match"] and not x["numeric_unresolved"] for x in rows)}
    return {**numeric_metrics, "n": len(rows), "n_proxy_scored": sum(x.get("proxy_score") is not None for x in rows),
            "n_judge_scored": sum(x.get("judge_score") is not None for x in rows),
            "n_pair_scored": len(graded), "n_gap_scored": len(paired), "n_unscored": len(rows) - len(graded),
            "accuracy": avg("correct"), "format_valid_rate": avg("format_valid"),
            "mean_proxy_score": avg("proxy_score"), "mean_judge_score": avg("judge_score"),
            "mean_gap": avg("gap"), "mean_predicted_gap": avg("predicted_gap"),
            "gap_mse": mean((gap - pred)**2), "gap_mae": mean(np.abs(gap - pred)),
            "zero_gap_baseline_mse": mean(gap**2),
            "proxy_judge_pearson": safe_corr([x["proxy_score"] for x in graded], [x["judge_score"] for x in graded]),
            "high_gap_rate": mean(true), "gap_threshold": threshold,
            "high_gap_auroc": auroc(true, pred), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mean_response_tokens": avg("response_tokens"), "length_cap_rate": avg("length_capped")}


def paired_bootstrap(left_rows, right_rows, samples=2000, seed=42, metric="correct"):
    left, right = {r["id"]: r for r in left_rows}, {r["id"]: r for r in right_rows}
    if set(left) != set(right) or len(left) != len(left_rows) or len(right) != len(right_rows):
        raise ValueError("Paired evaluation requires exactly one response per identical question ID.")
    ids = sorted(left)
    delta = np.array([float(left[k][metric]) - float(right[k][metric]) for k in ids])
    rng = np.random.default_rng(seed)
    boot = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(samples)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"metric": metric, "accuracy_difference": float(delta.mean()), "ci95": [float(lo), float(hi)], "n": len(ids),
            "left_only_correct": int((delta == 1).sum()), "right_only_correct": int((delta == -1).sum()),
            "uncertainty_scope": "paired bootstrap over questions; does not measure training-seed variability"}
