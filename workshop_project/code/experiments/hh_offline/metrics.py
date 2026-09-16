"""Gap fidelity and judge-reward ordering; no task-correctness labels are inferred."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and min(np.ptp(a), np.ptp(b)) > 0 else None


def summarize(frame, prediction, theta):
    truth, prediction = np.asarray(frame.gap, float), np.asarray(prediction, float)
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite and aligned with evaluation answers")
    mse = float(np.mean((truth - prediction) ** 2))
    variance = float(np.var(truth))
    labels = truth > theta
    judge = np.asarray(frame.judge_z, float)
    corrected = np.asarray(frame.proxy_z, float) - prediction
    # Within-conversation comparisons are restricted to unequal judge scores.
    # This agrees with the judge's preferences, not necessarily a human's.
    wins, pairs = 0., 0
    for indices in frame.groupby("group", sort=True).indices.values():
        for at, i in enumerate(indices):
            for j in indices[:at]:
                if judge[i] == judge[j]:
                    continue
                signed = (judge[i] - judge[j]) * (corrected[i] - corrected[j])
                wins += 1. if signed > 0 else .5 if signed == 0 else 0.
                pairs += 1
    return {"answers": len(frame), "conversation_groups": int(frame.group.nunique()),
            "high_gap_count": int(labels.sum()), "gap_mse": mse,
            "gap_rmse": float(np.sqrt(mse)), "gap_mae": float(np.mean(abs(truth - prediction))),
            "gap_r2": 1-mse/variance if np.ptp(truth) > 0 and variance > 0 else None,
            "gap_pearson": corr(truth, prediction), "gap_spearman": corr(rankdata(truth), rankdata(prediction)),
            "high_gap_auroc": float(roc_auc_score(labels, prediction)) if 0 < labels.sum() < len(labels) else None,
            "high_gap_ap": float(average_precision_score(labels, prediction)) if labels.any() else None,
            "corrected_judge_mse": mse,
            "corrected_judge_pearson": corr(judge, corrected),
            "judge_preference_pairs": pairs, "judge_preference_agreement": wins/pairs if pairs else None}


def paired_intervals(frame, knn, ridge, theta, samples, seed):
    """Paired question bootstrap, conditional on fitted predictors and saved policies."""
    groups = list(frame.groupby("group", sort=True).indices.values())
    truth = np.asarray(frame.gap, float)
    labels = truth > theta
    values = {"ridge_minus_knn_gap_mse": [], "ridge_minus_knn_high_gap_auroc": []}
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        ix = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        values["ridge_minus_knn_gap_mse"].append(float(np.mean((truth[ix]-ridge[ix])**2 - (truth[ix]-knn[ix])**2)))
        if 0 < labels[ix].sum() < len(ix):
            values["ridge_minus_knn_high_gap_auroc"].append(float(roc_auc_score(labels[ix], ridge[ix])-roc_auc_score(labels[ix], knn[ix])))
    return {key: {"ci95": np.quantile(v, [.025, .975]).tolist() if v else None,
                  "valid_resamples": len(v)} for key, v in values.items()}
