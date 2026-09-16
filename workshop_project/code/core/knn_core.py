"""CPU-only geometry, labeling, validation selection, and detector fitting."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from importlib import metadata
import hashlib
import io
import json
import math
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

MODEL_ID = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
MODEL_REVISION = "8c14a4e9e6321deaf572544339b16b8d6bbe8886"
BANK_HASH = "32632fcdfb17535e775db4b27c78b65e5433cab380c150bad25753b10ae41589"
SPLITS = ("training", "model_validation", "calibration", "selection", "test")
METHOD_NAMES = {
    "gap_knn": "kNN predicted signed gap (main)",
    "positive_prototypes": "High-gap prototype memory",
    "positive_knn": "High-gap kNN (uncompressed)",
    "mixed_knn": "Two-class kNN",
    "linear_probe": "Linear probe on frozen embeddings",
    "proxy_score": "Proxy score only",
    "answer_length": "Answer length only",
    "saved_two_head_gap_finder": "Saved two-head GapFinder (seed 42)",
    "saved_direct_gap_classifier": "Saved direct classifier (seed 42)",
    "saved_teacher_score_pair": "Saved teacher score + pairwise (seed 42)",
    "saved_teacher_score_only": "Saved teacher score only (seed 42)",
}


@dataclass(frozen=True)
class Settings:
    gap_k: tuple = (5, 15, 31, 63)
    gap_temperatures: tuple = (0.05, 0.10, 0.20)
    distance_gate_quantile: float = 0.95
    insertion_radii: tuple = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
    positive_k: tuple = (1, 3, 5, 10, 25)
    mixed_k: tuple = (5, 15, 31, 63)
    probe_c: tuple = (0.01, 0.1, 1.0, 10.0)
    distance_temperature: float = 0.1
    audit_fraction: float = 0.10
    bootstrap_draws: int = 1000
    bootstrap_seed: int = 20260909
    cpu_threads: int = 8
    query_batch_size: int = 128


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def versions(names=("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers")):
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_bank(root):
    root = Path(root)
    path = root / "data/candidate_bank.csv"
    if file_hash(path) != BANK_HASH:
        raise ValueError("The packaged candidate bank changed. Restore the original data/candidate_bank.csv.")
    bank = pd.read_csv(path, keep_default_na=False)
    calibration = json.loads((root / "data/calibration.json").read_text())
    if set(bank.split) != set(SPLITS) or bank.groupby("prompt").split.nunique().max() != 1:
        raise ValueError("Unexpected splits or prompt leakage.")
    if bank.duplicated(["prompt", "answer"]).any():
        raise ValueError("Duplicate prompt-answer rows.")
    for column in ("prompt", "answer"):
        if not bank[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError("Empty or invalid " + column)
    numeric = bank[["proxy_raw", "proxy_z", "judge_raw", "judge_z"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite reference rewards.")
    for model in ("proxy", "judge"):
        if calibration[model + "_std"] <= 0:
            raise ValueError("Invalid reference standard deviation.")
        expected = (bank[model + "_raw"] - calibration[model + "_mean"]) / calibration[model + "_std"]
        if not np.allclose(expected, bank[model + "_z"], atol=1e-7, rtol=1e-7):
            raise ValueError("Normalization mismatch for " + model)
    bank["row_id"] = [hashlib.sha256((p + "\0" + a).encode()).hexdigest()
                      for p, a in zip(bank.prompt, bank.answer)]
    bank["gap"] = bank.proxy_z - bank.judge_z
    bank["high_gap"] = (bank.gap > calibration["theta"]).astype(int)
    for split, frame in bank.groupby("split"):
        if frame.high_gap.nunique() != 2:
            raise ValueError(split + " does not contain both labels.")
    return bank, calibration


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Features must be a finite matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Zero-length feature vector.")
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def top_neighbors(query, reference, k, batch_size=128):
    """Exact cosine neighbors. Equal similarities use reference-row order."""
    query = np.asarray(query, np.float32)
    reference = np.asarray(reference, np.float32)
    if len(reference) == 0 or k < 1 or k > len(reference):
        raise ValueError("k must fit the nonempty reference memory.")
    if query.shape[1] != reference.shape[1]:
        raise ValueError("Feature dimensions differ.")
    similarities = np.empty((len(query), k), dtype=np.float32)
    indices = np.empty((len(query), k), dtype=np.int64)
    for start in range(0, len(query), batch_size):
        end = min(start + batch_size, len(query))
        matrix = np.clip(query[start:end] @ reference.T, -1.0, 1.0)
        # Stable complete sorting is inexpensive for this ~8k-row memory and
        # resolves ties even at the kth boundary reproducibly.
        order = np.argsort(-matrix, axis=1, kind="stable")[:, :k]
        similarities[start:end] = np.take_along_axis(matrix, order, axis=1)
        indices[start:end] = order
    return similarities, indices


def build_prototypes(positive_vectors, radius):
    """Greedy radius cover, fixed representatives; input must have stable order."""
    if radius < 0 or radius > 2 or not len(positive_vectors):
        raise ValueError("Invalid prototype radius or empty positive memory.")
    # radius=0 is the explicit uncompressed control, including exact duplicates.
    if radius == 0:
        ids = np.arange(len(positive_vectors))
        return ids, ids.copy()
    representatives, assignment = [], []
    for index, vector in enumerate(positive_vectors):
        if not representatives:
            representatives.append(index)
            assignment.append(0)
            continue
        similarities = np.clip(positive_vectors[representatives] @ vector, -1, 1)
        nearest = int(np.argmax(similarities))
        if 1.0 - float(similarities[nearest]) > radius:
            representatives.append(index)
            nearest = len(representatives) - 1
        assignment.append(nearest)
    return np.asarray(representatives, int), np.asarray(assignment, int)


def metrics(frame, scores, fraction=0.1, gate=None):
    scores = np.asarray(scores, float)
    labels = frame.high_gap.to_numpy(int)
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Risk scores do not align with evaluation labels.")
    if not len(labels) or not 0 < fraction <= 1:
        raise ValueError("Invalid evaluation set or audit fraction.")
    positives = int(labels.sum())
    k = max(1, int(math.ceil(fraction * len(labels))))
    order = np.lexsort((frame.row_id.to_numpy(str), -scores))
    true_positives = int(labels[order[:k]].sum())
    result = {
        "answers": len(labels), "prompts": int(frame.prompt.nunique()),
        "positives": positives, "prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)) if 0 < positives < len(labels) else None,
        "average_precision": float(average_precision_score(labels, scores)) if positives else None,
        "reviewed": k, "caught": true_positives, "precision_at_10pct": true_positives / k,
        "recall_at_10pct": true_positives / positives if positives else None,
        "lift_over_random": (true_positives / k) / labels.mean() if positives else None,
    }
    if gate is not None:
        flagged = scores > gate
        tp = int(labels[flagged].sum())
        result.update(gate=float(gate), gate_flagged=int(flagged.sum()),
                      gate_precision=tp / int(flagged.sum()) if flagged.any() else None,
                      gate_recall=tp / positives if positives else None)
    return result


def calibrate(scores, labels):
    """Monotonic Platt fit on the reserved calibration split only."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    mean, std = float(scores.mean()), max(float(scores.std()), 1e-8)
    z = (scores - mean) / std
    def objective(parameters):
        a, b = parameters
        logits = a * z + b
        error = expit(logits) - labels
        loss = np.mean(np.logaddexp(0, logits) - labels * logits) + 1e-4 * a * a
        gradient = np.array([np.mean(error * z) + 2e-4 * a, np.mean(error)])
        return loss, gradient
    initial_b = float(np.log(labels.mean() / (1 - labels.mean())))
    fitted = minimize(objective, [1.0, initial_b], jac=True, method="L-BFGS-B",
                      bounds=[(0, None), (None, None)])
    if not fitted.success:
        raise RuntimeError("Probability calibration failed: " + str(fitted.message))
    return {"mean": mean, "std": std, "a": float(fitted.x[0]), "b": float(fitted.x[1])}


def calibrated_probability(scores, parameters):
    return expit(parameters["a"] * (np.asarray(scores) - parameters["mean"]) / parameters["std"]
                 + parameters["b"])


def previous_baselines(root, test, calibration):
    """Saved seed-42 predictions; no model loading and no refitting."""
    archive = Path(root) / "baselines/previous_comparison.zip"
    result = {}
    with zipfile.ZipFile(archive) as z:
        protocol = json.loads(z.read("results/comparison_protocol.json"))
        if protocol["bank_sha256"] != BANK_HASH:
            raise ValueError("Saved comparison uses a different bank.")
        for key in ("theta", "proxy_mean", "proxy_std", "judge_mean", "judge_std"):
            if not np.isclose(calibration[key], protocol["calibration"][key], atol=1e-10, rtol=0):
                raise ValueError("Saved comparison normalization/threshold differs.")
        for method in ("two_head_gap_finder", "direct_gap_classifier", "teacher_score_pair", "teacher_score_only"):
            frame = pd.read_csv(io.BytesIO(z.read(f"results/{method}/seed=42/test_predictions.csv")))
            if frame.row_id.duplicated().any() or set(frame.row_id) != set(test.row_id):
                raise ValueError("Saved baseline has unmatched test rows.")
            result["saved_" + method] = frame.set_index("row_id").loc[test.row_id].risk_score.to_numpy(float)
    return result


def fit_detectors(bank, features, settings, output):
    """No test labels or test features are used to select a detector."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_ids = bank.index[bank.split == "training"].to_numpy()
    train_ids = train_ids[np.argsort(bank.iloc[train_ids].row_id.to_numpy(), kind="stable")]
    val_ids = bank.index[bank.split == "model_validation"].to_numpy()
    train = features[train_ids]
    y_train = bank.iloc[train_ids].high_gap.to_numpy(int)
    positive_local = np.flatnonzero(y_train)
    positive = train[positive_local]
    val = features[val_ids]
    y_val = bank.iloc[val_ids].high_gap.to_numpy(int)
    candidates, fitted = [], {}
    def record(method, parameters, scores, memory_size):
        row = {"method": method, "parameters": json.dumps(parameters, sort_keys=True),
               "validation_ap": float(average_precision_score(y_val, scores)),
               "validation_auroc": float(roc_auc_score(y_val, scores)),
               "memory_vectors": int(memory_size)}
        candidates.append(row)
        return row["validation_ap"]

    with threadpool_limits(limits=settings.cpu_threads):
        ks = [k for k in settings.gap_k if k <= len(train)]
        if not ks or any(t <= 0 for t in settings.gap_temperatures):
            raise ValueError("Invalid gap-regression k/temperature.")
        similarities, neighbors = top_neighbors(val, train, max(ks), settings.query_batch_size)
        training_gaps = bank.iloc[train_ids].gap.to_numpy(float)
        validation_gaps = bank.iloc[val_ids].gap.to_numpy(float)
        best = None
        for k in ks:
            for temperature in settings.gap_temperatures:
                weights = np.exp((similarities[:, :k] - similarities[:, :1]) / temperature)
                predicted = (weights * training_gaps[neighbors[:, :k]]).sum(1) / weights.sum(1)
                mse = float(np.mean((predicted - validation_gaps) ** 2))
                record("gap_knn", {"k": k, "temperature": temperature}, predicted, len(train))
                candidates[-1]["validation_gap_mse"] = mse
                key = (-mse, -k, -temperature)
                if best is None or key > best:
                    best = key
                    fitted["gap_knn"] = {"k": k, "temperature": temperature, "vectors": train,
                                         "gaps": training_gaps, "bank_ids": train_ids}
        best = None
        for radius in settings.insertion_radii:
            representatives, assignment = build_prototypes(positive, radius)
            similarities, _ = top_neighbors(val, positive[representatives], 1, settings.query_batch_size)
            ap = record("positive_prototypes", {"radius": radius}, similarities[:, 0], len(representatives))
            # AP primary, smaller memory secondary, radius tertiary. Test never consulted.
            key = (ap, -len(representatives), -radius)
            if best is None or key > best:
                best = key
                fitted["positive_prototypes"] = {
                    "radius": radius, "k": 1, "vectors": positive[representatives],
                    "bank_ids": train_ids[positive_local[representatives]],
                    "assignment": assignment, "positive_bank_ids": train_ids[positive_local],
                }
        ks = [k for k in settings.positive_k if k <= len(positive)]
        if not ks:
            raise ValueError("No positive k fits the training memory.")
        similarities, _ = top_neighbors(val, positive, max(ks), settings.query_batch_size)
        best = None
        for k in ks:
            ap = record("positive_knn", {"k": k}, similarities[:, :k].mean(axis=1), len(positive))
            if best is None or (ap, -k) > best:
                best = (ap, -k)
                fitted["positive_knn"] = {"k": k, "vectors": positive,
                                          "bank_ids": train_ids[positive_local]}
        ks = [k for k in settings.mixed_k if k <= len(train)]
        if not ks or settings.distance_temperature <= 0:
            raise ValueError("Invalid mixed k or temperature.")
        similarities, neighbors = top_neighbors(val, train, max(ks), settings.query_batch_size)
        best = None
        for k in ks:
            weights = np.exp((similarities[:, :k] - similarities[:, :1]) / settings.distance_temperature)
            scores = (weights * y_train[neighbors[:, :k]]).sum(axis=1) / weights.sum(axis=1)
            ap = record("mixed_knn", {"k": k, "temperature": settings.distance_temperature}, scores, len(train))
            if best is None or (ap, -k) > best:
                best = (ap, -k)
                fitted["mixed_knn"] = {"k": k, "temperature": settings.distance_temperature,
                                       "vectors": train, "labels": y_train, "bank_ids": train_ids}
        best = None
        for c in settings.probe_c:
            probe = LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                       max_iter=3000, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                probe.fit(train, y_train)
            scores = probe.decision_function(val)
            ap = record("linear_probe", {"C": c}, scores, 0)
            if best is None or (ap, -c) > best:
                best = (ap, -c)
                fitted["linear_probe"] = {"C": c, "coef": probe.coef_[0].copy(),
                                           "intercept": float(probe.intercept_[0])}
    pd.DataFrame(candidates).to_csv(output / "validation_sweep.csv", index=False)
    return fitted


def predict_detector(model, method, query, settings):
    if method == "linear_probe":
        return np.asarray(query @ model["coef"] + model["intercept"], float), {}
    similarities, indices = top_neighbors(query, model["vectors"], model["k"], settings.query_batch_size)
    details = {"nearest_bank_index": model["bank_ids"][indices[:, 0]],
               "nearest_cosine_distance": 1.0 - similarities[:, 0],
               "mean_neighbor_distance": (1.0 - similarities).mean(1)}
    if method in ("mixed_knn", "gap_knn"):
        weights = np.exp((similarities - similarities[:, :1]) / model["temperature"])
        targets = model["gaps"] if method == "gap_knn" else model["labels"]
        score = (weights * targets[indices]).sum(axis=1) / weights.sum(axis=1)
        if method == "gap_knn":
            details["neighbor_gap_std"] = np.sqrt(
                (weights * (targets[indices] - score[:, None]) ** 2).sum(1) / weights.sum(1))
            details["neighbor_indices"] = model["bank_ids"][indices]
            details["neighbor_weights"] = weights / weights.sum(1, keepdims=True)
    else:
        score = similarities.mean(axis=1)
    return np.asarray(score, float), details


def bootstrap_all(test, score_map, settings):
    """Identical prompt-cluster resamples for all methods; fixed fitted detectors."""
    groups = list(test.groupby("prompt", sort=False).indices.values())
    rng = np.random.default_rng(settings.bootstrap_seed)
    names = list(score_map)
    keys = ("auroc", "average_precision", "recall_at_10pct", "precision_at_10pct")
    values = {name: [] for name in names}
    for draw in range(settings.bootstrap_draws):
        ids = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        frame = test.iloc[ids]
        if frame.high_gap.nunique() != 2:
            continue
        for name in names:
            row = metrics(frame, np.asarray(score_map[name])[ids], settings.audit_fraction)
            values[name].append([row[k] for k in keys])
        if (draw + 1) % 200 == 0:
            print(f"Prompt bootstrap {draw + 1}/{settings.bootstrap_draws}", flush=True)
    intervals, comparisons = [], []
    for name in names:
        a = np.asarray(values[name])
        for i, key in enumerate(keys):
            lo, hi = np.quantile(a[:, i], [.025, .975])
            intervals.append({"method": name, "metric": key, "ci95_low": lo, "ci95_high": hi})
    for other in ("positive_prototypes", "mixed_knn", "linear_probe", "saved_teacher_score_pair"):
        if other not in values:
            continue
        delta = np.asarray(values["gap_knn"]) - np.asarray(values[other])
        left = metrics(test, score_map["gap_knn"], settings.audit_fraction)
        right = metrics(test, score_map[other], settings.audit_fraction)
        for i, key in enumerate(keys):
            lo, hi = np.quantile(delta[:, i], [.025, .975])
            comparisons.append({"comparison": "gap_knn minus " + other,
                                "metric": key, "delta": left[key] - right[key],
                                "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(intervals), pd.DataFrame(comparisons)
