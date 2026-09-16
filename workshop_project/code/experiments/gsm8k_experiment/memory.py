from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from knn_core import top_neighbors

from .common import atomic_json, finite, read_json


@dataclass
class Normalization:
    proxy_mean: float
    proxy_std: float
    judge_mean: float
    judge_std: float
    threshold: float

    @classmethod
    def fit(cls, proxy, judge, quantile, minimum_std):
        proxy, judge = np.asarray(proxy, float), np.asarray(judge, float)
        finite(proxy, "calibration proxy ratings")
        finite(judge, "calibration judge ratings")
        if len(proxy) != len(judge) or len(proxy) < 2:
            raise ValueError("Need aligned calibration scores.")
        mp, sp, mj, sj = float(proxy.mean()), float(proxy.std()), float(judge.mean()), float(judge.std())
        if min(sp, sj) < minimum_std:
            raise ValueError(f"Degenerate judge calibration: proxy std={sp:.5f}, judge std={sj:.5f}. This scoring protocol has insufficient variation. Inspect calibration responses; do not fabricate a gap by dividing by an epsilon.")
        gaps = (proxy - mp) / sp - (judge - mj) / sj
        return cls(mp, sp, mj, sj, float(np.quantile(gaps, quantile)))

    def proxy_z(self, values):
        return (np.asarray(values, dtype=float) - self.proxy_mean) / self.proxy_std

    def judge_z(self, values):
        return (np.asarray(values, dtype=float) - self.judge_mean) / self.judge_std

    def gap(self, proxy, judge):
        return self.proxy_z(proxy) - self.judge_z(judge)

    def save(self, path):
        atomic_json(path, self.__dict__)

    @classmethod
    def load(cls, path):
        return cls(**read_json(path))


class GapMemory:
    def __init__(self, embeddings, gaps, group_ids, k, temperature, encoder_identity):
        self.embeddings = np.asarray(embeddings, np.float32)
        self.gaps = np.asarray(gaps, np.float32)
        self.group_ids = np.asarray(group_ids, dtype=str)
        self.k, self.temperature, self.encoder_identity = int(k), float(temperature), encoder_identity
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.gaps) or len(self.gaps) != len(self.group_ids):
            raise ValueError("Memory arrays have incompatible shapes.")
        if self.k < 1 or len(self.gaps) < self.k or self.temperature <= 0:
            raise ValueError("Memory is too small for k or invalid temperature.")
        finite(self.embeddings, "memory embeddings")
        finite(self.gaps, "memory gaps")
        if not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-4):
            raise ValueError("Memory embeddings must be L2-normalized.")

    def predict(self, embeddings, query_ids=None, encoder_identity=None):
        if encoder_identity is not None and encoder_identity != self.encoder_identity:
            raise ValueError("Frozen proxy encoder/reward protocol mismatch; rebuild memory.")
        q = np.asarray(embeddings, np.float32)
        finite(q, "query embeddings")
        if q.ndim != 2 or q.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension differs from memory.")
        if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4):
            raise ValueError("Query embeddings must be L2-normalized.")
        predictions, similarities, neighbors = [], [], []
        for i, query in enumerate(q):
            valid = np.ones(len(self.embeddings), dtype=bool)
            if query_ids is not None:
                valid &= self.group_ids != str(query_ids[i])
            eligible = np.flatnonzero(valid)
            if len(eligible) < self.k:
                raise ValueError("Too few neighbors after excluding the entire query question group.")
            # Stable exact cosine top-k; no approximate-index accuracy confound.
            scores, indices = top_neighbors(query[None, :], self.embeddings[eligible], self.k)
            idx = eligible[indices[0]]
            weights = np.exp((scores[0] - scores[0].max()) / self.temperature)
            weights /= weights.sum()
            predictions.append(float(weights @ self.gaps[idx]))
            similarities.append(float(scores[0, 0]))
            neighbors.append(idx.tolist())
        return np.asarray(predictions), np.asarray(similarities), neighbors

    def extend(self, embeddings, gaps, group_ids):
        return GapMemory(np.concatenate([self.embeddings, embeddings]), np.concatenate([self.gaps, gaps]),
                         np.concatenate([self.group_ids, group_ids]), self.k, self.temperature, self.encoder_identity)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as f:
            np.savez_compressed(f, embeddings=self.embeddings, gaps=self.gaps, group_ids=self.group_ids,
                                k=self.k, temperature=self.temperature, encoder_identity=self.encoder_identity)
        temp.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as x:
            return cls(x["embeddings"], x["gaps"], x["group_ids"], x["k"].item(),
                       x["temperature"].item(), x["encoder_identity"].item())


def select_memory(embeddings, gaps, ids, selection_embeddings, selection_gaps, selection_ids, config, identity):
    candidates = []
    for k in config["knn"]["k_grid"]:
        for temperature in config["knn"]["temperature_grid"]:
            memory = GapMemory(embeddings, gaps, ids, k, temperature, identity)
            pred, _, _ = memory.predict(selection_embeddings, selection_ids, identity)
            candidates.append({"k": k, "temperature": temperature,
                               "selection_mse": float(np.mean((pred - selection_gaps)**2))})
    best = min(candidates, key=lambda x: (x["selection_mse"], x["k"], x["temperature"]))
    return GapMemory(embeddings, gaps, ids, best["k"], best["temperature"], identity), {"selected": best, "grid": candidates}


def corrected_reward(proxy_z, predicted_gap, mode="signed"):
    predicted_gap = np.asarray(predicted_gap)
    applied = predicted_gap if mode == "signed" else np.maximum(0, predicted_gap)
    return np.asarray(proxy_z) - applied
