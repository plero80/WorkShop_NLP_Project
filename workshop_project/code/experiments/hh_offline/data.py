"""Checked adapters for the original memory and completed HH evaluations."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import file_hash, read_json
from evaluation import load_completed
from knn_distillation.data import group


def vectors(values):
    x = np.asarray(values, np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite embedding matrix")
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4):
        raise ValueError("Expected the saved L2-normalized proxy embeddings")
    return np.ascontiguousarray(x)


def memory(bank_path, memory_dir):
    memory_dir = Path(memory_dir)
    complete = read_json(memory_dir / "complete.json")
    for name in ("protocol.json", "embedding_information.json", "detectors/gap_knn.npz"):
        if file_hash(memory_dir / name) != complete["artifact_sha256"][name]:
            raise ValueError("Memory artifact checksum mismatch: " + name)
    protocol = read_json(memory_dir / "protocol.json")
    if file_hash(bank_path) != protocol["bank_sha256"]:
        raise ValueError("Candidate-bank checksum mismatch")
    bank = pd.read_csv(bank_path, keep_default_na=False)
    with np.load(memory_dir / "detectors/gap_knn.npz", allow_pickle=False) as stored:
        ids, x, gaps = stored["bank_ids"], vectors(stored["vectors"]), stored["gaps"]
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(bank):
        raise ValueError("Invalid memory-to-bank indices")
    frame = bank.iloc[ids].reset_index(drop=True).copy()
    if not (frame.split == "training").all() or len(frame) != len(x):
        raise ValueError("Memory must contain aligned original training rows only")
    if not np.allclose(gaps, frame.proxy_z - frame.judge_z, atol=1e-12, rtol=0):
        raise ValueError("Memory gaps do not align with bank rows")
    frame["gap"] = gaps
    frame["group"] = frame.prompt.map(group)
    frame["bank_id"] = ids
    return frame, x, protocol


def evaluation(folders, calibration):
    frames, features = [], []
    for folder in folders:
        frame, x = load_completed(folder, features=True)
        x = vectors(x)
        frame["group"] = frame.prompt.map(group)
        if not (frame.group == frame.conversation_group).all():
            raise ValueError("Recorded conversation group differs from prompt text")
        if frame.duplicated(["group", "policy_id"]).any():
            raise ValueError("Duplicate conversation/policy in an evaluation")
        if frame.proxy_truncated.any() or frame.judge_truncated.any():
            raise ValueError("Truncated reward inputs cannot enter this comparison")
        for role in ("proxy", "judge"):
            expected = (frame[role + "_raw"] - calibration[role + "_mean"]) / calibration[role + "_std"]
            if not np.isfinite(expected).all() or not np.allclose(expected, frame[role + "_z"], atol=1e-10, rtol=0):
                raise ValueError("Evaluation normalization differs from the original memory")
        frame["gap"] = frame.proxy_z - frame.judge_z
        if not np.allclose(frame.gap, frame.actual_proxy_judge_gap, atol=1e-10, rtol=0):
            raise ValueError("Saved gap differs from proxy minus judge")
        if not np.array_equal(frame.gap > calibration["theta"], frame.high_gap):
            raise ValueError("Saved high-gap labels use a different cutoff")
        frames.append(frame)
        features.append(x)
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["group", "policy_id"]).any():
        raise ValueError("Overlapping evaluation folders or duplicate policy answers")
    return frame, np.concatenate(features)


def disjoint(*frames):
    seen = set()
    for frame in frames:
        groups = set(frame.group)
        if seen & groups:
            raise ValueError("Memory, validation and evaluation conversation groups overlap")
        seen.update(groups)


def nested_subsets(groups, fractions, seeds):
    """Uniform conversation sampling; no labels or scores affect subset choice."""
    groups = np.asarray(groups, str)
    unique = np.array(sorted(set(groups)))
    result = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(unique)
        for fraction in fractions:
            # Full memory is one identical fitted comparison, not three independent runs.
            if fraction == 1 and seed != seeds[0]:
                continue
            selected = order[:max(1, int(np.ceil(fraction * len(unique))))]
            indices = np.flatnonzero(np.isin(groups, selected))
            result.append({"fraction": fraction, "subset_seed": seed,
                           "memory_groups": len(selected), "indices": indices})
    return result
