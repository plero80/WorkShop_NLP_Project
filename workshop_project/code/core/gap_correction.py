"""Reusable frozen-memory correction; no judge or GPU is needed once vectors exist."""
from pathlib import Path
import json

import numpy as np
from threadpoolctl import threadpool_limits

from knn_core import Settings, MODEL_ID, MODEL_REVISION, file_hash, unit_vectors, predict_detector

CORRECTION_NAMES = {
    'proxy': 'Uncorrected proxy',
    'mean_gap': 'Training mean-gap correction',
    'signed_gated': 'kNN signed + distance gate (primary)',
    'positive_gated': 'kNN positive-only + distance gate',
    'signed': 'kNN signed, ungated',
    'positive': 'kNN positive-only, ungated',
    'saved_teacher_score_pair': 'Saved teacher score + pairwise',
    'saved_teacher_score_only': 'Saved teacher score only',
}


def apply_gap_correction(proxy_raw, predicted_gap, mean_distance, calibration, cutoff, mode='signed_gated'):
    """Return proxy-scale reward and normalized judge estimate, with diagnostics.

    g = z_proxy - z_judge. Therefore raw_corrected = raw_proxy - sigma_proxy * applied_g.
    The returned raw value is on the PROXY scale; it is not the judge's raw score.
    A gated-off query retains the proxy reward. This is a fallback, not a safety guarantee.
    """
    raw, gap, distance = [np.asarray(v, dtype=float) for v in (proxy_raw, predicted_gap, mean_distance)]
    if raw.ndim != 1 or gap.shape != raw.shape or distance.shape != raw.shape:
        raise ValueError('Provide aligned one-dimensional reward, gap, and distance arrays.')
    if not all(np.isfinite(v).all() for v in (raw, gap, distance)):
        raise ValueError('Non-finite inference input.')
    if mode not in ('signed', 'positive', 'signed_gated', 'positive_gated'):
        raise ValueError('Unknown correction mode: ' + mode)
    std, mean = float(calibration['proxy_std']), float(calibration['proxy_mean'])
    if not np.isfinite([std, mean, cutoff]).all() or std <= 0 or not 0 <= cutoff <= 2:
        raise ValueError('Invalid frozen calibration or cosine-distance cutoff.')
    supported = distance < cutoff
    applied = np.maximum(0.0, gap) if mode.startswith('positive') else gap.copy()
    if mode.endswith('_gated'):
        applied = np.where(supported, applied, 0.0)
    return {
        'reward_raw': raw - std * applied,
        'reward_z': (raw - mean) / std - applied,
        'predicted_gap': gap,
        'applied_gap': applied,
        'mean_neighbor_distance': distance,
        'within_distance_gate': supported,
    }


class GapCorrector:
    """Load exported memory and correct new frozen-encoder representations.

    This is the reward-side numerical adapter, not a TRL/PPO training loop.
    Only use input vectors extracted exactly as embedding_information.json specifies.
    Changing encoder weights, revision, pooling, or tokenization requires rebuilding memory.
    """
    def __init__(self, result_dir, cpu_threads=8, query_batch_size=128):
        self.result_dir = Path(result_dir)
        complete = json.loads((self.result_dir / 'complete.json').read_text())
        for name in ('protocol.json', 'locked_detectors.json', 'detectors/gap_knn.npz',
                     'embedding_information.json'):
            expected = complete['artifact_sha256'].get(name)
            if expected is None or file_hash(self.result_dir / name) != expected:
                raise ValueError('Missing or changed corrector artifact: ' + name)
        protocol = json.loads((self.result_dir / 'protocol.json').read_text())
        self.calibration = protocol['calibration']
        locked = json.loads((self.result_dir / 'locked_detectors.json').read_text())['gap_knn']
        self.cutoff = locked['distance_cutoff']
        with np.load(self.result_dir / 'detectors/gap_knn.npz', allow_pickle=False) as data:
            self.model = {name: data[name].copy() for name in data.files}
        self.model.update(locked['parameters'])
        information = json.loads((self.result_dir / 'embedding_information.json').read_text())
        self.encoder_spec = information['spec']
        if self.encoder_spec['model'] != MODEL_ID or self.encoder_spec['revision'] != MODEL_REVISION:
            raise ValueError('This corrector was not built with the supported frozen encoder.')
        self.settings = Settings(cpu_threads=cpu_threads, query_batch_size=query_batch_size)

    def correct_from_embeddings(self, proxy_raw, embeddings, *, mode='signed_gated'):
        """Embeddings are [batch, hidden_dim] reward-head inputs (pooled, then L2 normalized)."""
        vectors = unit_vectors(embeddings)
        with threadpool_limits(limits=self.settings.cpu_threads):
            gap, details = predict_detector(self.model, 'gap_knn', vectors, self.settings)
        result = apply_gap_correction(proxy_raw, gap, details['mean_neighbor_distance'],
                                      self.calibration, self.cutoff, mode)
        result['neighbor_gap_std'] = details['neighbor_gap_std']
        result['neighbor_bank_indices'] = details['neighbor_indices']
        result['neighbor_weights'] = details['neighbor_weights']
        return result
