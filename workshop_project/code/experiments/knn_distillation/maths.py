"""The original signed cosine-weighted kNN formula, without model imports."""
import numpy as np
from threadpoolctl import threadpool_limits


def predict_memory(vectors, memory, k=31, temperature=.05, threads=8):
    from knn_core import top_neighbors, unit_vectors
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('Positive finite kNN temperature required.')
    with threadpool_limits(limits=threads):
        sim, ix = top_neighbors(unit_vectors(vectors), memory['vectors'], k)
        weight = np.exp((sim - sim[:, :1]) / temperature)
        gap = (weight * memory['gaps'][ix]).sum(1) / weight.sum(1)
    return gap, (1 - sim).mean(1)
