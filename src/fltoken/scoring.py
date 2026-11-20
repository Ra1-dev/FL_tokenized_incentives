"""Non-mutating reward baselines, scored on one common update matrix.

Cosine scoring follows Sendros et al. (2026), Section 6. Multi-Krum
acceptance follows the squared-distance construction of Blanchard et al.
(2017), with the accepted set size explicitly fixed to K-f here.
"""

from __future__ import annotations

import numpy as np


def _matrix(deltas: np.ndarray) -> np.ndarray:
    x = np.asarray(deltas, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("deltas must be a nonempty (clients, parameters) matrix")
    if not np.isfinite(x).all():
        raise ValueError("deltas must be finite")
    return x


def equal_split(deltas: np.ndarray) -> np.ndarray:
    x = _matrix(deltas)
    return np.full(len(x), 1.0 / len(x), dtype=np.float64)


def cosine_to_reference(deltas: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rowwise cosine; either zero norm gives zero. Never alter the inputs."""
    x = _matrix(deltas)
    r = np.asarray(reference, dtype=np.float64)
    if r.shape != (x.shape[1],) or not np.isfinite(r).all():
        raise ValueError("reference must be one finite vector matching deltas")
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(r)
    dots = np.einsum("ij,j->i", x, r)
    return np.clip(np.divide(dots, norms, out=np.zeros(len(x)), where=norms > 0), -1, 1)


def cosine_sendros(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(deltas)
    cosines = cosine_to_reference(x, x.mean(axis=0))
    q = np.floor(100 * np.maximum(0, cosines)).astype(np.int64)
    shares = q.astype(np.float64) / q.sum() if q.sum() else np.zeros(len(x))
    return q, shares


def krum_scores(deltas: np.ndarray, f: int) -> np.ndarray:
    x = _matrix(deltas)
    k = len(x)
    if not isinstance(f, (int, np.integer)) or isinstance(f, (bool, np.bool_)) or f < 0:
        raise ValueError("f must be a nonnegative integer")
    if k < 2 * f + 3:
        raise ValueError("Multi-Krum requires K >= 2*f + 3")
    scores = np.empty(k, dtype=np.float64)
    # Rowwise differences avoid a K*K*d temporary and Gram-matrix cancellation.
    for i in range(k):
        distances = np.square(x - x[i]).sum(axis=1)
        distances = np.delete(distances, i)
        scores[i] = np.sort(distances)[:k - f - 2].sum()
    return scores


def multi_krum(deltas: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    scores = krum_scores(deltas, f)
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(scores, kind="stable")[:len(scores) - f]] = True
    return selected, selected.astype(np.float64) / selected.sum()
