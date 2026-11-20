"""Non-mutating reward baselines, scored on one common update matrix.

Cosine scoring follows Sendros et al. (2026), Section 6. Multi-Krum
acceptance follows the squared-distance construction of Blanchard et al.
(2017), with the accepted set size explicitly fixed to K-f here.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .ours import OursScorer


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
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        norms = np.linalg.norm(x, axis=1) * np.linalg.norm(r)
        dots = np.einsum("ij,j->i", x, r)
        result = np.divide(dots, norms, out=np.zeros(len(x)), where=norms > 0)
    nonzero = np.any(x != 0, axis=1) & np.any(r != 0)
    unsafe = nonzero & ((norms < np.finfo(float).tiny) | ~np.isfinite(norms) | ~np.isfinite(result))
    if unsafe.any():
        # Power-of-two scaling preserves binary significands and the cosine.
        reference = np.ldexp(r, -math.frexp(float(np.max(np.abs(r))))[1])
        for index in np.flatnonzero(unsafe):
            row = np.ldexp(x[index], -math.frexp(float(np.max(np.abs(x[index]))))[1])
            result[index] = np.dot(row, reference)/(np.linalg.norm(row)*np.linalg.norm(reference))
    return np.clip(result, -1, 1)


def cosine_sendros(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(deltas)
    scale = float(np.max(np.abs(x)))
    if scale and scale < math.sqrt(np.finfo(float).tiny):
        x = np.ldexp(x, -math.frexp(scale)[1])
    with np.errstate(over="ignore", invalid="ignore"):
        reference = x.mean(axis=0)
    if not np.isfinite(reference).all():
        reference = np.array([float(sum(map(Fraction, column), Fraction(0))/len(x)) for column in x.T])
    cosines = cosine_to_reference(x, reference)
    q = np.floor(100 * np.maximum(0, cosines)).astype(np.int64)
    shares = q.astype(np.float64) / q.sum() if q.sum() else np.zeros(len(x))
    return q, shares


def _distance_scores(deltas: np.ndarray, f: int):
    x = _matrix(deltas)
    k = len(x)
    if not isinstance(f, (int, np.integer)) or isinstance(f, (bool, np.bool_)) or f < 0:
        raise ValueError("f must be a nonnegative integer")
    if k < 2 * f + 3:
        raise ValueError("Multi-Krum requires K >= 2*f + 3")
    nonzero = np.abs(x[x != 0])
    limits = np.finfo(float)
    unsafe = len(nonzero) and (nonzero.max() > math.sqrt(limits.max/(4*x.size)) or
                               nonzero.min() < math.sqrt(limits.tiny)/limits.eps)
    if unsafe:
        # Exact common-denominator integers keep the distance ordering when
        # binary64 squares would overflow or erase small unequal distances.
        ratios = [[float(value).as_integer_ratio() for value in row] for row in x]
        denominator = max(divisor for row in ratios for _, divisor in row)
        integers = [[numerator*(denominator//divisor) for numerator, divisor in row] for row in ratios]
        return [sum(sorted(sum((a-b)**2 for a,b in zip(row, other))
                           for j,other in enumerate(integers) if j != i)[:len(x)-f-2])
                for i,row in enumerate(integers)]
    scores = np.empty(k, dtype=np.float64)
    # Rowwise differences avoid a K*K*d temporary and Gram-matrix cancellation.
    for i in range(k):
        distances = np.square(x - x[i]).sum(axis=1)
        distances = np.delete(distances, i)
        scores[i] = np.sort(distances)[:k - f - 2].sum()
    return scores


def krum_scores(deltas: np.ndarray, f: int) -> np.ndarray:
    """Distance sums; extreme-range diagnostics are divided by their maximum.

    Selection uses exact ordering before conversion of those diagnostics to
    binary64. Ordinary representable-distance scores keep their original units.
    """
    scores = _distance_scores(deltas, f)
    if isinstance(scores, list):
        largest = max(scores)
        return np.array([score/largest if largest else 0. for score in scores])
    return scores


def multi_krum(deltas: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    scores = _distance_scores(deltas, f)
    order = sorted(range(len(scores)), key=lambda i:scores[i]) if isinstance(scores, list) else np.argsort(scores, kind="stable")
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:len(scores) - f]] = True
    return selected, selected.astype(np.float64) / selected.sum()
