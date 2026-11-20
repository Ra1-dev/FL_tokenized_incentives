"""Update replacements and the deterministic label-poisoning transform."""

from __future__ import annotations

import numpy as np


def zero_update(delta: np.ndarray) -> np.ndarray:
    return np.zeros_like(delta)


def gaussian_noise(shape: tuple[int, ...], sigma: float, rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    return rng.normal(0.0, sigma, shape)


def sign_flip(delta: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative")
    return -scale * np.asarray(delta)


def label_flip(labels):
    """Reverse the ten-class label order; works on NumPy or Torch arrays."""
    return 9 - labels


def stale_round(round: int, lag: int = 5) -> int:
    """Global snapshot whose locally trained delta is replayed in this round."""
    if round < 1 or lag < 0:
        raise ValueError("round must be positive and lag nonnegative")
    return max(1, round - lag)


def replay_own(history: list[tuple[int, np.ndarray]], round: int, lag: int,
               template: np.ndarray) -> tuple[int, np.ndarray]:
    """Sensitivity variant: most recent reported delta old enough to replay.

    Source zero denotes no eligible report. Always return a copy so the
    caller cannot mutate history. Persistent stale clients with no initial
    honest reports consequently keep reporting zero in this variant.
    """
    if round < 1 or lag < 1:
        raise ValueError("round and replay lag must be positive")
    eligible = [(t, delta) for t, delta in history if t <= round - lag]
    if not eligible:
        return 0, np.zeros_like(template)
    source, delta = max(eligible, key=lambda item: item[0])
    return source, delta.copy()
