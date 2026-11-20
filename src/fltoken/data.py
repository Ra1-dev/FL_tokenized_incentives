"""Deterministic, independent random streams for partitioning and sampling."""

from __future__ import annotations

import numpy as np


def seed_for(seed: int, *coordinates: int) -> int:
    """Stable seed for a named integer-coordinate stream, independent of order."""
    return int(np.random.SeedSequence([seed, *coordinates]).generate_state(1, dtype=np.uint64)[0])


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> list[np.ndarray]:
    """Shuffle each class and split by cumulative Dirichlet proportions.

    Entire draws are retried until every client has at least ten samples.
    No sample is discarded, duplicated or transferred after a draw. Very sparse
    impossible/practically unattainable settings fail explicitly instead of
    looping forever or quietly substituting an IID partition.
    """
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("labels must be a nonempty vector")
    if not isinstance(num_clients, (int, np.integer)) or num_clients < 1:
        raise ValueError("num_clients must be a positive integer")
    if len(y) < 10 * num_clients:
        raise ValueError("at least ten examples per client are required")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == c) for c in np.unique(y)]
    for _ in range(10000):
        parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for indices in classes:
            shuffled = rng.permutation(indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(np.int64)
            for client, chunk in enumerate(np.split(shuffled, cuts)):
                parts[client].append(chunk)
        result = [np.concatenate(chunks).astype(np.int64) for chunks in parts]
        if min(map(len, result)) >= 10:
            return result
    raise ValueError("could not draw a partition with ten samples per client after 10000 attempts")


def sample_clients(num_clients: int, k: int, round: int, seed: int) -> np.ndarray:
    if not 1 <= k <= num_clients or round < 1:
        raise ValueError("require 1 <= k <= num_clients and round >= 1")
    rng = np.random.default_rng(np.random.SeedSequence([seed, round, 1]))
    return rng.choice(num_clients, k, replace=False)
