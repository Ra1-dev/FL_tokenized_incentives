"""Leave-one-out median rewards with participation-based reputation and penalties."""

from __future__ import annotations

import math

import numpy as np


def _unit(vector):
    scale = np.max(np.abs(vector))
    if scale == 0:
        return np.zeros_like(vector)
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


class OursScorer:
    """State is keyed by scientific client id, never by sampled-row position.

    Defaults are design priors for development. Empirical parameter selection
    must use a separately recorded pilot corpus before evaluation.
    """

    DEFAULTS = dict(reference=True, ema=True, penalties=True, beta=.5,
                    threshold=0., patience=3, initial_reputation=.5)

    def __init__(self, params=None):
        params = {} if params is None else dict(params)
        if params.keys() - self.DEFAULTS.keys():
            raise ValueError("unknown reward parameters")
        self.params = {**self.DEFAULTS, **params}
        for key in ["reference", "ema", "penalties"]:
            if type(self.params[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ["beta", "threshold", "initial_reputation"]:
            value = self.params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and within [0, 1]")
        patience = self.params["patience"]
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        self.reputation = {}
        self.low_count = {}
        self.last_round = 0

    def references(self, deltas):
        """Return the actual unnormalized reference for each input row."""
        x = np.asarray(deltas, dtype=np.float64)
        if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
            raise ValueError("updates must be a finite nonempty matrix")
        if not self.params["reference"]:
            # Dividing first avoids overflow for a representable mean.
            return np.broadcast_to((x / len(x)).sum(axis=0), x.shape).copy()
        refs = np.zeros_like(x)
        if len(x) == 1:
            return refs
        for i in range(len(x)):
            others = np.sort(np.delete(x, i, axis=0), axis=0)
            middle = len(others) // 2
            refs[i] = others[middle] if len(others) % 2 else .5*others[middle-1] + .5*others[middle]
        return refs

    def score(self, client_ids, deltas, round):
        x = np.asarray(deltas, dtype=np.float64)
        refs = self.references(x)
        quality = np.array([max(0., float(np.clip(_unit(row) @ _unit(ref), -1, 1))) for row, ref in zip(x, refs)])
        return self.allocate(client_ids, quality, round)

    def allocate(self, client_ids, quality, round):
        """Apply state transitions to precomputed qualities from the same reference.

        Pilot candidates share reference geometry, so computing it once avoids
        repeating median and cosine calculations for every economic parameter.
        """
        quality = np.asarray(quality, dtype=np.float64)
        if quality.ndim != 1 or not len(quality) or not np.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("qualities must be a finite nonempty vector within [0, 1]")
        ids = list(client_ids)
        if len(ids) != len(quality) or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in ids):
            raise ValueError("one integer client id is required per update")
        ids = [int(i) for i in ids]
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("client ids must be distinct and nonnegative")
        if isinstance(round, (bool, np.bool_)) or not isinstance(round, (int, np.integer)) or round <= self.last_round:
            raise ValueError("round numbers must strictly increase")
        weights = quality.copy()
        slash = np.zeros(len(quality), dtype=bool)
        for index, client in enumerate(ids):
            prior = self.reputation.get(client, self.params["initial_reputation"])
            reputation = self.params["beta"]*prior + (1-self.params["beta"])*quality[index]
            self.reputation[client] = float(reputation)
            if self.params["ema"]:
                weights[index] *= reputation
            low = self.low_count.get(client, 0)+1 if quality[index] <= self.params["threshold"] else 0
            if self.params["penalties"] and low >= self.params["patience"]:
                slash[index] = True
                weights[index] = 0
                low = 0
            self.low_count[client] = low
        total = math.fsum(weights)
        shares = weights / total if total else np.zeros(len(quality))
        self.last_round = int(round)
        return dict(shares=shares, scores=quality, selected=shares > 0, slash=slash,
                    state=dict(reputation=self.reputation.copy(), low_count=self.low_count.copy(), last_round=self.last_round))
