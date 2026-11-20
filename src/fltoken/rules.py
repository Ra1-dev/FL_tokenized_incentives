"""Reward registry; adding a rule does not change the training trajectory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from fltoken.scoring import cosine_sendros, equal_split, krum_scores, multi_krum


class Scorer(Protocol):
    def score(self, client_ids: np.ndarray, deltas: np.ndarray, round: int) -> dict: ...


class Baseline:
    def __init__(self, name: str, f: int):
        self.name, self.f = name, f

    def score(self, client_ids: np.ndarray, deltas: np.ndarray, round: int) -> dict:
        if self.name == "B0":
            shares = equal_split(deltas)
            scores = np.ones(len(deltas))
        elif self.name == "B1":
            scores, shares = cosine_sendros(deltas)
        else:
            scores = krum_scores(deltas, self.f)
            _, shares = multi_krum(deltas, self.f)
        return dict(scores=scores, shares=shares, selected=shares > 0,
                    slash=np.zeros(len(deltas), dtype=bool), state={})


REGISTRY: dict[str, Callable[[dict], Scorer]] = {
    name: (lambda config, name=name: Baseline(name, config["krum_f"])) for name in ["B0", "B1", "B2"]
}


def make_rules(config: dict) -> dict[str, Scorer]:
    return {name: REGISTRY[name](config) for name in config["rules"]}


def score_round(rules: dict[str, Scorer], client_ids: np.ndarray, deltas: np.ndarray, round: int) -> dict:
    # A read-only view catches accidental writes; one common matrix for all rules.
    x = deltas.view()
    x.flags.writeable = False
    results = {}
    for name, rule in rules.items():
        result = rule.score(client_ids, x, round)
        shares = np.asarray(result["shares"], dtype=np.float64)
        if shares.shape != (len(x),) or not np.isfinite(shares).all() or (shares < 0).any():
            raise ValueError(f"invalid shares from {name}")
        if shares.sum() != 0 and not np.isclose(shares.sum(), 1, rtol=0, atol=1e-12):
            raise ValueError(f"shares from {name} do not sum to one")
        results[name] = result
    return results
