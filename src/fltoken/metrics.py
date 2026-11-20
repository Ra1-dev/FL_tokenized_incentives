"""Integer payout accounting and metrics on the same logged participations."""

from __future__ import annotations

from fractions import Fraction
from operator import index

import numpy as np

SCALE = 10**18
ROUND_POOL = 1000 * SCALE
SETTLEMENT_VERSION = "exact-float-normalized-v1"
METRICS_VERSION = "integer-rational-metrics-v2"


def payout(shares: np.ndarray) -> tuple[list[int], list[int]]:
    """Normalize the exact binary-float values before flooring to base units.

    No client gets the residue of another client's rounding. Empty payment
    rounds leave the entire pool unminted; invalid shares fail closed.
    """
    values = np.asarray(shares, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("shares must be a finite nonnegative vector")
    exact = [Fraction(float(value)) for value in values]
    total = sum(exact, Fraction(0))
    units = [int(value * SCALE // total) for value in exact] if total else [0] * len(exact)
    return units, [ROUND_POOL * share // SCALE for share in units]


def gini(values: list[int]) -> float | None:
    """Exact rank-weighted Gini of integer payouts, rounded to float once."""
    x = sorted(index(value) for value in values)
    if any(value < 0 for value in x):
        raise ValueError("payouts must be nonnegative integers")
    total = sum(x)
    if not x or not total:
        return None
    numerator = sum((2*rank-len(x)-1)*value for rank, value in enumerate(x, 1))
    return float(Fraction(numerator, len(x)*total))


def summarize(rows: list[dict], ledger: dict, config: dict, attacks: dict[int, str]) -> dict:
    honest_ids = [i for i in range(config["num_clients"]) if i not in attacks]
    output = {}
    for name, rounds in ledger.items():
        totals = [0] * config["num_clients"]
        empty = 0
        for item in rounds:
            empty += sum(item["payouts"]) == 0
            for client, amount in zip(item["ids"], item["payouts"]):
                totals[client] += amount
        minted = sum(totals)
        tp = fp = fn = honest_count = 0
        slashed = set()
        for row in rows:
            flagged = Fraction(float(row[f"{name}_share"])) < Fraction(1, 2*config["clients_per_round"])
            adversary = bool(row["adversary"])
            tp += flagged and adversary
            fp += flagged and not adversary
            fn += not flagged and adversary
            honest_count += not adversary
            if not adversary and row[f"{name}_slash"]:
                slashed.add(row["client_id"])
        precision = float(Fraction(tp, tp+fp)) if tp+fp else 0.0
        recall = float(Fraction(tp, tp+fn)) if tp+fn else 0.0
        output[name] = dict(
            attacker_share=float(Fraction(sum(totals[i] for i in attacks), minted)) if minted else None,
            headcount_share=float(Fraction(len(attacks), config["num_clients"])),
            honest_false_penalty_rate=float(Fraction(fp, honest_count)) if honest_count else None,
            honest_ever_slashed=len(slashed), honest_gini=gini([totals[i] for i in honest_ids]),
            precision=precision, recall=recall,
            f1=float(Fraction(2*tp, 2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
            tp=int(tp), fp=int(fp), fn=int(fn), honest_participations=int(honest_count),
            empty_rounds=int(empty), total_minted=minted,
            rounding_remainder=len(rounds) * ROUND_POOL - minted,
            payouts_by_client=totals, final_accuracy=rows[-1]["accuracy"],
        )
    return output
