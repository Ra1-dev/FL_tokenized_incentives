"""Post-hoc sensitivity of the economic parameters of Mstar on the recorded grid.

The leave-one-out median quality of every participation is recorded in the
evaluation logs. Reputation, thresholds and penalties act only on those
qualities, and the global model never depends on rewards, so every parameter
setting can be replayed exactly without retraining. The frozen setting must
reproduce the published shares and metrics bit for bit before any other
setting is scored.

This is an exploratory analysis on the evaluation grid itself. It describes
how the trade-off moves with each parameter; it does not select a new method.
"""

import argparse
import csv
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from evaluation import scenario_key
from fltoken.io import sha256, write_json
from fltoken.metrics import payout, summarize
from fltoken.ours import OursScorer
from fltoken.statistics import sampled_bootstrap_mean, rational

FROZEN = Path("configs/method_frozen.json")
MANIFEST = Path("results/grid/manifest.json")
PLAN = Path("configs/evaluation_plan.json")
OUTPUT = Path("results/sensitivity_sweep/summary.json")

BETAS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
THRESHOLDS = [0.0, 0.05, 0.1, 0.2]
PATIENCES = [1, 2, 3, 5]
METRICS = ["attacker_share", "honest_false_penalty_rate"]


def settings(frozen):
    """Enumerate the declared grid; the frozen setting is always included."""
    base = dict(frozen, reference=True)
    grid = []
    for beta, threshold, patience in itertools.product(BETAS, THRESHOLDS, PATIENCES):
        grid.append(dict(base, ema=True, penalties=True, beta=beta, threshold=threshold, patience=patience))
    for beta in BETAS:
        grid.append(dict(base, ema=True, penalties=False, beta=beta))
    grid.append(dict(base, ema=False, penalties=False))
    unique = []
    for params in grid:
        if params not in unique:
            unique.append(params)
    if frozen not in unique:
        raise ValueError("the frozen setting must be part of the declared grid")
    return unique


def label(params):
    if not params["ema"]:
        return "reference only"
    if not params["penalties"]:
        return f"beta={params['beta']:g}, no penalties"
    return f"beta={params['beta']:g}, threshold={params['threshold']:g}, patience={params['patience']}"


def load_cell(cell):
    folder = Path(cell["path"])
    if sha256(folder/"per_round.csv") != cell["per_round_sha256"] or sha256(folder/"metadata.json") != cell["metadata_sha256"]:
        raise ValueError(f"recorded evaluation log differs from the manifest: {folder}")
    config = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))["config"]
    with (folder/"per_round.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["round"], row["client_id"] = int(row["round"]), int(row["client_id"])
        row["adversary"] = row["adversary"] == "True"
        row["accuracy"] = float(row["accuracy"])
    attacks = {int(k): v for k, v in json.loads((folder/"protocol.json").read_text(encoding="utf-8"))["adversaries"].items()}
    return config, rows, attacks


def replay(config, rows, attacks, params, check=False):
    """Score one run under one parameter setting from the recorded qualities."""
    scorer = OursScorer(params)
    ledger = []
    scored = []
    for round in range(1, config["rounds"]+1):
        selected = [row for row in rows if row["round"] == round]
        quality = np.array([float(row["Mstar_score"]) for row in selected])
        result = scorer.allocate([row["client_id"] for row in selected], quality, round)
        units, amounts = payout(result["shares"])
        ledger.append(dict(round=round, ids=[row["client_id"] for row in selected], payouts=amounts))
        for row, share, slash in zip(selected, result["shares"], result["slash"]):
            if check and (float(row["Mstar_share"]) != float(share) or (row["Mstar_slash"] == "True") != bool(slash)):
                raise ValueError("the frozen setting does not reproduce the recorded allocation")
            scored.append(dict(client_id=row["client_id"], adversary=row["adversary"], accuracy=row["accuracy"],
                               S_share=float(share), S_slash=bool(slash)))
    metrics = summarize(scored, {"S": ledger}, config, attacks)["S"]
    return {metric: metrics[metric] for metric in METRICS} | dict(honest_ever_slashed=metrics["honest_ever_slashed"])


def pooled(cells_by_scenario, values, seeds, statistics):
    """Average fixed scenarios within each seed, then resample seeds (as published)."""
    result = {}
    for metric in METRICS:
        scenarios = [key for key in cells_by_scenario if metric != "attacker_share" or key[0] != "none"]
        by_seed = [sum((rational(values[(key, seed)][metric]) for key in scenarios), Fraction(0))/len(scenarios)
                   for seed in seeds]
        result[metric] = sampled_bootstrap_mean(by_seed, statistics["confidence"],
                                                statistics["bootstrap_resamples"], statistics["bootstrap_seed"])
    return result


def pareto(points):
    """Indices of settings that no other setting beats on both pooled means."""
    front = []
    for i, a in enumerate(points):
        dominated = any(b["attacker_share"]["mean"] <= a["attacker_share"]["mean"] and
                        b["honest_false_penalty_rate"]["mean"] <= a["honest_false_penalty_rate"]["mean"] and
                        (b["attacker_share"]["mean"], b["honest_false_penalty_rate"]["mean"]) !=
                        (a["attacker_share"]["mean"], a["honest_false_penalty_rate"]["mean"])
                        for j, b in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return front


def sweep():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["params"]
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["method_sha256"] != sha256(FROZEN) or manifest["plan_sha256"] != sha256(PLAN):
        raise ValueError("grid manifest differs from the frozen comparison")
    seeds = plan["seeds"]
    cells_by_scenario = {}
    for cell in manifest["cells"]:
        if cell["horizon"] == "main":
            cells_by_scenario.setdefault(scenario_key(cell), {})[cell["seed"]] = cell
    if any(sorted(cells) != sorted(seeds) for cells in cells_by_scenario.values()):
        raise ValueError("complete seed coverage is required for every scenario")
    loaded = {(key, seed): load_cell(cell) for key, cells in cells_by_scenario.items() for seed, cell in cells.items()}
    rows = []
    for params in settings(frozen):
        values = {index: replay(*data, params, check=params == frozen) for index, data in loaded.items()}
        estimates = pooled(cells_by_scenario, values, seeds, plan["statistics"])
        rows.append(dict(label=label(params), params=params, frozen=params == frozen,
                         honest_ever_slashed=sum(v["honest_ever_slashed"] for v in values.values()), **estimates))
    front = pareto(rows)
    for index, row in enumerate(rows):
        row["pareto"] = index in front
    published = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))["evaluation"]["pooled"]
    reference = next(row for row in rows if row["frozen"])
    for metric in METRICS:
        if reference[metric] != published[metric]["rules"]["Mstar"]["estimate"]:
            raise ValueError(f"frozen setting does not reproduce the published pooled {metric}")
    result = dict(schema=1, description=__doc__.strip().splitlines()[0], exploratory=True,
                  frozen_sha256=sha256(FROZEN), grid_manifest_sha256=sha256(MANIFEST), plan_sha256=sha256(PLAN),
                  scenarios=len(cells_by_scenario), seeds=seeds, grid=dict(beta=BETAS, threshold=THRESHOLDS, patience=PATIENCES),
                  settings=rows)
    write_json(OUTPUT, result)
    return result


def main():
    argparse.ArgumentParser(description=__doc__.strip().splitlines()[0]).parse_args()
    result = sweep()
    rows = result["settings"]
    print(json.dumps(dict(settings=len(rows), pareto=sum(row["pareto"] for row in rows),
                          frozen_reproduced=True), indent=2))


if __name__ == "__main__":
    main()
