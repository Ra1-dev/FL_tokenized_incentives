"""Seed-level intervals and paired contrasts for a complete evaluation grid."""

import argparse
from fractions import Fraction
from pathlib import Path

from evaluation import ABLATIONS, evaluation_cells, published_method, read, scenario_key
from fltoken.io import sha256, write_json
from fltoken.statistics import sampled_bootstrap_mean, holm, paired_wilcoxon, rational
from rebuild_ledgers import rebuild


METRICS = ["attacker_share", "headcount_share", "honest_false_penalty_rate", "honest_ever_slashed",
           "honest_gini", "precision", "recall", "f1", "tp", "fp", "fn", "honest_participations",
           "empty_rounds", "total_minted", "rounding_remainder", "final_accuracy"]


def mean_complete(values):
    if not values or any(value is None for value in values):
        return None
    return sum((rational(value) for value in values), Fraction(0)) / len(values)


def interval(values, statistics):
    missing = sum(value is None for value in values)
    return dict(defined=len(values)-missing, undefined=missing,
                estimate=None if missing else sampled_bootstrap_mean(values, statistics["confidence"],
                    statistics["bootstrap_resamples"],statistics["bootstrap_seed"]))


def summarize_records(records, plan):
    """Require all planned runs; never turn incomplete coverage into a result."""
    planned = list(evaluation_cells(plan))
    key = lambda cell: (cell["horizon"], cell["seed"], cell["attack"], cell["fraction"])
    indexed = {key(record): record for record in records}
    if len(indexed) != len(records) or set(indexed) != {key(cell) for cell in planned}:
        raise ValueError("complete unique evaluation coverage is required")
    rules = plan["rules"] + list(ABLATIONS)
    statistics = plan["statistics"]
    if statistics["bootstrap_resamples"] < 10000:
        raise ValueError("the declared evaluation requires at least ten thousand bootstrap resamples")
    for record in records:
        if set(record["metrics"]) != set(rules):
            raise ValueError("evaluation rules differ from the declared comparison")
        if record["metrics"]["Mstar"] != record["metrics"]["Afull"]:
            raise ValueError("the final component stage must equal the fully enabled selected method")
        for rule in rules:
            for metric in METRICS:
                value = record["metrics"][rule][metric]
                if value is not None:
                    rational(value)
    nominal, distinct = [], {}
    for horizon in ["main", "long"]:
        scenarios = []
        for cell in planned:
            label = (cell["attack"], cell["fraction"])
            if cell["horizon"] == horizon and label not in scenarios:
                scenarios.append(label)
        for attack, fraction in scenarios:
            cells = [indexed[(horizon, seed, attack, fraction)] for seed in plan["seeds"]]
            identity = scenario_key(cells[0])
            canonical = (horizon, *identity)
            if canonical in distinct:
                original = distinct[canonical]
                if any(a["metrics"] != b["metrics"] or a["per_round_sha256"] != b["per_round_sha256"]
                       for a,b in zip(cells,original)):
                    raise ValueError("nominally identical fraction labels differ in their measured evidence")
            else:
                distinct[canonical] = cells
            nominal.append(dict(horizon=horizon, attack=attack, fraction=fraction,
                metrics={rule: {metric: interval([c["metrics"][rule][metric] for c in cells], statistics)
                                for metric in METRICS} for rule in rules}))
    pooled, contrasts = {}, []
    for metric in plan["statistics"]["primary_metrics"]:
        selected = [cells for (horizon,attack,fraction),cells in distinct.items()
                    if horizon == "main" and (metric != "attacker_share" or attack != "none")]
        values = {rule: [mean_complete([cells[index]["metrics"][rule][metric] for cells in selected])
                         for index in range(len(plan["seeds"]))] for rule in rules}
        pooled[metric] = dict(scenarios=len(selected), rules={rule: interval(v,statistics) for rule,v in values.items()})
        attacked = [cells for (horizon,attack,fraction),cells in distinct.items() if horizon == "main" and attack != "none"]
        pair_values = {rule: [cell["metrics"][rule][metric] for cells in attacked for cell in cells] for rule in rules}
        seed_values = {rule: [mean_complete([cells[index]["metrics"][rule][metric] for cells in attacked])
                              for index in range(len(plan["seeds"]))] for rule in rules}
        for baseline in ["B0", "B1", "B2"]:
            left, right = pair_values["Mstar"], pair_values[baseline]
            seed_left, seed_right = seed_values["Mstar"], seed_values[baseline]
            undefined = any(value is None for value in left+right)
            differences = [None if a is None or b is None else a-b for a,b in zip(seed_left,seed_right)]
            contrasts.append(dict(metric=metric, baseline=baseline,
                condition_seed_test=None if undefined else paired_wilcoxon(left,right),
                seed_test=None if undefined else paired_wilcoxon(seed_left,seed_right),
                paired_difference=interval(differences,statistics),
                undefined_condition_seed_pairs=sum(a is None or b is None for a,b in zip(left,right))))
    for test in ["condition_seed_test", "seed_test"]:
        adjusted = holm({f"{c['metric']}:{c['baseline']}": c[test]["p_value"] if c[test] else None for c in contrasts})
        for comparison in contrasts:
            comparison[test+"_holm_p"] = adjusted[f"{comparison['metric']}:{comparison['baseline']}"]
    return dict(nominal_cells=len(records), seed_count=len(plan["seeds"]), rules=rules,
                distinct_main_scenarios=sum(horizon == "main" for horizon,_,_ in distinct),
                intervals=nominal, pooled=pooled, contrasts=contrasts,
                statistical_plan=plan["statistics"], inference_status="complete fixed-design comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("configs/evaluation_plan.json"))
    args = parser.parse_args()
    plan = read(args.plan)
    published_method(full_evidence=True)
    manifest = read("results/grid/manifest.json")
    if manifest["plan_sha256"] != sha256(args.plan) or manifest["method_sha256"] != sha256("configs/method_frozen.json"):
        raise ValueError("grid provenance differs from the declared analysis")
    records = []
    evidence = []
    for cell in manifest["cells"]:
        folder = Path(cell["path"])
        if sha256(folder / "per_round.csv") != cell["per_round_sha256"] or sha256(folder / "metadata.json") != cell["metadata_sha256"]:
            raise ValueError("grid artifacts differ from the completed manifest")
        rebuild(folder)
        records.append(dict(**cell, metrics=read(folder / "metrics.json")))
        evidence.append(dict(path=cell["path"], metrics_sha256=sha256(folder / "metrics.json"),
                             participation_sha256=cell["per_round_sha256"]))
    result = summarize_records(records,plan)
    result["provenance"] = dict(plan_sha256=sha256(args.plan), method_sha256=sha256("configs/method_frozen.json"),
                                 analysis_sha256=sha256(__file__), statistics_sha256=sha256("src/fltoken/statistics.py"),
                                 inputs=evidence)
    write_json("results/grid/summary.json",result)
    summary = read("results/summary.json")
    summary["evaluation"] = result
    write_json("results/summary.json",summary)
    print(f"Summarized {len(records)} complete runs, {result['distinct_main_scenarios']} distinct main scenarios")


if __name__ == "__main__":
    main()
