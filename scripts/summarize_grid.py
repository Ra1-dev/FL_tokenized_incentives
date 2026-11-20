"""Seed-level intervals and paired contrasts for a complete evaluation grid."""

import argparse
import csv
import json
from fractions import Fraction
from pathlib import Path

from evaluation import ABLATIONS, comparison_output, evaluation_cells, published_bytes, published_method, read, scenario_key
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from fltoken.statistics import sampled_bootstrap_mean, holm, paired_wilcoxon, rational
from rebuild_ledgers import rebuild
from run_grid import checked_execution_clock
from grid_reproduction import verify_training_source


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
        if any(record["metrics"][rule]["final_accuracy"] != record["metrics"]["B0"]["final_accuracy"] for rule in rules):
            raise ValueError("shadow rules must share the same final model accuracy")
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
    return dict(nominal_cells=len(records), seed_count=len(plan["seeds"]), device=plan["device"], rules=rules,
                distinct_main_scenarios=sum(horizon == "main" for horizon,_,_ in distinct),
                intervals=nominal, pooled=pooled, contrasts=contrasts,
                statistical_plan=plan["statistics"], inference_status="complete fixed-design comparison")


def trajectory_intervals(records, plan):
    """Keep one shared accuracy curve per condition, with seeds as observations."""
    expected = {cell["path"]: cell for cell in evaluation_cells(plan)}
    indexed = {record["path"]: record for record in records}
    if len(indexed) != len(records) or indexed.keys() != expected.keys():
        raise ValueError("complete trajectory coverage is required")
    groups = {}
    for path, cell in expected.items():
        rows = indexed[path]["round_history"]
        if [row["round"] for row in rows] != list(range(1, cell["rounds"]+1)):
            raise ValueError("round history is incomplete or out of order")
        for row in rows:
            for metric in ["accuracy", "loss"]:
                rational(row[metric])
            if not 0 <= row["accuracy"] <= 1 or row["loss"] < 0:
                raise ValueError("invalid global evaluation metric")
            key = (cell["horizon"], cell["attack"], cell["fraction"], row["round"])
            groups.setdefault(key, []).append(row)
    return [dict(horizon=horizon, attack=attack, fraction=fraction, round=round,
                 **{metric: interval([row[metric] for row in rows], plan["statistics"])
                    for metric in ["accuracy", "loss"]})
            for (horizon, attack, fraction, round), rows in groups.items()]


def read_round_history(folder):
    expected_hash = read(folder/"hashes.json")["round_metrics.csv"]
    if sha256(folder/"round_metrics.csv") != expected_hash:
        raise ValueError("round history differs from its recorded hash")
    def signature(row):
        return (float(row["accuracy"]), float(row["loss"]), row["update_set_sha256"])
    with (folder/"round_metrics.csv").open(encoding="utf-8", newline="") as stream:
        rounds = list(csv.DictReader(stream))
    indexed = {int(row["round"]): signature(row) for row in rounds}
    if len(indexed) != len(rounds):
        raise ValueError("duplicate global evaluation round")
    seen = set()
    with (folder/"per_round.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            round = int(row["round"])
            if indexed.get(round) != signature(row):
                raise ValueError("global evaluation differs from the common participation log")
            seen.add(round)
    if seen != indexed.keys():
        raise ValueError("global evaluation lacks a common participation round")
    return [dict(round=int(row["round"]), accuracy=float(row["accuracy"]), loss=float(row["loss"])) for row in rounds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("configs/evaluation_plan.json"))
    args = parser.parse_args()
    plan = json.loads(published_bytes(args.plan))
    method = published_method(full_evidence=True)
    output = comparison_output(plan)
    manifest = read(output / "manifest.json")
    planned = {cell["path"]: cell for cell in evaluation_cells(plan)}
    if (len(manifest["cells"]) != len(planned) or
            {cell["path"] for cell in manifest["cells"]} != planned.keys() or
            manifest["expected_cells"] != len(planned) or manifest["completed_cells"] != len(planned)):
        raise ValueError("complete unique evaluation coverage is required")
    if manifest["plan_sha256"] != sha256(args.plan) or manifest["method_sha256"] != sha256("configs/method_frozen.json"):
        raise ValueError("grid provenance differs from the declared analysis")
    clock_path = output / "execution.json"
    if manifest.get("execution_sha256") != sha256(clock_path):
        raise ValueError("grid execution clock differs from its manifest")
    clock = checked_execution_clock(clock_path, dict(plan_sha256=sha256(args.plan),
                                     method_sha256=sha256("configs/method_frozen.json")))
    if clock["finished_utc"] is None or manifest.get("elapsed_wall_seconds") != clock["elapsed_wall_seconds"]:
        raise ValueError("complete grid wall time has not been recorded")
    records = []
    histories = []
    evidence = []
    checked_sources = set()
    implementation = None
    base = load_config(plan["config"])
    if base["dataset"] != plan["dataset"] or base["train_limit"] or base["test_limit"] or not base["save_updates"]:
        raise ValueError("comparison requires its declared complete dataset and retained updates")
    if "dataset_manifest" in plan:
        published_bytes(plan["dataset_manifest"])
        if clock.get("dataset_sha256") != sha256(plan["dataset_manifest"]):
            raise ValueError("comparison dataset checksums differ from the execution clock")
    for cell in manifest["cells"]:
        folder = Path(cell["path"])
        if any(cell.get(key) != value for key, value in planned[cell["path"]].items()):
            raise ValueError("evaluation cell differs from its declared condition")
        if sha256(folder / "per_round.csv") != cell["per_round_sha256"] or sha256(folder / "metadata.json") != cell["metadata_sha256"]:
            raise ValueError("grid artifacts differ from the completed manifest")
        metadata = read(folder / "metadata.json")
        expected = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction", "rounds"]},
                    **{key: plan[key] for key in ["device", "workers", "gpu_fraction"]},
                    "rules": plan["rules"] + list(ABLATIONS), "ours_params": method["params"]}
        if (metadata["status"] != "complete" or metadata["config"] != expected or
                metadata["source_config"] != plan["config"] or metadata["source_git_dirty"] or
                metadata["git_commit"] != cell["source_commit"] or
                metadata.get("method_sha256") != sha256("configs/method_frozen.json") or
                metadata["source_sha256"] != clock["training_source_sha256"]):
            raise ValueError("evaluation scientific inputs differ from their declared provenance")
        current = (metadata["source_sha256"], metadata["versions"])
        if implementation is not None and current != implementation:
            raise ValueError("evaluation runs used different sources or library versions")
        implementation = current
        source_key = (metadata["git_commit"], json.dumps(metadata["source_sha256"], sort_keys=True))
        if source_key not in checked_sources:
            verify_training_source(metadata)
            checked_sources.add(source_key)
        rebuild(folder)
        records.append(dict(**cell, metrics=read(folder / "metrics.json")))
        histories.append(dict(path=cell["path"], round_history=read_round_history(folder)))
        evidence.append(dict(path=cell["path"], metrics_sha256=sha256(folder / "metrics.json"),
                             round_metrics_sha256=sha256(folder / "round_metrics.csv"),
                             participation_sha256=cell["per_round_sha256"]))
    result = summarize_records(records,plan)
    result["runtime"] = dict(training_wall_seconds=manifest["wall_seconds"],
                             elapsed_wall_seconds=clock["elapsed_wall_seconds"],
                             started_utc=clock["started_utc"], finished_utc=clock["finished_utc"],
                             execution_sha256=sha256(clock_path))
    result["accuracy_trajectories"] = trajectory_intervals(histories,plan)
    result["provenance"] = dict(plan_sha256=sha256(args.plan), method_sha256=sha256("configs/method_frozen.json"),
                                 analysis_sha256=sha256(__file__), statistics_sha256=sha256("src/fltoken/statistics.py"),
                                 inputs=evidence)
    if manifest["wall_seconds"] != sum(read(Path(cell["path"])/"metadata.json")["wall_seconds"] for cell in manifest["cells"]):
        raise ValueError("successful run runtime differs from the completed metadata")
    result["dataset"] = plan["dataset"]
    write_json(output / "summary.json",result)
    summary = read("results/summary.json")
    summary["evaluation" if output.name == "grid" else "fashion"] = result
    write_json("results/summary.json",summary)
    print(f"Summarized {len(records)} complete runs, {result['distinct_main_scenarios']} distinct main scenarios")


if __name__ == "__main__":
    main()
