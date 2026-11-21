"""Seed-level reward and self-inclusion evidence from complete offline replays."""

from fractions import Fraction
from pathlib import Path

from evaluation import evaluation_cells, published_bytes, recorded_method, read
from extension_replay import FACTORIES, replay_geometry
from fltoken.io import sha256, write_json
from fltoken.statistics import holm, paired_wilcoxon, rational
from reproduction_analysis import exact_quantile
from select_extensions import FROZEN_PATH, PLAN_PATH, recorded_extensions, scientific_bindings
from grid_reproduction import verify_training_source
from summarize_grid import METRICS, interval
from rebuild_ledgers import rebuild


def mean(values):
    return float(sum((rational(value) for value in values), Fraction(0))/len(values)) if values else None


def noise_statistics(rows):
    rows = [row for row in rows if row["attack"] == "noise"]
    defined = all(row["bias_defined"] for row in rows)
    relative = [abs(row["cross_residual"])/abs(row["cosine"]) if row["cosine"] else float("inf") for row in rows]
    return dict(participations=len(rows), eligible=sum(row["eligible_before"] for row in rows),
                undefined_geometry=sum(not row["bias_defined"] for row in rows),
                zero_measured_cosine=sum(row["cosine"] == 0 for row in rows),
                mean_cosine=mean([row["cosine"] for row in rows]),
                mean_self_inclusion=mean([row["self_inclusion"] for row in rows]) if defined else None,
                mean_excluding_cosine=mean([row["excluding_cosine"] for row in rows]),
                mean_inclusive_minus_excluding=mean([float(Fraction(row["cosine"])-Fraction(row["excluding_cosine"])) for row in rows]),
                median_relative_error=exact_quantile(relative, .5) if defined else None,
                fraction_positive_cosine=mean([int(row["cosine"] > 0) for row in rows]),
                fraction_positive_excluding_cosine=mean([int(row["excluding_cosine"] > 0) for row in rows]))


BIAS_METRICS = ["mean_cosine", "mean_self_inclusion", "mean_excluding_cosine", "mean_inclusive_minus_excluding",
                "median_relative_error", "fraction_positive_cosine", "fraction_positive_excluding_cosine"]
PRIMARY_METRICS = ["attacker_share", "honest_false_penalty_rate"]


def measured_rule(cell, rule):
    measured = dict(cell["extensions"][rule]["metrics"] if rule in FACTORIES else cell["baseline_metrics"][rule])
    if rule in FACTORIES and measured["undefined_state_rounds"]:
        measured.update({metric: None for metric in
                         ["attacker_share", "honest_false_penalty_rate", "honest_gini", "precision", "recall", "f1"]})
    return measured


def comparison_statistics(records, plan):
    """One eight-contrast family, with whole-seed sensitivity alongside paired cells."""
    statistics = plan["statistics"]
    rules = ["B1", "Mstar", "CGSV", "RFFL"]
    main = [cell for cell in records if cell["horizon"] == "main"]
    attacked = [cell for cell in main if cell["attack"] != "none"]
    if not attacked:
        raise ValueError("extension paired comparisons require attacked main conditions")
    def complete_mean(values):
        return None if not values or any(value is None for value in values) else mean(values)
    pooled, contrasts = {}, []
    for metric in PRIMARY_METRICS:
        cells = attacked if metric == "attacker_share" else main
        values = {rule: [complete_mean([measured_rule(cell, rule)[metric] for cell in cells if cell["seed"] == seed])
                         for seed in plan["seeds"]] for rule in rules}
        pooled[metric] = dict(scenarios=len(cells)//len(plan["seeds"]),
                             rules={rule: interval(rows, statistics) for rule, rows in values.items()})
        paired = {rule: [measured_rule(cell, rule)[metric] for cell in attacked] for rule in rules}
        seed_means = {rule: [complete_mean([measured_rule(cell, rule)[metric] for cell in attacked if cell["seed"] == seed])
                            for seed in plan["seeds"]] for rule in rules}
        for family in FACTORIES:
            for baseline in ["B1", "Mstar"]:
                left, right = paired[family], paired[baseline]
                seed_left, seed_right = seed_means[family], seed_means[baseline]
                undefined = sum(a is None or b is None for a, b in zip(left, right))
                differences = [None if a is None or b is None else a-b for a, b in zip(seed_left, seed_right)]
                contrasts.append(dict(metric=metric, family=family, baseline=baseline,
                    condition_seed_test=None if undefined else paired_wilcoxon(left, right),
                    seed_test=None if undefined else paired_wilcoxon(seed_left, seed_right),
                    paired_difference=interval(differences, statistics),
                    undefined_condition_seed_pairs=undefined, planned_condition_seed_pairs=len(attacked)))
    for test in ["condition_seed_test", "seed_test"]:
        key = lambda row: f"{row['metric']}:{row['family']}:{row['baseline']}"
        adjusted = holm({key(row): row[test]["p_value"] if row[test] else None for row in contrasts})
        for row in contrasts:
            row[test+"_holm_p"] = adjusted[key(row)]
    return pooled, contrasts


def summarize_records(records, plan, source_seeds):
    planned = list(evaluation_cells(plan))
    indexed = {row["path"]: row for row in records}
    if len(indexed) != len(records) or indexed.keys() != {cell["path"] for cell in planned}:
        raise ValueError("extension inference requires every unique planned evaluation cell")
    if len(set(source_seeds)) != len(source_seeds) or not set(source_seeds) <= set(plan["seeds"]):
        raise ValueError("source seeds must be a unique subset of the evaluation seeds")
    statistics = plan["statistics"]
    if statistics["bootstrap_resamples"] < 10000:
        raise ValueError("extension seed intervals require the declared bootstrap sample count")
    rules = ["B1", "Mstar", "CGSV", "RFFL"]
    observations, per_seed = [], []
    for cell in planned:
        record = indexed[cell["path"]]
        if any(record.get(key) != value for key, value in cell.items()):
            raise ValueError("extension cell identity differs from the fixed evaluation plan")
        if set(record["extensions"]) != set(FACTORIES):
            raise ValueError("every cell must include both reputation rules")
        for family, result in record["extensions"].items():
            if result["training_device"] != plan["device"]:
                raise ValueError("extension comparisons cannot mix training devices")
            for subset, rows in [("all_noise", result["rows"]),
                                 ("eligible_noise", [row for row in result["rows"] if row["eligible_before"]])]:
                measured = noise_statistics(rows)
                if result["metrics"]["undefined_state_rounds"]:
                    measured.update({metric: None for metric in BIAS_METRICS})
                per_seed.append(dict(**cell, family=family, subset=subset,
                                     undefined_state_rounds=result["metrics"]["undefined_state_rounds"], **measured))
            for row in result["rows"]:
                if row["attack"] == "noise":
                    observations.append(dict(seed=cell["seed"], horizon=cell["horizon"], attack=cell["attack"],
                        fraction=cell["fraction"], family=family, **{key: row[key] for key in
                        ["round", "client_id", "cosine", "self_inclusion", "cross_residual", "excluding_cosine",
                         "bias_defined", "eligible_before", "reference_weights", "reference_norm", "update_set_sha256"]}))
    scenarios = list(dict.fromkeys((cell["horizon"], cell["attack"], cell["fraction"]) for cell in planned))
    reward_intervals, bias_intervals = [], []
    for horizon, attack, fraction in scenarios:
        cells = [indexed[cell["path"]] for cell in planned
                 if (cell["horizon"], cell["attack"], cell["fraction"]) == (horizon, attack, fraction)]
        metrics = {rule: [] for rule in rules}
        for cell in cells:
            for rule in rules:
                metrics[rule].append(measured_rule(cell, rule))
        reward_intervals.append(dict(horizon=horizon, attack=attack, fraction=fraction,
            metrics={rule: {metric: interval([row[metric] for row in rows], statistics) for metric in METRICS}
                     for rule, rows in metrics.items()}))
        if attack not in {"noise", "sendros_mix"}:
            continue
        for family in FACTORIES:
            for subset in ["all_noise", "eligible_noise"]:
                rows = [row for row in per_seed if (row["horizon"], row["attack"], row["fraction"], row["family"], row["subset"])
                        == (horizon, attack, fraction, family, subset)]
                bias_intervals.append(dict(horizon=horizon, attack=attack, fraction=fraction, family=family, subset=subset,
                    participations=sum(row["participations"] for row in rows),
                    undefined_geometry=sum(row["undefined_geometry"] for row in rows),
                    zero_measured_cosine=sum(row["zero_measured_cosine"] for row in rows),
                    metrics={metric: interval([row[metric] for row in rows], statistics) for metric in BIAS_METRICS}))
    pooled, contrasts = comparison_statistics([indexed[cell["path"]] for cell in planned], plan)
    return dict(runs=len(planned), device=plan["device"], seeds=plan["seeds"], source_seeds=source_seeds,
                rules=rules, reward_intervals=reward_intervals, bias_intervals=bias_intervals,
                pooled=pooled, contrasts=contrasts,
                per_seed=per_seed, noise_observations=observations,
                undefined_state_rounds={family: sum(row["extensions"][family]["metrics"]["undefined_state_rounds"] for row in records)
                                        for family in FACTORIES},
                statistical_plan=dict(statistics,
                    paired_tests="Two-sided exact conditional Wilcoxon over attacked main condition-seed pairs; separate complete-seed-mean sensitivity",
                    multiple_testing="One Holm family of eight comparisons: two extensions by B1/Mstar by attacker share/honest false penalties; missing tests retain their family slots"),
                interpretation="Intervals resample complete seeds. Conditions sharing a seed are dependent; paired cell tests are descriptive, and complete-seed tests are the sensitivity check. Bias contrasts are arithmetic reference counterfactuals on the same updates, not separately trained models. Undefined observations are counted and never replaced with zero.")


def main():
    frozen = recorded_extensions(full_evidence=True)
    method = recorded_method()
    plan_path = Path("configs/evaluation_plan.json")
    published_bytes(plan_path)
    plan = read(plan_path)
    manifest_path = Path("results/extensions/manifest.json")
    manifest = read(manifest_path)
    if manifest.get("validation_source_sha256") != scientific_bindings(read(PLAN_PATH)):
        raise ValueError("extension validation sources differ; reconstruct recorded outputs before summarizing")
    if (not manifest["complete"] or manifest["completed_cells"] != manifest["expected_cells"] or
            manifest["frozen_sha256"] != sha256(FROZEN_PATH) or manifest["plan_sha256"] != sha256(plan_path) or
            manifest["grid_manifest_sha256"] != sha256("results/grid/manifest.json")):
        raise ValueError("complete extension coverage must bind the current frozen grid")
    grid = read("results/grid/manifest.json")
    grid_cells = {cell["path"]: cell for cell in grid["cells"]}
    if len(grid_cells) != len(grid["cells"]) or grid["completed_cells"] != grid["expected_cells"]:
        raise ValueError("the underlying training grid must be complete and unique")
    records, evidence, checked = [], [], set()
    for cell in manifest["cells"]:
        folder = Path(cell["path"])
        original = grid_cells.get(cell["path"])
        if original is None or any(cell.get(key) != value for key, value in original.items()):
            raise ValueError("extension cell differs from the common training manifest")
        if sha256(folder/"metadata.json") != original["metadata_sha256"] or sha256(folder/"per_round.csv") != original["per_round_sha256"]:
            raise ValueError("original training inputs differ from the common grid")
        metadata = read(folder/"metadata.json")
        if metadata["git_commit"] != original["source_commit"]:
            raise ValueError("original training commit differs from the common grid")
        binding = (metadata["git_commit"], tuple(sorted(metadata["source_sha256"].items())), metadata["source_git_dirty"])
        if binding not in checked:
            verify_training_source(metadata, method=method)
            checked.add(binding)
        outputs = {}
        for family, params in frozen["params"].items():
            path = Path("results/extensions")/folder.name/f"{family}.json"
            geometry = path.parent/"geometry"/f"{family}.json"
            binding = cell["outputs"][family]
            if binding["path"] != path.as_posix() or sha256(path) != binding["sha256"] or sha256(geometry) != binding["geometry_sha256"]:
                raise ValueError("extension output differs from its manifest")
            result = replay_geometry(folder, geometry, params)
            if result != read(path) or result["metrics"] != cell["metrics"][family]:
                raise ValueError("extension states and allocations do not reproduce their stored output")
            outputs[family] = result
        rebuild(folder)
        baseline = read(folder/"metrics.json")
        records.append(dict(**cell, extensions=outputs, baseline_metrics={name: baseline[name] for name in ["B1", "Mstar"]}))
        evidence.append(dict(path=cell["path"], outputs=cell["outputs"], baseline_metrics_sha256=sha256(folder/"metrics.json")))
    source = read("results/literature/sendros.json")
    result = summarize_records(records, plan, source["setup"]["seeds"])
    result["provenance"] = dict(inputs=evidence, manifest_sha256=sha256(manifest_path),
        frozen_sha256=sha256(FROZEN_PATH), source_sha256={name: sha256(name) for name in
        ["scripts/summarize_extensions.py", "scripts/summarize_grid.py", "src/fltoken/statistics.py", "scripts/rebuild_ledgers.py"]})
    write_json("results/extensions/summary.json", result)
    summary = read("results/summary.json")
    summary["extensions"] = result
    write_json("results/summary.json", summary)
    print(f"Summarized {result['runs']} complete offline runs; {len(result['noise_observations'])} noise-rule participations")


if __name__ == "__main__":
    main()
