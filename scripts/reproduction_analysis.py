"""Shared exact analysis of source-comparison and evaluation-mixture runs."""

import csv
import json
import math
from fractions import Fraction
from pathlib import Path
from statistics import mean, pstdev, stdev

from fltoken.io import sha256
from rebuild_ledgers import rebuild

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def exact_quantile(values, probability):
    """Linear quantile with exact rational interpolation, rounded once."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1)*Fraction(str(probability))
    lower = position.numerator//position.denominator
    fraction = position-lower
    if not math.isfinite(ordered[lower]):
        return None
    if not fraction:
        return ordered[lower]
    if not math.isfinite(ordered[lower+1]):
        return None
    return float((1-fraction)*Fraction(ordered[lower])+fraction*Fraction(ordered[lower+1]))


def collect_run(folder, seed):
    folder = Path(folder)
    metadata = read_json(folder / "metadata.json")
    config = metadata["config"]
    noise_rows, trajectories, shares, provenance = [], [], [], []
    hashes = read_json(folder / "hashes.json")
    for filename in ["per_round.csv", "round_metrics.csv", "ledger.json"]:
        if hashes.get(filename) != sha256(folder / filename):
            raise AssertionError(f"recorded artifact hash mismatch: {folder}, {filename}")
    rebuild(folder)
    with (folder / "per_round.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != config["rounds"] * config["clients_per_round"]:
        raise ValueError(f"{folder}: incomplete participation log")
    for row in rows:
        if row["attack"] != "noise":
            continue
        measured, predicted = float(row["cosine"]), float(row["predicted_bias"])
        noise_rows.append(dict(seed=seed, round=int(row["round"]), client_id=int(row["client_id"]),
                               delta_norm=float(row["delta_norm"]), aggregate_norm=float(row["aggregate_norm"]),
                               predicted=predicted, measured=measured, absolute_error=abs(predicted - measured),
                               relative_error=abs(predicted - measured) / abs(measured) if measured else None))
    for round in range(1, config["rounds"] + 1):
        selected = [row for row in rows if int(row["round"]) == round]
        honest = [Fraction(r["B1_score"]) for r in selected if r["adversary"] == "False"]
        noise = [Fraction(r["B1_score"]) for r in selected if r["attack"] == "noise"]
        trajectories.append(dict(seed=seed, round=round, honest_mean_q=float(mean(honest)) if honest else None,
                                 noise_mean_q=float(mean(noise)) if noise else None,
                                 honest_q_sum=int(sum(honest)), noise_q_sum=int(sum(noise)),
                                 honest_count=len(honest), noise_count=len(noise)))
    metrics = read_json(folder / "metrics.json")["B1"]
    all_metrics = read_json(folder / "metrics.json")
    attack_map = read_json(folder / "protocol.json")["adversaries"]
    by_attack, participations = {}, {}
    for row in rows:
        participations[row["attack"]] = participations.get(row["attack"], 0) + 1
    for item in read_json(folder / "ledger.json")["B1"]:
        for client, amount in zip(item["ids"], item["payouts"]):
            attack = attack_map.get(str(client), "none")
            by_attack[attack] = by_attack.get(attack, 0) + amount
    minted = sum(by_attack.values())
    shares.append(dict(seed=seed, attacker_share=metrics["attacker_share"], honest_gini=metrics["honest_gini"],
                       final_accuracy=metrics["final_accuracy"], empty_rounds=metrics["empty_rounds"],
                       sampled_attacker_fraction=float(Fraction(len(rows)-participations.get("none", 0),len(rows))),
                       baseline_attacker_shares={rule: all_metrics[rule]["attacker_share"] for rule in ["B0", "B1", "B2"]},
                       participations_by_attack=participations,
                       reward_share_by_attack={name:float(Fraction(amount,minted)) for name, amount in by_attack.items()}))
    provenance.append(dict(seed=seed, path=folder.as_posix(), git_commit=metadata["git_commit"],
                           per_round_sha256=sha256(folder / "per_round.csv"), wall_seconds=metadata["wall_seconds"]))
    return dict(noise=noise_rows, trajectory=trajectories, share=shares[0], provenance=provenance[0])


def summarize_runs(runs, source, device):
    noise_rows = [row for run in runs for row in run["noise"]]
    trajectories = [row for run in runs for row in run["trajectory"]]
    shares = [run["share"] for run in runs]
    provenance = [run["provenance"] for run in runs]
    errors = [r["relative_error"] if r["relative_error"] is not None else math.inf for r in noise_rows]
    quantiles = {str(p): exact_quantile(errors,p) for p in [0, .25, .5, .75, .9, .95, 1]}
    actual = [row["attacker_share"] for row in shares]
    result = dict(device=device, noise_participations=len(noise_rows), zero_measured_cosines=sum(math.isinf(e) for e in errors),
                  relative_error_quantiles=quantiles, median_error_target=.03,
                  median_error_pass=quantiles["0.5"] is not None and quantiles["0.5"] <= .03,
                  attacker_share_mean=mean(actual), attacker_share_sample_sd=stdev(actual),
                  attacker_share_population_sd=pstdev(actual),
                  published=source["standard"], attacker_share_difference=mean(actual)-source["standard"]["attacker_share_mean"],
                  runs=provenance, quality_trajectory=trajectories,
                  noise_observations=noise_rows, per_seed=shares)
    result["quality_windows"] = {}
    for label, start, stop in [("early", 1, 5), ("late", 16, 20)]:
        window = [row for row in trajectories if start <= row["round"] <= stop]
        result["quality_windows"][label] = dict(start=start, stop=stop)
        for group in ["honest", "noise"]:
            count = sum(row[f"{group}_count"] for row in window)
            result["quality_windows"][label][f"{group}_participations"] = count
            result["quality_windows"][label][f"{group}_mean_q"] = (
                float(Fraction(sum(row[f"{group}_q_sum"] for row in window),count)) if count else None)
    return result
