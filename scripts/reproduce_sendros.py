"""Reproduce the source comparison, retaining all noise participations."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from fltoken.config import load_config
from fltoken.io import sha256, write_json
from rebuild_ledgers import rebuild


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-run", action="store_true", help="fail if any complete full run is missing")
    args = parser.parse_args()
    source = read_json("results/literature/sendros.json")
    noise_rows, trajectories, shares, provenance = [], [], [], []
    for seed in source["setup"]["seeds"]:
        folder = Path(f"results/runs/mnist-sendros_mix-s{seed}-{args.device}-t20")
        complete = (folder / "metadata.json").exists() and read_json(folder / "metadata.json")["status"] == "complete"
        if not complete:
            if args.no_run:
                raise RuntimeError(f"full run missing: {folder}")
            subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", "configs/sendros_mix.yaml", "--seed", str(seed), "--device", args.device], check=True)
        metadata = read_json(folder / "metadata.json")
        config = metadata["config"]
        expected_config = load_config("configs/sendros_mix.yaml")
        expected_config.update(seed=seed, device=args.device)
        execution_keys = {"workers", "gpu_fraction", "data_dir", "save_updates"}
        for key, expected in expected_config.items():
            if key not in execution_keys and {"stale_mode": "train_snapshot", **config}.get(key) != expected:
                raise ValueError(f"{folder}: {key} differs from full reproduction protocol")
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
            honest = [float(r["B1_score"]) for r in selected if r["adversary"] == "False"]
            noise = [float(r["B1_score"]) for r in selected if r["attack"] == "noise"]
            trajectories.append(dict(seed=seed, round=round, honest_mean_q=float(np.mean(honest)) if honest else None,
                                     noise_mean_q=float(np.mean(noise)) if noise else None,
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
                           sampled_attacker_fraction=1-participations.get("none", 0)/len(rows),
                           baseline_attacker_shares={rule: entry["attacker_share"] for rule, entry in all_metrics.items()},
                           participations_by_attack=participations,
                           reward_share_by_attack={name: amount/minted for name, amount in by_attack.items()}))
        provenance.append(dict(seed=seed, path=folder.as_posix(), git_commit=metadata["git_commit"],
                               per_round_sha256=sha256(folder / "per_round.csv"), wall_seconds=metadata["wall_seconds"]))
    errors = np.array([r["relative_error"] if r["relative_error"] is not None else np.inf for r in noise_rows])
    quantiles = {str(p): float(np.quantile(errors, p)) for p in [0, .25, .5, .75, .9, .95, 1]}
    quantiles = {p: value if np.isfinite(value) else None for p, value in quantiles.items()}
    actual = np.array([row["attacker_share"] for row in shares])
    result = dict(device=args.device, noise_participations=len(noise_rows), zero_measured_cosines=int(np.isinf(errors).sum()),
                  relative_error_quantiles=quantiles, median_error_target=.03,
                  median_error_pass=bool(np.median(errors) <= .03),
                  attacker_share_mean=float(actual.mean()), attacker_share_sample_sd=float(actual.std(ddof=1)),
                  attacker_share_population_sd=float(actual.std(ddof=0)),
                  published=source["standard"], attacker_share_difference=float(actual.mean() - source["standard"]["attacker_share_mean"]),
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
                sum((row[f"{group}_mean_q"] or 0)*row[f"{group}_count"] for row in window)/count if count else None)
    out = Path("results/reproduction")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "noise_participations.csv", noise_rows)
    write_csv(out / "quality_trajectory.csv", trajectories)
    write_csv(out / "per_seed.csv", shares)
    write_json(out / "summary.json", result)
    summary_path = Path("results/summary.json")
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary["reproduction"] = result
    write_json(summary_path, summary)
    print(json.dumps({k: v for k, v in result.items() if k not in {"quality_trajectory", "noise_observations"}}, indent=2))
    if not result["median_error_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
