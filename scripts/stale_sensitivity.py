"""Run the predeclared own-report replay sensitivity without changing the main protocol."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from fltoken.io import sha256, write_json
from rebuild_ledgers import rebuild


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    source = read("results/literature/sendros.json")
    runs = []
    for seed in source["setup"]["seeds"]:
        main_folder = Path(f"results/runs/mnist-sendros_mix-s{seed}-cpu-t20")
        folder = Path(f"results/sensitivity/replay-own-s{seed}")
        if not (folder / "metadata.json").exists() or read(folder / "metadata.json")["status"] != "complete":
            if args.no_run:
                raise RuntimeError(f"complete sensitivity run required: {folder}")
            subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", "configs/sendros_replay_own.yaml",
                            "--seed", str(seed), "--device", "cpu", "--output", str(folder)], check=True)
        metadata = read(folder / "metadata.json")
        original_metadata = read(main_folder / "metadata.json")
        for checked_folder in [main_folder, folder]:
            hashes = read(checked_folder / "hashes.json")
            for filename in ["per_round.csv", "round_metrics.csv", "ledger.json"]:
                if hashes.get(filename) != sha256(checked_folder / filename):
                    raise AssertionError(f"recorded artifact hash mismatch: {checked_folder}, {filename}")
            # Recompute both the integer ledger and derived metrics from the
            # floating shares, instead of trusting a cached attacker fraction.
            rebuild(checked_folder)
        expected = {"stale_mode": "train_snapshot", **original_metadata["config"]}
        expected["stale_mode"] = "replay_own"
        if metadata["config"] != expected:
            raise AssertionError("sensitivity must change only stale_mode")
        main_protocol, protocol = [read(p / "protocol.json") for p in [main_folder, folder]]
        if main_protocol != protocol:
            raise AssertionError("partition, attack assignment or aggregation changed")
        with (folder / "per_round.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (main_folder / "per_round.csv").open(encoding="utf-8", newline="") as stream:
            original = list(csv.DictReader(stream))
        schedule = lambda values: [(r["round"], r["client_id"], r["attack"]) for r in values]
        if schedule(rows) != schedule(original):
            raise AssertionError("participation schedule changed")
        stale = [row for row in rows if row["attack"] == "stale"]
        # With persistent stale identities and no warm-up report, recursive
        # replay of an initially missing report remains zero by induction.
        if not stale or any(float(row["delta_norm"]) != 0 for row in stale):
            raise AssertionError("persistent own-report replay did not preserve the declared zero initial condition")
        a, b = [read(p / "metrics.json")["B1"]["attacker_share"] for p in [main_folder, folder]]
        runs.append(dict(seed=seed, main_attacker_share=a, variant_attacker_share=b,
                         difference=b-a, path=folder.as_posix(),
                         wall_seconds=metadata["wall_seconds"], git_commit=metadata["git_commit"],
                         per_round_sha256=sha256(folder / "per_round.csv"), stale_participations=len(stale),
                         stale_nonzero_reports=0))
    a = np.array([row["main_attacker_share"] for row in runs])
    b = np.array([row["variant_attacker_share"] for row in runs])
    published = source["standard"]["attacker_share_mean"]
    result = dict(variant="replay_own", device="cpu", published_attacker_share=published,
                  main_mean=float(a.mean()), variant_mean=float(b.mean()),
                  variant_sample_sd=float(b.std(ddof=1)), paired_mean_difference=float((b-a).mean()),
                  main_absolute_source_gap=float(abs(a.mean()-published)),
                  variant_absolute_source_gap=float(abs(b.mean()-published)),
                  closer_to_source=bool(abs(b.mean()-published) < abs(a.mean()-published)),
                  semantics="Latest own reported delta from a round at or before t-lag; zero if none. Persistent stale identities have no honest warm-up, so every stale report is zero.",
                  runs=runs)
    write_json("results/sensitivity/summary.json", result)
    summary = read("results/summary.json")
    summary["stale_sensitivity"] = result
    write_json("results/summary.json", summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
