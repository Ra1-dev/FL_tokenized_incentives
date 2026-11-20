"""Recompute or verify settlement from immutable recorded floating shares."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from fltoken.io import sha256, write_json
from fltoken.metrics import ROUND_POOL, SCALE, SETTLEMENT_VERSION, payout, summarize


def rebuild(folder, rewrite=False):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    with (folder / "per_round.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != config["rounds"] * config["clients_per_round"]:
        raise ValueError(f"incomplete participation log: {folder}")
    for row in rows:
        for key in ["round", "client_id"]:
            row[key] = int(row[key])
        row["adversary"] = row["adversary"] == "True"
        row["accuracy"] = float(row["accuracy"])
        for name in config["rules"]:
            row[f"{name}_share"] = float(row[f"{name}_share"])
            row[f"{name}_slash"] = row[f"{name}_slash"] == "True"
    ledger = {name: [] for name in config["rules"]}
    for round in range(1, config["rounds"] + 1):
        selected = [row for row in rows if row["round"] == round]
        if len(selected) != config["clients_per_round"]:
            raise ValueError(f"incorrect round size: {folder}, round {round}")
        for name in ledger:
            units, amounts = payout(np.array([row[f"{name}_share"] for row in selected]))
            ledger[name].append(dict(round=round, ids=[row["client_id"] for row in selected],
                                     shares1e18=units, payouts=amounts,
                                     slash=[row[f"{name}_slash"] for row in selected],
                                     remainder=ROUND_POOL - sum(amounts)))
    original = json.loads((folder / "ledger.json").read_text(encoding="utf-8"))
    attacks = {int(k): v for k, v in json.loads((folder / "protocol.json").read_text(encoding="utf-8"))["adversaries"].items()}
    metrics = summarize(rows, ledger, config, attacks)
    if rewrite:
        provenance_path = folder / "ledger_provenance.json"
        if not provenance_path.exists() and original != ledger:
            write_json(provenance_path, dict(
                previous_ledger_sha256=sha256(folder / "ledger.json"),
                training_log_sha256=sha256(folder / "per_round.csv"),
                training_commit=metadata["git_commit"],
                previous_conversion="literal-float-times-scale",
                conversion=SETTLEMENT_VERSION,
                conversion_source_sha256=sha256("src/fltoken/metrics.py"),
                rebuild_source_sha256=sha256(__file__),
                derivation_base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            ))
        write_json(folder / "ledger.json", ledger)
        write_json(folder / "metrics.json", metrics)
        write_json(folder / "hashes.json", {name: sha256(folder / name) for name in ["per_round.csv", "round_metrics.csv", "ledger.json"]})
    elif original != ledger or json.loads((folder / "metrics.json").read_text(encoding="utf-8")) != metrics:
        raise AssertionError(f"ledger or metrics mismatch: {folder}; run with --rewrite for a declared conversion migration")
    items = [item for rounds in ledger.values() for item in rounds]
    assert all(sum(item["shares1e18"]) <= SCALE and item["remainder"] >= 0 for item in items)
    return dict(path=folder.as_posix(), round_rule_cases=len(items), overshoots=0,
                minimum_remainder=min(item["remainder"] for item in items),
                maximum_remainder=max(item["remainder"] for item in items),
                per_round_sha256=sha256(folder / "per_round.csv"), ledger_sha256=sha256(folder / "ledger.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--output", default="results/validation/settlement.json")
    args = parser.parse_args()
    folders = args.paths or sorted(p.parent for p in Path("results").rglob("metadata.json")
                                   if json.loads(p.read_text(encoding="utf-8")).get("status") == "complete")
    results = [rebuild(folder, args.rewrite) for folder in folders]
    output = dict(conversion=SETTLEMENT_VERSION, runs=len(results),
                  round_rule_cases=sum(item["round_rule_cases"] for item in results), overshoots=0, checks=results)
    write_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
