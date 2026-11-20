"""Compare independent repeats or explicitly different execution devices."""

import argparse
import csv
import json
from pathlib import Path

from fltoken.io import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--cross-device", action="store_true")
    parser.add_argument("--output", default="results/validation/repeatability.json")
    args = parser.parse_args()
    paths = [args.first, args.second]
    metadata = [json.loads((p / "metadata.json").read_text(encoding="utf-8")) for p in paths]
    if any(m["status"] != "complete" for m in metadata):
        raise ValueError("both runs must be complete")
    if metadata[0]["config"]["seed"] != metadata[1]["config"]["seed"]:
        raise ValueError("repeatability comparison requires the same seed")
    ignored = {"workers", "data_dir", "save_updates"}
    if args.cross_device:
        ignored |= {"device", "gpu_fraction"}
    configs = [{k: v for k, v in m["config"].items() if k not in ignored} for m in metadata]
    if configs[0] != configs[1]:
        raise ValueError("scientific configurations differ")
    hashes = {name: [sha256(p / name) for p in paths] for name in ["per_round.csv", "round_metrics.csv", "ledger.json"]}
    with (paths[0] / "per_round.csv").open(encoding="utf-8", newline="") as first, (paths[1] / "per_round.csv").open(encoding="utf-8", newline="") as second:
        left, right = list(csv.DictReader(first)), list(csv.DictReader(second))
    if len(left) != len(right) or any((a["round"], a["client_id"]) != (b["round"], b["client_id"]) for a, b in zip(left, right)):
        raise AssertionError("client participation schedules differ")
    differences = {key: max(abs(float(a[key]) - float(b[key])) for a, b in zip(left, right)) for key in ["accuracy", "loss", "cosine", "B1_score", "B1_share"]}
    final_difference = abs(float(left[-1]["accuracy"]) - float(right[-1]["accuracy"]))
    passed = final_difference <= .02 if args.cross_device else all(pair[0] == pair[1] for pair in hashes.values())
    result = dict(paths=[p.as_posix() for p in paths], hashes=hashes, byte_identical=all(pair[0] == pair[1] for pair in hashes.values()),
                  max_absolute_differences=differences, final_accuracy_difference=final_difference,
                  mode="cross-device" if args.cross_device else "exact-repeat", passed=passed)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
