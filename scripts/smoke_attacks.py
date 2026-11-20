"""Exercise every attack through Flower, including a genuinely lagged snapshot."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from fltoken.io import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="out/attack-smoke")
    parser.add_argument("--no-run", action="store_true", help="validate existing complete smoke outputs")
    args = parser.parse_args()
    checks = []
    for attack in ["zero", "noise", "stale", "label_flip", "sign_flip"]:
        output = Path(args.output) / attack
        if not args.no_run:
            subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", f"configs/smoke_{attack}.yaml",
                            "--seed", "100", "--device", args.device, "--output", str(output)], check=True)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        config = metadata["config"]
        if metadata["status"] != "complete" or config["attack"] != attack or config["device"] != args.device:
            raise AssertionError(f"{attack}: complete run with matching device and attack required")
        hashes = json.loads((output / "hashes.json").read_text(encoding="utf-8"))
        if any(sha256(output / filename) != digest for filename, digest in hashes.items()):
            raise AssertionError(f"{attack}: recorded output hash mismatch")
        with (output / "per_round.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        malicious = [r for r in rows if r["attack"] == attack]
        if len(rows) != config["rounds"] * config["clients_per_round"]:
            raise AssertionError(f"{attack}: incomplete participation log")
        if not malicious:
            raise AssertionError(f"{attack} was not sampled")
        if attack == "zero":
            assert all(float(r["delta_norm"]) == 0 for r in malicious)
        if attack == "stale":
            assert any(int(r["round"]) >= 7 and int(r["source_round"]) >= 2 for r in malicious)
        checks.append(dict(attack=attack, participations=len(malicious), rounds=metadata["config"]["rounds"],
                           status=metadata["status"], wall_seconds=metadata["wall_seconds"],
                           path=output.as_posix(), hashes=hashes))
    write_json(Path(args.output) / "checks.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
