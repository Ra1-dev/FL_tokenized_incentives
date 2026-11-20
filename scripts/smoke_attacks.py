"""Exercise every attack through Flower, including a genuinely lagged snapshot."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from fltoken.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="out/attack-smoke")
    args = parser.parse_args()
    checks = []
    for attack in ["zero", "noise", "stale", "label_flip", "sign_flip"]:
        output = Path(args.output) / attack
        subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", f"configs/smoke_{attack}.yaml",
                        "--seed", "100", "--device", args.device, "--output", str(output)], check=True)
        with (output / "per_round.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        malicious = [r for r in rows if r["attack"] == attack]
        if not malicious:
            raise AssertionError(f"{attack} was not sampled")
        if attack == "zero":
            assert all(float(r["delta_norm"]) == 0 for r in malicious)
        if attack == "stale":
            assert any(int(r["round"]) >= 7 and int(r["source_round"]) >= 2 for r in malicious)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        checks.append(dict(attack=attack, participations=len(malicious), rounds=metadata["config"]["rounds"],
                           status=metadata["status"], wall_seconds=metadata["wall_seconds"]))
    write_json(Path(args.output) / "checks.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
