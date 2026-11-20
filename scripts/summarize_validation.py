"""Collect recorded full-run timings and independently generated validation evidence."""

import json
from pathlib import Path

from fltoken.io import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    timings = []
    failures = []
    for path in sorted(Path("results").rglob("metadata.json")):
        m = read(path)
        c = m["config"]
        if m["status"] == "failed":
            failures.append(dict(path=path.parent.as_posix(), device=c["device"],
                                 workers=c["workers"], gpu_fraction=c["gpu_fraction"],
                                 wall_seconds=m["wall_seconds"], error=m["error"]))
        if (m["status"] != "complete" or c["rounds"] != 20 or c["train_limit"] or c["test_limit"]
                or c["local_epochs"] != 5 or c["attack"] != "sendros_mix" or c.get("stale_mode", "train_snapshot") != "train_snapshot"):
            continue
        bound = 600 if c["device"] == "cuda" else 1800
        timings.append(dict(path=path.parent.as_posix(), seed=c["seed"], device=c["device"],
                            device_name=m["device_name"], workers=c["workers"], gpu_fraction=c["gpu_fraction"],
                            wall_seconds=m["wall_seconds"], bound_seconds=bound, within_bound=m["wall_seconds"] <= bound,
                            git_commit=m["git_commit"], source_git_dirty=m.get("source_git_dirty", m["git_dirty"])))
    evidence = {}
    for name in ["repeatability", "cross_device", "gpu_scheduling"]:
        path = Path("results/validation") / (name + ".json")
        item = read(path)
        if not item["passed"]:
            raise AssertionError(f"comparison failed: {path}")
        for filename in ["per_round.csv", "round_metrics.csv", "ledger.json"]:
            actual = [sha256(Path(folder) / filename) for folder in item["paths"]]
            if actual != item["hashes"].get(filename):
                raise AssertionError(f"stale comparison evidence: {path}, {filename}")
        evidence[name] = item
    smoke = Path("results/validation/attack-smoke/checks.json")
    if smoke.exists():
        evidence["attack_smokes"] = read(smoke)
        for item in evidence["attack_smokes"]:
            folder = Path(item["path"])
            if read(folder / "metadata.json")["status"] != "complete":
                raise AssertionError(f"incomplete smoke run: {folder}")
            if any(sha256(folder / name) != digest for name, digest in item["hashes"].items()):
                raise AssertionError(f"stale smoke evidence: {folder}")
    result = dict(timings=timings, failed_runs=failures, checks=evidence)
    write_json("results/validation/summary.json", result)
    summary = read("results/summary.json")
    summary["validation"] = result
    write_json("results/summary.json", summary)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
