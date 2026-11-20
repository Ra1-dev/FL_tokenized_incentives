"""Compare full chain evidence across platforms; elapsed time is not exact."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def equivalent(left, right, elapsed=False):
    left, right = dict(left), dict(right)
    if elapsed:
        for item in [left,right]:
            seconds = item.pop("wall_seconds")
            if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("a measured positive elapsed time is required")
    return left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks",type=Path,required=True)
    parser.add_argument("--replays",type=Path,required=True)
    parser.add_argument("--stress",type=Path,required=True)
    parser.add_argument("--gas",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    canonical = read("results/contracts/checks.json")
    current = read(args.checks)
    checks = {"checks_"+key: canonical[key] == current[key]
              for key in ["tests_passed", "tests_failed", "fuzz_runs", "fuzz_seed", "tests", "static_counts", "source_sha256", "provenance"]}
    evidence = []
    for path in sorted(Path("results/onchain").glob("*.json")):
        if path.name in {"stress.json","summary.json"}:
            continue
        other = args.replays/path.name
        left, right = read(path), read(other)
        checks[path.name] = equivalent(left,right,elapsed=True)
        evidence.append(dict(case=path.name, canonical_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             compared_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
                             canonical_seconds=left["wall_seconds"], compared_seconds=right["wall_seconds"],
                             exact_except_elapsed=checks[path.name]))
    checks["stress"] = read("results/onchain/stress.json") == read(args.stress)
    checks["gas"] = read("results/gas/summary.json") == read(args.gas/"summary.json")
    checks["boundary"] = read("results/gas/boundary/boundary.json") == read(args.gas/"boundary/boundary.json")
    canonical_inputs = ["results/contracts/checks.json", "results/onchain/stress.json",
                        "results/gas/summary.json", "results/gas/boundary/boundary.json"]
    compared_inputs = [args.checks, args.stress, args.gas/"summary.json", args.gas/"boundary/boundary.json"]
    report = dict(platform=sys.platform, python=platform.python_version(), checks=checks,
                  canonical_input_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in canonical_inputs},
                  compared_input_sha256={Path(name).as_posix(): hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in compared_inputs},
                  replay_cases=evidence, replay_round_rule_cases=sum(read(args.replays/e["case"])["rounds"] for e in evidence),
                  passed=all(checks.values()), excluded_field="wall_seconds only in full replay records; all states, gas and scientific provenance compare exactly",
                  comparison_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(dict(passed=report["passed"], checks=len(checks), replay_round_rule_cases=report["replay_round_rule_cases"])))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
