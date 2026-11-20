"""Verify portable metric reconstruction from the same committed training logs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folders = sorted(p.parent for p in Path("results").rglob("metadata.json")
                     if json.loads(p.read_text(encoding="utf-8")).get("status") == "complete")
    artifacts = [folder/name for folder in folders for name in ["per_round.csv", "round_metrics.csv", "ledger.json", "metrics.json"]]
    artifacts += sorted(Path("results/reproduction").glob("*.json"))+sorted(Path("results/reproduction").glob("*.csv"))
    artifacts += [Path("results/sensitivity/summary.json"), Path("results/validation/settlement.json")]
    before = {p.as_posix():digest(p) for p in artifacts}
    commands = [["scripts/rebuild_ledgers.py"], ["scripts/reproduce_sendros.py", "--no-run"],
                ["scripts/stale_sensitivity.py", "--no-run"]]
    env = {**os.environ, "PYTHONPATH":str(Path("src").resolve()), "PYTHONUTF8":"1"}
    executed = []
    for command in commands:
        result = subprocess.run([sys.executable, *command], env=env, capture_output=True, text=True, encoding="utf-8")
        executed.append(dict(command=["python", *command], exit_code=result.returncode))
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    after = {p.as_posix():digest(p) for p in artifacts}
    changed = [name for name in before if before[name] != after[name]]
    if changed:
        raise AssertionError(f"reconstructed artifacts changed: {changed}")
    migrations = [folder/"metrics_provenance.json" for folder in folders if (folder/"metrics_provenance.json").exists()]
    changed_fields = {}
    for path in migrations:
        for item in json.loads(path.read_text(encoding="utf-8"))["migrations"]:
            for fields in item["changed_fields"].values():
                for field in fields:
                    changed_fields[field] = changed_fields.get(field,0)+1
    result = dict(passed=True, platform=platform.platform(), python=platform.python_version(),
                  versions={name:importlib.metadata.version(name) for name in ["numpy", "PyYAML"]},
                  runs=len(folders), artifacts=len(after), artifact_sha256=after, changed_artifacts=changed,
                  migration_runs=len(migrations), migrated_rule_fields=changed_fields,
                  commands=executed, source_sha256={p: digest(Path(p)) for p in ["src/fltoken/metrics.py",
                      "scripts/rebuild_ledgers.py", "scripts/reproduce_sendros.py", "scripts/stale_sensitivity.py",
                      "scripts/reproduction_analysis.py", "scripts/grid_reproduction.py", "src/fltoken/statistics.py",
                      "scripts/check_derived_metrics.py"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"artifact_sha256","source_sha256","commands"}},sort_keys=True))


if __name__ == "__main__":
    main()
