"""Resume exact chain replay without treating partial grid coverage as complete."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from contract_evidence import provenance
from evaluation import evaluation_cells, published_bytes, recorded_method, read
from fltoken.io import sha256, write_json
from grid_reproduction import verify_training_source
from summarize_contracts import validate_states


def replay_jobs(plan):
    """All cosine/full-method ledgers, plus a declared sample of other baselines."""
    for cell in evaluation_cells(plan):
        rules = ["B1", "Mstar"]
        sample = cell["seed"] == plan["seeds"][0] and (
            cell["attack"] in {"none", "sendros_mix"} or cell["fraction"] == max(plan["fractions"]))
        if sample:
            rules += ["B0", "B2"]
        for rule in rules:
            yield dict(**cell, rule=rule)


def covered_cells(manifest, plan, available):
    planned = {cell["path"]: cell for cell in evaluation_cells(plan)}
    supplied = {cell["path"]: cell for cell in manifest["cells"]}
    if (len(supplied) != len(manifest["cells"]) or not supplied.keys() <= planned.keys()
            or manifest["completed_cells"] != len(supplied) or manifest["expected_cells"] != len(planned)):
        raise ValueError("grid manifest has invalid or duplicated coverage")
    for path, record in supplied.items():
        if any(record.get(key) != value for key, value in planned[path].items()):
            raise ValueError("grid cell differs from the declared plan")
    if not available and supplied.keys() != planned.keys():
        raise ValueError("complete grid required; partial replay must be explicitly requested")
    return supplied


def validate_case(case, job, folder, delay):
    expected = dict(path=job["path"], rule=job["rule"], rounds=job["rounds"],
                    clients=read(folder/"metadata.json")["config"]["num_clients"], vesting_delay=delay, exact=True)
    if any(case.get(key) != value for key, value in expected.items()):
        raise ValueError("cached replay describes another run, rule or settlement")
    if case["provenance"] != provenance("scripts/replay_onchain.py"):
        raise ValueError("cached replay source provenance differs")
    for field, source in [("pool_source_sha256", "contracts/src/RewardPool.sol"),
                          ("token_source_sha256", "contracts/src/RewardToken.sol")]:
        if case[field] != sha256(source):
            raise ValueError("cached replay contract digest differs")
    if case["ledger_sha256"] != sha256(folder/"ledger.json") or case["participation_sha256"] != sha256(folder/"per_round.csv"):
        raise ValueError("cached replay input hashes differ")
    count = validate_states(expected["clients"], delay, read(folder/"ledger.json")[job["rule"]], case["states"])
    if count != job["rounds"]:
        raise ValueError("cached replay has incomplete state history")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("configs/evaluation_plan.json"))
    parser.add_argument("--chain-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=Path("results/grid_onchain"))
    parser.add_argument("--available", action="store_true", help="replay completed cells while retaining partial coverage")
    parser.add_argument("--no-run", action="store_true", help="validate existing on-chain evidence without launching a chain")
    args = parser.parse_args()
    published_bytes(args.plan)
    published_bytes("configs/method_frozen.json")
    plan, method = read(args.plan), recorded_method()
    manifest_bytes = Path("results/grid/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest["plan_sha256"] != sha256(args.plan) or manifest["method_sha256"] != sha256("configs/method_frozen.json"):
        raise ValueError("grid manifest differs from the published plan or method")
    settlement = method["settlement"]
    if (settlement["stake_tokens"], settlement["slash_numerator"], settlement["slash_denominator"]) != (100, 1, 10):
        raise ValueError("frozen economics differ from the deployed contract")
    delay = settlement["vesting_rounds"]
    supplied = covered_cells(manifest, plan, args.available)
    jobs = list(replay_jobs(plan))
    completed = []
    checked_sources = set()
    source_hash = sha256(__file__)
    for job in jobs:
        if job["path"] not in supplied:
            continue
        folder, record = Path(job["path"]), supplied[job["path"]]
        if sha256(folder/"metadata.json") != record["metadata_sha256"] or sha256(folder/"per_round.csv") != record["per_round_sha256"]:
            raise ValueError("grid run differs from the completed manifest")
        metadata = read(folder/"metadata.json")
        if metadata["status"] != "complete" or metadata["method_sha256"] != manifest["method_sha256"]:
            raise ValueError("only complete frozen-method runs may be replayed")
        source_key = (metadata["git_commit"], json.dumps(metadata["source_sha256"], sort_keys=True),
                      metadata["source_git_dirty"])
        if source_key not in checked_sources:
            verify_training_source(metadata, method=method)
            checked_sources.add(source_key)
        filename = job["path"].replace("/", "_")+f"-{job['rule']}.json"
        target = args.output/filename
        if not target.exists():
            if args.no_run:
                raise ValueError(f"missing on-chain replay: {target}")
            subprocess.run([str(args.chain_python.resolve()), "scripts/replay_onchain.py", str(folder),
                            "--rule", job["rule"], "--vesting-delay", str(delay), "--output", str(args.output)], check=True)
        case = read(target)
        count = validate_case(case, job, folder, delay)
        completed.append(dict(path=job["path"], rule=job["rule"], rounds=count,
                              evidence=target.as_posix(), evidence_sha256=sha256(target)))
        if source_hash != sha256(__file__):
            raise ValueError("replay driver changed while processing the grid")
        result = dict(expected_jobs=len(jobs), completed_jobs=len(completed), complete=len(completed)==len(jobs),
                      expected_round_rule_cases=sum(item["rounds"] for item in jobs),
                      round_rule_cases=sum(item["rounds"] for item in completed), cases=completed,
                      plan_sha256=sha256(args.plan), method_sha256=manifest["method_sha256"],
                      grid_manifest_sha256=manifest_hash, driver_sha256=source_hash)
        write_json(args.output/"coverage.json", result)
        print(f"Validated chain jobs: {len(completed)}/{len(jobs)}", flush=True)
    if not completed:
        raise ValueError("no completed grid cells are available to replay")


if __name__ == "__main__":
    main()
