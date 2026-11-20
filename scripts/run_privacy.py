"""Run the declared honest-noise comparison with one fixed pilot clipping bound."""

import argparse
from datetime import datetime, timezone
import itertools
from pathlib import Path
import subprocess
import sys

from evaluation import published_bytes, published_method, read
from extension_replay import read_training
from fltoken.cache import load_update_rounds
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from fltoken.privacy import account_schedule
from grid_reproduction import verify_training_source
from privacy_protocol import PLAN, FROZEN, calibration, published_privacy, settings
from rebuild_ledgers import rebuild
from run_grid import checked_execution_clock, finish_execution_clock
from run_pilots import training_sources


def privacy_cells(plan):
    for seed, multiplier in itertools.product(plan["seeds"], plan["noise_multipliers"]):
        name = f"{plan['dataset']}-sigma{multiplier:g}-s{seed}-{plan['device']}-t{plan['rounds']}"
        yield dict(seed=seed, noise_multiplier=multiplier, path=f"results/privacy/{name}")


def expected_config(plan, cell, frozen, method):
    base = load_config(plan["config"])
    return {**base, **{key: plan[key] for key in ["dataset", "attack", "fraction", "rounds", "device", "workers", "gpu_fraction", "rules"]},
            "seed": cell["seed"], "ours_params": method["params"],
            "privacy": settings(frozen, cell["noise_multiplier"])}


def verify_cell(folder, expected, frozen_path=FROZEN, verify_cache=True):
    """Reconstruct accounting from public identities and verify every raw matrix."""
    metadata, attacks, rounds, inputs = read_training(folder)
    if (attacks or metadata.get("source_git_dirty") is not False or metadata["config"] != expected or
            metadata.get("privacy_sha256") != sha256(frozen_path) or
            metadata.get("method_sha256") != sha256("configs/method_frozen.json")):
        raise ValueError("privacy training differs from its frozen scientific configuration")
    account = account_schedule(expected, [ids for _, ids, _ in rounds])
    if read(folder/"privacy.json") != account:
        raise ValueError("privacy accounting differs from the actual public release schedule")
    rebuild(folder)
    if verify_cache and sum(1 for _ in load_update_rounds(folder)) != expected["rounds"]:
        raise ValueError("every privacy round must retain its common perturbed update matrix")
    return metadata, dict(**inputs, **{name: sha256(folder/name) for name in ["privacy.json", "metrics.json", "timing.json"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    published_bytes(PLAN)
    frozen = published_privacy(full_evidence=True)
    calibration()
    published_privacy(full_evidence=True)
    method = published_method(full_evidence=True)
    plan = read(PLAN)
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs", "scripts"], text=True).strip():
        raise ValueError("commit the privacy experiment sources before running the comparison")
    planned = list(privacy_cells(plan))
    if not planned or len({cell["path"] for cell in planned}) != len(planned):
        raise ValueError("privacy cells must be nonempty and unique")
    base = load_config(plan["config"])
    if base["dataset"] != plan["dataset"] or base["train_limit"] or base["test_limit"] or not base["save_updates"]:
        raise ValueError("privacy comparison requires full datasets and retained perturbed updates")
    sources = training_sources(plan)
    binding = dict(plan_sha256=sha256(PLAN), frozen_sha256=sha256(FROZEN),
                   method_sha256=sha256("configs/method_frozen.json"), training_source_sha256=sources)
    clock_path = Path("results/privacy/execution.json")
    if not clock_path.exists():
        if args.no_run or any((Path(cell["path"])/"metadata.json").exists() for cell in planned):
            raise ValueError("retain the original privacy execution clock on resumption")
        write_json(clock_path, dict(**binding, started_utc=datetime.now(timezone.utc).isoformat(),
                                   finished_utc=None, elapsed_wall_seconds=None))
    clock = checked_execution_clock(clock_path, binding)
    completed, checked, common_versions = [], set(), None

    def record():
        if (args.no_run or clock["finished_utc"] is not None) and len(completed) != len(planned):
            return
        write_json("results/privacy/manifest.json", dict(schema=1, **binding, expected_cells=len(planned),
                   completed_cells=len(completed), complete=len(completed) == len(planned),
                   execution_sha256=sha256(clock_path), elapsed_wall_seconds=clock["elapsed_wall_seconds"],
                   training_device=plan["device"], successful_run_seconds=sum(cell["wall_seconds"] for cell in completed),
                   cells=completed))

    for cell in planned:
        if training_sources(plan) != sources:
            raise ValueError("privacy training sources changed during the comparison")
        folder = Path(cell["path"])
        expected = expected_config(plan, cell, frozen, method)
        existing = read(folder/"metadata.json") if (folder/"metadata.json").exists() else {}
        if existing.get("status") != "complete":
            if args.no_run or clock["finished_utc"] is not None:
                raise ValueError("missing complete privacy cell; restore or finish its original execution")
            command = [sys.executable, "scripts/run_experiment.py", "--config", plan["config"],
                       "--method-config", "configs/method_frozen.json", "--privacy-config", str(FROZEN),
                       "--noise-multiplier", str(cell["noise_multiplier"]), "--seed", str(cell["seed"]),
                       "--rules", *plan["rules"], "--output", str(folder)]
            for key in ["attack", "fraction", "rounds", "device", "workers", "gpu_fraction"]:
                command.extend(["--"+key.replace("_", "-"), str(plan[key])])
            subprocess.run(command, check=True)
        metadata, inputs = verify_cell(folder, expected)
        if metadata["source_sha256"] != sources or metadata["source_config"] != plan["config"]:
            raise ValueError("privacy training sources differ between cells")
        if common_versions is not None and common_versions != metadata["versions"]:
            raise ValueError("privacy library versions differ between cells")
        common_versions = metadata["versions"]
        if metadata["git_commit"] not in checked:
            verify_training_source(metadata)
            checked.add(metadata["git_commit"])
        completed.append(dict(**cell, inputs=inputs, source_commit=metadata["git_commit"], wall_seconds=metadata["wall_seconds"]))
        record()
        print(f"Validated privacy cells: {len(completed)}/{len(planned)}", flush=True)
    if args.no_run and clock["finished_utc"] is None:
        raise ValueError("privacy completion was not recorded; resume before reporting elapsed time")
    clock = finish_execution_clock(clock_path, binding)
    record()


if __name__ == "__main__":
    main()
