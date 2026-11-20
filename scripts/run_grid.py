"""Run the complete published evaluation plan with independent method states."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from evaluation import ABLATIONS, comparison_output, evaluation_cells, published_bytes, published_method, read
from fltoken.config import load_config
from fltoken.cache import load_update_rounds
from fltoken.io import sha256, write_json
from rebuild_ledgers import rebuild


def verify_dataset(plan, base):
    """Check separately prepared input files without downloading during training."""
    if "dataset_manifest" not in plan:
        return {}
    from fltoken.datasets import DATASETS
    record = read(plan["dataset_manifest"])
    published_bytes(plan["dataset_manifest"])
    cls, mean, std = DATASETS[base["dataset"]]
    if record["dataset"] != base["dataset"] or record["normalization"] != dict(mean=mean, std=std):
        raise ValueError("dataset normalization differs from the prepared input manifest")
    raw = Path(base["data_dir"]) / cls.__name__ / "raw"
    names = [f"{prefix}-{kind}-idx{dimension}-ubyte{suffix}"
             for prefix in ["train", "t10k"] for kind, dimension in [("images", 3), ("labels", 1)]
             for suffix in ["", ".gz"]]
    if set(record["sha256"]) != set(names):
        raise ValueError("dataset input checksum coverage is incomplete")
    for name, digest in record["sha256"].items():
        if not (raw/name).is_file() or sha256(raw/name) != digest:
            raise ValueError(f"prepared dataset checksum differs: {name}")
    return dict(dataset_sha256=sha256(plan["dataset_manifest"]))


def checked_execution_clock(path, binding):
    clock = read(path)
    if any(clock.get(key) != value for key, value in binding.items()):
        raise ValueError("execution clock belongs to different scientific inputs")
    start = datetime.fromisoformat(clock["started_utc"])
    if start.utcoffset() is None:
        raise ValueError("execution timestamps require an explicit timezone")
    if clock["finished_utc"] is not None:
        stop = datetime.fromisoformat(clock["finished_utc"])
        if stop.utcoffset() is None or stop < start or clock["elapsed_wall_seconds"] != (stop-start).total_seconds():
            raise ValueError("execution elapsed time differs from its recorded endpoints")
    elif clock["elapsed_wall_seconds"] is not None:
        raise ValueError("unfinished execution cannot claim a completed elapsed time")
    return clock


def finish_execution_clock(path, binding):
    clock = checked_execution_clock(path, binding)
    if clock["finished_utc"] is None:
        stop = datetime.now(timezone.utc)
        start = datetime.fromisoformat(clock["started_utc"])
        if stop < start:
            raise ValueError("system time moved before the recorded execution start")
        clock.update(finished_utc=stop.isoformat(), elapsed_wall_seconds=(stop-start).total_seconds())
        write_json(path, clock)
    return clock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("configs/evaluation_plan.json"))
    parser.add_argument("--method", type=Path, default=Path("configs/method_frozen.json"))
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    published_bytes(args.plan)
    method = published_method(args.method, full_evidence=True)
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", "configs", "scripts"], text=True).strip():
        raise ValueError("commit the evaluation sources before executing the grid")
    subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run([sys.executable, "scripts/select_method.py", "--recorded-only"], check=True)
    # Reconstructing the complete pilot comparison must leave the published
    # winner and all of its bindings byte-identical before evaluation starts.
    published_method(args.method, full_evidence=True)
    base = load_config(plan["config"])
    if base["dataset"] != plan["dataset"]:
        raise ValueError("comparison dataset differs from its training configuration")
    if base["train_limit"] or base["test_limit"] or not base["save_updates"]:
        raise ValueError("the evaluation grid requires complete datasets and retained update matrices")
    rules = plan["rules"] + list(ABLATIONS)
    completed = []
    planned = list(evaluation_cells(plan))
    source_paths = sorted(Path("src/fltoken").glob("*.py")) + [Path("scripts/run_experiment.py"), Path("scripts/evaluation.py"), Path(plan["config"])]
    sources = {path.as_posix(): sha256(path) for path in source_paths}
    for path in source_paths:
        published_bytes(path)
    binding = dict(plan_sha256=sha256(args.plan), method_sha256=sha256(args.method), training_source_sha256=sources,
                   **verify_dataset(plan, base))
    output = comparison_output(plan)
    clock_path = output / "execution.json"
    if not clock_path.exists():
        if args.no_run or any((Path(cell["path"])/"metadata.json").exists() for cell in planned):
            raise ValueError("retain the original grid execution clock when resuming or validating runs")
        write_json(clock_path, dict(**binding, started_utc=datetime.now(timezone.utc).isoformat(),
                                    finished_utc=None, elapsed_wall_seconds=None))
    clock = checked_execution_clock(clock_path, binding)
    common_versions = None
    def record_manifest():
        if (args.no_run or clock["finished_utc"] is not None) and len(completed) != len(planned):
            return
        write_json(output / "manifest.json", dict(plan_sha256=binding["plan_sha256"],
                   method_sha256=binding["method_sha256"], expected_cells=len(planned), completed_cells=len(completed),
                   wall_seconds=sum(item["wall_seconds"] for item in completed),
                   execution_sha256=sha256(clock_path), elapsed_wall_seconds=clock["elapsed_wall_seconds"], cells=completed))
    for cell in planned:
        if sources != {path.as_posix(): sha256(path) for path in source_paths}:
            raise ValueError("training sources changed during evaluation")
        folder = Path(cell["path"])
        existing = read(folder / "metadata.json") if (folder / "metadata.json").exists() else {}
        if existing.get("status") != "complete":
            if args.no_run:
                raise ValueError(f"missing complete evaluation cell: {folder}")
            if clock["finished_utc"] is not None:
                raise ValueError("a completed execution is missing run artifacts; restore its original evidence")
            command = [sys.executable, "scripts/run_experiment.py", "--config", plan["config"],
                       "--method-config", str(args.method), "--rules", *rules, "--output", str(folder)]
            for key in ["seed", "attack", "fraction", "rounds"]:
                command.extend(["--" + key, str(cell[key])])
            for key in ["device", "workers", "gpu_fraction"]:
                command.extend(["--" + key.replace("_", "-"), str(plan[key])])
            subprocess.run(command, check=True)
            existing = read(folder / "metadata.json")
        expected = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction", "rounds"]},
                    **{key: plan[key] for key in ["device", "workers", "gpu_fraction"]},
                    "rules": rules, "ours_params": method["params"]}
        if existing["config"] != expected or existing["source_git_dirty"] or existing["source_sha256"] != sources:
            raise ValueError(f"evaluation cell has different scientific inputs: {folder}")
        if existing.get("method_sha256") != sha256(args.method):
            raise ValueError("evaluation cell does not bind the published method")
        if common_versions is not None and existing["versions"] != common_versions:
            raise ValueError("evaluation runs must use identical library versions")
        common_versions = existing["versions"]
        rebuild(folder)
        if sum(1 for _ in load_update_rounds(folder)) != cell["rounds"]:
            raise ValueError("every evaluation round must retain its verified common update matrix")
        completed.append(dict(**cell, source_commit=existing["git_commit"],
                              wall_seconds=existing["wall_seconds"],
                              per_round_sha256=sha256(folder / "per_round.csv"),
                              metadata_sha256=sha256(folder / "metadata.json")))
        record_manifest()
        print(f"Validated evaluation cells: {len(completed)}/{len(planned)}", flush=True)
    if args.no_run and clock["finished_utc"] is None:
        raise ValueError("execution completion was not recorded; resume the runner before claiming total wall time")
    clock = finish_execution_clock(clock_path, binding)
    record_manifest()


if __name__ == "__main__":
    main()
