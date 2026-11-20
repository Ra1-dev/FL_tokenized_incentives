"""Run the complete published evaluation plan with independent method states."""

import argparse
from pathlib import Path
import subprocess
import sys

from evaluation import ABLATIONS, evaluation_cells, published_bytes, published_method, read
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from rebuild_ledgers import rebuild


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
    subprocess.run([sys.executable, "scripts/select_method.py", "--recorded-only"], check=True)
    # Reconstructing the complete pilot comparison must leave the published
    # winner and all of its bindings byte-identical before evaluation starts.
    published_method(args.method, full_evidence=True)
    base = load_config(plan["config"])
    if base["train_limit"] or base["test_limit"]:
        raise ValueError("the evaluation grid requires the complete datasets")
    rules = plan["rules"] + list(ABLATIONS)
    completed = []
    planned = list(evaluation_cells(plan))
    source_paths = sorted(Path("src/fltoken").glob("*.py")) + [Path("scripts/run_experiment.py"), Path("scripts/evaluation.py"), Path(plan["config"])]
    sources = {path.as_posix(): sha256(path) for path in source_paths}
    for cell in planned:
        if sources != {path.as_posix(): sha256(path) for path in source_paths}:
            raise ValueError("training sources changed during evaluation")
        folder = Path(cell["path"])
        existing = read(folder / "metadata.json") if (folder / "metadata.json").exists() else {}
        if existing.get("status") != "complete":
            if args.no_run:
                raise ValueError(f"missing complete evaluation cell: {folder}")
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
        rebuild(folder)
        completed.append(dict(**cell, source_commit=existing["git_commit"],
                              wall_seconds=existing["wall_seconds"],
                              per_round_sha256=sha256(folder / "per_round.csv"),
                              metadata_sha256=sha256(folder / "metadata.json")))
        write_json("results/grid/manifest.json", dict(plan_sha256=sha256(args.plan),
                   method_sha256=sha256(args.method), expected_cells=len(planned), completed_cells=len(completed),
                   wall_seconds=sum(item["wall_seconds"] for item in completed), cells=completed))
        print(f"Validated evaluation cells: {len(completed)}/{len(planned)}", flush=True)


if __name__ == "__main__":
    main()
