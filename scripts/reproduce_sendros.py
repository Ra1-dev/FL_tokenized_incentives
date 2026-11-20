"""Reproduce the source comparison, retaining all noise participations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fltoken.config import load_config
from fltoken.io import write_json
from reproduction_analysis import collect_run, exact_quantile, read_json, summarize_runs, write_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--no-run", action="store_true", help="fail if any complete full run is missing")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--grid", action="store_true", help="analyze the published evaluation mixture corpus without training")
    mode.add_argument("--legacy", action="store_true", help="explicitly regenerate the original standalone reproduction")
    args = parser.parse_args()
    source = read_json("results/literature/sendros.json")
    existing = read_json("results/summary.json") if Path("results/summary.json").exists() else {}
    use_grid = args.grid or (not args.legacy and existing.get("reproduction", {}).get("basis") == "evaluation_grid")
    if use_grid:
        from grid_reproduction import refresh_grid_reproduction
        refresh_grid_reproduction(args.device)
        return
    device = args.device or "cpu"
    runs = []
    for seed in source["setup"]["seeds"]:
        folder = Path(f"results/runs/mnist-sendros_mix-s{seed}-{device}-t20")
        complete = (folder / "metadata.json").exists() and read_json(folder / "metadata.json")["status"] == "complete"
        if not complete:
            if args.no_run:
                raise RuntimeError(f"full run missing: {folder}")
            subprocess.run([sys.executable, "scripts/run_experiment.py", "--config", "configs/sendros_mix.yaml", "--seed", str(seed), "--device", device], check=True)
        metadata = read_json(folder / "metadata.json")
        config = metadata["config"]
        expected_config = load_config("configs/sendros_mix.yaml")
        expected_config.update(seed=seed, device=device)
        execution_keys = {"workers", "gpu_fraction", "data_dir", "save_updates"}
        for key, expected in expected_config.items():
            if key not in execution_keys and {"stale_mode": "train_snapshot", **config}.get(key) != expected:
                raise ValueError(f"{folder}: {key} differs from full reproduction protocol")
        runs.append(collect_run(folder, seed))
    result = summarize_runs(runs, source, device)
    out = Path("results/reproduction")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "noise_participations.csv", result["noise_observations"])
    write_csv(out / "quality_trajectory.csv", result["quality_trajectory"])
    write_csv(out / "per_seed.csv", result["per_seed"])
    write_json(out / "summary.json", result)
    summary_path = Path("results/summary.json")
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary["reproduction"] = result
    summary.pop("reproduction_overall", None)
    for name in ["summary.json", "noise_participations.csv", "quality_trajectory.csv", "per_seed.csv"]:
        (out/f"overall_{name}").unlink(missing_ok=True)
    write_json(summary_path, summary)
    print(json.dumps({k: v for k, v in result.items() if k not in {"quality_trajectory", "noise_observations"}}, indent=2))
    if not result["median_error_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
