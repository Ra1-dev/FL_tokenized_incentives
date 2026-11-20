"""Run a recorded Flower simulation; no dataset download occurs here."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROCESS_STARTED = time.perf_counter()

# Set before importing Flower or initializing CUDA in any process.
os.environ["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONUTF8"] = "1"

import torch
from flwr.simulation import run_simulation

from fltoken.config import load_config, validate_config
from fltoken.experiment import client_app, make_server
from fltoken.io import sha256, write_json
from fltoken.metrics import SETTLEMENT_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sendros_mix.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output")
    parser.add_argument("--attack")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpu-fraction", type=float, help="Ray resource reservation per client")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ["seed", "device", "attack", "fraction", "workers", "gpu_fraction"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config = validate_config(config)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run_id = f"{config['dataset']}-{config['attack']}-s{config['seed']}-{config['device']}-t{config['rounds']}"
    if config["train_limit"]:
        run_id += "-smoke"
    output = Path(args.output or f"results/runs/{run_id}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ["metrics.json", "ledger.json", "hashes.json", "timing.json", "per_round.csv", "round_metrics.csv", "protocol.json"]:
        (output / name).unlink(missing_ok=True)
    metadata = dict(config=config.copy(), source_config=args.config,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                    versions={name: importlib.metadata.version(name) for name in ["flwr", "ray", "torch", "torchvision", "numpy", "PyYAML"]},
                    platform=platform.platform(), python=platform.python_version(),
                    device_name=torch.cuda.get_device_name(0) if config["device"] == "cuda" else platform.processor(),
                    started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                    settlement_version=SETTLEMENT_VERSION)
    source_paths = sorted(Path("src/fltoken").glob("*.py")) + [Path(__file__).relative_to(Path.cwd()), Path(args.config)]
    metadata["source_sha256"] = {path.as_posix(): sha256(path) for path in source_paths}
    metadata["source_git_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configs", "scripts/run_experiment.py"], text=True).strip())
    write_json(output / "metadata.json", metadata)
    config["data_dir"] = str(Path(config["data_dir"]).resolve())
    started = time.perf_counter()
    try:
        run_simulation(
            server_app=make_server(config, output), client_app=client_app,
            num_supernodes=config["num_clients"],
            backend_config={
                "init_args": {"num_cpus": config["workers"], "include_dashboard": False,
                              "log_to_driver": True, "ignore_reinit_error": True},
                "client_resources": {"num_cpus": 1, "num_gpus": config["gpu_fraction"] if config["device"] == "cuda" else 0.0},
            },
        )
        if not (output / "metrics.json").exists():
            raise RuntimeError("simulation returned without completed metrics")
        metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["runtime_seconds"] = time.perf_counter() - started
        metadata["wall_seconds"] = time.perf_counter() - PROCESS_STARTED
        write_json(output / "metadata.json", metadata)
    print(f"Completed {output.name} in {metadata['wall_seconds']:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
