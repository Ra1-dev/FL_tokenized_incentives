"""Run a recorded Flower simulation; no dataset download occurs here."""

from __future__ import annotations

import argparse
import json
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
from fltoken.metrics import METRICS_VERSION, SETTLEMENT_VERSION
from fltoken.runtime import host_information, ray_initialization
from evaluation import ABLATIONS, published_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sendros_mix.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output")
    parser.add_argument("--attack")
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--rules", nargs="+")
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--privacy-config", type=Path)
    parser.add_argument("--noise-multiplier", type=float)
    parser.add_argument("--object-store-mb", type=int, default=128)
    parser.add_argument("--task-memory-mb", type=int, default=512)
    parser.add_argument("--gpu-fraction", type=float, help="Ray resource reservation per client")
    parser.add_argument("--study-cell", type=int, help="cell of the published mixture-noise and transfer plan")
    args = parser.parse_args()
    study, rule_factory = None, None
    if args.study_cell is not None:
        if any(getattr(args, key) is not None for key in ['seed', 'device', 'attack', 'fraction', 'workers',
                'gpu_fraction', 'rounds', 'rules', 'method_config', 'privacy_config', 'noise_multiplier', 'output']):
            raise ValueError('study cells cannot override their published configuration or output')
        from study_protocol import prepare_cell
        config, study, rule_factory = prepare_cell(args.study_cell)
        args.config, args.output = study['plan']['config'], study['cell']['path']
        args.method_config = Path('configs/method_frozen.json')
        args.privacy_config = Path('configs/privacy_frozen.json') if 'privacy' in config else None
        if Path(args.output).exists():
            raise ValueError('preserve an existing study cell before any retry')
    else:
        config = load_config(args.config)
        for key in ["seed", "device", "attack", "fraction", "workers", "gpu_fraction", "rounds", "rules"]:
            value = getattr(args, key)
            if value is not None:
                config[key] = value
        if set(config["rules"]) & ({"Mstar"} | ABLATIONS.keys()):
            if args.method_config is None:
                raise ValueError("method scoring requires a published frozen configuration")
            config["ours_params"] = published_method(args.method_config)["params"]
        elif args.method_config is not None:
            raise ValueError("a method configuration was supplied without a method rule")
        if args.privacy_config is not None:
            from privacy_protocol import published_privacy, settings
            if args.noise_multiplier is None:
                raise ValueError("the privacy comparison requires an explicit noise multiplier")
            selected = settings(published_privacy(args.privacy_config), args.noise_multiplier)
            if "privacy" in config and config["privacy"] != selected:
                raise ValueError("configured privacy parameters differ from the published calibration")
            config["privacy"] = selected
        elif args.noise_multiplier is not None or "privacy" in config:
            raise ValueError("privacy experiments require a published frozen calibration")
        config = validate_config(config)
    runtime_resources = ray_initialization(config["workers"], args.object_store_mb, args.task_memory_mb)
    host = host_information()
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    run_id = f"{config['dataset']}-{config['attack']}-s{config['seed']}-{config['device']}-t{config['rounds']}"
    if config["train_limit"]:
        run_id += "-smoke"
    if "privacy" in config:
        run_id += f"-sigma{config['privacy']['noise_multiplier']:g}"
    output = Path(args.output or f"results/runs/{run_id}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ["metrics.json", "ledger.json", "hashes.json", "timing.json", "per_round.csv", "round_metrics.csv", "protocol.json", "privacy.json", "failure.json", "prefix_metrics.json"]:
        (output / name).unlink(missing_ok=True)
    metadata = dict(config=config.copy(), source_config=args.config,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                    versions={name: importlib.metadata.version(name) for name in ["flwr", "ray", "torch", "torchvision", "numpy", "PyYAML"]},
                    platform=host["platform"], python=platform.python_version(), host_information=host,
                    runtime_resources=runtime_resources,
                    device_name=torch.cuda.get_device_name(0) if config["device"] == "cuda" else host["processor"],
                    started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                    settlement_version=SETTLEMENT_VERSION, metrics_version=METRICS_VERSION)
    source_paths = sorted(Path("src/fltoken").glob("*.py")) + [Path(__file__).relative_to(Path.cwd()), Path("scripts/evaluation.py"), Path(args.config)]
    metadata["source_sha256"] = {path.as_posix(): sha256(path) for path in source_paths}
    if study is not None:
        from study_protocol import PLAN, sources
        metadata.update(source_sha256=sources(study['plan']), effective_method=study['effective'],
                        study_plan_sha256=sha256(PLAN), study_cell=study['cell'],
                        extensions_sha256=sha256('configs/extensions_frozen.json'))
    if args.method_config:
        metadata["method_sha256"] = sha256(args.method_config)
    if args.privacy_config:
        metadata["privacy_sha256"] = sha256(args.privacy_config)
    metadata["source_git_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configs", *(['scripts'] if study is not None else ['scripts/run_experiment.py', 'scripts/evaluation.py'])], text=True).strip())
    if study is not None and metadata['source_git_dirty']:
        raise ValueError('commit study sources before training')
    write_json(output / "metadata.json", metadata)
    config["data_dir"] = str(Path(config["data_dir"]).resolve())
    started = time.perf_counter()
    try:
        run_simulation(
            server_app=make_server(config, output, rule_factory=rule_factory), client_app=client_app,
            num_supernodes=config["num_clients"],
            backend_config={
                "init_args": runtime_resources,
                "client_resources": {"num_cpus": 1, "num_gpus": config["gpu_fraction"] if config["device"] == "cuda" else 0.0},
            },
        )
        if (output / "failure.json").exists():
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            metadata.update(status="failed_numerical", completed_rounds=failure["completed_rounds"],
                            failure_sha256=sha256(output / "failure.json"))
        elif not (output / "metrics.json").exists():
            raise RuntimeError("simulation returned without completed metrics")
        else:
            metadata["status"] = "complete"
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["runtime_seconds"] = time.perf_counter() - started
        metadata["wall_seconds"] = time.perf_counter() - PROCESS_STARTED
        write_json(output / "metadata.json", metadata)
    print(f"{metadata['status']}: {output.name} in {metadata['wall_seconds']:.2f} seconds", flush=True)
    return 20 if metadata["status"] == "failed_numerical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
