"""Adaptive-attacker study: ALIE and a public-information mimic on the unchanged protocol.

Without arguments the driver runs every declared cell serially in fresh
processes, skipping cells that already completed. ``--cell N`` runs one cell.
``--smoke`` runs two CPU rounds on a 3000-example subset into out/ for a quick
installation check; smoke outputs are never analyzed.

The server, client training, aggregation and scoring are the published
``fltoken.experiment`` code. The two attacks replace only the reported update
rows of the sampled attackers, through hooks on the round's client sample and
on the stacking of the reported updates, so the honest path is byte-identical.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

PROCESS_STARTED = time.perf_counter()
os.environ["FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION"] = "1"
os.environ["FLWR_TELEMETRY_ENABLED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONUTF8"] = "1"

import numpy as np

PLAN = Path("configs/adaptive_study.json")
ROOT = Path("results/adaptive_study")
ATTACKS = ("alie", "mimic")


def read_plan():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (plan["schema"] != 1 or set(plan["attacks"]) != set(ATTACKS) or plan["rules"] != ["B0", "B1", "B2", "Mstar", "CGSV", "RFFL"]
            or plan["dataset"] != "mnist" or not plan["seeds"] or not plan["fractions"]):
        raise ValueError("adaptive study plan differs from the declared comparison")
    return plan


def study_cells(plan):
    cells = []
    for seed in plan["seeds"]:
        for attack in ATTACKS:
            for fraction in plan["fractions"]:
                name = f"{plan['dataset']}-{attack}-f{round(fraction*100):02d}-s{seed}-{plan['device']}-t{plan['rounds']}"
                cells.append(dict(index=len(cells), seed=seed, attack=attack, fraction=fraction, path=(ROOT/name).as_posix()))
    return cells


def alie_z(sampled: int, attackers: int) -> float:
    """Baruch et al.: s = floor(n/2 + 1) - m and z = Phi^{-1}((n - s)/n), with s kept at least one."""
    if not 0 < attackers < sampled:
        raise ValueError("ALIE needs at least one attacker and one honest client in the round")
    supporters = max(math.floor(sampled/2 + 1) - attackers, 1)
    return NormalDist().inv_cdf((sampled - supporters)/sampled)


def alie_update(honest: np.ndarray, z: float) -> np.ndarray:
    """Coordinate-wise mean minus z standard deviations of the round's honest updates."""
    return honest.mean(axis=0) - z*honest.std(axis=0)


class Hooks:
    """Replace sampled attackers' reported rows; everything else is delegated to NumPy."""

    def __init__(self, attack_map: dict[int, str], sample_clients):
        self.attack_map = attack_map
        self.sample_clients = sample_clients
        self.ids = None
        self.previous_aggregate = None
        self.log = []

    def sample(self, *args, **kwargs):
        self.ids = self.sample_clients(*args, **kwargs)
        return self.ids

    def replace(self, rows: np.ndarray) -> np.ndarray:
        if self.ids is None or len(self.ids) != len(rows):
            raise RuntimeError("reported updates do not match the sampled clients")
        attacked = np.array([self.attack_map.get(int(i)) is not None for i in self.ids])
        if not attacked.any():
            self.log.append(dict(attackers=0))
            return rows
        kinds = {self.attack_map[int(i)] for i in np.asarray(self.ids)[attacked]}
        if len(kinds) != 1:
            raise RuntimeError("one adaptive attack per run")
        kind = kinds.pop()
        rows = np.array(rows, dtype=np.float64, copy=True)
        if kind == "alie":
            z = alie_z(len(rows), int(attacked.sum()))
            rows[attacked] = alie_update(rows[~attacked], z)
            self.log.append(dict(attackers=int(attacked.sum()), z=z))
        elif kind == "mimic":
            rows[attacked] = 0.0 if self.previous_aggregate is None else self.previous_aggregate
            self.log.append(dict(attackers=int(attacked.sum()), previous_aggregate=self.previous_aggregate is not None))
        else:
            raise RuntimeError(f"unsupported adaptive attack {kind}")
        return rows

    def numpy(self):
        hooks = self

        class NumPy:
            def __getattr__(self, name):
                return getattr(np, name)

            @staticmethod
            def stack(arrays, *args, **kwargs):
                return hooks.replace(np.stack(arrays, *args, **kwargs))

            @staticmethod
            def average(*args, **kwargs):
                result = np.average(*args, **kwargs)
                if "weights" in kwargs:
                    hooks.previous_aggregate = np.array(result, dtype=np.float64, copy=True)
                return result

        return NumPy()


def install(experiment, attack_map):
    """Attach hooks to the published server module for one run."""
    hooks = Hooks(attack_map, experiment.sample_clients)
    experiment.sample_clients = hooks.sample
    experiment.np = hooks.numpy()
    return hooks


def cell_config(plan, cell, smoke=False):
    from evaluation import recorded_method
    from fltoken.config import load_config, validate_config
    base = load_config(plan["config"])
    method = recorded_method()
    config = {**base, "rounds": 2 if smoke else plan["rounds"], "device": "cpu" if smoke else plan["device"],
              "workers": plan["workers"], "gpu_fraction": plan["gpu_fraction"], "rules": plan["rules"],
              "dataset": plan["dataset"], "seed": cell["seed"], "fraction": cell["fraction"],
              "ours_params": method["params"], "save_updates": False}
    if smoke:
        config.update(train_limit=3000, test_limit=1000)
    # The published validator knows only the original attacks; validate the shared fields, then name the attack.
    config = validate_config({**config, "attack": "sign_flip"})
    config["attack"] = cell["attack"]
    return config, method


def run_cell(index, smoke=False):
    import torch
    from flwr.simulation import run_simulation
    from compare_method import historical_scorer
    from fltoken import experiment
    from fltoken.io import sha256, write_json
    from fltoken.metrics import METRICS_VERSION, SETTLEMENT_VERSION
    from fltoken.runtime import host_information, ray_initialization
    from select_extensions import runtime_extensions
    from study_protocol import study_rules
    from functools import partial

    plan = read_plan()
    cell = study_cells(plan)[index]
    config, method = cell_config(plan, cell, smoke)
    output = Path("out/adaptive-smoke")/Path(cell["path"]).name if smoke else Path(cell["path"])
    if not smoke and (output/"metadata.json").exists():
        raise ValueError("preserve an existing cell before any retry")
    output.mkdir(parents=True, exist_ok=True)
    factory, revision = historical_scorer(method)
    extensions = runtime_extensions()
    rule_factory = partial(study_rules, method_factory=factory, extension_parameters=extensions["params"])
    attack_map = experiment.adversaries(config)
    hooks = install(experiment, attack_map)
    runtime_resources = ray_initialization(config["workers"], 128, 512)
    host = host_information()
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    sources = ["scripts/run_adaptive_study.py", "src/fltoken/experiment.py", "scripts/study_protocol.py",
               "scripts/compare_method.py", "scripts/select_extensions.py", str(PLAN), plan["config"],
               "configs/method_frozen.json", "configs/extensions_frozen.json"]
    metadata = dict(config=config.copy(), study_cell=cell, smoke=smoke, plan_sha256=sha256(PLAN),
                    source_sha256={path: sha256(path) for path in sources},
                    effective_method=dict(path="src/fltoken/ours.py", revision=revision, sha256=method["scorer_sha256"]),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--", *sources], text=True).strip()),
                    versions={name: importlib.metadata.version(name) for name in ["flwr", "ray", "torch", "torchvision", "numpy"]},
                    platform=host["platform"], python=platform.python_version(), runtime_resources=runtime_resources,
                    device_name=torch.cuda.get_device_name(0) if config["device"] == "cuda" else host["processor"],
                    started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                    settlement_version=SETTLEMENT_VERSION, metrics_version=METRICS_VERSION)
    write_json(output/"metadata.json", metadata)
    config["data_dir"] = str(Path(config["data_dir"]).resolve())
    started = time.perf_counter()
    try:
        run_simulation(server_app=experiment.make_server(config, output, rule_factory=rule_factory),
                       client_app=experiment.client_app, num_supernodes=config["num_clients"],
                       backend_config={"init_args": runtime_resources,
                                       "client_resources": {"num_cpus": 1, "num_gpus": config["gpu_fraction"] if config["device"] == "cuda" else 0.0}})
        if (output/"failure.json").exists():
            metadata["status"] = "failed_numerical"
        elif not (output/"metrics.json").exists():
            raise RuntimeError("simulation returned without completed metrics")
        else:
            metadata["status"] = "complete"
        if len(hooks.log) != config["rounds"] and metadata["status"] == "complete":
            raise RuntimeError("the attack hook did not see every round")
        write_json(output/"attack_log.json", hooks.log)
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["runtime_seconds"] = time.perf_counter() - started
        metadata["wall_seconds"] = time.perf_counter() - PROCESS_STARTED
        write_json(output/"metadata.json", metadata)
    print(f"{metadata['status']}: {output.name} in {metadata['wall_seconds']:.1f} seconds", flush=True)
    return 20 if metadata["status"] == "failed_numerical" else 0


def drive():
    plan = read_plan()
    cells = study_cells(plan)
    for cell in cells:
        metadata = Path(cell["path"])/"metadata.json"
        if metadata.exists():
            status = json.loads(metadata.read_text(encoding="utf-8"))["status"]
            if status in {"complete", "failed_numerical"}:
                print(f"Already recorded ({status}): {Path(cell['path']).name}", flush=True)
                continue
            raise ValueError(f"interrupted cell {cell['path']}: move it to out/ before resuming")
        print(f"Cell {cell['index']+1}/{len(cells)}: {Path(cell['path']).name}", flush=True)
        process = subprocess.run([sys.executable, __file__, "--cell", str(cell["index"])])
        if process.returncode not in {0, 20}:
            raise SystemExit(f"cell {cell['index']} failed with exit code {process.returncode}")
    print(f"All {len(cells)} adaptive cells recorded", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--cell", type=int)
    parser.add_argument("--smoke", action="store_true", help="two CPU rounds of the first ALIE and mimic cells into out/")
    args = parser.parse_args()
    if args.smoke:
        plan = read_plan()
        cells = study_cells(plan)
        if args.cell is not None:
            return run_cell(args.cell, smoke=True)
        for index in [next(c["index"] for c in cells if c["attack"] == attack and c["fraction"] == max(plan["fractions"]))
                      for attack in ATTACKS]:
            process = subprocess.run([sys.executable, __file__, "--smoke", "--cell", str(index)])
            if process.returncode:
                return process.returncode
        return 0
    if args.cell is not None:
        return run_cell(args.cell)
    drive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
