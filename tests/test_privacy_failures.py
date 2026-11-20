"""Numerical failures preserve evidence and never select surviving seeds."""

import csv
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from fltoken.config import DEFAULTS
from fltoken.data import sample_clients
from fltoken.io import sha256, write_json


@pytest.fixture
def numerical_run(tmp_path, monkeypatch):
    import fltoken.experiment as experiment
    from flwr.supercore.task_identity import TaskIdentity
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    monkeypatch.chdir(tmp_path)
    for field in ["_run_id", "_task_id", "_node_id"]:
        monkeypatch.setattr(TaskIdentity, field, 1)
    config = {**DEFAULTS, "attack": "none", "fraction": 0., "num_clients": 5,
              "clients_per_round": 3, "rounds": 3, "seed": 1373,
              "rules": ["B0", "B1", "B2", "Mstar", "Aref"], "ours_params": {},
              "privacy": dict(clipping_bound=.7, noise_multiplier=.5, delta=1e-5)}
    images, labels = torch.zeros(100, 1, 28, 28), torch.arange(100) % 10
    monkeypatch.setattr(experiment, "load_tensors", lambda *args: (images, labels))
    monkeypatch.setattr(experiment, "dirichlet_partition", lambda *args: [np.arange(i*20, (i+1)*20) for i in range(5)])
    monkeypatch.setattr(experiment, "LeNet5", lambda: None)
    monkeypatch.setattr(experiment, "flatten", lambda _: np.zeros(7, dtype=np.float64))
    calls = []

    def run(stage, failed_round, reverse=True):
        output = tmp_path/f"{stage}-{failed_round}"
        output.mkdir()
        target = int(sample_clients(5, 3, failed_round, config["seed"])[0])
        def local(initial, images, labels, config, client, number, poisoned):
            calls.append((number, client))
            if number == failed_round and stage == "local_training" and client == target:
                return np.array([np.nan, np.inf, 0., 0., 0., 0., 0.])
            if number == failed_round and stage == "transport":
                raise RuntimeError("transport failure fixture")
            return np.full(7, 1e40) if number == failed_round and stage == "aggregation" else np.arange(1., 8.)*(client+1)
        monkeypatch.setattr(experiment, "local_update", local)
        if stage == "aggregation":
            monkeypatch.setattr(experiment, "release_update", lambda delta, *args: delta)
        evaluations = []
        def evaluate(*args):
            evaluations.append(1)
            return .5, np.inf if stage == "evaluation" and len(evaluations) == failed_round else 1.
        monkeypatch.setattr(experiment, "evaluate", evaluate)
        class LocalTransport:
            def get_node_ids(self):
                return [11, 12, 13, 14, 15]
            def send_and_receive(self, messages, timeout):
                return [experiment.train(message, None) for message in (reversed(messages) if reverse else messages)]
        experiment.make_server(config, output)(LocalTransport(), None)
        failure = importlib.import_module("evaluation").read(output/"failure.json")
        write_json("configs/method_frozen.json", dict(synthetic_fixture=True))
        write_json("configs/privacy_frozen.json", dict(synthetic_fixture=True))
        write_json(output/"metadata.json", dict(status="failed_numerical", config=config,
                   synthetic_fixture=True, source_git_dirty=False, completed_rounds=failure["completed_rounds"],
                   failure_sha256=sha256(output/"failure.json"),
                   method_sha256=sha256("configs/method_frozen.json"), privacy_sha256=sha256("configs/privacy_frozen.json")))
        return output, config, failure, calls
    return run


@pytest.mark.parametrize("stage", ["local_training", "aggregation", "evaluation"])
@pytest.mark.parametrize("failed_round", [1, 2, 3])
def test_numerical_outcome_keeps_only_the_completed_prefix(numerical_run, stage, failed_round):
    with np.errstate(over="ignore"):
        folder, config, failure, calls = numerical_run(stage, failed_round)
    module = importlib.import_module("run_privacy")
    metadata, inputs = module.verify_cell(folder, config)
    assert metadata["status"] == "failed_numerical"
    assert failure["completed_rounds"] == failed_round-1
    assert max(number for number, _ in calls) == failed_round
    assert all(item["stage"] == stage for item in failure["failures"])
    assert not (folder/"metrics.json").exists() and not (folder/"privacy.json").exists()
    with (folder/"per_round.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3*(failed_round-1)
    ledger = module.read(folder/"ledger.json")
    assert all(len(entries) == failed_round-1 for entries in ledger.values())
    assert module.verify_cell(folder, config, verify_cache=False)[1] == inputs
    if stage == "local_training":
        assert failure["failures"][0]["nonfinite_count"] == 2


def test_unrelated_runtime_errors_are_not_numerical_outcomes(numerical_run):
    with pytest.raises(RuntimeError, match="transport failure"):
        numerical_run("transport", 2)
    assert not Path("transport-2/failure.json").exists()


def test_failure_evidence_rejects_rehashed_fabricated_metrics_and_missing_rounds(numerical_run):
    folder, config, _, _ = numerical_run("local_training", 3)
    module = importlib.import_module("run_privacy")
    assert module.verify_cell(folder, config)
    write_json(folder/"metrics.json", {})
    with pytest.raises(ValueError, match="strict prefix"):
        module.verify_cell(folder, config)
    (folder/"metrics.json").unlink()
    path = folder/"prefix_metrics.json"
    original = path.read_bytes()
    metrics = module.read(path)
    metrics["B1"]["honest_false_penalty_rate"] = .123456
    write_json(path, metrics)
    hashes = module.read(folder/"hashes.json")
    write_json(folder/"hashes.json", {**hashes, "prefix_metrics.json": sha256(path)})
    with pytest.raises(ValueError, match="do not reconstruct"):
        module.verify_cell(folder, config)
    path.write_bytes(original)
    write_json(folder/"hashes.json", hashes)
    failure = module.read(folder/"failure.json")
    write_json(folder/"failure.json", {**failure, "failed_round": 2})
    with pytest.raises(ValueError, match="strict prefix"):
        module.verify_cell(folder, config)


def test_missing_seed_never_becomes_a_survivor_only_interval(numerical_run):
    folder, config, failure, _ = numerical_run("local_training", 2)
    module = importlib.import_module("summarize_privacy")
    plan = module.read(Path(__file__).resolve().parents[1]/"configs/privacy_plan.json")
    plan.update(seeds=[1373, 1381], noise_multipliers=[.5], rounds=3, rules=config["rules"])
    planned = list(module.privacy_cells(plan))
    prefix = module.read(folder/"prefix_metrics.json")
    account = module.planned_accounting(config)
    records = [dict(**planned[0], status="complete", metrics=prefix, accounting=account),
               dict(**planned[1], status="failed_numerical", metrics=None, accounting=account,
                    failure=failure, prefix_metrics=prefix)]
    result = module.summarize_records(records, plan, config["privacy"])
    assert result["runs"] == 2 and result["completed_runs"] == result["numerical_failures"] == 1
    row = result["cells"][0]
    for rule in plan["rules"]:
        for value in row["metrics"][rule].values():
            assert value["estimate"] is None and value["undefined"] >= 1
    assert row["paired_false_penalty_difference_from_B1"]["Mstar"]["estimate"] is None
    assert row["outcomes"][1]["completed_rounds"] == 1
    assert "not covered" in result["accounting_scope"]
    with pytest.raises(ValueError, match="every unique declared cell"):
        module.summarize_records(records[:1], plan, config["privacy"])
    records[1]["metrics"] = prefix
    with pytest.raises(ValueError, match="cannot supply full-horizon"):
        module.summarize_records(records, plan, config["privacy"])


def test_runner_retains_failed_seed_on_resume_and_analysis_rejects_false_counts(numerical_run, monkeypatch):
    import shutil
    import sys
    folder, config, _, _ = numerical_run("local_training", 1)
    runner = importlib.import_module("run_privacy")
    analysis = importlib.import_module("summarize_privacy")
    plan = runner.read(Path(__file__).resolve().parents[1]/"configs/privacy_plan.json")
    plan.update(config="configs/test.yaml", seeds=[config["seed"]], noise_multipliers=[.5, 1.],
                **{key: config[key] for key in ["dataset", "attack", "fraction", "rounds", "device", "workers", "gpu_fraction", "rules"]})
    write_json(plan["config"], {key: value for key, value in config.items() if key not in ["privacy", "ours_params"]})
    frozen = dict(clipping_bound=.7, delta=1e-5, noise_multipliers=[.5, 1.])
    write_json(runner.PLAN, plan)
    write_json(runner.FROZEN, frozen)
    planned = list(runner.privacy_cells(plan))
    target = Path(planned[0]["path"])
    shutil.copytree(folder, target)
    metadata = runner.read(target/"metadata.json")
    metadata.update(privacy_sha256=sha256(runner.FROZEN), source_config=plan["config"],
                    source_sha256={"synthetic": "bound"}, versions={"synthetic": True}, git_commit="a"*40, wall_seconds=1.)
    write_json(target/"metadata.json", metadata)
    binding = dict(plan_sha256=sha256(runner.PLAN), frozen_sha256=sha256(runner.FROZEN),
                   method_sha256=sha256("configs/method_frozen.json"), training_source_sha256={"synthetic": "bound"})
    write_json("results/privacy/execution.json", dict(**binding, started_utc="2026-01-01T00:00:00+00:00",
               finished_utc=None, elapsed_wall_seconds=None))
    for module in [runner, analysis]:
        monkeypatch.setattr(module, "published_bytes", lambda path: Path(path).read_bytes())
        monkeypatch.setattr(module, "published_privacy" if module is runner else "recorded_privacy", lambda **kwargs: frozen)
        monkeypatch.setattr(module, "published_method" if module is runner else "recorded_method", lambda **kwargs: dict(params={}))
        monkeypatch.setattr(module, "verify_training_source", lambda value, **kwargs: None)
    monkeypatch.setattr(runner, "calibration", lambda: None)
    monkeypatch.setattr(runner, "training_sources", lambda plan: {"synthetic": "bound"})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "")
    launched = []
    def simulated_numerical_exit(command):
        destination = Path(command[command.index("--output")+1])
        assert destination == Path(planned[1]["path"]), "an existing numerical outcome must not be silently retried"
        assert not destination.exists()
        launched.append(destination)
        shutil.copytree(folder, destination)
        expected = runner.expected_config(plan, planned[1], frozen, dict(params={}))
        write_json(destination/"metadata.json", {**metadata, "config": expected})
        return runner.subprocess.CompletedProcess(command, 20)
    monkeypatch.setattr(runner.subprocess, "run", simulated_numerical_exit)
    monkeypatch.setattr(sys, "argv", ["run_privacy.py"])
    runner.main()
    path = Path("results/privacy/manifest.json")
    manifest = runner.read(path)
    assert manifest["complete"] and manifest["attempted_cells"] == manifest["numerical_failures"] == 2
    assert manifest["completed_cells"] == manifest["successful_run_seconds"] == 0
    assert manifest["attempted_run_seconds"] == 2 and len(launched) == 1
    before = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["run_privacy.py", "--no-run"])
    runner.main()
    assert path.read_bytes() == before and len(launched) == 1
    write_json("results/summary.json", dict(synthetic_fixture=True))
    analysis.main()
    result = runner.read("results/privacy/summary.json")
    assert result["numerical_failures"] == 2 and result["completed_runs"] == 0
    assert result["cells"][0]["outcomes"][0]["completed_rounds"] == 0
    write_json(path, {**manifest, "completed_cells": 1, "numerical_failures": 0})
    with pytest.raises(ValueError, match="completion and failure counts"):
        analysis.main()
