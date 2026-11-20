"""A dataset transfer retains the frozen method, complete seeds and shared updates."""

import copy
import csv
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from fltoken.config import load_config
from fltoken.data import sample_clients
from fltoken.io import sha256, write_json
from fltoken.metrics import payout, ROUND_POOL, summarize
from fltoken.rules import make_rules, score_round


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    return importlib.import_module("summarize_grid"), importlib.import_module("run_grid")


def test_transfer_plan_has_six_unique_cells_and_leaves_main_grid_unchanged(modules):
    analysis, _ = modules
    plan = analysis.read("configs/fashion_plan.json")
    cells = list(analysis.evaluation_cells(plan))
    assert len(cells) == 6 and len({cell["path"] for cell in cells}) == 6
    assert {cell["seed"] for cell in cells} == {42, 43, 44}
    assert {(cell["attack"], cell["fraction"]) for cell in cells} == {("sendros_mix", 1/6), ("label_flip", .3)}
    assert all(cell["horizon"] == "main" and cell["rounds"] == 20 and cell["path"].startswith("results/fashion/") for cell in cells)
    assert load_config(plan["config"])["dataset"] == "fashion_mnist"
    main = list(analysis.evaluation_cells(analysis.read("configs/evaluation_plan.json")))
    assert len(main) == 90 and sum(cell["horizon"] == "long" for cell in main) == 5
    assert all(cell["path"].startswith("results/grid/") for cell in main)
    with pytest.raises(ValueError, match="output"):
        list(analysis.evaluation_cells({**plan, "output_root": "../results/fashion"}))


def test_prepared_dataset_checks_complete_hashes_and_normalization(modules, tmp_path, monkeypatch):
    _, runner = modules
    monkeypatch.chdir(tmp_path)
    names = [f"{prefix}-{kind}-idx{dimension}-ubyte{suffix}"
             for prefix in ["train", "t10k"] for kind, dimension in [("images", 3), ("labels", 1)]
             for suffix in ["", ".gz"]]
    raw = Path("data/FashionMNIST/raw")
    raw.mkdir(parents=True)
    for name in names:
        (raw/name).write_text("synthetic input: "+name, encoding="utf-8")
    record = dict(dataset="fashion_mnist", normalization=dict(mean=.286, std=.353),
                  sha256={name:sha256(raw/name) for name in names})
    write_json("results/datasets/fashion_mnist.json", record)
    plan = dict(dataset_manifest="results/datasets/fashion_mnist.json")
    base = dict(dataset="fashion_mnist", data_dir="data")
    monkeypatch.setattr(runner, "published_bytes", lambda path: Path(path).read_bytes())
    assert runner.verify_dataset(plan, base) == dict(dataset_sha256=sha256(plan["dataset_manifest"]))
    (raw/names[0]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        runner.verify_dataset(plan, base)
    record["normalization"]["mean"] = .1307
    write_json(plan["dataset_manifest"], record)
    with pytest.raises(ValueError, match="normalization"):
        runner.verify_dataset(plan, base)
    record["normalization"]["mean"] = .286
    del record["sha256"][names[-1]]
    write_json(plan["dataset_manifest"], record)
    with pytest.raises(ValueError, match="coverage"):
        runner.verify_dataset(plan, base)


@pytest.fixture
def corpus(modules, tmp_path, monkeypatch):
    analysis, runner = modules
    root = Path(__file__).resolve().parents[1]
    for name in ["configs/fashion_plan.json", "configs/fashion_mnist.yaml", "configs/sendros_mix.yaml",
                 "results/datasets/fashion_mnist.json", "scripts/summarize_grid.py", "src/fltoken/statistics.py"]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, path)
    monkeypatch.chdir(tmp_path)
    plan = analysis.read("configs/fashion_plan.json")
    method = dict(synthetic_fixture=True, params=dict(reference=True, ema=True, penalties=True,
                  beta=.5, threshold=.05, patience=2, initial_reputation=.5))
    write_json("configs/method_frozen.json", method)
    # These are small-dimensional artificial update traces, not trained results.
    records = []
    for cell in analysis.evaluation_cells(plan):
        folder = Path(cell["path"])
        folder.mkdir(parents=True)
        config = {**load_config(plan["config"]),
                  **{key:cell[key] for key in ["seed", "attack", "fraction", "rounds"]},
                  **{key:plan[key] for key in ["device", "workers", "gpu_fraction"]},
                  "rules":plan["rules"]+list(analysis.ABLATIONS), "ours_params":method["params"]}
        rules = make_rules({**config, "krum_f": 2 if cell["attack"] == "sendros_mix" else 3})
        rows, history = [], []
        ledger = {rule:[] for rule in rules}
        adversaries = {client:cell["attack"] for client in range(5 if cell["attack"] == "sendros_mix" else 9)}
        rng = np.random.default_rng(cell["seed"])
        for number in range(1, 21):
            ids = sample_clients(30, 10, number, cell["seed"])
            updates = rng.normal(1., .1, size=(10, 13))
            updates[[client in adversaries for client in ids]] = rng.normal(0., 1., size=(sum(client in adversaries for client in ids), 13))
            outputs = score_round(rules, ids, updates, number)
            common = dict(round=number, accuracy=.4+number/100+(cell["seed"]-42)/1000-(number/250 if cell["attack"] == "label_flip" else 0),
                          loss=2.-number/20, update_set_sha256=sha256_bytes(updates.tobytes()))
            history.append(common)
            for index, client in enumerate(ids):
                row = dict(**common, client_id=int(client), adversary=int(client) in adversaries,
                           attack=adversaries.get(int(client), "none"))
                for rule, values in outputs.items():
                    row[f"{rule}_share"] = float(values["shares"][index])
                    row[f"{rule}_slash"] = bool(values["slash"][index])
                rows.append(row)
            for rule, values in outputs.items():
                units, amounts = payout(values["shares"])
                ledger[rule].append(dict(round=number, ids=ids.tolist(), shares1e18=units, payouts=amounts,
                                         slash=values["slash"].tolist(), remainder=ROUND_POOL-sum(amounts)))
        for name, values in [("per_round.csv", rows), ("round_metrics.csv", history)]:
            with (folder/name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(values)
        write_json(folder/"ledger.json", ledger)
        write_json(folder/"metrics.json", summarize(rows, ledger, config, adversaries))
        write_json(folder/"protocol.json", dict(adversaries=adversaries))
        write_json(folder/"hashes.json", {name:sha256(folder/name) for name in ["per_round.csv", "round_metrics.csv", "ledger.json"]})
        write_json(folder/"metadata.json", dict(status="complete", synthetic_fixture=True, config=config,
                   source_config=plan["config"], source_sha256={}, source_git_dirty=False, versions={"synthetic":True},
                   git_commit="a"*40, wall_seconds=1., method_sha256=sha256("configs/method_frozen.json")))
        records.append(dict(**cell, source_commit="a"*40, wall_seconds=1.,
                            metadata_sha256=sha256(folder/"metadata.json"), per_round_sha256=sha256(folder/"per_round.csv")))
    binding = dict(plan_sha256=sha256("configs/fashion_plan.json"), method_sha256=sha256("configs/method_frozen.json"))
    write_json("results/fashion/execution.json", dict(**binding, training_source_sha256={},
               dataset_sha256=sha256(plan["dataset_manifest"]), started_utc="2026-01-01T00:00:00+00:00",
               finished_utc="2026-01-01T00:01:00+00:00", elapsed_wall_seconds=60.))
    write_json("results/fashion/manifest.json", dict(**binding, expected_cells=6, completed_cells=6, cells=records,
               wall_seconds=6., elapsed_wall_seconds=60., execution_sha256=sha256("results/fashion/execution.json")))
    write_json("results/summary.json", dict(synthetic_fixture=True))
    checked = []
    monkeypatch.setattr(analysis, "recorded_method", lambda **kwargs: method)
    monkeypatch.setattr(analysis, "published_bytes", lambda path: Path(path).read_bytes())
    monkeypatch.setattr(analysis, "verify_training_source", lambda metadata, method: checked.append(metadata["git_commit"]))
    monkeypatch.setattr(sys, "argv", ["summarize_grid.py", "--plan", "configs/fashion_plan.json"])
    return analysis, runner, records, checked


def sha256_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def test_transfer_summary_rebuilds_all_seeds_and_is_byte_stable(corpus):
    analysis, _, records, checked = corpus
    analysis.main()
    result = analysis.read("results/summary.json")["fashion"]
    assert result["nominal_cells"] == 6 and result["seed_count"] == 3
    assert result["distinct_main_scenarios"] == 2 and result["device"] == "cuda"
    assert len(result["accuracy_trajectories"]) == 40 and checked == ["a"*40]
    assert all(row["accuracy"]["estimate"]["observations"] == 3 for row in result["accuracy_trajectories"])
    assert all(row["condition_seed_test"]["pairs"] == 6 and row["seed_test"]["pairs"] == 3 for row in result["contrasts"])
    first = Path("results/summary.json").read_bytes()
    analysis.main()
    assert Path("results/summary.json").read_bytes() == first
    attacker = result["intervals"][0]["metrics"]
    assert len({attacker[rule]["attacker_share"]["estimate"]["mean"] for rule in ["B0", "B1", "B2", "Mstar"]}) > 2
    manifest = analysis.read("results/fashion/manifest.json")
    manifest["cells"].pop()
    write_json("results/fashion/manifest.json", manifest)
    with pytest.raises(ValueError, match="coverage"):
        analysis.main()


@pytest.mark.parametrize("field,value", [("source_git_dirty", True), ("versions", {"different":True}),
                                         ("source_sha256", {"changed":"source"}), ("method_sha256", "0"*64)])
def test_rehashed_metadata_cannot_hide_mixed_scientific_inputs(corpus, field, value):
    analysis, _, records, _ = corpus
    manifest = analysis.read("results/fashion/manifest.json")
    cell = manifest["cells"][-1]
    path = Path(cell["path"])/"metadata.json"
    metadata = analysis.read(path)
    metadata[field] = value
    write_json(path, metadata)
    cell["metadata_sha256"] = sha256(path)
    write_json("results/fashion/manifest.json", manifest)
    with pytest.raises(ValueError, match="provenance|versions"):
        analysis.main()


def test_shadow_accuracy_cannot_be_claimed_as_a_rule_specific_gain(modules):
    analysis, _ = modules
    plan = analysis.read("configs/fashion_plan.json")
    records = []
    for cell in analysis.evaluation_cells(plan):
        metrics = {rule:dict.fromkeys(analysis.METRICS, .5) for rule in plan["rules"]+list(analysis.ABLATIONS)}
        records.append(dict(**cell, metrics=metrics, per_round_sha256="synthetic"))
    records[0]["metrics"]["Mstar"]["final_accuracy"] = .6
    records[0]["metrics"]["Afull"]["final_accuracy"] = .6
    with pytest.raises(ValueError, match="same final model"):
        analysis.summarize_records(records, plan)


def test_transfer_runner_revalidates_every_cell_without_training(corpus, monkeypatch):
    analysis, runner, records, _ = corpus
    root = Path(__file__).resolve().parents[1]
    for name in ["scripts/run_experiment.py", "scripts/evaluation.py"]:
        shutil.copyfile(root/name, name)
    sources = {path.as_posix():sha256(path) for path in
               [*sorted(Path("src/fltoken").glob("*.py")), Path("scripts/run_experiment.py"),
                Path("scripts/evaluation.py"), Path("configs/fashion_mnist.yaml")]}
    clock_path = Path("results/fashion/execution.json")
    clock = analysis.read(clock_path)
    clock["training_source_sha256"] = sources
    write_json(clock_path, clock)
    for cell in records:
        path = Path(cell["path"])/"metadata.json"
        metadata = analysis.read(path)
        metadata["source_sha256"] = sources
        write_json(path, metadata)
    method = analysis.read("configs/method_frozen.json")
    monkeypatch.setattr(runner, "published_method", lambda *args, **kwargs:method)
    published = []
    def publication(path):
        published.append(Path(path).as_posix())
        return Path(path).read_bytes()
    monkeypatch.setattr(runner, "published_bytes", publication)
    monkeypatch.setattr(runner, "verify_dataset", lambda plan, base:dict(dataset_sha256=sha256(plan["dataset_manifest"])))
    commands = []
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:commands.append(command))
    matrices = []
    def retained(folder):
        matrices.append(folder.as_posix())
        return iter(range(20))
    monkeypatch.setattr(runner, "load_update_rounds", retained)
    monkeypatch.setattr(sys, "argv", ["run_grid.py", "--plan", "configs/fashion_plan.json", "--no-run"])
    runner.main()
    assert commands == [["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"],
                        [sys.executable, "scripts/select_method.py", "--recorded-only"]]
    assert matrices == [cell["path"] for cell in records]
    assert set(sources) <= set(published)
    assert analysis.read(clock_path) == clock
    manifest = analysis.read("results/fashion/manifest.json")
    assert manifest["expected_cells"] == manifest["completed_cells"] == 6
    assert manifest["elapsed_wall_seconds"] == 60. and manifest["wall_seconds"] == 6.
    clock_path.unlink()
    with pytest.raises(ValueError, match="original grid execution clock"):
        runner.main()


def test_fashion_loader_uses_declared_normalization_without_download(monkeypatch):
    import torch
    from fltoken import datasets
    calls = []
    class Images:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.data = torch.tensor([[[0, 255], [128, 64]]], dtype=torch.uint8)
            self.targets = torch.tensor([3])
    monkeypatch.setitem(datasets.DATASETS, "fashion_mnist", (Images, .286, .353))
    datasets.load_tensors.cache_clear()
    try:
        images, labels = datasets.load_tensors("fashion_mnist", "unused-synthetic-path", True)
        assert images.shape == (1, 1, 2, 2) and labels.tolist() == [3]
        expected = torch.tensor([[[[0., 255.], [128., 64.]]]]).div(255).sub(.286).div(.353)
        assert torch.equal(images, expected)
        assert calls == [dict(root="unused-synthetic-path", train=True, download=False)]
    finally:
        datasets.load_tensors.cache_clear()


def test_transfer_assets_preserve_units_counts_and_undefined_values(corpus):
    analysis, _, _, _ = corpus
    analysis.main()
    result = analysis.read("results/summary.json")["fashion"]
    assets = importlib.import_module("fashion_assets")
    tables = []
    assets.fashion_tables(result, lambda name, columns, header, rows, context:
                          tables.append((name, columns, header, rows, context)))
    assert len(tables) == 5
    assert all(len(columns) == len(header) and all(len(row) == len(columns) for row in rows)
               for _, columns, header, rows, _ in tables)
    assert len(next(rows for name, _, _, rows, _ in tables if name == "fashion_accuracy")) == 2
    gini = next((header, rows) for name, _, header, rows, _ in tables if name == "fashion_honest_gini")
    assert gini[0][2:4] == ["Mean Gini", "SD"]
    assert all(0 <= float(row[2]) <= 1 for row in gini[1])
    root = Path(__file__).resolve().parents[1]
    summary = analysis.read(root/"results/summary.json")
    summary["fashion"] = copy.deepcopy(result)
    summary["fashion"]["intervals"][0]["metrics"]["Mstar"]["attacker_share"]["estimate"] = None
    write_json("results/summary.json", summary)
    write_json("results/literature/sendros.json", analysis.read(root/"results/literature/sendros.json"))
    importlib.import_module("make_macros").main()
    macros = Path("paper/generated/numbers.tex").read_text()
    assert r"\newcommand{\FashionRuns}{6}" in macros
    assert r"\newcommand{\FashionMixtureOursAttackerShareMean}{undefined}" in macros
    assert r"\newcommand{\FashionLabelFlipSharedAccuracyMean}{52.100}" in macros
    write_json("out/synthetic-fashion-summary.json", result)
