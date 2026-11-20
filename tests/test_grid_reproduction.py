"""Keep the published source comparison distinct from the larger mixture cohort."""

import copy
import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest


@pytest.fixture
def analysis(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    return importlib.import_module("grid_reproduction"), json.loads((root/"configs/evaluation_plan.json").read_text())


def test_mixture_cohorts_keep_all_seeds_without_long_run_duplication(analysis):
    module, plan = analysis
    manifest = {"cells": list(module.evaluation_cells(plan))}
    selected = module.mixture_cells(manifest, plan)
    assert [cell["seed"] for cell in selected] == [42,43,44,45,46]
    assert {cell["rounds"] for cell in selected} == {20}
    for corrupt in [manifest["cells"]+[selected[0]],
                    [cell for cell in manifest["cells"] if cell != selected[-1]]]:
        with pytest.raises(ValueError, match="one complete"):
            module.mixture_cells({"cells":corrupt}, plan)


def test_source_and_overall_summaries_use_their_declared_seed_sets(analysis):
    module, plan = analysis
    runs = [dict(share=dict(seed=seed, attacker_share=seed/1000), provenance=dict(seed=seed),
                 noise=[dict(relative_error=.01)],
                 trajectory=[dict(round=1, honest_count=2, noise_count=1, honest_q_sum=2*seed, noise_q_sum=10)])
            for seed in [46,43,45,42,44]]
    primary, overall = module.comparison_groups(runs, [42,43,44], plan["seeds"])
    assert [row["share"]["seed"] for row in primary] == [42,43,44]
    source = {"standard":{"attacker_share_mean":.1027}}
    first = module.summarize_runs(primary, source, "cuda")
    second = module.summarize_runs(overall, source, "cuda")
    assert first["noise_participations"] == 3 and second["noise_participations"] == 5
    assert first["attacker_share_mean"] == pytest.approx(.043)
    assert second["attacker_share_mean"] == pytest.approx(.044)
    assert first["quality_windows"]["early"]["honest_mean_q"] == 43
    assert second["quality_windows"]["early"]["honest_mean_q"] == 44
    with pytest.raises(ValueError, match="exactly once"):
        module.comparison_groups(runs+[runs[0]], [42,43,44], plan["seeds"])


def test_training_provenance_rejects_unpublished_or_changed_sources(analysis, monkeypatch):
    module, _ = analysis
    blobs = {"src/fltoken/model.py":b"model source", "scripts/run_experiment.py":b"training entry point",
             "scripts/evaluation.py":b"publication gate", "configs/sendros_mix.yaml":b"scientific protocol"}
    metadata = dict(git_commit="a"*40, source_git_dirty=False, source_config="configs/sendros_mix.yaml",
                    source_sha256={name:hashlib.sha256(value).hexdigest() for name,value in blobs.items()})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    def output(command, **kwargs):
        if command[1] == "ls-tree":
            return "src/fltoken/model.py\n"
        return blobs[command[2].split(":",1)[1]]
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module.subprocess, "check_output", output)
    module.verify_training_source(metadata)
    assert calls == [["git","merge-base","--is-ancestor","a"*40,"origin/main"]]
    changed = copy.deepcopy(metadata)
    changed["source_sha256"]["src/fltoken/model.py"] = "0"*64
    with pytest.raises(ValueError, match="digest differs"):
        module.verify_training_source(changed)
    changed = copy.deepcopy(metadata)
    del changed["source_sha256"]["src/fltoken/model.py"]
    with pytest.raises(ValueError, match="coverage differs"):
        module.verify_training_source(changed)
    def unreachable(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(module.subprocess, "run", unreachable)
    with pytest.raises(ValueError, match="reachable"):
        module.verify_training_source(metadata)


def test_default_regeneration_preserves_the_grid_analysis_basis(analysis, monkeypatch, tmp_path):
    module, _ = analysis
    (tmp_path/"results/literature").mkdir(parents=True)
    (tmp_path/"results/literature/sendros.json").write_text("{}", encoding="utf-8")
    (tmp_path/"results/summary.json").write_text(json.dumps({"reproduction":{"basis":"evaluation_grid"}}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(module, "refresh_grid_reproduction", lambda device:calls.append(device))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["reproduce_sendros.py","--no-run"])
    importlib.import_module("reproduce_sendros").main()
    assert calls == [None]


def test_explicit_legacy_regeneration_removes_stale_overall_assets(analysis, monkeypatch, tmp_path):
    module, _ = analysis
    root = Path(__file__).resolve().parents[1]
    for name in ["configs/sendros_mix.yaml", "results/literature/sendros.json"]:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    for seed in [42,43,44]:
        name = f"results/runs/mnist-sendros_mix-s{seed}-cpu-t20"
        shutil.copytree(root/name, tmp_path/name)
    (tmp_path/"results/summary.json").write_text(json.dumps(
        dict(reproduction=dict(basis="evaluation_grid"),reproduction_overall=dict(stale=True))),encoding="utf-8")
    (tmp_path/"results/reproduction").mkdir()
    for name in ["summary.json","noise_participations.csv","quality_trajectory.csv","per_seed.csv"]:
        (tmp_path/"results/reproduction"/f"overall_{name}").write_text("stale",encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,"argv",["reproduce_sendros.py","--legacy","--no-run"])
    importlib.import_module("reproduce_sendros").main()
    assert "reproduction_overall" not in module.read("results/summary.json")
    assert not list(Path("results/reproduction").glob("overall_*"))
    legacy = module.read("results/reproduction/summary.json")
    assert legacy["device"] == "cpu"
    assert [row["seed"] for row in legacy["per_seed"]] == [42,43,44]
    assert all(row["path"].startswith("results/runs/") for row in legacy["runs"])
    Path("paper/generated").mkdir(parents=True)
    Path("paper/generated/reproduction_grid_cohorts.tex").write_text("stale cohort",encoding="utf-8")
    importlib.import_module("make_tables").main()
    assert not Path("paper/generated/reproduction_grid_cohorts.tex").exists()


def test_grid_refresh_rebuilds_all_cohorts_and_is_byte_stable(analysis, monkeypatch, tmp_path):
    module, plan = analysis
    root = Path(__file__).resolve().parents[1]
    for name in ["configs/sendros_mix.yaml", "configs/evaluation_plan.json", "results/literature/sendros.json",
                 "scripts/reproduce_sendros.py", "scripts/grid_reproduction.py", "scripts/reproduction_analysis.py",
                 "scripts/rebuild_ledgers.py", "src/fltoken/metrics.py", "src/fltoken/statistics.py"]:
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name, target)
    method = {"params":{}}
    (tmp_path/"configs/method_frozen.json").write_text(json.dumps(method), encoding="utf-8")
    (tmp_path/"results/summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Publication and source-hash gates have separate tests. This fixture exercises
    # complete stored logs, exact ledger reconstruction and both summary cohorts.
    monkeypatch.setattr(module, "published_bytes", lambda path:Path(path).read_bytes())
    monkeypatch.setattr(module, "published_method", lambda:method)
    verified = []
    monkeypatch.setattr(module, "verify_training_source", lambda metadata:verified.append(metadata["git_commit"]))
    from fltoken.metrics import ROUND_POOL, payout, summarize
    cells = []
    rules = plan["rules"]+list(module.ABLATIONS)
    for cell in [c for c in module.evaluation_cells(plan) if c["horizon"] == "main" and c["attack"] == "sendros_mix"]:
        folder = Path(cell["path"])
        folder.mkdir(parents=True)
        config = {**module.load_config(plan["config"]), **{key:cell[key] for key in ["seed","attack","fraction","rounds"]},
                  **{key:plan[key] for key in ["device","workers","gpu_fraction"]}, "rules":rules, "ours_params":{}}
        rows, ledger = [], {rule:[] for rule in rules}
        attacks = {0:"noise"}
        share = cell["seed"]/1000
        allocation = [share]+[(1-share)/9]*9
        units, amounts = payout(allocation)
        for round_number in range(1,21):
            for client in range(10):
                row = dict(round=round_number, client_id=client, attack=attacks.get(client,"none"),
                           adversary=client in attacks, accuracy=.8, delta_norm=1, aggregate_norm=1,
                           cosine=.1, predicted_bias=.101, B1_score=10)
                row.update({f"{rule}_share":allocation[client] for rule in rules})
                row.update({f"{rule}_slash":False for rule in rules})
                rows.append(row)
            for rule in rules:
                ledger[rule].append(dict(round=round_number, ids=list(range(10)), shares1e18=units,
                                         payouts=amounts, slash=[False]*10, remainder=ROUND_POOL-sum(amounts)))
        with (folder/"per_round.csv").open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (folder/"round_metrics.csv").write_text("synthetic fixture\n",encoding="utf-8")
        module.write_json(folder/"ledger.json",ledger)
        module.write_json(folder/"metrics.json",summarize(rows,ledger,config,attacks))
        module.write_json(folder/"protocol.json",dict(adversaries=attacks))
        module.write_json(folder/"hashes.json",{name:module.sha256(folder/name) for name in
                                               ["per_round.csv","round_metrics.csv","ledger.json"]})
        module.write_json(folder/"metadata.json",dict(status="complete", config=config, synthetic_fixture=True,
            source_config=plan["config"], source_sha256={}, source_git_dirty=False, versions={},
            git_commit="a"*40, wall_seconds=1, method_sha256=module.sha256("configs/method_frozen.json")))
        cells.append(dict(**cell,source_commit="a"*40,metadata_sha256=module.sha256(folder/"metadata.json"),
                          per_round_sha256=module.sha256(folder/"per_round.csv")))
    module.write_json("results/grid/manifest.json",dict(cells=cells,plan_sha256=module.sha256("configs/evaluation_plan.json"),
                                                       method_sha256=module.sha256("configs/method_frozen.json")))
    module.refresh_grid_reproduction()
    summary = module.read("results/summary.json")
    assert summary["reproduction"]["seeds"] == [42,43,44]
    assert summary["reproduction_overall"]["seeds"] == [42,43,44,45,46]
    assert summary["reproduction"]["noise_participations"] == 60
    assert summary["reproduction_overall"]["noise_participations"] == 100
    assert summary["reproduction"]["attacker_share_mean"] == pytest.approx(.043)
    assert summary["reproduction_overall"]["attacker_share_mean"] == pytest.approx(.044)
    assert len(verified) == 1
    paths = [Path("results/summary.json"),*Path("results/reproduction").iterdir()]
    before = {path:path.read_bytes() for path in paths}
    module.refresh_grid_reproduction()
    assert {path:path.read_bytes() for path in paths} == before
    importlib.import_module("make_tables").main()
    table = Path("paper/generated/reproduction_grid_cohorts.tex").read_text()
    assert "Source seeds & 3 & 4.300" in table and "All evaluation seeds & 5 & 4.400" in table
    assert "Device: CUDA" in table
    summary["reproduction"]["relative_error_quantiles"]["1"] = None
    module.write_json("results/summary.json",summary)
    importlib.import_module("make_macros").main()
    macros = Path("paper/generated/numbers.tex").read_text()
    assert r"\newcommand{\BiasMaxError}{undefined}" in macros
    assert r"\newcommand{\OverallMixtureShare}{4.400}" in macros
    metadata_path = Path(cells[-1]["path"])/"metadata.json"
    metadata_path.write_bytes(metadata_path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="metadata differs"):
        module.refresh_grid_reproduction()
