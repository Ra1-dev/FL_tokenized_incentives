"""Adaptive attacks replace only the sampled attackers' rows of the unchanged protocol."""

import importlib
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    monkeypatch.chdir(ROOT)
    return importlib.import_module("run_adaptive_study")


@pytest.mark.parametrize("attackers,expected", [(1, 0.0), (2, NormalDist().inv_cdf(.6)), (3, NormalDist().inv_cdf(.7)),
                                                 (6, NormalDist().inv_cdf(.9)), (9, NormalDist().inv_cdf(.9))])
def test_alie_z_follows_the_published_formula_and_clamps_a_corrupted_majority(study, attackers, expected):
    assert study.alie_z(10, attackers) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("attackers", [0, 10])
def test_alie_requires_attackers_and_honest_clients(study, attackers):
    with pytest.raises(ValueError):
        study.alie_z(10, attackers)


def hooks(study, attack_map, ids):
    result = study.Hooks(attack_map, lambda *args, **kwargs: np.array(ids))
    result.sample(0)
    return result


def test_alie_replaces_only_attacker_rows_with_mean_minus_z_std(study):
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(5, 7))
    original = rows.copy()
    h = hooks(study, {3: "alie", 9: "alie"}, [1, 3, 5, 9, 11])
    replaced = h.replace(rows)
    honest = original[[0, 2, 4]]
    expected = honest.mean(axis=0) - study.alie_z(5, 2)*honest.std(axis=0)
    assert np.array_equal(rows, original)
    assert np.array_equal(replaced[[0, 2, 4]], honest)
    assert np.allclose(replaced[1], expected) and np.allclose(replaced[3], expected)
    assert h.log == [dict(attackers=2, z=study.alie_z(5, 2))]


def test_mimic_reports_zero_first_then_the_previous_weighted_aggregate(study):
    h = hooks(study, {2: "mimic"}, [1, 2, 3])
    proxy = h.numpy()
    first = proxy.stack([np.ones(4), np.full(4, 5.), np.full(4, 2.)])
    assert np.array_equal(first[1], np.zeros(4)) and np.array_equal(first[0], np.ones(4))
    aggregate = proxy.average(first, axis=0, weights=[1, 1, 2])
    proxy.mean(first, axis=0)
    second = proxy.stack([np.ones(4), np.full(4, 5.), np.full(4, 2.)])
    assert np.array_equal(second[1], aggregate)
    assert proxy.float64 is np.float64


def test_rounds_without_attackers_pass_through_unchanged(study):
    h = hooks(study, {7: "alie"}, [1, 2, 3])
    rows = np.arange(12.).reshape(3, 4)
    assert h.replace(rows) is rows and h.log == [dict(attackers=0)]


def test_mismatched_rows_are_rejected(study):
    h = hooks(study, {1: "alie"}, [1, 2, 3])
    with pytest.raises(RuntimeError, match="do not match"):
        h.replace(np.zeros((2, 4)))


def test_declared_cells_are_seed_major_and_unique(study):
    plan = study.read_plan()
    cells = study.study_cells(plan)
    assert len(cells) == len(plan["seeds"])*len(study.ATTACKS)*len(plan["fractions"])
    assert len({cell["path"] for cell in cells}) == len(cells)
    assert [cell["seed"] for cell in cells] == sorted(cell["seed"] for cell in cells)
    assert all(cell["path"].startswith("results/adaptive_study/") for cell in cells)


def test_installed_hooks_leave_the_honest_server_code_unchanged(study, monkeypatch):
    from fltoken import experiment
    monkeypatch.setattr(experiment, "np", experiment.np)
    monkeypatch.setattr(experiment, "sample_clients", experiment.sample_clients)
    original = experiment.sample_clients
    h = study.install(experiment, {0: "alie"})
    assert experiment.sample_clients == h.sample and h.sample_clients is original
    ids = experiment.sample_clients(30, 10, 1, 42)
    assert np.array_equal(ids, original(30, 10, 1, 42))


def test_analysis_uses_only_complete_seeds_and_pools_fractions(monkeypatch, tmp_path):
    import json
    import shutil
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    assets = importlib.import_module("adaptive_assets")
    runner = importlib.import_module("run_adaptive_study")
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    shutil.copyfile(ROOT/"configs/adaptive_study.json", "configs/adaptive_study.json")
    plan = runner.read_plan()
    from fltoken.io import sha256
    for cell in runner.study_cells(plan):
        if cell["seed"] not in (42, 43) and not (cell["seed"] == 44 and cell["attack"] == "alie"):
            continue
        folder = Path(cell["path"])
        folder.mkdir(parents=True)
        config = dict(attack=cell["attack"], fraction=cell["fraction"], seed=cell["seed"], rounds=plan["rounds"],
                      rules=plan["rules"], device="cuda")
        (folder/"metadata.json").write_text(json.dumps(dict(status="complete", study_cell=cell, plan_sha256=sha256(runner.PLAN),
                                                            smoke=False, config=config, git_commit="a"*40,
                                                            source_git_dirty=False, wall_seconds=1.0)), encoding="utf-8")
        share = .1 if cell["seed"] == 42 else .3
        metrics = {rule: dict(attacker_share=share, honest_false_penalty_rate=.05, final_accuracy=.9, headcount_share=cell["fraction"])
                   for rule in plan["rules"]}
        (folder/"metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (folder/"attack_log.json").write_text(json.dumps([dict(attackers=1)]*plan["rounds"]), encoding="utf-8")
    complete = assets.recorded(plan)
    assert list(complete) == [42, 43]
    summary = assets.summarize(plan, complete)
    assert summary["complete_cells"] == 12 and summary["seeds"] == [42, 43]
    assert summary["pooled"]["alie"]["Mstar"]["attacker_share"]["mean"] == pytest.approx(.2)
    assert summary["pooled"]["mimic"]["CGSV"]["final_accuracy"]["mean"] == pytest.approx(.9)
