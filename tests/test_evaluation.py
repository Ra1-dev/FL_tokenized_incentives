"""Coverage, publication and uncertainty must not overstate the experiments."""

import copy
import importlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from fltoken.rules import make_rules, score_round


@pytest.fixture
def analysis(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    module = importlib.import_module("summarize_grid")
    plan = json.loads((root / "configs/evaluation_plan.json").read_text())
    return module, plan


def synthetic(module, plan):
    records = []
    for cell in module.evaluation_cells(plan):
        seed_index = plan["seeds"].index(cell["seed"])
        attack_index = plan["attacks"].index(cell["attack"])
        fraction = .1 if cell["attack"] in {"none", "sendros_mix"} else cell["fraction"]
        measurements = {}
        for index, rule in enumerate(plan["rules"] + list(module.ABLATIONS)):
            value = (seed_index+1)/100 + attack_index/100 + fraction/10 + index/1000
            measurements[rule] = {metric: value for metric in module.METRICS}
        measurements["Afull"] = dict(measurements["Mstar"])
        for rule in measurements:
            measurements[rule]["final_accuracy"] = .5 + seed_index/100
        records.append(dict(**cell, metrics=measurements,
                            per_round_sha256=f"synthetic:{cell['seed']}:{cell['attack']}:{fraction}:{cell['horizon']}"))
    return records


def test_complete_grid_preserves_unique_conditions_and_seed_units(analysis):
    module, plan = analysis
    records = synthetic(module, plan)
    result = module.summarize_records(records, plan)
    assert result["nominal_cells"] == 90
    assert result["distinct_main_scenarios"] == 17
    assert result["pooled"]["attacker_share"]["scenarios"] == 16
    assert len(result["contrasts"]) == 6
    for comparison in result["contrasts"]:
        assert comparison["seed_test"]["pairs"] == 5
        assert comparison["seed_test"]["p_value"] == .0625
        assert comparison["paired_difference"]["estimate"]["bootstrap_samples"] == 20000
        assert comparison["condition_seed_test"]["pairs"] == 80
        assert comparison["condition_seed_test_holm_p"] >= comparison["condition_seed_test"]["p_value"]
    with pytest.raises(ValueError, match="coverage"):
        module.summarize_records(records[:-1], plan)
    with pytest.raises(ValueError, match="coverage"):
        module.summarize_records(records+[records[0]], plan)
    changed = copy.deepcopy(records)
    changed[1]["metrics"]["Afull"]["attacker_share"] += .1
    with pytest.raises(ValueError, match="final component stage"):
        module.summarize_records(changed, plan)


def test_undefined_cell_is_visible_and_never_dropped_from_a_claim(analysis):
    module, plan = analysis
    records = synthetic(module, plan)
    target = next(cell for cell in records if cell["attack"] == "noise")
    target["metrics"]["Mstar"]["attacker_share"] = None
    target["metrics"]["Afull"]["attacker_share"] = None
    result = module.summarize_records(records, plan)
    pooled = result["pooled"]["attacker_share"]["rules"]["Mstar"]
    assert pooled == dict(defined=4, undefined=1, estimate=None)
    for comparison in result["contrasts"]:
        if comparison["metric"] == "attacker_share":
            assert comparison["condition_seed_test"] is None
            assert comparison["seed_test_holm_p"] is None
            assert comparison["undefined_condition_seed_pairs"] == 1


def test_publication_gate_rejects_missing_or_changed_artifacts(tmp_path, monkeypatch, analysis):
    module, plan = analysis
    publication = importlib.import_module("evaluation")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(publication, "REPOSITORY", tmp_path)
    path = Path("frozen.json")
    path.write_bytes(b'{}\n')
    monkeypatch.setattr(publication.subprocess, "check_output", lambda *args, **kwargs: b'{}\n')
    assert publication.published_bytes(path) == b'{}\n'
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ValueError, match="differs"):
        publication.published_bytes(path)
    with pytest.raises(ValueError, match="relative"):
        publication.published_bytes("../outside.json")
    def absent(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])
    monkeypatch.setattr(publication.subprocess, "check_output", absent)
    with pytest.raises(RuntimeError, match="Git exited 128"):
        publication.published_bytes(path)


def test_ablations_have_independent_state_and_one_common_input():
    params = dict(reference=True, ema=True, penalties=True, beta=.5, threshold=.05, patience=2, initial_reputation=.5)
    config = dict(rules=["B0", "B1", "B2", "Mstar", "Aref", "Aema", "Afull"], krum_f=1, ours_params=params)
    rules = make_rules(config)
    ids = np.arange(7)
    x = np.random.default_rng(817).normal(1., .1, size=(7,31))
    before = x.copy()
    outputs = score_round(rules,ids,x,1)
    assert np.array_equal(x,before)
    assert np.array_equal(outputs["Mstar"]["shares"], outputs["Afull"]["shares"])
    assert rules["Mstar"].reputation is not rules["Afull"].reputation
    assert rules["Aref"].params["ema"] is False
    assert rules["Aema"].params["penalties"] is False
    score_round(rules,ids,np.zeros_like(x),2)
    outputs = score_round(rules,ids,np.zeros_like(x),3)
    assert outputs["Afull"]["slash"].all()
    assert not outputs["Aref"]["slash"].any()
    assert not outputs["Aema"]["slash"].any()


def test_accuracy_uncertainty_uses_one_observation_per_seed_and_round(analysis):
    module, plan = analysis
    records=[dict(path=cell["path"],round_history=[dict(round=round,accuracy=.5+(cell["seed"]-42)/100,loss=1.)
             for round in range(1,cell["rounds"]+1)]) for cell in module.evaluation_cells(plan)]
    curves=module.trajectory_intervals(records,plan)
    assert len(curves)==390
    for row in curves:
        assert row["accuracy"]["estimate"]["observations"]==5
        assert row["accuracy"]["estimate"]["mean"]==.52
        assert row["loss"]["estimate"]["lower"]==row["loss"]["estimate"]["upper"]==1.
    records[0]["round_history"].pop()
    with pytest.raises(ValueError,match="round history"):
        module.trajectory_intervals(records,plan)


def test_accuracy_curve_must_match_the_shared_participation_log(analysis, tmp_path):
    module, plan = analysis
    text="round,accuracy,loss,update_set_sha256\n1,0.7,0.4,common\n"
    for name in ["round_metrics.csv", "per_round.csv"]:
        (tmp_path/name).write_text(text,encoding="utf-8")
    def bind():
        (tmp_path/"hashes.json").write_text(json.dumps({"round_metrics.csv":module.sha256(tmp_path/"round_metrics.csv")}))
    bind()
    assert module.read_round_history(tmp_path)==[dict(round=1,accuracy=.7,loss=.4)]
    (tmp_path/"round_metrics.csv").write_text(text.replace("0.7","0.9"),encoding="utf-8")
    bind()
    with pytest.raises(ValueError,match="common participation log"):
        module.read_round_history(tmp_path)
