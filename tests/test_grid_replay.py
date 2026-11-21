"""Full replay coverage and cached evidence must retain their actual scope."""

import importlib
import hashlib
import json
from pathlib import Path
import sys

import pytest


def replay_module(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    return importlib.import_module("replay_grid"), json.loads((root/"configs/evaluation_plan.json").read_text())


def test_replay_covers_all_primary_ledgers_and_declared_baseline_sample(monkeypatch):
    module, plan = replay_module(monkeypatch)
    jobs = list(module.replay_jobs(plan))
    assert len(jobs) == 196
    assert sum(job["rounds"] for job in jobs) == 4280
    for rule in ["B1", "Mstar"]:
        required=[job for job in jobs if job["rule"]==rule]
        assert len(required)==90 and sum(job["rounds"] for job in required)==1950
    for rule in ["B0", "B2"]:
        sample=[job for job in jobs if job["rule"]==rule]
        assert len(sample)==8 and {job["attack"] for job in sample}==set(plan["attacks"])
        assert {job["seed"] for job in sample}=={plan["seeds"][0]}


def test_partial_replay_cannot_be_mistaken_for_complete_coverage(monkeypatch):
    module, plan = replay_module(monkeypatch)
    planned=list(module.evaluation_cells(plan))
    manifest=dict(expected_cells=len(planned), completed_cells=1, cells=[planned[0]])
    assert len(module.covered_cells(manifest,plan,True))==1
    with pytest.raises(ValueError,match="complete grid"):
        module.covered_cells(manifest,plan,False)
    manifest["cells"].append(planned[0])
    manifest["completed_cells"]=2
    with pytest.raises(ValueError,match="duplicated coverage"):
        module.covered_cells(manifest,plan,True)
    manifest=dict(expected_cells=len(planned), completed_cells=len(planned), cells=planned)
    assert len(module.covered_cells(manifest,plan,False))==90
    manifest["cells"][0]["rounds"]+=1
    with pytest.raises(ValueError,match="declared plan"):
        module.covered_cells(manifest,plan,False)


@pytest.mark.parametrize("change", ["rule", "rounds", "vesting_delay", "provenance", "ledger_sha256", "state"])
def test_cached_replay_rejects_mismatched_or_tampered_evidence(monkeypatch, tmp_path, change):
    module, plan = replay_module(monkeypatch)
    oracle=importlib.import_module("chain_ledger").ChainLedger(range(2),1)
    item=dict(round=1,ids=[0,1],shares1e18=[10**18,0],payouts=[1000*10**18,0],slash=[False,False],remainder=0)
    transition=oracle.settle(item)
    (tmp_path/"metadata.json").write_text(json.dumps(dict(config=dict(num_clients=2))))
    (tmp_path/"ledger.json").write_text(json.dumps(dict(B1=[item])))
    (tmp_path/"per_round.csv").write_text("synthetic recorded input\n")
    job=dict(path=tmp_path.as_posix(),rule="B1",rounds=1)
    case=dict(**job,clients=2,vesting_delay=1,exact=True,provenance=module.provenance("scripts/replay_onchain.py"),
              ledger_sha256=module.sha256(tmp_path/"ledger.json"),participation_sha256=module.sha256(tmp_path/"per_round.csv"),
              pool_source_sha256=module.sha256("contracts/src/RewardPool.sol"),
              token_source_sha256=module.sha256("contracts/src/RewardToken.sol"),
              states=json.loads(json.dumps([dict(**transition,state=oracle.snapshot())])))
    assert module.validate_case(case,job,tmp_path,1)==1
    if change=="state":case["states"][0]["state"]["accounts"]["0"]["stake"]+=1
    elif change=="provenance":case["provenance"].pop("scripts/replay_onchain.py")
    elif change=="ledger_sha256":case[change]="0"*64
    elif change=="rule":case[change]="Mstar"
    else:case[change]+=1
    with pytest.raises(ValueError):module.validate_case(case,job,tmp_path,1)


@pytest.mark.parametrize("producer", ["frozen", "current_library", "missing", "altered"])
def test_grid_replay_requires_the_frozen_scorer_before_chain_execution(monkeypatch, tmp_path, producer):
    module, plan = replay_module(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root/"configs/method_frozen.json").read_text(encoding="utf-8"))
    pilots = json.loads((root/"results/pilots/manifest.json").read_text(encoding="utf-8"))
    # Use a real historical source record; the local participation text is only
    # a placeholder because rejection must happen before any replay is loaded.
    metadata = json.loads((root/pilots["cells"][0]["path"]/"metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_sha256"]["src/fltoken/ours.py"] == frozen["scorer_sha256"]
    if producer == "current_library":
        current = hashlib.sha256((root/"src/fltoken/ours.py").read_bytes()).hexdigest()
        assert current != frozen["scorer_sha256"]
        metadata["source_sha256"]["src/fltoken/ours.py"] = current
    elif producer == "missing":
        metadata["source_sha256"].pop("src/fltoken/ours.py")
    elif producer == "altered":
        metadata["source_sha256"]["src/fltoken/ours.py"] = "0"*64
    cell = next(module.evaluation_cells(plan))
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/evaluation_plan.json").write_bytes((root/"configs/evaluation_plan.json").read_bytes())
    Path("configs/method_frozen.json").write_bytes((root/"configs/method_frozen.json").read_bytes())
    metadata["method_sha256"] = module.sha256("configs/method_frozen.json")
    folder = Path(cell["path"])
    folder.mkdir(parents=True)
    (folder/"metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder/"per_round.csv").write_text("synthetic placeholder\n", encoding="utf-8")
    record = dict(**cell, metadata_sha256=module.sha256(folder/"metadata.json"),
                  per_round_sha256=module.sha256(folder/"per_round.csv"))
    manifest = dict(plan_sha256=module.sha256("configs/evaluation_plan.json"),
                    method_sha256=metadata["method_sha256"], cells=[record], completed_cells=1,
                    expected_cells=len(list(module.evaluation_cells(plan))))
    Path("results/grid/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["replay_grid.py", "--available", "--no-run"])
    expected = "missing on-chain replay" if producer == "frozen" else "training scorer differs"
    with pytest.raises(ValueError, match=expected):
        module.main()
