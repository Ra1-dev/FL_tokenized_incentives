"""Full replay coverage and cached evidence must retain their actual scope."""

import importlib
import json
from pathlib import Path

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
