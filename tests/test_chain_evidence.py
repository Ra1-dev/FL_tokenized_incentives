"""Recorded settlement evidence must survive independent state reconstruction."""

from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest


def fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    module = importlib.import_module("summarize_contracts")
    oracle = module.ChainLedger(range(2), 1)
    inputs, states = [], []
    for round in range(1, 5):
        shares = [10**18//2]*2 if round != 3 else [0, 0]
        amounts = [1000*s for s in shares]
        item = dict(round=round, ids=[0, 1], shares1e18=shares, payouts=amounts,
                    slash=[round == 2, round == 4], remainder=1000*10**18-sum(amounts))
        transition = oracle.settle(item)
        if round % 2 == 0:
            oracle.claim(0)
        inputs.append(item)
        states.append(json.loads(json.dumps(dict(transition=transition, state=oracle.snapshot()))))
    return module, inputs, states


def test_recorded_delayed_claims_reconstruct_exactly(monkeypatch):
    module, inputs, states = fixture(monkeypatch)
    assert module.validate_states(2, 1, inputs, states) == 4


def test_impossible_population_is_rejected_before_building_accounts(monkeypatch):
    module, inputs, states = fixture(monkeypatch)
    with pytest.raises(ValueError, match="account coverage"):
        module.validate_states(2**128, 1, inputs, states)


@pytest.mark.parametrize("field", ["wallet", "stake", "claimable", "unvested"])
def test_recorded_balance_tampering_is_rejected(monkeypatch, field):
    module, inputs, states = fixture(monkeypatch)
    states = deepcopy(states)
    states[2]["state"]["accounts"]["0"][field] += 1
    with pytest.raises(ValueError):
        module.validate_states(2, 1, inputs, states)


def test_recorded_burn_tampering_is_rejected(monkeypatch):
    module, inputs, states = fixture(monkeypatch)
    states[1]["transition"]["burned"] += 1
    with pytest.raises(ValueError, match="mint, burn"):
        module.validate_states(2, 1, inputs, states)


@pytest.mark.parametrize("field,value", [("round", True), ("supply", 1200e18), ("unminted", False)])
def test_numerically_equal_noninteger_state_is_rejected(monkeypatch, field, value):
    module, inputs, states = fixture(monkeypatch)
    assert states[0]["state"][field] == value
    states[0]["state"][field] = value
    with pytest.raises(ValueError, match="unsigned integer"):
        module.validate_states(2, 1, inputs, states)


@pytest.mark.parametrize("container,field,value", [("account", "wallet", False), ("transition", "burned", 0.0)])
def test_nested_noninteger_state_is_rejected(monkeypatch, container, field, value):
    module, inputs, states = fixture(monkeypatch)
    target = states[0]["state"]["accounts"]["0"] if container == "account" else states[0]["transition"]
    assert target[field] == value
    target[field] = value
    with pytest.raises(ValueError, match="unsigned integer"):
        module.validate_states(2, 1, inputs, states)


@pytest.mark.parametrize("field,value", [("round", True), ("ids", [False, 1]),
                                       ("payouts", [500e18, 500e18]), ("remainder", 0.0)])
def test_noninteger_input_ledger_is_rejected(monkeypatch, field, value):
    module, inputs, states = fixture(monkeypatch)
    assert inputs[0][field] == value
    inputs[0][field] = value
    with pytest.raises(ValueError, match="unsigned integer"):
        module.validate_states(2, 1, inputs, states)


@pytest.mark.parametrize("removed", ["contracts/src/RewardPool.sol", "scripts/replay_onchain.py",
                                    "contracts/test/RewardPool.t.sol"])
def test_measurement_source_coverage_is_complete(monkeypatch, removed):
    module, _, _ = fixture(monkeypatch)
    program, extra = "scripts/check_contracts.py", ["contracts/test/RewardPool.t.sol"]
    sources = {**module.provenance(program), **{name: module.sha256(name) for name in extra}}
    module.validate_measurement_sources(sources, program, extra)
    del sources[removed]
    with pytest.raises(ValueError, match="source coverage"):
        module.validate_measurement_sources(sources, program, extra)


def test_measurement_source_mutation_is_rejected(monkeypatch):
    module, _, _ = fixture(monkeypatch)
    program = "scripts/replay_onchain.py"
    sources = module.provenance(program)
    sources["contracts/src/RewardPool.sol"] = "0"*64
    with pytest.raises(ValueError, match="source coverage"):
        module.validate_measurement_sources(sources, program)


@pytest.mark.parametrize("mutation", ["checks", "truthy_check", "canonical_input", "missing_case",
                                     "duplicate_case", "round_count", "elapsed_time"])
def test_platform_evidence_requires_complete_coverage(monkeypatch, mutation):
    module, _, _ = fixture(monkeypatch)
    recorded = module.read("results/contracts/portability-linux.json")
    replays = {case["case"]: module.read(Path("results/onchain")/case["case"])
               for case in recorded["replay_cases"]}
    module.validate_portability(recorded, replays)
    if mutation == "checks":
        recorded["checks"] = {}
    elif mutation == "truthy_check":
        recorded["checks"][next(iter(recorded["checks"]))] = 1
    elif mutation == "canonical_input":
        del recorded["canonical_input_sha256"]["results/contracts/checks.json"]
    elif mutation == "missing_case":
        recorded["replay_cases"].pop()
    elif mutation == "duplicate_case":
        recorded["replay_cases"][-1] = recorded["replay_cases"][0]
    elif mutation == "round_count":
        recorded["replay_round_rule_cases"] -= 1
    else:
        recorded["replay_cases"][0]["compared_seconds"] = True
    with pytest.raises(ValueError, match="platform comparison"):
        module.validate_portability(recorded, replays)


def test_platform_comparison_excludes_only_elapsed_time(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    module = importlib.import_module("check_contract_portability")
    left = dict(wall_seconds=12., states=[{"balance": 10**24}], gas=[12345])
    right = dict(wall_seconds=19., states=[{"balance": 10**24}], gas=[12345])
    assert module.equivalent(left,right,elapsed=True)
    right["states"][0]["balance"] += 1
    assert not module.equivalent(left,right,elapsed=True)
    with pytest.raises(ValueError):
        module.equivalent(left,{**left,"wall_seconds":0},elapsed=True)
