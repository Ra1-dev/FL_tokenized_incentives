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
