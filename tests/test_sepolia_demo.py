"""The integer model used to check the public-testnet demonstration follows the pool's settlement order."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("web3")


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    return importlib.import_module("sepolia_demo")


def test_free_rider_is_flagged_only_after_the_frozen_patience(demo):
    flags = [demo.scores(round, [None]*3)[1] for round in range(1, 7)]
    assert [f[2] for f in flags] == [False, False, True, False, False, True]
    assert all(not f[0] and not f[1] for f in flags)
    assert sum(demo.scores(1, [None]*3)[0]) == demo.SCALE


def test_expected_model_matures_before_slashing_and_floors_burns(demo):
    model = demo.Expected(["a", "b", "c"], vesting=1)
    model.settle(1, ["a", "b", "c"], [demo.SCALE//2, demo.SCALE//4, demo.SCALE//4], [False, False, False])
    assert model.tranches["c"] == [(250*demo.SCALE, 2)] and model.claimable["c"] == 0
    model.settle(2, ["a", "b", "c"], [demo.SCALE//2, demo.SCALE//2, 0], [False, False, True])
    # Round-1 tranche matured first and is protected; the slash hits stake only.
    assert model.claimable["c"] == 250*demo.SCALE
    assert model.stake["c"] == 90*demo.SCALE and model.burned == 10*demo.SCALE
    model.claim("c")
    assert model.claimed["c"] == 250*demo.SCALE and model.claimable["c"] == 0
    assert model.minted == 2000*demo.SCALE


def test_slash_burns_a_tenth_of_unvested_tranches(demo):
    model = demo.Expected(["a"], vesting=3)
    model.settle(1, ["a"], [demo.SCALE], [False])
    model.settle(2, ["a"], [0], [True])
    assert model.tranches["a"] == [(900*demo.SCALE, 4)]
    assert model.burned == 10*demo.SCALE + 100*demo.SCALE


def test_gas_comparison_uses_medians_per_operation(demo):
    assert demo.median([5, 1, 3]) == 3 and demo.median([4, 1, 3, 2]) == 2
    record = {"transactions": [dict(operation="settle", gas_used=300), dict(operation="settle", gas_used=500),
                               dict(operation="sweep", gas_used=21_000)]}
    reference = {"transactions": [dict(operation="settle", gas_used=100)]}
    rows = demo.gas_table(record, reference)
    assert "| settle | 100 | 400 | 4.00x |" in rows
    assert rows[-1] == "| total (without sweeps) | 100 | 800 | 8.00x |"
