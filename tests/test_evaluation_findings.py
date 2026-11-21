"""Wins and costs must come from the same estimates and preserve missing data."""
import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def findings(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return importlib.import_module("evaluation_findings")


def estimates():
    make=lambda mean:dict(estimate=dict(mean=mean,lower=mean-.01,upper=mean+.01,sample_sd=.02))
    return dict(pooled={"attacker_share":dict(scenarios=16,rules={rule:make(value) for rule,value in
                         [("B0",.2),("B1",.18),("B2",.12),("Mstar",.05)]}),
                       "honest_false_penalty_rate":dict(scenarios=17,rules={rule:make(value) for rule,value in
                         [("B0",0),("B1",.06),("B2",.11),("Mstar",.15)]})},
                intervals=[dict(horizon="main",attack="stale",fraction=fraction,
                          metrics={rule:dict(attacker_share=make(value)) for rule,value in
                                   [("B0",fraction),("B1",fraction+.01),("B2",fraction/2),("Mstar",fraction*.9)]})
                           for fraction in [.1,.2,.3]])


def test_headlines_put_honest_cost_and_every_stale_fraction_next_to_pooled_gain(findings):
    result=estimates()
    text=findings.headline_text(result)
    assert all(word in text for word in ["Attackers, pooled","Honest penalties, pooled","Stale 10%","Stale 20%","Stale 30%"])
    assert "exceeds B2 at 3 of 3" in text and "belongs to Mstar" in text
    changed=copy.deepcopy(result)
    changed["pooled"]["honest_false_penalty_rate"]["rules"]["B1"]["estimate"]["mean"]=.4
    changed["intervals"][0]["metrics"]["Mstar"]["attacker_share"]["estimate"]["mean"]=0
    text=findings.headline_text(changed)
    assert "belongs to B1" in text and "exceeds B2 at 2 of 3" in text


def test_missing_estimates_do_not_become_ranked_wins_or_zero_values(findings):
    result=estimates()
    result["pooled"]["honest_false_penalty_rate"]["rules"]["B1"]["estimate"]=None
    result["pooled"]["attacker_share"]["rules"]["B1"]["estimate"]=None
    result["intervals"][0]["metrics"]["B2"]["attacker_share"]["estimate"]=None
    text=findings.headline_text(result)
    assert "Undefined" in text and "prevents a complete ranking" in text
    assert "prevents a complete reward comparison" in text and "reduces pooled" not in text
    assert "2 of 2 defined fractions (three planned)" in text


@pytest.mark.parametrize("change",["missing","duplicate"])
def test_stale_coverage_cannot_be_silently_pruned(findings,change):
    result=estimates()
    if change=="missing":result["intervals"].pop()
    else:result["intervals"][-1]=result["intervals"][0]
    with pytest.raises(ValueError,match="all three"):
        findings.headline_rows(result)
