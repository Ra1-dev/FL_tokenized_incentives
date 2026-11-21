"""Numerical-variant claims require the original frozen implementation and full coverage."""
import copy
import importlib
from pathlib import Path

import pytest


def records():
    paths = {"results/grid/a": 2, "results/grid/b": 3}
    weights = dict(complete_grid=True, runs=2, rounds=5, original_scorer_sha256="frozen",
                   original_reconstruction_matches_recorded=True, stages={"Afull": {"erased_product_rounds": 0}})
    cases = [dict(corpus="grid", path=path, rounds=count, identical=True, original_matches_recorded=True,
                  stages={name: dict(maximum_share_difference=0., maximum_payout_difference=0,
                                     changed=dict(shares=0, payouts=0)) for name in ["Aref", "Aema", "Afull"]})
             for path, count in paths.items()]
    comparison = dict(complete_corpus=True, synthetic_only=False, original_scorer_sha256="frozen",
                      current_scorer_sha256="variant", original_matches_recorded=True, cases=cases)
    return weights, comparison, paths


@pytest.mark.parametrize("damage", ["scorer", "missing_run", "original_mismatch", "partial"])
def test_variant_claim_rejects_changed_provenance_or_coverage(monkeypatch, damage):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    module = importlib.import_module("document_numerical_comparison")
    weights, comparison, paths = records()
    if damage == "scorer": comparison["original_scorer_sha256"] = "variant"
    elif damage == "missing_run": comparison["cases"].pop()
    elif damage == "original_mismatch": comparison["cases"][0]["original_matches_recorded"] = False
    else: weights["complete_grid"] = False
    with pytest.raises(ValueError):
        module.comparison_summary(weights, comparison, paths, "frozen")


def test_zero_underflow_does_not_hide_a_different_numerical_variant(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    module = importlib.import_module("document_numerical_comparison")
    weights, comparison, paths = records()
    before = copy.deepcopy((weights, comparison))
    assert module.comparison_summary(weights, comparison, paths, "frozen")["identical_on_grid"]
    assert (weights, comparison) == before
    comparison["cases"][1]["identical"] = False
    comparison["cases"][1]["stages"]["Afull"]["maximum_payout_difference"] = 7
    comparison["cases"][1]["stages"]["Afull"]["changed"]["payouts"] = 1
    result = module.comparison_summary(weights, comparison, paths, "frozen")
    assert result["changed_grid_runs"] == 1 and not result["identical_on_grid"]
    assert result["maximum_payout_difference"] == 7


def test_variant_identity_cannot_override_changed_payouts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    module = importlib.import_module("document_numerical_comparison")
    weights, comparison, paths = records()
    comparison["cases"][0]["stages"]["Afull"]["changed"]["payouts"] = 1
    with pytest.raises(ValueError, match="identity differs"):
        module.comparison_summary(weights, comparison, paths, "frozen")
