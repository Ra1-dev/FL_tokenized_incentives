"""Integer reward metrics must not depend on floating reduction implementations."""

from fractions import Fraction
import importlib
from pathlib import Path
import random

import pytest

from fltoken.metrics import gini, summarize


def test_gini_matches_independent_pairwise_integer_definition():
    rng = random.Random(2519)
    for count in [1, 2, 7, 25, 63]:
        values = [rng.randrange(10**30) for _ in range(count)]
        numerator = sum(abs(a-b) for a in values for b in values)
        expected = float(Fraction(numerator, 2*count*sum(values)))
        assert gini(values) == expected
        assert gini([value*10**400 for value in reversed(values)]) == expected
    assert gini([]) is None and gini([0, 0]) is None
    assert gini([17]*25) == 0.
    assert gini([0, 0, 9]) == float(Fraction(2, 3))


def test_gini_rejects_noninteger_or_negative_payouts():
    with pytest.raises(ValueError):
        gini([0, -1])
    with pytest.raises(TypeError):
        gini([1., 2.])


def test_detection_f1_uses_integer_counts_once():
    rows = [dict(client_id=i, adversary=i<3, accuracy=.9, R_share=share, R_slash=False)
            for i,share in enumerate([0.,0.,1.,0.,1.])]
    ledger = {"R":[dict(ids=list(range(5)), payouts=[0,0,500,0,500])]}
    result = summarize(rows,ledger,dict(num_clients=5,clients_per_round=5),{0:"noise",1:"noise",2:"noise"})["R"]
    assert (result["tp"],result["fp"],result["fn"]) == (2,1,1)
    assert result["f1"] == float(Fraction(4,6))
    assert result["attacker_share"] == .5
    assert result["honest_gini"] == .5


def test_quantiles_interpolate_exactly_and_keep_undefined_values(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    quantile = importlib.import_module("reproduce_sendros").exact_quantile
    values = [.1,.2,.3,.4]
    assert quantile(values,.9) == float(Fraction(.3)*Fraction(3,10)+Fraction(.4)*Fraction(7,10))
    assert quantile([0.,float("inf")],.5) is None
    assert quantile([], .5) is None
    assert quantile([.1], .5) == .1
