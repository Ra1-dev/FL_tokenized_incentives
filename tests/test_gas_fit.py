"""Gas capacity must follow the measurements and exact regression arithmetic."""

from fractions import Fraction
import importlib
from pathlib import Path

import pytest


def test_exact_gas_fit_and_cap_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    fit=importlib.import_module("gas_fit").fit
    points=[(1,10),(2,12),(3,17)]
    expected=dict(intercept=6.,slope=3.5,r_squared=float(Fraction(49,52)),max_absolute_residual=1.,estimated_cap_n=3)
    assert fit(points,cap=17)==expected
    assert fit(list(reversed(points)),cap=17)==expected
    assert fit([(x,y*10**100) for x,y in points])["r_squared"]==expected["r_squared"]
    assert fit([(1,7),(3,7)],cap=17)["estimated_cap_n"] is None
    with pytest.raises(ValueError):
        fit([(2,7),(2,8)])
