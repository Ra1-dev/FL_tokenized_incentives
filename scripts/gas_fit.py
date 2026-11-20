"""Exact least-squares derivation from integer transaction measurements."""

from fractions import Fraction
from operator import index


def fit(points, cap=None):
    points = [(index(x), index(y)) for x, y in points]
    if len(points) < 2:
        raise ValueError("a gas fit requires at least two measurements")
    count = len(points)
    sx, sy = sum(x for x, _ in points), sum(y for _, y in points)
    denominator = count*sum(x*x for x, _ in points)-sx*sx
    if not denominator:
        raise ValueError("client counts must vary")
    slope = Fraction(count*sum(x*y for x, y in points)-sx*sy, denominator)
    intercept = Fraction(sy, count)-slope*Fraction(sx, count)
    residuals = [Fraction(y)-intercept-slope*x for x, y in points]
    total = sum((Fraction(y)-Fraction(sy, count))**2 for _, y in points)
    r_squared = 1-sum(r*r for r in residuals)/total if total else Fraction(1)
    result = dict(intercept=float(intercept), slope=float(slope), r_squared=float(r_squared),
                  max_absolute_residual=float(max(abs(r) for r in residuals)))
    if cap is not None:
        result["estimated_cap_n"] = (index(cap)-intercept)//slope if slope > 0 else None
    return result


def gas_fits(cases, rounds, cap):
    result = {}
    for round in range(1, rounds+1):
        for operation in ["openRound", "postScores", "settle"]:
            points = [(c["clients"], next(g["gas_used"] for g in c["gas"]
                       if g["round"] == round and g["operation"] == operation)) for c in cases]
            result[f"{operation}_round{round}"] = fit(points)
        result[f"settle_estimated_limit_round{round}"] = fit(
            [(c["clients"], c["rounds"][round-1]["settle_estimated_limit"]) for c in cases], cap)
        result[f"total_round{round}"] = fit(
            [(c["clients"], c["rounds"][round-1]["total_gas"]) for c in cases])
    return result
