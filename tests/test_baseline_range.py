"""Finite extreme updates retain the baseline definitions and stable tie order."""

from decimal import Decimal, localcontext

import numpy as np
import pytest


from fltoken import scoring


def decimal_krum(matrix, f):
    with localcontext() as context:
        context.prec = 1500
        rows = [[Decimal.from_float(float(value)) for value in row] for row in matrix]
        scores = [sum(sorted(sum((a-b)**2 for a,b in zip(row, other))
                             for j, other in enumerate(rows) if i != j)[:len(rows)-f-2])
                  for i, row in enumerate(rows)]
        order = sorted(range(len(rows)), key=lambda i:scores[i])
        return np.array([i in order[:len(rows)-f] for i in range(len(rows))])


def decimal_cosine(matrix):
    with localcontext() as context:
        context.prec = 1500
        rows = [[Decimal.from_float(float(value)) for value in row] for row in matrix]
        reference = [sum(column)/len(rows) for column in zip(*rows)]
        reference_norm = sum(value*value for value in reference).sqrt()
        result = []
        for row in rows:
            divisor = sum(value*value for value in row).sqrt()*reference_norm
            cosine = sum(a*b for a,b in zip(row,reference))/divisor if divisor else Decimal(0)
            # The API exposes binary64 cosine before integer quantization.
            # Angles below that resolution round to one before flooring.
            result.append(int(100*float(min(Decimal(1),max(Decimal(0),cosine)))))
        return result


@pytest.mark.parametrize('f,dimension', [(0,1),(1,2),(2,17),(4,203)])
def test_common_scale_keeps_cosine_scores_and_distance_order(f, dimension):
    random = np.random.default_rng(701+f)
    for _ in range(7):
        matrix = random.normal(size=(2*f+3, dimension))
        matrix[-1] = 0
        original_q, original_shares = scoring.cosine_sendros(matrix)
        original_selected, _ = scoring.multi_krum(matrix, f)
        for exponent in [-900, -550, 0, 550, 900]:
            scaled = np.ldexp(matrix, exponent)
            before = scaled.copy()
            q, shares = scoring.cosine_sendros(scaled)
            selected, rewards = scoring.multi_krum(scaled, f)
            assert np.isfinite(scoring.krum_scores(scaled,f)).all()
            np.testing.assert_array_equal(q, original_q)
            np.testing.assert_array_equal(shares, original_shares)
            np.testing.assert_array_equal(selected, original_selected)
            assert rewards.sum() == pytest.approx(1)
            np.testing.assert_array_equal(scaled, before)


@pytest.mark.parametrize('size', [1,2,3,5,7,10,11])
def test_maximum_finite_mean_is_still_finite(size):
    matrix = np.full((size,1),np.finfo(float).max)
    with np.errstate(over='raise',invalid='raise'):
        q, shares = scoring.cosine_sendros(matrix)
    assert q.tolist() == [100]*size
    assert shares.tolist() == [1/size]*size


def test_large_and_small_clients_both_keep_their_directions():
    matrix = np.array([[1e308],[1e308],[1e-200],[0.],[0.]])
    q, shares = scoring.cosine_sendros(matrix)
    assert q.tolist() == [100,100,100,0,0]
    assert shares.tolist() == [1/3,1/3,1/3,0,0]
    # The large outlier cannot erase unequal distances among small rows.
    mixed = np.array([[99e-200],[4e-200],[1e-200],[2e-200],[3e-200],[5e-200],[1e308]])
    selected, _ = scoring.multi_krum(mixed,2)
    np.testing.assert_array_equal(selected, decimal_krum(mixed,2))
    assert selected.tolist() == [False,True,True,True,True,True,False]
    assert np.isfinite(scoring.krum_scores(mixed,2)).all()
    assert scoring.krum_scores(mixed,2)[-1] == 1


@pytest.mark.parametrize('f', [0,1,2,4])
def test_exact_fallback_preserves_distance_ties(f):
    matrix = np.full((2*f+3,3),np.finfo(float).max)
    selected, _ = scoring.multi_krum(matrix,f)
    assert selected.tolist() == [True]*(len(matrix)-f)+[False]*f
    np.testing.assert_array_equal(selected, decimal_krum(matrix,f))


@pytest.mark.parametrize('matrix', [
    np.array([[1e308,2e-300,1.], [1e308,3e-300,-1.], [-1e308,4e-300,0.], [0.,7e-300,3.], [-2.,1e-300,-4.]]),
    np.array([[1e-320,2e-320],[-1e-320,-2e-320],[3e-320,1e-320],[0.,0.],[1e-320,-3e-320]])])
def test_extreme_cosine_matches_high_precision_definition(matrix):
    assert scoring.cosine_sendros(matrix)[0].tolist() == decimal_cosine(matrix)


def test_registry_outputs_remain_finite_at_extreme_ranges(monkeypatch):
    import fltoken.rules as registry
    for name in ['cosine_sendros','krum_scores','multi_krum']:
        monkeypatch.setattr(registry,name,getattr(scoring,name))
    rules = registry.make_rules(dict(rules=['B0','B1','B2'],krum_f=2))
    matrix = np.array([[99e-200],[4e-200],[1e-200],[2e-200],[3e-200],[5e-200],[1e308]])
    results = registry.score_round(rules,np.arange(len(matrix)),matrix,1)
    for result in results.values():
        assert np.isfinite(result['scores']).all()
        assert np.isfinite(result['shares']).all()
        assert np.all(result['shares'] >= 0)
