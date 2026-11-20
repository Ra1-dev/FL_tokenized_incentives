"""Relative allocation survives positive-weight underflow without changing state."""
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from fltoken.metrics import payout, ROUND_POOL
from fltoken.ours import OursScorer


def decimal_allocation(quality, reputation, slash):
    # Independently evaluate the specified product and normalization at enough
    # precision to cover the entire binary64 exponent range.
    with localcontext() as context:
        context.prec = 2500
        weights = [Decimal.from_float(float(q))*Decimal.from_float(float(r)) if not flag else Decimal(0)
                   for q, r, flag in zip(quality, reputation, slash)]
        total = sum(weights, Decimal(0))
        return np.array([float(weight/total) if total else 0. for weight in weights])


@pytest.mark.parametrize('beta',[0.,.5,.9,1.])
def test_positive_near_orthogonal_updates_receive_rewards(beta):
    small = np.nextafter(0.,1.)
    matrix = np.array([[small,1.],[1.,0.]])
    result = OursScorer(dict(beta=beta)).score([3,7],matrix,1)
    np.testing.assert_array_equal(result['scores'],[small,small])
    np.testing.assert_array_equal(result['shares'],[.5,.5])
    assert not result['slash'].any() and result['selected'].all()
    units, amounts = payout(result['shares'])
    assert units == [10**18//2]*2 and amounts == [ROUND_POOL//2]*2


@pytest.mark.parametrize('beta',[0.,.5,.9,1.])
@pytest.mark.parametrize('initial',[0.,np.nextafter(0.,1.),.5,1.])
def test_mixed_exponents_match_independent_product_normalization(beta,initial):
    rng = np.random.default_rng(726253)
    scorer = OursScorer(dict(beta=beta,initial_reputation=initial,penalties=False))
    for number in range(1,9):
        ids = rng.choice(17,7,replace=False)
        quality = np.ldexp(rng.integers(1,8,7).astype(float),rng.choice([-1074,-1070,-1025,-1022,-600,-10],7))
        quality[0] = np.nextafter(0.,1.)
        before = quality.copy()
        quality.flags.writeable = False
        result = scorer.allocate(ids,quality,number)
        reputation = [result['state']['reputation'][int(client)] for client in ids]
        expected = decimal_allocation(quality,reputation,result['slash'])
        # This trace always includes a positive subnormal product unless all
        # recorded reputations are zero. Ordinary products retain their path.
        np.testing.assert_allclose(result['shares'],expected,rtol=3e-15,atol=0.)
        np.testing.assert_array_equal(result['shares']==0,expected==0)
        np.testing.assert_array_equal(quality,before)
        assert math.fsum(result['shares']) == pytest.approx(1. if any(reputation) else 0.,abs=2e-16)


def test_partial_subnormal_rounding_preserves_unequal_weights():
    small = np.nextafter(0.,1.)
    result = OursScorer(dict(beta=1.,initial_reputation=.5)).allocate([0,1],[small,3*small],1)
    np.testing.assert_array_equal(result['shares'],[.25,.75])


def test_absence_permutation_and_penalties_preserve_exact_state():
    small = np.nextafter(0.,1.)
    settings = dict(beta=.5,threshold=small,patience=2)
    left,right = OursScorer(settings),OursScorer(settings)
    rounds = [([3,7,9],[small,2*small,0.]),([7,9],[4*small,0.]),
              ([3,7,9],[small,3*small,small]),([3,7,9],[2*small,small,small])]
    old_snapshot = None
    for number,(ids,values) in enumerate(rounds,1):
        quality = np.array(values)
        first = left.allocate(ids,quality,number)
        second = right.allocate(ids[::-1],quality[::-1],number)
        for field in ['scores','shares','slash','selected']:
            np.testing.assert_array_equal(first[field],second[field][::-1])
        assert first['state'] == second['state']
        expected = decimal_allocation(quality,[first['state']['reputation'][client] for client in ids],first['slash'])
        np.testing.assert_array_equal(first['shares'],expected)
        if number == 1:
            old_snapshot = first['state']
        if number == 2:
            assert first['state']['reputation'][3] == old_snapshot['reputation'][3]
            assert first['state']['low_count'][3] == 1
        if number == 3:
            assert first['slash'].tolist() == [True,False,False]
    assert old_snapshot['last_round'] == 1 and old_snapshot['low_count'][3] == 1


@pytest.mark.parametrize('params,quality,expected',[
    (dict(beta=0.),[0.,0.],[0.,0.]),
    (dict(beta=1.,initial_reputation=0.),[.5,1.],[0.,0.]),
    (dict(threshold=1.,patience=1),[np.nextafter(0.,1.)]*2,[0.,0.]),
    (dict(ema=False),[np.nextafter(0.,1.)]*2,[.5,.5]),
    (dict(beta=0.),[np.nextafter(0.,1.),1.],[0.,1.]),
])
def test_zero_state_disabled_ema_and_unrepresentable_share(params,quality,expected):
    result = OursScorer(params).allocate([0,1],quality,1)
    np.testing.assert_array_equal(result['shares'],expected)
    np.testing.assert_array_equal(result['selected'],np.array(expected)>0)


def test_subnormal_share_is_not_clipped_to_zero_or_promoted():
    small = np.nextafter(0.,1.)
    result = OursScorer(dict(beta=1.,initial_reputation=small)).allocate([0,1],[small,1.],1)
    np.testing.assert_array_equal(result['shares'],[small,1.])
    units, amounts = payout(result['shares'])
    # Binary64 can represent a share which the integer token unit cannot.
    assert units == [0,10**18-1] and sum(amounts) < ROUND_POOL


def test_ordinary_stateful_allocations_keep_original_float_arithmetic():
    rng = np.random.default_rng(628732)
    for beta in [0.,.5,.9,1.]:
        scorer = OursScorer(dict(beta=beta,threshold=.2,patience=2))
        reputation,counts = {},{}
        for number in range(1,13):
            ids = rng.choice(23,10,replace=False)
            quality = rng.random(10)
            weights = quality.copy()
            flags = []
            for index,client in enumerate(ids):
                reputation[client] = beta*reputation.get(client,.5)+(1-beta)*quality[index]
                weights[index] *= reputation[client]
                counts[client] = counts.get(client,0)+1 if quality[index] <= .2 else 0
                flags.append(counts[client] >= 2)
                if flags[-1]:
                    weights[index] = 0.
                    counts[client] = 0
            expected = weights/math.fsum(weights)
            result = scorer.allocate(ids,quality,number)
            np.testing.assert_array_equal(result['shares'],expected)
            np.testing.assert_array_equal(result['slash'],flags)
            assert result['state']['reputation'] == reputation
            assert result['state']['low_count'] == counts


def test_invalid_cached_quality_does_not_modify_prior_state():
    scorer = OursScorer()
    before = scorer.allocate([3,7],[np.nextafter(0.,1.)]*2,1)['state']
    with pytest.raises(ValueError):
        scorer.allocate([3,3],[np.nextafter(0.,1.)]*2,2)
    assert scorer.last_round == before['last_round']
    assert scorer.reputation == before['reputation'] and scorer.low_count == before['low_count']
