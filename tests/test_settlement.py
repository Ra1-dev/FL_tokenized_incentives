"""Conservation and exact conversion at the off-chain settlement boundary."""

from fractions import Fraction

import numpy as np
import pytest

from fltoken.metrics import ROUND_POOL, SCALE, payout
from fltoken.scoring import cosine_sendros, equal_split, multi_krum


def test_random_matrix_settlement_never_overshoots():
    rng = np.random.default_rng(613)
    positive_rounds = empty_rounds = 0
    for case in range(1000):
        k, d = int(rng.integers(5, 31)), int(rng.integers(1, 65))
        matrix = rng.normal(size=(k, d))
        if case % 11 == 0:
            matrix[:] = 0
        elif case % 7 == 0:
            matrix[1:] = matrix[0]
        elif case % 5 == 0:
            matrix[1:] = 0
        f = int(rng.integers(0, (k - 3) // 2 + 1))
        for shares in [equal_split(matrix), cosine_sendros(matrix)[1], multi_krum(matrix, f)[1]]:
            original = shares.copy()
            units, amounts = payout(shares)
            assert all(type(unit) is int and unit >= 0 for unit in units)
            assert sum(units) <= SCALE
            assert 0 <= sum(amounts) <= ROUND_POOL
            assert all(amount == ROUND_POOL * unit // SCALE for unit, amount in zip(units, amounts))
            rational = list(map(lambda x: Fraction(float(x)), shares))
            total = sum(rational)
            if total:
                positive_rounds += 1
                assert 0 <= SCALE - sum(units) < k
                for value, unit in zip(rational, units):
                    assert unit <= value / total * SCALE < unit + 1
            else:
                empty_rounds += 1
                assert not any(units) and not any(amounts)
            np.testing.assert_array_equal(shares, original)
    assert positive_rounds and empty_rounds


def test_conversion_preserves_exact_floats_and_zero_rounds():
    assert payout(np.array([.1] * 10))[0] == [SCALE // 10] * 10
    assert payout(np.zeros(10)) == ([0] * 10, [0] * 10)
    assert payout(np.array([0., 1., 0.]))[0] == [0, SCALE, 0]
    assert payout(np.array([1., 3.]))[0] == [SCALE // 4, 3 * SCALE // 4]
    # Exact normalized binary floats need not equal the source integer ratio.
    shares = np.array([1 / 3, 2 / 3])
    assert payout(shares)[0] == [SCALE // 3, 2 * SCALE // 3]
    q = np.array([1, 4, 7])
    units = payout(q / q.sum())[0]
    exact = [Fraction(float(x)) for x in q / q.sum()]
    assert units == [int(x * SCALE // sum(exact)) for x in exact]


@pytest.mark.parametrize("shares", [[-1, 2], [np.nan], [np.inf], [[.5, .5]]])
def test_invalid_shares_are_rejected(shares):
    with pytest.raises(ValueError, match="finite nonnegative vector"):
        payout(shares)
