"""Uncertainty calculations must retain the seed unit and exact paired null."""

from fractions import Fraction
from itertools import product

import pytest

from fltoken.statistics import bootstrap_mean, holm, paired_wilcoxon, sampled_bootstrap_mean


def test_grouped_bootstrap_matches_every_ordered_resample():
    values = [Fraction(1, 10), Fraction(1, 5), Fraction(9, 10)]
    draws = sorted(sum(sample)/len(sample) for sample in product(values, repeat=len(values)))
    def quantile(p):
        position = (len(draws)-1)*p
        lower = position.numerator//position.denominator
        weight = position-lower
        return float(draws[lower]*(1-weight)+draws[min(lower+1,len(draws)-1)]*weight)
    result = bootstrap_mean(values, .5)
    assert result['lower'] == quantile(Fraction(1, 4))
    assert result['upper'] == quantile(Fraction(3, 4))
    assert result == bootstrap_mean(list(reversed(values)), .5)
    assert result['bootstrap_samples'] == 27
    equal = bootstrap_mean([7]*5)
    assert equal['lower'] == equal['upper'] == 7
    assert equal['bootstrap_samples'] == 3125
    with pytest.raises((TypeError, ValueError)):
        bootstrap_mean([1, None, 3])


@pytest.mark.parametrize('differences', [[1,1,-2,0], [1,-3,3,4,-5], [1,2,3,4,5], [0,0]])
def test_signed_rank_matches_enumerated_conditional_signs(differences):
    magnitudes = sorted(abs(d) for d in differences if d)
    ranks = {value: Fraction(sum(i+1 for i,x in enumerate(magnitudes) if x == value), magnitudes.count(value))
             for value in magnitudes}
    observed = sum(ranks[abs(d)] for d in differences if d > 0)
    total = sum(ranks[value] for value in magnitudes)
    cutoff = min(observed, total-observed)
    signs = list(product([0,1], repeat=len(magnitudes)))
    extreme = sum(sum(sign*ranks[value] for sign,value in zip(pattern,magnitudes)) <= cutoff for pattern in signs)
    expected = float(min(Fraction(1), Fraction(2*extreme, len(signs))))
    result = paired_wilcoxon(differences, [0]*len(differences))
    assert result['p_value'] == expected
    assert result['statistic'] == float(cutoff)
    assert paired_wilcoxon([1]*5,[0]*5)['p_value'] == .0625
    assert paired_wilcoxon([0]*3,[0]*3)['p_value'] == 1


def test_holm_retains_the_complete_family_and_order():
    observed = holm({'a':Fraction(1,100),'b':Fraction(4,100),'c':Fraction(3,100)})
    assert observed == {'a':.03,'b':.06,'c':.06}
    assert holm({'a':Fraction(1,100),'b':None,'c':Fraction(3,100)}) == {'a':.03,'b':None,'c':.06}
    with pytest.raises(ValueError):
        holm({'bad':1.1})


def test_fixed_seed_bootstrap_matches_uncompressed_draws():
    import random
    from math import sqrt
    from fltoken.statistics import weighted_quantile
    values=[Fraction(1,10),Fraction(1,3),Fraction(8,10),Fraction(9,10),Fraction(1)]
    rng=random.Random(137)
    draws=[sum((values[rng.randrange(5)] for _ in range(5)),Fraction(0))/5 for _ in range(10000)]
    result=sampled_bootstrap_mean(values,resamples=10000,seed=137)
    assert result["lower"] == float(weighted_quantile([(v,1) for v in draws],Fraction(1,40)))
    assert result["upper"] == float(weighted_quantile([(v,1) for v in draws],Fraction(39,40)))
    assert result["mean"] == float(sum(values)/5)
    average=sum(values)/len(values)
    assert result["sample_sd"] == pytest.approx(sqrt(float(sum((v-average)**2 for v in values)/(len(values)-1))))
    assert result == sampled_bootstrap_mean(values,resamples=10000,seed=137)
