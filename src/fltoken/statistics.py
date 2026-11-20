"""Exact small-seed bootstrap and conditional paired comparisons."""

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial, isfinite
import random
from statistics import stdev


def rational(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not isfinite(value):
        raise ValueError("statistical observations must be finite")
    return Fraction(float(value))


def weighted_quantile(values, probability):
    """Linear quantile of an integer-weighted empirical sample, exactly."""
    values = sorted(values)
    count = sum(weight for _, weight in values)
    position = (count-1)*probability
    lower = position.numerator//position.denominator
    fraction = position-lower
    wanted = [lower, min(lower+1, count-1)]
    found = []
    for rank in wanted:
        cumulative = 0
        for value, weight in values:
            cumulative += weight
            if cumulative > rank:
                found.append(value)
                break
    return (1-fraction)*found[0]+fraction*found[1]


def sample_sd(values):
    if len(values) < 2:
        return None
    return stdev(values)


@lru_cache(maxsize=32)
def bootstrap_indices(count, resamples, seed):
    rng = random.Random(seed)
    return tuple(Counter(tuple(sorted(rng.randrange(count) for _ in range(count)))
                         for _ in range(resamples)).items())


def sampled_bootstrap_mean(values, confidence=.95, resamples=20000, seed=20261120):
    """Fixed-seed percentile bootstrap, retaining exact means and sample SD."""
    values = [rational(value) for value in values]
    count = len(values)
    confidence = Fraction(str(confidence))
    if not 1 <= count <= 8 or not 0 < confidence < 1 or type(resamples) is not int or resamples < 1:
        raise ValueError("invalid seed bootstrap design")
    distribution = [(sum((values[index] for index in sample),Fraction(0))/count, weight)
                    for sample,weight in bootstrap_indices(count,resamples,seed)]
    tail = (1-confidence)/2
    return dict(mean=float(sum(values)/count), sample_sd=sample_sd(values),
                lower=float(weighted_quantile(distribution,tail)),
                upper=float(weighted_quantile(distribution,1-tail)), observations=count,
                bootstrap_samples=resamples, bootstrap_seed=seed, confidence=float(confidence),
                method="percentile; Python Random MT19937 indices; exact rational sample means")


def bootstrap_mean(values, confidence=.95):
    """Enumerate the complete bootstrap distribution, grouped by multiplicity.

    The small-seed design has no bootstrap RNG or Monte Carlo error. Values must
    contain every planned observation; callers retain undefined cells separately.
    """
    values = [rational(value) for value in values]
    n = len(values)
    confidence = Fraction(str(confidence))
    if not 1 <= n <= 8 or not 0 < confidence < 1:
        raise ValueError("exhaustive bootstrap requires one to eight observations and valid confidence")
    distribution = []
    for sample in combinations_with_replacement(range(n), n):
        weight = factorial(n)
        for frequency in Counter(sample).values():
            weight //= factorial(frequency)
        distribution.append((sum((values[i] for i in sample), Fraction(0))/n, weight))
    tail = (1-confidence)/2
    return dict(mean=float(sum(values)/n), sample_sd=sample_sd(values), lower=float(weighted_quantile(distribution, tail)),
                upper=float(weighted_quantile(distribution, 1-tail)), observations=n,
                bootstrap_samples=n**n, confidence=float(confidence))


def paired_wilcoxon(left, right):
    """Exact two-sided conditional signed-rank test, including tied ranks."""
    left, right = list(left), list(right)
    if not left or len(left) != len(right):
        raise ValueError("nonempty paired samples must have equal lengths")
    differences = [rational(a)-rational(b) for a, b in zip(left, right)]
    nonzero = sorted((abs(d), d > 0) for d in differences if d)
    doubled_ranks, positive, ties = [], 0, 0
    start = 0
    while start < len(nonzero):
        stop = start+1
        while stop < len(nonzero) and nonzero[stop][0] == nonzero[start][0]:
            stop += 1
        # Average of one-based ranks (start+1)..stop, multiplied by two.
        rank = start+1+stop
        doubled_ranks.extend([rank]*(stop-start))
        positive += rank*sum(sign for _, sign in nonzero[start:stop])
        ties += stop-start > 1
        start = stop
    total = sum(doubled_ranks)
    statistic = min(positive, total-positive)
    distribution = {0: 1}
    for rank in doubled_ranks:
        updated = distribution.copy()
        for value, count in distribution.items():
            updated[value+rank] = updated.get(value+rank, 0)+count
        distribution = updated
    probability = min(Fraction(1), Fraction(2*sum(count for value, count in distribution.items()
                                                  if value <= statistic), 2**len(nonzero)))
    return dict(statistic=statistic/2, positive_rank_sum=positive/2, negative_rank_sum=(total-positive)/2,
                p_value=float(probability), pairs=len(left), nonzero_pairs=len(nonzero),
                zero_pairs=len(left)-len(nonzero), tie_groups=ties,
                mean_difference=float(sum(differences)/len(differences)),
                rank_biserial=float(Fraction(2*positive-total,total)) if total else 0.,
                left_lower=sum(d < 0 for d in differences), left_higher=sum(d > 0 for d in differences))


def holm(p_values):
    """Adjust one complete family; missing comparisons retain their multiplicity."""
    supplied = {name: rational(value) for name, value in p_values.items() if value is not None}
    if any(not 0 <= value <= 1 for value in supplied.values()):
        raise ValueError("p-values must lie in the unit interval")
    adjusted = {name: None for name in p_values}
    previous = Fraction(0)
    for index, (name, value) in enumerate(sorted(supplied.items(), key=lambda item: (item[1], item[0]))):
        previous = max(previous, min(Fraction(1), (len(p_values)-index)*value))
        adjusted[name] = float(previous)
    return adjusted
