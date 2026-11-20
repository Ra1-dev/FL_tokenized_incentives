"""Leave-one-out median rewards with participation-based reputation and penalties."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def _unit(vector):
    scale = np.max(np.abs(vector))
    if scale == 0:
        return np.zeros_like(vector)
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _range_safe_mean(x):
    with np.errstate(over="ignore", under="ignore"):
        result = (x / len(x)).sum(axis=0)
    small = ((x != 0) & (np.abs(x) < np.finfo(np.float64).tiny * len(x))).any(axis=0)
    for index in np.flatnonzero(small | ~np.isfinite(result)):
        result[index] = float(sum((Fraction(float(value)) for value in x[:, index]), Fraction(0)) / len(x))
    return result


def _range_safe_midpoint(a, b):
    result = .5*a + .5*b
    small = ((a != 0) & (np.abs(a) < 2*np.finfo(np.float64).tiny)) | ((b != 0) & (np.abs(b) < 2*np.finfo(np.float64).tiny))
    for index in np.flatnonzero(small):
        result[index] = float((Fraction(float(a[index])) + Fraction(float(b[index]))) / 2)
    return result


def _integer_rows(matrix):
    # Binary64 coordinates are dyadic rationals. A common power-of-two unit
    # represents them exactly; its value cancels out of every cosine.
    ratios = [[float(value).as_integer_ratio() for value in row] for row in matrix]
    exponent = max(denominator.bit_length()-1 for row in ratios for numerator, denominator in row)
    return [[numerator << (exponent-denominator.bit_length()+1) for numerator, denominator in row] for row in ratios]


def _integer_reference(rows, index, median):
    selected = [row for position, row in enumerate(rows) if not median or position != index]
    if not selected:
        return [0]*len(rows[index])
    if not median:
        # Dividing every coordinate by the same positive count leaves direction unchanged.
        return [sum(column) for column in zip(*selected)]
    middle = len(selected)//2
    ordered = [sorted(column) for column in zip(*selected)]
    if len(selected)%2:
        return [column[middle] for column in ordered]
    # Likewise, the midpoint's common factor one-half cancels from the cosine.
    return [column[middle-1]+column[middle] for column in ordered]


def _integer_quality(row, reference):
    """Round the exact positive cosine once, including the subnormal boundary."""
    dot = sum(a*b for a, b in zip(row,reference))
    if dot <= 0:
        return 0.
    numerator = dot*dot
    denominator = sum(a*a for a in row)*sum(b*b for b in reference)
    exponent = numerator.bit_length()-denominator.bit_length()
    below = numerator < denominator<<exponent if exponent >= 0 else numerator<<-exponent < denominator
    if below:
        exponent -= 1
    # The square root's binade determines its binary64 spacing. Compare
    # squared midpoints using integers to avoid a second floating conversion.
    spacing = max(-1074, exponent//2-52)
    scaled = numerator << (-2*spacing)
    lower = math.isqrt(scaled//denominator)
    midpoint = denominator*(2*lower+1)**2
    rounded = lower + (4*scaled>midpoint or 4*scaled==midpoint and lower%2==1)
    return min(1., math.ldexp(float(rounded), spacing))


class OursScorer:
    """State is keyed by scientific client id, never by sampled-row position.

    Defaults are design priors for development. Empirical parameter selection
    must use a separately recorded pilot corpus before evaluation.
    """

    DEFAULTS = dict(reference=True, ema=True, penalties=True, beta=.5,
                    threshold=0., patience=3, initial_reputation=.5)

    def __init__(self, params=None):
        params = {} if params is None else dict(params)
        if params.keys() - self.DEFAULTS.keys():
            raise ValueError("unknown reward parameters")
        self.params = {**self.DEFAULTS, **params}
        for key in ["reference", "ema", "penalties"]:
            if type(self.params[key]) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ["beta", "threshold", "initial_reputation"]:
            value = self.params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and within [0, 1]")
        patience = self.params["patience"]
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        self.reputation = {}
        self.low_count = {}
        self.last_round = 0

    def references(self, deltas):
        """Return float64 approximations of the unnormalized reference coordinates.

        A tiny midpoint can round to zero in this diagnostic representation.
        Scoring retains the exact averaged direction before computing cosine.
        """
        x = np.asarray(deltas, dtype=np.float64)
        if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
            raise ValueError("updates must be a finite nonempty matrix")
        if not self.params["reference"]:
            # Dividing first avoids overflow for a representable mean.
            return np.broadcast_to(_range_safe_mean(x), x.shape).copy()
        refs = np.zeros_like(x)
        if len(x) == 1:
            return refs
        for i in range(len(x)):
            others = np.sort(np.delete(x, i, axis=0), axis=0)
            middle = len(others) // 2
            refs[i] = others[middle] if len(others) % 2 else _range_safe_midpoint(others[middle-1], others[middle])
        return refs

    def score(self, client_ids, deltas, round):
        x = np.asarray(deltas, dtype=np.float64)
        refs = self.references(x)
        quality = []
        integer_rows = None
        inclusive_reference = None
        for index, (row, ref) in enumerate(zip(x, refs)):
            a, b = _unit(row), _unit(ref)
            dot = float(a@b)
            # Cancellation can leave a positive rounding residue at exact zero.
            # The absolute term also covers products whose relative error bound
            # underflows. This selects exact arithmetic; it never clips quality.
            error_bound = 4*len(row)*(np.finfo(float).eps*float(np.sum(np.abs(a*b))) + np.nextafter(0.,1.))
            averaging = not self.params["reference"] or len(x) > 1 and len(x)%2
            exceptional = averaging or (abs(dot) <= error_bound and np.any(row) and np.any(ref))
            if exceptional:
                if integer_rows is None:
                    integer_rows = _integer_rows(x)
                if not self.params["reference"]:
                    if inclusive_reference is None:
                        inclusive_reference = _integer_reference(integer_rows, index, False)
                    exact_reference = inclusive_reference
                else:
                    exact_reference = _integer_reference(integer_rows, index, True)
                value = _integer_quality(integer_rows[index], exact_reference)
            else:
                value = max(0., float(np.clip(dot, -1, 1)))
            quality.append(value)
        return self.allocate(client_ids, np.array(quality), round)

    def allocate(self, client_ids, quality, round):
        """Apply state transitions to precomputed qualities from the same reference.

        Pilot candidates share reference geometry, so computing it once avoids
        repeating median and cosine calculations for every economic parameter.
        """
        quality = np.asarray(quality, dtype=np.float64)
        if quality.ndim != 1 or not len(quality) or not np.isfinite(quality).all() or ((quality < 0) | (quality > 1)).any():
            raise ValueError("qualities must be a finite nonempty vector within [0, 1]")
        ids = list(client_ids)
        if len(ids) != len(quality) or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in ids):
            raise ValueError("one integer client id is required per update")
        ids = [int(i) for i in ids]
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("client ids must be distinct and nonnegative")
        if isinstance(round, (bool, np.bool_)) or not isinstance(round, (int, np.integer)) or round <= self.last_round:
            raise ValueError("round numbers must strictly increase")
        weights = quality.copy()
        slash = np.zeros(len(quality), dtype=bool)
        for index, client in enumerate(ids):
            prior = self.reputation.get(client, self.params["initial_reputation"])
            reputation = self.params["beta"]*prior + (1-self.params["beta"])*quality[index]
            self.reputation[client] = float(reputation)
            if self.params["ema"]:
                weights[index] *= reputation
            low = self.low_count.get(client, 0)+1 if quality[index] <= self.params["threshold"] else 0
            if self.params["penalties"] and low >= self.params["patience"]:
                slash[index] = True
                weights[index] = 0
                low = 0
            self.low_count[client] = low
        total = math.fsum(weights)
        shares = weights / total if total else np.zeros(len(quality))
        self.last_round = int(round)
        return dict(shares=shares, scores=quality, selected=shares > 0, slash=slash,
                    state=dict(reputation=self.reputation.copy(), low_count=self.low_count.copy(), last_round=self.last_round))
