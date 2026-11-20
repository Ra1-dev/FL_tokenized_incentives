"""Clipped-update noise and analytic accounting for individually visible releases.

The accountant describes an ideal Gaussian mechanism under replacement of one
client's data with fixed public registration and record counts. It claims no
sampling amplification. Seeded floating-point simulations are not certified
privacy mechanisms. Composition and conversion follow Mironov (CSF 2017),
https://arxiv.org/abs/1702.07476, Propositions 1, 3 and 7.
"""

from decimal import Decimal, localcontext
import math
from operator import index

import numpy as np


def validate_privacy(settings):
    """Require an explicit fixed scale, multiplier and accounting delta."""
    keys = {"clipping_bound", "noise_multiplier", "delta"}
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError("privacy settings require clipping_bound, noise_multiplier and delta")
    result = {}
    for key in sorted(keys):
        value = settings[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{key} must be a finite real number")
        result[key] = float(value)
        if not math.isfinite(result[key]):
            raise ValueError(f"{key} must be a finite real number")
    if result["clipping_bound"] <= 0 or result["noise_multiplier"] < 0 or not 0 < result["delta"] < 1:
        raise ValueError("privacy requires C>0, sigma>=0 and delta in (0,1)")
    if not math.isfinite(result["clipping_bound"]*result["noise_multiplier"]):
        raise ValueError("noise standard deviation exceeds the numerical range")
    if result["noise_multiplier"] > 0 and result["clipping_bound"]*result["noise_multiplier"] == 0:
        raise ValueError("positive noise standard deviation underflows the numerical range")
    if not math.isfinite(2*result["clipping_bound"]):
        raise ValueError("replacement sensitivity exceeds the numerical range")
    return result


def release_update(delta, config, client, round):
    """Apply the reproducible simulation mechanism before a client's reply."""
    if "privacy" not in config:
        return delta
    from fltoken.data import seed_for
    settings = validate_privacy(config["privacy"])
    rng = np.random.default_rng(seed_for(config["seed"], round, client, 4))
    return perturb_update(delta, settings["clipping_bound"], settings["noise_multiplier"], rng)


def account_schedule(config, observed_ids):
    """Account only for the public release schedule, never raw private norms."""
    settings = validate_privacy(config["privacy"])
    num_clients, k, rounds = (config[key] for key in ["num_clients", "clients_per_round", "rounds"])
    if len(observed_ids) != rounds:
        raise ValueError("privacy accounting requires every completed round")
    counts = np.zeros(num_clients, dtype=np.int64)
    for ids in observed_ids:
        if (len(ids) != k or len(set(ids)) != k or
                any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or not 0 <= i < num_clients for i in ids)):
            raise ValueError("release schedule contains invalid client identities")
        counts[list(ids)] += 1
    budgets = [dict(client_id=i, **privacy_budget(int(count), settings["noise_multiplier"], settings["delta"]))
               for i, count in enumerate(counts)]
    return dict(schema=1, settings=settings, clients=budgets,
                conservative_all_rounds=privacy_budget(rounds, settings["noise_multiplier"], settings["delta"]),
                sensitivity=2*settings["clipping_bound"],
                noise_standard_deviation=settings["noise_multiplier"]*settings["clipping_bound"],
                adjacency="replacement of one client's complete data with fixed public identities, partition and record counts",
                released_transcript="identified clipped and perturbed client updates, their post-processing and public sampling schedule",
                sampling_amplification=False,
                simulation_only=True,
                limitation="Public-seeded floating-point draws do not provide deployment privacy; bounds describe ideal undisclosed Gaussian randomness conditional on a clipping bound and auxiliary protocol metadata fixed independently of the protected data. Public-benchmark pilot calibration is not included in this accountant")


def clip_update(delta, bound):
    """Project the complete update onto an L2 ball without overflowing its norm."""
    vector = np.asarray(delta, dtype=np.float64)
    bound = float(bound)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("a finite nonempty flattened update is required")
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("clipping bound must be positive and finite")
    scale = np.max(np.abs(vector))
    if not scale:
        return vector.copy()
    unit_scale = vector/scale
    scaled_norm = np.linalg.norm(unit_scale)
    if scale <= bound/scaled_norm:
        return vector.copy()
    return unit_scale*(bound/scaled_norm)


def perturb_update(delta, bound, noise_multiplier, rng):
    """Add noise with per-coordinate SD sigma*C, without dimension rescaling."""
    sigma = float(noise_multiplier)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("noise multiplier must be finite and nonnegative")
    clipped = clip_update(delta, bound)
    if sigma == 0:
        return clipped
    standard_deviation = sigma*float(bound)
    if not math.isfinite(standard_deviation):
        raise ValueError("noise standard deviation exceeds the numerical range")
    if not standard_deviation:
        raise ValueError("positive noise standard deviation underflows the numerical range")
    result = clipped+rng.normal(0., standard_deviation, size=clipped.shape)
    if not np.isfinite(result).all():
        raise ValueError("perturbed update exceeds the numerical range")
    return result


def _inputs(participations, noise_multiplier):
    count = index(participations)
    sigma = float(noise_multiplier)
    if isinstance(participations, (bool, np.bool_)) or count < 0:
        raise ValueError("participations must be a nonnegative integer")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("noise multiplier must be finite and nonnegative")
    return count, sigma


def _upper(value):
    result = float(value)
    if not math.isfinite(result):
        raise OverflowError("the finite privacy bound exceeds binary64 range")
    return math.nextafter(result, math.inf) if value > 0 else 0.


def gaussian_rdp(order, noise_multiplier, participations):
    """RDP at one order for sensitivity 2C and noise SD sigma*C per release."""
    count, sigma = _inputs(participations, noise_multiplier)
    order = float(order)
    if not math.isfinite(order) or order <= 1:
        raise ValueError("Renyi order must be finite and greater than one")
    if not count:
        return 0.
    if not sigma:
        return math.inf
    with localcontext() as context:
        context.prec = 80
        return _upper(2*Decimal(count)*Decimal.from_float(order)/(Decimal.from_float(sigma)**2))


def privacy_budget(participations, noise_multiplier, delta):
    """Minimize the standard RDP conversion over all real orders above one.

For rho=2*t/sigma**2, minimize rho*a + log(1/delta)/(a-1).
The analytic optimum is a=1+sqrt(log(1/delta)/rho). The bound is
rho+2*sqrt(rho*log(1/delta)), with upward-rounded binary64 serialization.
This is an upper bound, not an exact privacy-loss distribution calculation.
"""
    count, sigma = _inputs(participations, noise_multiplier)
    delta = float(delta)
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    result = dict(participations=count, noise_multiplier=sigma, delta=delta,
                  accountant="unsampled Gaussian RDP; replacement sensitivity twice the clipping bound",
                  sampling_amplification=False)
    if not count:
        return dict(**result, epsilon=0., rdp_slope=0., optimal_order=None, finite=True)
    if not sigma:
        return dict(**result, epsilon=None, rdp_slope=None, optimal_order=None, finite=False)
    with localcontext() as context:
        context.prec = 80
        rho = 2*Decimal(count)/(Decimal.from_float(sigma)**2)
        logarithm = -Decimal.from_float(delta).ln()
        optimum = 1+(logarithm/rho).sqrt()
        epsilon = _upper(rho+2*(rho*logarithm).sqrt())
        order = float(optimum)
        return dict(**result, epsilon=epsilon, rdp_slope=_upper(rho),
                    optimal_order=order if math.isfinite(order) else None, finite=True)
