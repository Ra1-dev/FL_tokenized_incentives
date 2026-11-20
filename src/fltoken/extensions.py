"""Token-share adaptations of CGSV and RFFL with explicit bias diagnostics."""

from operator import index
from decimal import Decimal, localcontext
from math import fsum
import numpy as np


def _unit_rows(values):
    scale = np.max(np.abs(values), axis=1)
    scaled = np.divide(values, scale[:, None], out=np.zeros_like(values), where=scale[:, None] > 0)
    lengths = np.sqrt(np.einsum("ij,ij->i", scaled, scaled))
    return np.divide(scaled, lengths[:, None], out=np.zeros_like(scaled), where=lengths[:, None] > 0)


def _weights(values):
    largest = values.max()
    scaled = values/largest if largest else np.zeros_like(values)
    total = fsum(scaled)
    return scaled/total if total else scaled


def _tanh(values, beta):
    """Fixed Decimal precision avoids platform-specific vector-math rounding."""
    result = []
    with localcontext() as context:
        context.prec = 80
        coefficient = Decimal.from_float(beta)
        for value in values:
            x = coefficient*Decimal.from_float(float(value))
            # Beyond this argument the correctly rounded binary64 value is one.
            if x >= 20:
                result.append(1.)
                continue
            if x < Decimal("0.001"):
                # expm1's series keeps very small reputations instead of
                # cancelling them when exp(2*x) rounds to one at this precision.
                term = total = 2*x
                order = 1
                while term:
                    order += 1
                    term *= 2*x/order
                    updated = total+term
                    if updated == total:
                        break
                    total = updated
                exponential_minus_one = total
            else:
                exponential_minus_one = (2*x).exp()-1
            result.append(float(exponential_minus_one/(exponential_minus_one+2)))
    return np.array(result)


def inclusive_geometry(deltas, weights, gamma=.5):
    """Decompose cosine into its weighted self term and cross-client residual.

    Nonzero updates have norm gamma. Reference weights are nonnegative and
    normalized over the supplied sample. A zero reference has cosine zero,
    with an explicit false bias_defined flag rather than a defined bias claim.
    Unrepresentable self terms are also marked undefined, never emitted as inf.
    """
    matrix = np.asarray(deltas, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape) or not np.isfinite(matrix).all():
        raise ValueError("updates must be a nonempty finite matrix")
    if weights.shape != (len(matrix),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("reference weights must be finite, nonnegative and match the sample")
    if not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("normalization magnitude must lie in (0, 1]")
    unit = _unit_rows(matrix)
    normalized = _weights(weights)
    updates = gamma*unit
    reference = np.einsum("i,ij->j", normalized, updates)
    reference_unit = _unit_rows(reference[None, :])[0]
    reference_scale = np.max(np.abs(reference))
    reference_norm = float(reference_scale*np.linalg.norm(reference/reference_scale)) if reference_scale else 0.
    nonzero = np.any(matrix != 0, axis=1)
    measured = np.clip(unit@reference_unit, -1., 1.)
    defined = nonzero & (reference_norm > 0)
    with np.errstate(over="ignore", divide="ignore"):
        self_term = np.divide(normalized*gamma*nonzero, reference_norm,
                              out=np.zeros_like(normalized), where=reference_norm > 0)
    defined &= np.isfinite(self_term)
    self_term = np.where(defined,self_term,0.)
    excluded = reference[None, :]-normalized[:, None]*updates
    without_self = np.clip(np.einsum("ij,ij->i", unit, _unit_rows(excluded)), -1., 1.)
    return dict(cosine=measured, reference_weights=normalized, reference_norm=reference_norm,
                normalized_update_norm=gamma*nonzero, self_inclusion=self_term,
                cross_residual=measured-self_term, excluding_cosine=without_self,
                bias_defined=defined)


GEOMETRY_FIELDS = {"cosine","reference_weights","reference_norm","normalized_update_norm",
                   "self_inclusion","cross_residual","excluding_cosine","bias_defined"}


class _ReputationRule:
    def __init__(self, params, family):
        allowed = {"num_clients","alpha","gamma"}
        allowed |= {"beta","floor"} if family == "cgsv" else {"threshold_divisor"}
        if set(params)-allowed:
            raise ValueError("unknown reputation-rule parameter")
        self.num_clients = index(params["num_clients"])
        if isinstance(params["num_clients"],bool) or self.num_clients < 1:
            raise ValueError("registered client count must be positive")
        self.family = family
        self.alpha, self.gamma = float(params.get("alpha",.95)), float(params.get("gamma",.5))
        self.beta, self.floor = float(params.get("beta",1)), float(params.get("floor",.001))
        divisor = index(params.get("threshold_divisor",3))
        if isinstance(params.get("threshold_divisor",3),bool) or divisor < 1:
            raise ValueError("threshold divisor must be a positive integer")
        self.threshold = 1/(divisor*self.num_clients)
        if not np.isfinite([self.alpha,self.gamma,self.beta,self.floor,self.threshold]).all():
            raise ValueError("rule parameters must be finite")
        if not 0 <= self.alpha < 1 or not 0 < self.gamma <= 1 or self.beta < 1 or not 0 < self.floor <= 1:
            raise ValueError("invalid normalization, memory or reward parameters")
        if not 0 < self.threshold <= 1/self.num_clients:
            raise ValueError("reputation threshold must be positive and no larger than the equal proportion")
        # A declared partial-participation adaptation: identities start with
        # equal prior mass, including those not sampled in the first round.
        self.reputation = np.full(self.num_clients,1/self.num_clients)
        self.active = np.ones(self.num_clients,dtype=bool)
        self.last_round = 0
        self.undefined_rounds = 0

    def _clients(self, client_ids, round):
        ids = np.asarray(client_ids)
        if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer):
            raise ValueError("client ids must be a nonempty integer vector")
        if len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= self.num_clients).any():
            raise ValueError("client ids must be distinct registered identities")
        if isinstance(round,bool) or index(round) != self.last_round+1:
            raise ValueError("score each round exactly once in order")
        return ids

    def score(self, client_ids, deltas, round):
        ids = self._clients(client_ids,round)
        weights = np.where(self.active[ids],self.reputation[ids],0.)
        geometry = inclusive_geometry(deltas,weights,self.gamma)
        return self.score_recorded(ids,geometry,round)

    def score_recorded(self, client_ids, geometry, round):
        """Reconstruct allocations from an externally hash-validated trace.

        This verifies the reference's reputation weights. Raw matrix hashes and
        the trace's source/input bindings remain the recording pipeline's duty.
        """
        ids = self._clients(client_ids,round)
        if set(geometry) != GEOMETRY_FIELDS:
            raise ValueError("recorded geometry fields differ from the declared trace")
        geometry = {key:float(value) if key == "reference_norm" else np.asarray(value).copy()
                    for key,value in geometry.items()}
        for key,value in geometry.items():
            if not np.isfinite(value).all() or (key != "reference_norm" and value.shape != ids.shape):
                raise ValueError("recorded geometry must be finite and match the sampled identities")
        if geometry["bias_defined"].dtype != np.bool_:
            raise ValueError("bias definition flags must be boolean")
        if geometry["reference_norm"] < 0 or geometry["reference_norm"] > self.gamma*(1+2e-14):
            raise ValueError("recorded reference norm is outside normalized geometry")
        if any((np.abs(geometry[key]) > 1).any() for key in ["cosine","excluding_cosine"]):
            raise ValueError("recorded cosine is outside its mathematical range")
        expected = _weights(np.where(self.active[ids],self.reputation[ids],0.))
        if not np.array_equal(geometry["reference_weights"],expected):
            raise ValueError("recorded reference weights differ from the current reputation state")
        norms = geometry["normalized_update_norm"]
        if not ((norms == 0)|(norms == self.gamma)).all():
            raise ValueError("normalized update norms must be zero or the declared magnitude")
        with np.errstate(over="ignore",divide="ignore"):
            own = np.divide(geometry["reference_weights"]*norms,geometry["reference_norm"],
                            out=np.zeros(len(ids)),where=geometry["reference_norm"] > 0)
        defined = (norms > 0)&(geometry["reference_norm"] > 0)&np.isfinite(own)
        own = np.where(defined,own,0.)
        if not np.array_equal(geometry["bias_defined"],defined) or not np.array_equal(geometry["self_inclusion"],own):
            raise ValueError("recorded self-inclusion differs from weights and norms")
        if not np.array_equal(geometry["cross_residual"],geometry["cosine"]-own):
            raise ValueError("recorded cross residual does not complete the cosine decomposition")
        if (geometry["cosine"][norms == 0] != 0).any() or (geometry["reference_norm"] == 0 and geometry["cosine"].any()):
            raise ValueError("zero-vector cosine must be zero")
        eligible_before = self.active[ids].copy()
        previous = self.reputation.copy()
        proposed = previous.copy()
        observed = ids[eligible_before]
        proposed[observed] = self.alpha*previous[observed]+(1-self.alpha)*geometry["cosine"][eligible_before]
        if self.family == "cgsv":
            proposed[observed] = np.maximum(proposed[observed],self.floor)
        mass = fsum(proposed)
        undefined = bool(len(observed) and mass <= 0)
        removed = np.zeros(len(ids),dtype=bool)
        if undefined:
            # Signed normalization has no valid positive-mass interpretation.
            # Preserve history, pay nothing and expose the invalid round so a
            # complete empirical comparison cannot silently use this fallback.
            self.undefined_rounds += 1
        elif mass > 0:
            proposed /= mass
            if self.family == "rffl":
                removed = eligible_before & (proposed[ids] < self.threshold)
                self.active[ids[removed]] = False
                proposed[~self.active] = 0.
                surviving_mass = fsum(proposed)
                if surviving_mass:
                    proposed /= surviving_mass
            self.reputation = proposed
        quota = _tanh(self.reputation,self.beta) if self.family == "cgsv" else self.reputation.copy()
        maximum = quota.max()
        quota = quota[ids]/maximum if maximum else np.zeros(len(ids))
        quota = np.where(self.active[ids],quota,0.)
        total = fsum(quota)
        shares = quota/total if total and geometry["reference_norm"] > 0 and not undefined else np.zeros(len(ids))
        self.last_round = index(round)
        return dict(**geometry, scores=geometry["cosine"].copy(), shares=shares, quota=quota,
                    slash=np.zeros(len(ids),dtype=bool), removed=removed, eligible=self.active[ids].copy(),
                    reputation_before=previous[ids], reputation_after=self.reputation[ids].copy(),
                    undefined_state=undefined,
                    state=dict(round=self.last_round,reputation=self.reputation.tolist(),
                               active=self.active.tolist(),undefined_rounds=self.undefined_rounds))


class CGSVScorer(_ReputationRule):
    """Approximate cosine value, EMA importance and normalized tanh quotas."""

    def __init__(self, params):
        super().__init__(params,"cgsv")


class RFFLScorer(_ReputationRule):
    """Reputation quotas with permanent exclusion from the reward reference."""

    def __init__(self, params):
        super().__init__(params,"rffl")
