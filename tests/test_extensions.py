"""Analytic bias identities for normalized reputation-weighted references."""

import numpy as np
import pytest

from fltoken.extensions import CGSVScorer, GEOMETRY_FIELDS, RFFLScorer, inclusive_geometry, _tanh


def test_quota_transform_preserves_tiny_positive_reputations():
    values = np.array([0., 5e-324, 1e-250, 1e-100, 1e-20, 1e-8, .001, .2, 1.])
    result = _tanh(values, 1.)
    np.testing.assert_array_equal(result[:6], values[:6])
    np.testing.assert_allclose(result[6:], np.tanh(values[6:]), atol=2e-16, rtol=0)
    assert _tanh([1.], 20.)[0] == 1.


def test_orthogonal_updates_have_exact_weighted_self_inclusion():
    weights = np.array([1.,2.,3.,4.])
    expected = weights/np.linalg.norm(weights)
    output = inclusive_geometry(np.eye(4),weights)
    np.testing.assert_allclose(output["cosine"],expected,atol=2e-15,rtol=0)
    np.testing.assert_allclose(output["self_inclusion"],expected,atol=2e-15,rtol=0)
    np.testing.assert_allclose(output["cross_residual"],0,atol=2e-15,rtol=0)
    np.testing.assert_array_equal(output["excluding_cosine"],0)
    assert output["bias_defined"].all()


def test_correlated_updates_retain_the_cross_term_and_excluding_counterfactual():
    x = np.array([[3.,4.,0.],[1.,0.,2.],[-2.,1.,1.],[0.,0.,0.]])
    weights = np.array([.2,.5,.3,0.])
    before = x.copy()
    result = inclusive_geometry(x,weights)
    unit = np.zeros_like(x)
    unit[:3] = x[:3]/np.linalg.norm(x[:3],axis=1)[:,None]
    reference = sum(weight*row for weight,row in zip(weights,unit))
    for i in range(3):
        others = sum(weights[j]*unit[j] for j in range(4) if j != i)
        cross = np.dot(unit[i],others)/np.linalg.norm(reference)
        assert result["cross_residual"][i] == pytest.approx(cross,abs=2e-15)
        assert result["excluding_cosine"][i] == pytest.approx(np.dot(unit[i],others)/np.linalg.norm(others),abs=2e-15)
    assert np.max(np.abs(result["cross_residual"])) > .1
    np.testing.assert_array_equal(x,before)
    assert not result["bias_defined"][-1]


def test_geometry_is_invariant_to_scale_and_equivariant_to_sample_order():
    rng = np.random.default_rng(713)
    x, weights = rng.normal(size=(7,31)), rng.uniform(.1,1,size=7)
    expected = inclusive_geometry(x,weights)
    permutation = rng.permutation(7)
    for scale in [1e-240,1e240]:
        result = inclusive_geometry(x[permutation]*scale,weights[permutation]*scale)
        for key in ["cosine","reference_weights","self_inclusion","cross_residual","excluding_cosine"]:
            np.testing.assert_allclose(result[key],expected[key][permutation],atol=3e-14,rtol=0)


def test_zero_and_cancelled_references_remain_explicit():
    for x, weights in [(np.zeros((3,5)),[1,1,1]),(np.array([[1.,0.],[-1.,0.]]),[1,1]),
                       (np.eye(3),[0,0,0])]:
        result = inclusive_geometry(x,weights)
        assert result["reference_norm"] == 0
        assert not result["bias_defined"].any()
        np.testing.assert_array_equal(result["cosine"],0)
        for value in result.values():
            assert np.isfinite(value).all()
    with pytest.raises(ValueError,match="weights"):
        inclusive_geometry(np.eye(2),[1,-1])


def test_near_cancellation_does_not_underflow_the_reference_norm():
    x = np.array([[1.,0.],[-1.,0.],[0.,1.]])
    finite = inclusive_geometry(x,[1,1,1e-200])
    assert finite["reference_norm"] > 0
    assert finite["bias_defined"].all()
    np.testing.assert_array_equal(finite["cosine"],[0,0,1])
    assert finite["self_inclusion"][0] > 1e199
    overflow = inclusive_geometry(x,[1,1,1e-320])
    assert not overflow["bias_defined"][:2].any()
    assert overflow["bias_defined"][2]
    for value in overflow.values():
        assert np.isfinite(value).all()


def test_cgsv_altruism_changes_quotas_on_unequal_correlated_contributions():
    x = np.array([[1.,0.],[1.,0.],[0.,1.]])
    outputs = [CGSVScorer(dict(num_clients=3,beta=beta)).score([0,1,2],x,1)
               for beta in [1.,2.]]
    for beta, output in zip([1.,2.],outputs):
        importance = .95/3+.05*np.array([2.,2.,1.])/np.sqrt(5)
        importance /= importance.sum()
        quota = np.tanh(beta*importance)
        np.testing.assert_allclose(output["reputation_after"],importance,atol=1e-15)
        np.testing.assert_allclose(output["shares"],quota/quota.sum(),atol=1e-15)
    assert outputs[1]["shares"][2] > outputs[0]["shares"][2]


def test_rffl_exclusion_is_permanent_and_does_not_burn_stake():
    scorer = RFFLScorer(dict(num_clients=3))
    for round_number in range(1,5):
        first = scorer.score([0,1,2],[[1.,0.],[1.,0.],[-1.,0.]],round_number)
    np.testing.assert_array_equal(first["removed"],[False,False,True])
    np.testing.assert_array_equal(first["shares"],[.5,.5,0])
    assert not first["slash"].any()
    second = scorer.score([0,1,2],[[1.,0.]]*3,5)
    assert second["scores"][2] == 1
    assert second["shares"][2] == second["reference_weights"][2] == 0
    assert not second["eligible"][2]


@pytest.mark.parametrize("factory",[CGSVScorer,RFFLScorer])
def test_partial_participation_does_not_remove_unobserved_clients(factory):
    scorer = factory(dict(num_clients=4))
    first = scorer.score([0,1],np.eye(2),1)
    assert first["state"]["active"][2:]==[True,True]
    assert all(value > 0 for value in first["state"]["reputation"][2:])
    second = scorer.score([2,3],np.eye(2),2)
    assert second["shares"].sum() == pytest.approx(1.)
    before = scorer.reputation.copy()
    with pytest.raises(ValueError,match="exactly once"):
        scorer.score([2,3],np.eye(2),2)
    np.testing.assert_array_equal(scorer.reputation,before)


def test_zero_reference_pays_nothing_and_invalid_signed_mass_is_visible():
    cgsv = CGSVScorer(dict(num_clients=3)).score([0,1,2],np.zeros((3,5)),1)
    np.testing.assert_array_equal(cgsv["shares"],0)
    assert not cgsv["bias_defined"].any()
    scorer = RFFLScorer(dict(num_clients=3,alpha=0))
    rffl = scorer.score([0,1,2],np.zeros((3,5)),1)
    assert rffl["undefined_state"] and rffl["state"]["undefined_rounds"]==1
    np.testing.assert_array_equal(rffl["shares"],0)
    np.testing.assert_array_equal(scorer.reputation,[1/3]*3)
    valid = scorer.score([0,1,2],np.eye(3),2)
    assert not valid["undefined_state"]
    assert valid["shares"].sum() == pytest.approx(1.)


@pytest.mark.parametrize("factory",[CGSVScorer,RFFLScorer])
def test_stateful_rules_preserve_permutation_and_common_scaling(factory):
    left, right = factory(dict(num_clients=7)), factory(dict(num_clients=7))
    rng = np.random.default_rng(491)
    for round_number in range(1,5):
        ids = rng.choice(7,5,replace=False)
        x = rng.normal(size=(5,17))
        order = rng.permutation(5)
        a = left.score(ids,x,round_number)
        b = right.score(ids[order],x[order]*1e200,round_number)
        np.testing.assert_allclose(a["shares"][order],b["shares"],atol=2e-14,rtol=0)
        np.testing.assert_allclose(left.reputation,right.reputation,atol=2e-14,rtol=0)
        np.testing.assert_array_equal(left.active,right.active)


@pytest.mark.parametrize("factory",[CGSVScorer,RFFLScorer])
def test_recorded_geometry_replays_exact_states_and_rejects_stale_weights(factory):
    direct, replay = factory(dict(num_clients=9)), factory(dict(num_clients=9))
    rng = np.random.default_rng(907)
    for round_number in range(1,6):
        ids = rng.choice(9,6,replace=False)
        expected = direct.score(ids,rng.normal(size=(6,37)),round_number)
        geometry = {key:expected[key] for key in GEOMETRY_FIELDS}
        changed = {**geometry,"reference_weights":np.zeros(6)}
        with pytest.raises(ValueError,match="weights differ"):
            replay.score_recorded(ids,changed,round_number)
        actual = replay.score_recorded(ids,geometry,round_number)
        assert actual["state"] == expected["state"]
        np.testing.assert_array_equal(actual["shares"],expected["shares"])
