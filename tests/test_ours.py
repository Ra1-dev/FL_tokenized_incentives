import numpy as np
import pytest

from fltoken.scoring import OursScorer
from fltoken.rules import make_rules, score_round


def test_reference_excludes_own_update_and_does_not_mutate():
    rng = np.random.default_rng(591)
    x = rng.normal(size=(7, 19))
    original = x.copy()
    scorer = OursScorer()
    refs = scorer.references(x)
    for i in range(len(x)):
        altered = x.copy()
        altered[i] = 1e250
        np.testing.assert_array_equal(scorer.references(altered)[i], refs[i])
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("scale", [1e-250, 1e-100, 1., 1e100, 1e250])
def test_stateful_permutation_and_scale(scale):
    rng = np.random.default_rng(6211)
    a, b = OursScorer(), OursScorer()
    for round in range(1, 12):
        ids = rng.choice(12, 7, replace=False)
        x = rng.normal(size=(7, 23))
        x[0] = 0
        order = rng.permutation(7)
        left = a.score(ids, x, round)
        right = b.score(ids[order], scale*x[order], round)
        for name in ["shares", "scores", "slash"]:
            np.testing.assert_allclose(right[name], left[name][order], atol=2e-15, rtol=2e-14)
        assert left["state"]["low_count"] == right["state"]["low_count"]
        for client in left["state"]["reputation"]:
            assert left["state"]["reputation"][client] == pytest.approx(right["state"]["reputation"][client], abs=2e-15)


def test_zero_round_absence_recovery_and_state_snapshot():
    scorer = OursScorer({"patience": 2})
    first = scorer.score([3, 9], np.zeros((2, 5)), 1)
    assert not first["slash"].any() and not first["shares"].any()
    scorer.score([9], np.ones((1, 5)), 2)
    third = scorer.score([3, 9], np.zeros((2, 5)), 3)
    assert third["slash"].tolist() == [True, False]
    assert first["state"]["low_count"] == {3: 1, 9: 1}
    recovered = scorer.score([3, 9], np.ones((2, 5)), 4)
    assert not recovered["slash"].any()
    assert recovered["shares"].sum() == pytest.approx(1)
    assert recovered["state"]["low_count"] == {3: 0, 9: 0}


def test_components_are_switchable_and_repeat_is_deterministic():
    x = np.array([[1., 0.], [0., 1.], [0., 0.]])
    reference = OursScorer({"ema": False, "penalties": False})
    inclusive = OursScorer({"reference": False, "ema": False, "penalties": False})
    assert not reference.score([0, 1, 2], x, 1)["shares"].any()
    np.testing.assert_array_equal(inclusive.score([0, 1, 2], x, 1)["shares"], [.5, .5, 0.])
    a, b = OursScorer(), OursScorer()
    for round in range(1, 8):
        left, right = a.score([0, 1, 2], x, round), b.score([0, 1, 2], x, round)
        for key in ["shares", "scores", "slash"]:
            np.testing.assert_array_equal(left[key], right[key])
        assert left["state"] == right["state"]


def test_invalid_call_does_not_advance_state():
    scorer = OursScorer()
    with pytest.raises(ValueError):
        scorer.score([1, 1], np.ones((2, 3)), 1)
    assert scorer.last_round == 0 and not scorer.reputation
    scorer.score([1, 2], np.ones((2, 3)), 1)
    with pytest.raises(ValueError):
        scorer.score([1, 2], np.ones((2, 3)), 1)


def test_reputation_changes_allocation_and_absence_retains_it():
    x = np.array([[1., 0.], [.5, .5], [0., 1.]])
    memory = OursScorer({"beta": .9, "penalties": False})
    immediate = OursScorer({"ema": False, "penalties": False})
    first = memory.score([1, 2, 3], x, 1)
    immediate.score([1, 2, 3], x, 1)
    remembered = first["state"]["reputation"][2]
    memory.score([4, 5], np.ones((2, 2)), 2)
    assert memory.reputation[2] == remembered
    a = memory.score([1, 2, 3], np.ones((3, 2)), 3)
    b = immediate.score([1, 2, 3], np.ones((3, 2)), 3)
    assert a["shares"][1] > a["shares"][0]
    np.testing.assert_allclose(b["shares"], np.full(3, 1/3))


def test_penalty_switch_changes_flags_and_renormalizes():
    params = dict(threshold=.4, patience=1, ema=False)
    enabled = OursScorer(params)
    disabled = OursScorer({**params, "penalties": False})
    x = np.array([[1., 0.], [.5, .5], [0., 1.]])
    a = enabled.score([1, 2, 3], x, 1)
    b = disabled.score([1, 2, 3], x, 1)
    assert a["slash"].tolist() == [True, False, True]
    np.testing.assert_array_equal(a["shares"], [0., 1., 0.])
    assert not b["slash"].any() and (b["shares"] > 0).all()


def test_identical_updates_with_one_zero_and_single_client():
    x = np.ones((9, 12))
    x[3] = 0
    result = OursScorer().score(np.arange(9), x, 1)
    assert result["shares"][3] == 0
    np.testing.assert_allclose(np.delete(result["shares"], 3), np.full(8, 1/8))
    assert OursScorer().score([5], np.ones((1, 12)), 1)["shares"].tolist() == [0.]


def test_registry_scores_one_readonly_matrix_with_every_rule():
    x = np.random.default_rng(7201).normal(size=(7,19))
    original = x.copy()
    rules = make_rules(dict(rules=["B0","B1","B2","Mstar"],krum_f=2,ours_params={"ema":False}))
    outputs = score_round(rules,np.arange(7),x,1)
    assert set(outputs)==set(rules)
    for output in outputs.values():
        assert output["shares"].shape==(7,)
        assert output["selected"].shape==(7,)
    np.testing.assert_array_equal(x,original)


def test_shared_geometry_matches_independent_candidate_scoring():
    rng = np.random.default_rng(9177)
    for beta in [0., .5, .9]:
        direct = OursScorer({"beta":beta,"threshold":.05,"patience":2})
        cached = OursScorer({"beta":beta,"threshold":.05,"patience":2})
        for round in range(1,5):
            ids = rng.choice(12,7,replace=False)
            x = rng.normal(size=(7,19))
            q = OursScorer({"ema":False,"penalties":False}).score(ids,x,1)["scores"]
            a,b = direct.score(ids,x,round),cached.allocate(ids,q,round)
            for key in ["shares","scores","slash"]:
                np.testing.assert_array_equal(a[key],b[key])
            assert a["state"]==b["state"]
