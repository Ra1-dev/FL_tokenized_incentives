"""Stateful rewards agree with closed-form histories across changing attendance."""

import math

import numpy as np
import pytest

from fltoken.ours import OursScorer


def test_allocation_matches_closed_form_history_and_low_streaks():
    rng = np.random.default_rng(17781)
    observations = []
    for number in range(1, 41):
        ids = rng.choice(23, 7, replace=False).tolist()
        quality = rng.choice([0., .01, .04, .2, .7, 1.], size=7).tolist()
        if number % 7 == 0:
            quality = [0.]*7
        observations.append((ids, quality))

    signatures = {}
    for beta in [0., .4, 1.]:
        for threshold, patience in [(0., 2), (.05, 3), (.4, 1)]:
            for ema, penalties in [(False, False), (True, False), (True, True)]:
                scorer = OursScorer(dict(beta=beta, threshold=threshold, patience=patience,
                                         ema=ema, penalties=penalties, initial_reputation=.5))
                histories, signature = {}, []
                for number, (ids, quality) in enumerate(observations, 1):
                    for client, value in zip(ids, quality):
                        histories.setdefault(client, []).append(value)
                    # Evaluate the full weighted history, not the implementation's
                    # one-step recurrence. Attendance determines each history's age.
                    reputation = {client: beta**len(history)*.5 + (1-beta)*math.fsum(
                        beta**(len(history)-index-1)*value for index, value in enumerate(history))
                        for client, history in histories.items()}
                    streaks = {}
                    for client, history in histories.items():
                        last_good = max((index for index, value in enumerate(history) if value > threshold), default=-1)
                        streaks[client] = len(history)-last_good-1
                    flags = [penalties and streaks[client] > 0 and streaks[client] % patience == 0 for client in ids]
                    weights = [0. if flagged else value*(reputation[client] if ema else 1.)
                               for client, value, flagged in zip(ids, quality, flags)]
                    total = math.fsum(weights)
                    expected = [weight/total for weight in weights] if total else [0.]*len(ids)
                    result = scorer.allocate(ids, quality, number)
                    np.testing.assert_allclose(result['shares'], expected, atol=3e-15, rtol=3e-14)
                    assert result['slash'].tolist() == flags
                    assert result['state']['last_round'] == number
                    assert result['state']['reputation'] == pytest.approx(reputation, abs=3e-15, rel=3e-14)
                    assert result['state']['low_count'] == {
                        client: streak % patience if penalties else streak for client, streak in streaks.items()}
                    signature.extend(result['shares'].tolist())
                signatures[beta, threshold, patience, ema, penalties] = signature

    # The shared trace distinguishes temporal memory and penalties; an oracle
    # exercise in which every enabled variant pays identically is insufficient.
    assert signatures[0., .05, 3, True, False] != signatures[.4, .05, 3, True, False]
    assert signatures[.4, .4, 1, True, False] != signatures[.4, .4, 1, True, True]
