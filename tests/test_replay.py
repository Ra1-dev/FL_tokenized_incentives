"""The declared stale sensitivity variant cannot use a future report."""

import numpy as np

from fltoken.attacks import replay_own


def test_replay_own_uses_latest_eligible_report_and_returns_copy():
    history = [(2, np.ones(3)), (7, np.full(3, 7)), (4, np.full(3, 4))]
    source, delta = replay_own(history, 10, 5, np.ones(3))
    assert source == 4
    np.testing.assert_array_equal(delta, [4, 4, 4])
    delta[:] = 0
    np.testing.assert_array_equal(history[-1][1], [4, 4, 4])
    assert replay_own(history, 7, 5, np.ones(3))[0] == 2
    source, delta = replay_own(history, 6, 5, np.ones(3))
    assert source == 0 and not delta.any()


def test_persistent_replay_without_initial_report_remains_zero():
    history = []
    for round in [1, 3, 7, 9, 14, 20]:
        source, delta = replay_own(history, round, 5, np.ones(4))
        assert not delta.any()
        assert source == 0 or source <= round - 5
        history.append((round, delta))
