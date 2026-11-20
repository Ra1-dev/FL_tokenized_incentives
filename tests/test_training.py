"""Local replay and shadow-mode contracts, without downloads or network."""

import hashlib

import numpy as np
import pytest
import torch

from fltoken.config import DEFAULTS
from fltoken.models import LeNet5
from fltoken.rules import make_rules, score_round
from fltoken.training import configure_torch, flatten, local_update


def test_local_update_repeats_and_label_flip_changes_training():
    configure_torch(109)
    initial = flatten(LeNet5())
    generator = torch.Generator().manual_seed(801)
    x = torch.randn(13, 1, 28, 28, generator=generator)
    y = torch.arange(13) % 10
    config = {**DEFAULTS, "seed": 109, "local_epochs": 2, "batch_size": 5}
    first = local_update(initial, x, y, config, 4, 2)
    # Replay is independent of intervening global Torch state and local calls.
    local_update(initial, x, y, config, 6, 3)
    again = local_update(initial, x, y, config, 4, 2)
    poisoned = local_update(initial, x, y, config, 4, 2, poisoned=True)
    np.testing.assert_array_equal(first, again)
    assert first.dtype == np.float64
    assert not np.array_equal(first, poisoned)
    assert np.linalg.norm(first) > 0


def test_all_rules_score_identical_immutable_updates():
    x = np.random.default_rng(809).normal(size=(9, 47))
    digest = hashlib.sha256(x.tobytes()).hexdigest()
    rules = make_rules({"rules": ["B0", "B1", "B2"], "krum_f": 3})
    outputs = score_round(rules, np.arange(9), x, 1)
    assert set(outputs) == {"B0", "B1", "B2"}
    assert hashlib.sha256(x.tobytes()).hexdigest() == digest

    class Mutator:
        def score(self, client_ids, deltas, round):
            deltas[0] = 0

    with pytest.raises(ValueError, match="read-only"):
        score_round({"bad": Mutator()}, np.arange(9), x, 1)
