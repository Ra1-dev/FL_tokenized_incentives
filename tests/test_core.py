"""Boundary contracts that complement independent numerical evaluation."""

import numpy as np
import pytest
import torch

from fltoken.attacks import gaussian_noise, label_flip, sign_flip, stale_round, zero_update
from fltoken.data import dirichlet_partition, sample_clients
from fltoken.models import LeNet5
from fltoken.scoring import cosine_sendros, equal_split, multi_krum


def test_model_contract():
    model = LeNet5()
    assert sum(p.numel() for p in model.parameters()) == 44426
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.parametrize("f", [0, 1, 2, 4])
def test_krum_minimum_size_and_duplicate_ties(f):
    k = 2 * f + 3
    x = np.ones((k, 9))
    selected, shares = multi_krum(x, f)
    np.testing.assert_array_equal(selected, np.arange(k) < k - f)
    assert shares.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        multi_krum(x[:-1], f)


def test_degenerate_cosines_and_no_mutation():
    for x in [np.zeros((5, 11)), np.eye(5, 11), np.vstack([np.ones((1, 11)), np.zeros((4, 11))])]:
        original = x.copy()
        q, shares = cosine_sendros(x)
        assert q.dtype == np.int64
        assert shares.dtype == np.float64
        assert np.isfinite(shares).all()
        assert shares.sum() == pytest.approx(1 if q.sum() else 0)
        equal_split(x)
        multi_krum(x, 1)
        np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(cosine_sendros(np.array([[1., 0.], [-1., 0.]]))[0], [0, 0])


def test_rules_discriminate_constructed_attack():
    x = np.array([[1., 0.], [1., .1], [1., -.1], [1., .2], [-10., 0.]])
    b0 = equal_split(x)
    _, b1 = cosine_sendros(x)
    selected, b2 = multi_krum(x, 1)
    assert b0[-1] > 0
    assert b1[-1] == 1
    assert not selected[-1] and b2[-1] == 0


@pytest.mark.parametrize("x", [np.empty((0, 4)), np.empty((3, 0)), np.array([1., 2.]), np.array([[np.nan]])])
def test_invalid_scoring_inputs(x):
    with pytest.raises(ValueError):
        cosine_sendros(x)


def test_attack_contracts():
    x = np.arange(20, dtype=np.float64).reshape(4, 5)
    saved = x.copy()
    assert not zero_update(x).any()
    np.testing.assert_array_equal(sign_flip(x, 2), -2 * x)
    a = gaussian_noise(x.shape, .1, np.random.default_rng(8))
    b = np.random.default_rng(8).normal(0, .1, x.shape)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, saved)
    np.testing.assert_array_equal(label_flip(np.arange(10)), np.arange(9, -1, -1))
    assert [stale_round(t) for t in [1, 5, 6, 7, 20]] == [1, 1, 1, 2, 15]


def test_partition_and_sampling_general_shapes():
    y = np.repeat([2, 5, 9], 200)
    a = dirichlet_partition(y, 7, .7, 602)
    b = dirichlet_partition(y, 7, .7, 602)
    assert min(map(len, a)) >= 10
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(len(y)))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    for t in range(1, 8):
        sampled = sample_clients(7, 7, t, 602)
        np.testing.assert_array_equal(np.sort(sampled), np.arange(7))
    with pytest.raises(ValueError):
        dirichlet_partition(y[:5], 7, .5, 0)
