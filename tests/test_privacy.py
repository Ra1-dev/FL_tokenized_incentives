"""Sensitivity, noise calibration and Gaussian-composition contracts."""

import math

import numpy as np
import pytest

from fltoken.privacy import (account_schedule, clip_update, perturb_update, gaussian_rdp,
                            privacy_budget, release_update, validate_privacy)


def test_flat_clipping_preserves_small_updates_and_bounds_replacement_distance():
    rng = np.random.default_rng(1171)
    for size in [1, 7, 44426]:
        left, right = rng.normal(size=size), rng.normal(size=size)
        before = left.copy()
        a, b = clip_update(left, .7), clip_update(right, .7)
        assert np.linalg.norm(a) <= .7*(1+2e-15)
        assert np.linalg.norm(a-b) <= 1.4*(1+2e-15)
        np.testing.assert_array_equal(left, before)
    np.testing.assert_array_equal(clip_update([.03, .04], .1), [.03, .04])
    np.testing.assert_array_equal(clip_update([0., 0.], .1), [0., 0.])
    assert np.linalg.norm(clip_update([1., 0.], .7)-clip_update([-1., 0.], .7)) == 1.4


def test_clipping_handles_extreme_norms_and_common_scaling():
    vector = np.array([3., -4., 5.])
    expected = clip_update(vector, 2.)
    for scale in [1e-250, 1e250]:
        np.testing.assert_allclose(clip_update(vector*scale, 2*scale)/scale, expected, rtol=2e-15, atol=0)
    np.testing.assert_allclose(clip_update([1e308, -1e308], 1.), [1/math.sqrt(2), -1/math.sqrt(2)], rtol=2e-15)


def test_noise_scale_is_per_coordinate_and_zero_noise_does_not_consume_randomness():
    rng, untouched = np.random.default_rng(1181), np.random.default_rng(1181)
    np.testing.assert_allclose(perturb_update([3., 4.], 1., 0., rng), [.6, .8], atol=2e-16, rtol=0)
    assert rng.random() == untouched.random()
    noise = perturb_update(np.zeros(100000), 2., .5, np.random.default_rng(1187))
    assert abs(noise.mean()) < .03
    assert .97 < noise.std() < 1.03
    assert np.linalg.norm(noise) > 300
    for scale in [.1, 10.]:
        left = perturb_update([3., 4.], 2., .5, np.random.default_rng(1193))
        right = perturb_update(np.array([3., 4.])*scale, 2*scale, .5, np.random.default_rng(1193))
        np.testing.assert_allclose(right, left*scale, rtol=2e-15, atol=0)


def test_rdp_accounts_for_replacement_sensitivity_and_composition():
    assert gaussian_rdp(2., 1., 1) == pytest.approx(4.)
    assert gaussian_rdp(3., .5, 7) == pytest.approx(168.)
    assert gaussian_rdp(4., 1., 5) == pytest.approx(5*gaussian_rdp(4., 1., 1))
    assert gaussian_rdp(2., 0., 0) == 0
    assert math.isinf(gaussian_rdp(2., 0., 1))


def test_continuous_order_accountant_beats_every_fixed_order_conversion():
    for count in [1, 7, 20]:
        for sigma in [.5, 1., 1.5]:
            result = privacy_budget(count, sigma, 1e-5)
            rho = 2*count/sigma**2
            expected = rho+2*math.sqrt(rho*math.log(1e5))
            assert result["epsilon"] == pytest.approx(expected, rel=2e-15)
            for order in [1.1, 1.5, 2., 4., 8., 16., 64.]:
                assert result["epsilon"] <= gaussian_rdp(order, sigma, count)+math.log(1e5)/(order-1)+1e-12
            optimum = result["optimal_order"]
            assert result["epsilon"] == pytest.approx(gaussian_rdp(optimum, sigma, count)+math.log(1e5)/(optimum-1))
            assert result["finite"] and not result["sampling_amplification"]


def test_nonprivate_and_empty_release_states_are_explicit():
    empty = privacy_budget(0, 0., 1e-5)
    nonprivate = privacy_budget(1, 0., 1e-5)
    assert empty["epsilon"] == 0 and empty["finite"]
    assert nonprivate["epsilon"] is None and not nonprivate["finite"]
    with pytest.raises(ValueError):
        privacy_budget(-1, 1., 1e-5)
    with pytest.raises(ValueError):
        privacy_budget(True, 1., 1e-5)
    with pytest.raises(ValueError):
        privacy_budget(1, 1., 0.)
    with pytest.raises(ValueError):
        perturb_update([1.], 1., -1., np.random.default_rng(1))
    with pytest.raises(ValueError, match="underflows"):
        perturb_update([1.], 1e-250, 1e-250, np.random.default_rng(1))


def test_optional_privacy_preserves_legacy_config_shape_and_rejects_ambiguous_settings(tmp_path):
    from fltoken.config import DEFAULTS, load_config, validate_config
    assert "privacy" not in validate_config(DEFAULTS)
    base = {**DEFAULTS, "attack": "none", "fraction": 0.,
            "privacy": dict(clipping_bound=.7, noise_multiplier=.5, delta=1e-5)}
    assert validate_config(base)["privacy"] == base["privacy"]
    path = tmp_path/"config.yaml"
    path.write_text("attack: none\nfraction: 0\nprivacy:\n  clipping_bound: 0.7\n  noise_multiplier: 0.5\n  delta: 0.00001\n", encoding="utf-8")
    assert load_config(path) == base
    for settings in [None, {}, {**base["privacy"], "clip": .5},
                     {**base["privacy"], "clipping_bound": True},
                     {**base["privacy"], "noise_multiplier": float("nan")},
                     {**base["privacy"], "delta": 1.}]:
        with pytest.raises(ValueError):
            validate_privacy(settings)
    with pytest.raises(ValueError, match="honest"):
        validate_config({**base, "attack": "noise"})


def test_release_randomness_is_separate_and_identity_addressable():
    raw = np.arange(1., 101.)
    assert release_update(raw, {}, 1, 1) is raw
    config = dict(seed=1201, privacy=dict(clipping_bound=.7, noise_multiplier=.5, delta=1e-5))
    first = release_update(raw, config, 3, 2)
    np.testing.assert_array_equal(first, release_update(raw, config, 3, 2))
    assert not np.array_equal(first, release_update(raw, config, 4, 2))
    assert not np.array_equal(first, release_update(raw, config, 3, 3))
    np.testing.assert_array_equal(raw, np.arange(1., 101.))


def test_public_schedule_accounting_includes_unsampled_clients_and_all_round_bound():
    config = dict(num_clients=5, clients_per_round=3, rounds=2,
                  privacy=dict(clipping_bound=.7, noise_multiplier=1., delta=1e-5))
    result = account_schedule(config, [[0, 1, 2], [0, 2, 3]])
    assert [client["participations"] for client in result["clients"]] == [2, 1, 2, 1, 0]
    assert result["clients"][4]["epsilon"] == 0
    assert max(client["epsilon"] for client in result["clients"]) == result["conservative_all_rounds"]["epsilon"]
    assert result["sensitivity"] == 1.4 and result["noise_standard_deviation"] == .7
    assert result["simulation_only"] and not result["sampling_amplification"]
    for schedule in [[[0, 1, 2]], [[0, 0, 2], [0, 2, 3]], [[0, 1, 2], [0, 2, 5]], [[0, 1, 2], [False, 2, 3]]]:
        with pytest.raises(ValueError):
            account_schedule(config, schedule)


def test_flower_replies_and_training_consume_the_same_perturbed_matrices(tmp_path, monkeypatch):
    import csv
    import hashlib
    import json
    import torch
    import fltoken.experiment as experiment
    from fltoken.config import DEFAULTS
    from fltoken.data import sample_clients
    from fltoken.scoring import cosine_sendros
    from flwr.supercore.task_identity import TaskIdentity

    monkeypatch.chdir(tmp_path)
    for field in ["_run_id", "_task_id", "_node_id"]:
        monkeypatch.setattr(TaskIdentity, field, 1)
    config = {**DEFAULTS, "attack": "none", "fraction": 0., "num_clients": 5,
              "clients_per_round": 3, "rounds": 2, "seed": 1213,
              "rules": ["B0", "B1", "B2", "Mstar", "Aref"], "ours_params": {},
              "privacy": dict(clipping_bound=.7, noise_multiplier=.5, delta=1e-5)}
    images, labels = torch.zeros(100, 1, 28, 28), torch.arange(100) % 10
    monkeypatch.setattr(experiment, "load_tensors", lambda *args: (images, labels))
    partitions = [np.arange(i*20, (i+1)*20) for i in range(5)]
    monkeypatch.setattr(experiment, "dirichlet_partition", lambda *args: partitions)
    monkeypatch.setattr(experiment, "LeNet5", lambda: None)
    monkeypatch.setattr(experiment, "flatten", lambda _: np.zeros(7, dtype=np.float64))
    starts, evaluated = {}, []

    def local(initial, images, labels, config, client, round, poisoned):
        starts[round, client] = initial.copy()
        assert not poisoned
        return np.arange(1., 8.)*(client+1)

    monkeypatch.setattr(experiment, "local_update", local)

    def evaluate(vector, *args):
        evaluated.append(vector.copy())
        return .5, 1.

    monkeypatch.setattr(experiment, "evaluate", evaluate)

    class LocalTransport:
        def get_node_ids(self):
            return [11, 12, 13, 14, 15]

        def send_and_receive(self, messages, timeout):
            return [experiment.train(message, None) for message in reversed(messages)]

    output = tmp_path/"run"
    output.mkdir()
    experiment.make_server(config, output)(LocalTransport(), None)
    with (output/"per_round.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    global_vector = np.zeros(7)
    for round in [1, 2]:
        ids = sample_clients(5, 3, round, config["seed"])
        expected = np.array([release_update(np.arange(1., 8.)*(int(client)+1), config, int(client), round) for client in ids])
        with np.load(tmp_path/f"results/cache/run/round-{round:03d}.npz") as cached:
            np.testing.assert_array_equal(cached["deltas"], expected)
        digest = hashlib.sha256(expected.astype("<f8").tobytes()).hexdigest()
        for client in ids:
            np.testing.assert_array_equal(starts[round, int(client)], global_vector)
        global_vector = (global_vector+expected.mean(axis=0)).astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(evaluated[round-1], global_vector)
        selected = [row for row in rows if int(row["round"]) == round]
        assert all(row["update_set_sha256"] == digest for row in selected)
        np.testing.assert_array_equal([float(row["B1_share"]) for row in selected], cosine_sendros(expected)[1])
        assert all(float(row["delta_norm"]) == np.linalg.norm(expected[i]) for i, row in enumerate(selected))
    account = json.loads((output/"privacy.json").read_text(encoding="utf-8"))
    assert sum(client["participations"] for client in account["clients"]) == 6
    import importlib
    from pathlib import Path
    from fltoken.io import sha256, write_json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    runner = importlib.import_module("run_privacy")
    write_json("configs/method_frozen.json", dict(synthetic_fixture=True))
    write_json("configs/privacy_frozen.json", dict(synthetic_fixture=True))
    write_json(output/"metadata.json", dict(status="complete", synthetic_fixture=True,
               config=config, source_git_dirty=False, method_sha256=sha256("configs/method_frozen.json"),
               privacy_sha256=sha256("configs/privacy_frozen.json")))
    _, inputs = runner.verify_cell(output, config)
    assert inputs["privacy.json"] == sha256(output/"privacy.json")
    first = output/"privacy.json"
    original = first.read_bytes()
    changed = json.loads(original)
    changed["clients"][0]["participations"] += 1
    write_json(first, changed)
    with pytest.raises(ValueError, match="actual public release"):
        runner.verify_cell(output, config)
    first.write_bytes(original)
    metadata = json.loads((output/"metadata.json").read_text())
    write_json(output/"metadata.json", {**metadata, "source_git_dirty": True})
    with pytest.raises(ValueError, match="frozen scientific"):
        runner.verify_cell(output, config)
    write_json(output/"metadata.json", metadata)
    cache = tmp_path/"results/cache/run/round-001.npz"
    original_cache = cache.read_bytes()
    with np.load(cache) as saved:
        damaged = saved["deltas"].copy()
        damaged[0, 0] += 1
        np.savez(cache, client_ids=saved["client_ids"], deltas=damaged)
    with pytest.raises(ValueError, match="common matrix"):
        runner.verify_cell(output, config)
    cache.write_bytes(original_cache)
    for path in (tmp_path/"results/cache/run").glob("*.npz"):
        path.unlink()
    # Published analysis reconstructs schedules and ledgers without raw arrays.
    _, recorded_inputs = runner.verify_cell(output, config, verify_cache=False)
    assert recorded_inputs == inputs

    # Exercise complete published-record analysis from this actual message trace.
    import shutil
    analysis = importlib.import_module("summarize_privacy")
    plan = json.loads((Path(__file__).resolve().parents[1]/"configs/privacy_plan.json").read_text())
    plan.update(config="configs/test.yaml", seeds=[config["seed"]], noise_multipliers=[.5],
                **{key: config[key] for key in ["dataset", "attack", "fraction", "rounds", "device", "workers", "gpu_fraction", "rules"]})
    write_json(plan["config"], {key: value for key, value in config.items() if key not in ["privacy", "ours_params"]})
    write_json("configs/privacy_plan.json", plan)
    frozen = dict(clipping_bound=.7, delta=1e-5, noise_multipliers=[.5])
    write_json("configs/privacy_frozen.json", frozen)
    cell = list(runner.privacy_cells(plan))[0]
    destination = Path(cell["path"])
    shutil.copytree(output, destination)
    write_json(destination/"metadata.json", {**metadata, "privacy_sha256": sha256("configs/privacy_frozen.json"),
               "source_config": plan["config"], "git_commit": "a"*40, "source_sha256": {"synthetic": "bound"},
               "versions": {"synthetic": True}})
    _, complete_inputs = runner.verify_cell(destination, config, verify_cache=False)
    binding = dict(plan_sha256=sha256("configs/privacy_plan.json"), frozen_sha256=sha256("configs/privacy_frozen.json"),
                   method_sha256=sha256("configs/method_frozen.json"), training_source_sha256={"synthetic": "bound"})
    write_json("results/privacy/execution.json", dict(**binding, started_utc="2026-01-01T00:00:00+00:00",
               finished_utc="2026-01-01T00:00:01+00:00", elapsed_wall_seconds=1.))
    manifest = dict(**binding, complete=True, expected_cells=1, completed_cells=1, elapsed_wall_seconds=1.,
                    execution_sha256=sha256("results/privacy/execution.json"),
                    cells=[dict(**cell, inputs=complete_inputs, source_commit="a"*40)])
    write_json("results/privacy/manifest.json", manifest)
    write_json("results/summary.json", dict(synthetic_fixture=True))
    monkeypatch.setattr(analysis, "published_bytes", lambda path: Path(path).read_bytes())
    monkeypatch.setattr(analysis, "recorded_privacy", lambda **kwargs: frozen)
    monkeypatch.setattr(analysis, "recorded_method", lambda: dict(params={}))
    verified = []
    monkeypatch.setattr(analysis, "verify_training_source", lambda value, method: verified.append(value["git_commit"]))
    analysis.main()
    assert verified == ["a"*40]
    first_summary = Path("results/privacy/summary.json").read_bytes()
    assert json.loads(first_summary)["runs"] == 1
    analysis.main()
    assert Path("results/privacy/summary.json").read_bytes() == first_summary
    write_json("results/privacy/manifest.json", {**manifest, "complete": False})
    with pytest.raises(ValueError, match="complete published"):
        analysis.main()
