"""The declared runs preserve their producers and perturb only honest replies."""
import csv
from functools import partial
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module('study_protocol')


def test_exact_eighteen_cells_and_unchanged_calibration(study):
    plan = study.read(ROOT/study.PLAN)
    cells = list(study.study_cells(plan))
    assert len(cells) == len({cell['path'] for cell in cells}) == 18
    assert [cell['kind'] for cell in cells] == ['privacy']*12+['fashion']*6
    assert {(cell['seed'], cell['noise_multiplier']) for cell in cells[:12]} == {
        (seed, sigma) for seed in [42, 43, 44] for sigma in [0., .5, 1., 1.5]}
    assert {cell['fraction'] for cell in cells[12:] if cell['attack'] == 'label_flip'} == {.2}
    privacy = study.read(ROOT/'configs/privacy_frozen.json')
    method = study.read(ROOT/'configs/method_frozen.json')
    for cell in cells:
        config = study.expected_config(plan, cell, method, privacy)
        assert config['ours_params'] == method['params']
        if cell['kind'] == 'privacy':
            assert config['privacy']['clipping_bound'] == privacy['clipping_bound']
            assert config['privacy_scope'] == 'honest'
            with pytest.raises(ValueError, match='every client to be honest'):
                study.validate_config(config)
        else:
            assert 'privacy' not in config
    with pytest.raises(ValueError, match='predeclared'):
        study.expected_config(plan, {**cells[0], 'noise_multiplier':.75}, method, privacy)


def test_historical_producer_is_checked_before_any_source_scan(study, monkeypatch):
    monkeypatch.setattr(study, 'sha256', lambda path:'bound-plan')
    def forbidden(*args):
        raise AssertionError('a substituted historical producer reached later validation')
    monkeypatch.setattr(study, 'sources', forbidden)
    metadata = dict(git_commit='a'*40, source_git_dirty=False, study_plan_sha256='bound-plan',
                    effective_method=dict(sha256='changed'))
    with pytest.raises(ValueError, match='exact historical scorer'):
        study.verify_sources(metadata, {}, dict(sha256='frozen'))


@pytest.mark.parametrize('failed', [False, True])
def test_mixture_messages_common_matrices_and_terminal_evidence(study, tmp_path, monkeypatch, failed):
    import torch
    import fltoken.experiment as experiment
    from fltoken.config import DEFAULTS
    from fltoken.data import sample_clients, seed_for
    from fltoken.io import sha256, write_json
    from fltoken.privacy import release_update
    from fltoken.rules import score_round
    from flwr.supercore.task_identity import TaskIdentity

    method = study.read(ROOT/'configs/method_frozen.json')
    factory, revision = study.historical_scorer(method)
    assert revision and factory.__module__ == 'recorded_reward_arithmetic'
    extension = study.read(ROOT/'configs/extensions_frozen.json')['params']
    monkeypatch.chdir(tmp_path)
    for field in ['_run_id', '_task_id', '_node_id']:
        monkeypatch.setattr(TaskIdentity, field, 1)
    config = dict(DEFAULTS, num_clients=30, clients_per_round=30, rounds=3 if failed else 2,
                  rules=['B0', 'B1', 'B2', 'Mstar', 'CGSV', 'RFFL'], ours_params=method['params'],
                  privacy=dict(clipping_bound=.7, noise_multiplier=.5, delta=1e-5), privacy_scope='honest')
    attacks = experiment.adversaries(config)
    target = next(client for client in range(30) if client not in attacks)
    images, labels = torch.zeros(300, 1, 28, 28), torch.arange(300) % 10
    monkeypatch.setattr(experiment, 'load_tensors', lambda *args:(images, labels))
    monkeypatch.setattr(experiment, 'dirichlet_partition', lambda *args:[np.arange(i*10, (i+1)*10) for i in range(30)])
    monkeypatch.setattr(experiment, 'LeNet5', lambda:None)
    monkeypatch.setattr(experiment, 'flatten', lambda model:np.zeros(7))
    raw = lambda client, number:np.arange(1., 8.)*(client+1)*number
    def local(initial, images, labels, config, client, number, poisoned):
        assert not poisoned
        return np.full(7, np.nan) if failed and number == 2 and client == target else raw(client, number)
    monkeypatch.setattr(experiment, 'local_update', local)
    vectors = []
    def evaluate(vector, *args):
        vectors.append(vector.copy())
        return .5, 1.
    monkeypatch.setattr(experiment, 'evaluate', evaluate)
    replies = {}
    class Transport:
        def get_node_ids(self):
            return list(range(100, 130))
        def send_and_receive(self, messages, timeout):
            result = []
            for message in messages:
                reply = experiment.train(message, None)
                if 'arrays' in reply.content:
                    control = message.content['control']
                    replies[len(vectors)+1, control['client_id']] = reply.content['arrays'].to_numpy_ndarrays()[0]
                result.append(reply)
            return list(reversed(result))
    output = Path('run')
    output.mkdir()
    make = partial(study.study_rules, method_factory=factory, extension_parameters=extension)
    experiment.make_server(config, output, rule_factory=make)(Transport(), None)
    rules = make({**config, 'krum_f':5})
    with (output/'per_round.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    vector = np.zeros(7)
    for number in range(1, len(vectors)+1):
        ids = sample_clients(30, 30, number, config['seed'])
        expected = []
        for client in map(int, ids):
            attack = attacks.get(client, 'none')
            source = max(1, number-config['stale_lag']) if attack == 'stale' else number
            released = raw(client, source) if attack != 'none' else release_update(raw(client, source), config, client, source)
            np.testing.assert_array_equal(replies[number, client], released)
            if attack == 'zero':
                released = np.zeros(7)
            elif attack == 'noise':
                released = np.random.default_rng(seed_for(config['seed'], number, client, 3)).normal(0, config['noise_sigma'], 7)
            expected.append(released)
        expected = np.array(expected)
        with np.load(f'results/cache/run/round-{number:03d}.npz') as cached:
            np.testing.assert_array_equal(cached['deltas'], expected)
        vector = (vector+expected.mean(axis=0)).astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(vectors[number-1], vector)
        scores = score_round(rules, ids, expected, number)
        observed = [row for row in rows if int(row['round']) == number]
        for rule, result in scores.items():
            np.testing.assert_array_equal([float(row[rule+'_share']) for row in observed], result['shares'])
        assert all(row['CGSV_undefined_state'] in {'True', 'False'} for row in observed)
    for path in ['method', 'privacy', 'extensions']:
        write_json(f'configs/{path}_frozen.json', dict(synthetic_fixture=True))
    metadata = dict(config=config, status='failed_numerical' if failed else 'complete',
                    method_sha256=sha256('configs/method_frozen.json'),
                    privacy_sha256=sha256('configs/privacy_frozen.json'),
                    extensions_sha256=sha256('configs/extensions_frozen.json'))
    if failed:
        failure = study.read(output/'failure.json')
        assert failure['completed_rounds'] == 1
        metadata.update(completed_rounds=1, failure_sha256=sha256(output/'failure.json'))
    write_json(output/'metadata.json', metadata)
    _, inputs = study.verify_cell(output, config)
    assert inputs and study.verify_cell(output, config, verify_cache=False)[1] == inputs
    if not failed:
        account = study.read(output/'privacy.json')
        assert len(account['clients']) == 25 and account['unprotected_client_ids'] == sorted(attacks)
        assert sum(client['participations'] for client in account['clients']) == 50
        changed = {**account, 'unprotected_client_ids':[]}
        write_json(output/'privacy.json', changed)
        with pytest.raises(ValueError, match='privacy accounting'):
            study.verify_cell(output, config, verify_cache=False)
