"""Coordinate noise preserves clipping, historical rewards and terminal evidence."""
from copy import deepcopy
import importlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module('coordinate_protocol')


def test_nine_cells_same_clipping_and_coordinate_scale(protocol):
    from fltoken.data import seed_for
    from fltoken.privacy import clip_update, release_update, privacy_budget
    plan = protocol.original.read(ROOT/protocol.PLAN)
    method = protocol.original.read(ROOT/'configs/method_frozen.json')
    privacy = protocol.original.read(ROOT/'configs/privacy_frozen.json')
    frozen = deepcopy(privacy)
    cells = list(protocol.study_cells(plan))
    assert len(cells) == len({cell['path'] for cell in cells}) == 9
    assert {(c['seed'], c['coordinate_standard_deviation']) for c in cells} == {
        (seed, tau) for seed in [42, 43, 44] for tau in [.001, .003, .01]}
    factory, _ = protocol.original.historical_scorer(method)
    for cell in cells:
        config = protocol.expected_config(plan, cell, method, privacy)
        assert config['ours_params'] == method['params'] and config['privacy_scope'] == 'honest'
        assert config['privacy']['clipping_bound'] == privacy['clipping_bound']
        rules = protocol.study_rules({**config, 'krum_f':2}, factory)
        assert list(rules) == ['B0', 'B1', 'B2', 'Mstar']
        assert type(rules['Mstar']).__module__ == 'recorded_reward_arithmetic'
        tau = cell['coordinate_standard_deviation']
        for dimension in [7, 44426]:
            raw = np.arange(dimension, dtype=float)+1
            expected = clip_update(raw, privacy['clipping_bound'])+np.random.default_rng(
                seed_for(config['seed'], 1, 7, 4)).normal(0., tau, dimension)
            np.testing.assert_allclose(release_update(raw, config, 7, 1), expected, rtol=0., atol=1e-16)
        budget = privacy_budget(1, config['privacy']['noise_multiplier'], config['privacy']['delta'])
        assert budget['epsilon'] > 27000 and not budget['sampling_amplification']
    assert privacy == frozen and privacy['noise_multipliers'] == [0., .5, 1., 1.5]
    with pytest.raises(ValueError, match='not declared'):
        protocol.expected_config(plan, {**cells[0], 'coordinate_standard_deviation':.02}, method, privacy)


def test_substituted_producer_rejected_first(protocol, monkeypatch):
    monkeypatch.setattr(protocol, 'sha256', lambda path:'plan')
    monkeypatch.setattr(protocol, 'sources', lambda plan:pytest.fail('substituted producer reached source verification'))
    with pytest.raises(ValueError, match='exact historical scorer'):
        protocol.verify_sources(dict(git_commit='a'*40, source_git_dirty=False,
                                     coordinate_plan_sha256='plan', effective_method={}), {}, {'sha256':'frozen'})


@pytest.mark.parametrize('terminal', ['complete', 'failed_numerical', 'missing'])
def test_worker_records_real_configuration_and_terminal_status(protocol, monkeypatch, tmp_path, terminal):
    import run_coordinate_study as runner
    import run_experiment as runtime
    from fltoken.io import write_json
    plan = protocol.original.read(ROOT/protocol.PLAN)
    method = protocol.original.read(ROOT/'configs/method_frozen.json')
    privacy = protocol.original.read(ROOT/'configs/privacy_frozen.json')
    factory, revision = protocol.original.historical_scorer(method)
    effective = {'revision':revision, 'sha256':method['scorer_sha256']}
    config = protocol.expected_config(plan, next(protocol.study_cells(plan)), method, privacy)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, 'published_study', lambda:(plan, method, privacy, {}, factory, effective))
    monkeypatch.setattr(runner, 'expected_config', lambda *args:deepcopy(config))
    monkeypatch.setattr(runner, 'sources', lambda plan:{'published':'digest'})
    monkeypatch.setattr(runner, 'sha256', lambda path:'digest')
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda args, **kw:'a'*40 if 'rev-parse' in args else '')
    monkeypatch.setattr(runtime.torch.cuda, 'is_available', lambda:True)
    monkeypatch.setattr(runtime.torch.cuda, 'get_device_name', lambda number:'synthetic device')
    monkeypatch.setattr(runtime, 'ray_initialization', lambda *args:{'num_cpus':1})
    observed = {}
    def server(actual, output, rule_factory):
        observed.update(config=actual, output=output, rules=rule_factory({**actual, 'krum_f':2}))
        return 'server fixture'
    monkeypatch.setattr(runtime, 'make_server', server)
    def simulate(**kwargs):
        assert kwargs['num_supernodes'] == 30
        assert kwargs['backend_config']['client_resources'] == {'num_cpus':1, 'num_gpus':1.}
        assert list(observed['rules']) == ['B0', 'B1', 'B2', 'Mstar']
        assert observed['config']['privacy_scope'] == 'honest'
        if terminal == 'complete':
            write_json(observed['output']/'metrics.json', {'fixture':True})
        elif terminal == 'failed_numerical':
            write_json(observed['output']/'failure.json', {'completed_rounds':1})
    monkeypatch.setattr(runtime, 'run_simulation', simulate)
    if terminal == 'missing':
        with pytest.raises(RuntimeError, match='without completed metrics'):
            runner.run_cell(0)
    else:
        assert runner.run_cell(0) == (0 if terminal == 'complete' else 20)
    metadata = protocol.original.read(observed['output']/'metadata.json')
    assert metadata['config'] == config and metadata['effective_method'] == effective
    assert metadata['coordinate_cell'] == next(protocol.study_cells(plan))
    assert metadata['requested_coordinate_standard_deviation'] == .001
    assert metadata['actual_coordinate_standard_deviation'] == config['privacy']['noise_multiplier']*privacy['clipping_bound']
    assert metadata['status'] == ('failed' if terminal == 'missing' else terminal)
    with pytest.raises(ValueError, match='existing coordinate cell'):
        runner.run_cell(0)
