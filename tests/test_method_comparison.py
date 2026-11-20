"""Exact differences must remain visible when float tolerances would hide them."""
import importlib
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fltoken.ours import OursScorer


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('compare_method')


def inputs():
    matrices = [np.array([[1.,0.],[.5,.5],[0.,1.]]), np.ones((2,2)), np.zeros((3,2))]
    return [(number, np.array([0,2] if len(matrix)==2 else [0,1,2]), matrix, [])
            for number,matrix in enumerate(matrices,1)]


def economics():
    return dict(stake_tokens=100, slash_numerator=1, slash_denominator=10, vesting_rounds=3)


def test_identical_arithmetic_keeps_every_scalar_and_integer_state(comparison):
    result = comparison.compare_rounds(inputs(), {}, economics(), 4, OursScorer)
    assert result['identical'] and result['rounds'] == 3 and result['participations'] == 8
    assert all(not any(row['changed'].values()) for row in result['stages'].values())
    assert [row['ids'] for row in result['input_rounds']] == [[0,1,2],[0,2],[0,1,2]]


def test_agreeing_implementations_do_not_imply_agreement_with_recorded_quality(comparison):
    matrix = np.ones((2,3))
    recorded = [dict(calibration_quality='.25') for _ in range(2)]
    result = comparison.compare_rounds([(1,np.array([0,1]),matrix,recorded)], {}, economics(), 3, OursScorer)
    assert result['identical'] and not result['original_matches_recorded']
    assert result['recorded_quality_entries'] == result['original_recorded_quality_differences'] == 2


def test_recorded_full_method_alias_is_checked_too(comparison):
    matrix = np.ones((2,3))
    recorded = [{},{}]
    for rule,switches in {**comparison.ABLATIONS,'Mstar':comparison.ABLATIONS['Afull']}.items():
        output = OursScorer(switches).score([0,1],matrix,1)
        for index,row in enumerate(recorded):
            for field in ['scores','shares','slash','selected']:
                suffix = {'scores':'score','shares':'share'}.get(field,field)
                row[f'{rule}_{suffix}'] = str(output[field][index])
    result = comparison.compare_rounds([(1,np.array([0,1]),matrix,recorded)], {}, economics(), 3, OursScorer)
    assert result['original_matches_recorded']
    recorded[0]['Mstar_slash'] = 'True'
    result = comparison.compare_rounds([(1,np.array([0,1]),matrix,recorded)], {}, economics(), 3, OursScorer)
    assert result['identical'] and not result['original_matches_recorded']
    assert result['original_recorded_output_differences'] == 1


@pytest.mark.parametrize('mutation',['missing','extra','different_order','different_source'])
def test_saved_calibration_quality_is_bound_to_every_original_participation(comparison,tmp_path,monkeypatch,mutation):
    from fltoken.io import sha256, write_json
    monkeypatch.chdir(tmp_path)
    folder = Path('results/pilots/fixture')
    folder.mkdir(parents=True)
    (folder/'per_round.csv').write_text('synthetic identity fixture\n',encoding='utf-8')
    output = Path('results/pilot_selection/geometry')/folder.name
    output.mkdir(parents=True)
    rows = [dict(round='1',client_id=str(client),update_set_sha256='matrix') for client in [0,1]]
    quality = [dict(**row,quality='.5') for row in rows]
    def bind():
        with (output/'quality.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=['round','client_id','update_set_sha256','quality'])
            writer.writeheader()
            writer.writerows(quality)
        write_json(output/'provenance.json',dict(geometry_source_sha256='a'*64,
                   quality_sha256=sha256(output/'quality.csv'),participation_sha256=sha256(folder/'per_round.csv')))
    bind()
    monkeypatch.setattr(comparison,'load_update_rounds',lambda *args:iter([(1,np.array([0,1]),np.ones((2,3)),rows)]))
    method = dict(scorer_sha256='a'*64)
    valid = list(comparison.comparison_rounds(folder,'pilots',method,'cache'))
    assert [row['calibration_quality'] for row in valid[0][3]] == ['.5','.5']
    if mutation == 'missing':quality.pop()
    elif mutation == 'extra':quality.append(quality[-1])
    elif mutation == 'different_order':quality.reverse()
    else:method['scorer_sha256'] = 'b'*64
    bind()
    with pytest.raises(ValueError,match='pilot'):
        list(comparison.comparison_rounds(folder,'pilots',method,'cache'))


def test_one_float_step_is_not_integer_payout_equivalence(comparison):
    class Shifted(OursScorer):
        def allocate(self, *args):
            result = super().allocate(*args)
            result['shares'][0] = np.nextafter(result['shares'][0], 1.)
            return result
    matrix = np.ones((2,3))
    left, right = OursScorer().score([0,1],matrix,1), Shifted().score([0,1],matrix,1)
    assert np.allclose(left['shares'], right['shares'])
    result = comparison.compare_rounds([(1,np.array([0,1]),matrix,[])], {}, economics(), 3, OursScorer, Shifted)
    assert not result['identical']
    for stage in result['stages'].values():
        assert stage['changed']['shares'] == 1 and stage['changed']['payouts'] == 2
        assert stage['absolute_payout_difference'] > 0 and stage['maximum_payout_difference'] > 0
        assert stage['changed']['settlement_state'] > 0


def test_same_minted_payout_does_not_imply_same_wealth(comparison):
    class Slashed(OursScorer):
        def allocate(self, *args):
            result = super().allocate(*args)
            result['slash'][0] = True
            return result
    result = comparison.compare_rounds([(1,np.array([0,1]),np.zeros((2,3)),[])], {}, economics(), 3, OursScorer, Slashed)
    for stage in result['stages'].values():
        assert stage['changed']['payouts'] == 0
        assert stage['changed']['slash'] == 1
        assert stage['changed']['settlement_state'] > 0
    assert not result['identical']


@pytest.mark.parametrize('rounds', [[], [(2,np.array([0,1]),np.ones((2,3)),[])]])
def test_missing_rounds_cannot_be_called_equivalent(comparison, rounds):
    with pytest.raises(ValueError, match='round'):
        comparison.compare_rounds(rounds, {}, economics(), 3, OursScorer)


def test_other_economics_cannot_reuse_the_fixed_integer_oracle(comparison):
    with pytest.raises(ValueError, match='economics'):
        comparison.compare_rounds(inputs(), {}, {**economics(), 'slash_denominator':20}, 3, OursScorer)


def test_historical_class_is_loaded_from_its_verified_recorded_bytes(comparison):
    method = comparison.recorded_method()
    original, revision = comparison.historical_scorer(method)
    assert len(revision) == 40
    sample = np.ones((2,3))
    output = original().score([0,1],sample,1)
    np.testing.assert_array_equal(output['shares'], [.5,.5])


def test_synthetic_diagnostics_cannot_claim_complete_corpus(comparison, tmp_path, monkeypatch):
    method = dict(params={**OursScorer.DEFAULTS}, settlement=economics(), scorer_sha256='a'*64)
    monkeypatch.setattr(comparison, 'recorded_method', lambda full_evidence:method)
    monkeypatch.setattr(comparison, 'historical_scorer', lambda value:(OursScorer, 'b'*40))
    monkeypatch.setattr(comparison, 'sha256', lambda path:'c'*64)
    def no_corpus(*args):
        raise AssertionError('synthetic diagnostics must not read empirical cells')
    monkeypatch.setattr(comparison, 'corpus_cells', no_corpus)
    target = tmp_path/'synthetic.json'
    monkeypatch.setattr(sys,'argv',['compare_method.py','--synthetic','--output',str(target)])
    comparison.main()
    result = json.loads(target.read_text())
    assert result['synthetic_only'] and not result['complete_corpus'] and result['runs'] == 0
    assert result['synthetic_cases'] == len(result['cases']) == 11
    assert result['manifest_sha256'] == {} and result['identical']


def test_both_corpora_are_checked_before_any_matrix_is_rescored(comparison,monkeypatch):
    monkeypatch.setattr(comparison,'recorded_method',lambda **kwargs:{})
    monkeypatch.setattr(comparison,'historical_scorer',lambda method:(OursScorer,'a'*40))
    monkeypatch.setattr(comparison,'published_bytes',lambda *args,**kwargs:b'')
    def corpus(kind,method):
        if kind == 'grid':
            raise ValueError('grid is incomplete')
        return [(Path('results/pilots/fixture'),{})]
    def no_score(*args,**kwargs):
        raise AssertionError('coverage must be checked first')
    monkeypatch.setattr(comparison,'corpus_cells',corpus)
    monkeypatch.setattr(comparison,'compare_rounds',no_score)
    monkeypatch.setattr(sys,'argv',['compare_method.py'])
    with pytest.raises(ValueError,match='grid is incomplete'):
        comparison.main()


@pytest.mark.parametrize('mutation', ['missing','duplicate','changed_source','unfinished_clock'])
def test_comparison_rejects_incomplete_or_relabelled_corpus(comparison, tmp_path, monkeypatch, mutation):
    from fltoken.io import sha256, write_json
    plan = dict(config='configs/base.yaml', dataset='mnist', output_root='results/grid',
                seeds=[941,947], attacks=['none'], fractions=[.2], rounds=2,
                device='cpu', workers=1, gpu_fraction=1., rules=['B0','B1','B2','Mstar'])
    method = dict(params={}, scorer_sha256='a'*64)
    monkeypatch.chdir(tmp_path)
    write_json('configs/evaluation_plan.json', plan)
    write_json('configs/method_frozen.json', method)
    Path('configs/base.yaml').write_text('num_clients: 10\nclients_per_round: 10\nrounds: 2\n',encoding='utf-8')
    sources = {'src/fltoken/ours.py':'a'*64}
    cells = list(comparison.evaluation_cells(plan))
    for cell in cells:
        folder = Path(cell['path'])
        config = {**comparison.load_config(plan['config']), **{key:cell[key] for key in ['seed','attack','fraction','rounds']},
                  **{key:plan[key] for key in ['device','workers','gpu_fraction']},
                  'rules':plan['rules']+list(comparison.ABLATIONS), 'ours_params':method['params']}
        write_json(folder/'metadata.json', dict(status='complete', config=config, source_config=plan['config'],
                   git_commit='b'*40, source_sha256=sources, source_git_dirty=False, versions={},
                   method_sha256=sha256('configs/method_frozen.json')))
        (folder/'per_round.csv').write_text('synthetic corpus identity fixture\n',encoding='utf-8')
        cell.update(source_commit='b'*40, metadata_sha256=sha256(folder/'metadata.json'),
                    per_round_sha256=sha256(folder/'per_round.csv'))
    binding = dict(plan_sha256=sha256('configs/evaluation_plan.json'), method_sha256=sha256('configs/method_frozen.json'))
    clock_path = Path('results/grid/execution.json')
    clock = dict(**binding, training_source_sha256=sources, started_utc='2026-01-01T00:00:00+00:00',
                 finished_utc='2026-01-01T00:00:01+00:00', elapsed_wall_seconds=1.)
    write_json(clock_path, clock)
    manifest = dict(**binding, expected_cells=2, completed_cells=2, cells=cells,
                    execution_sha256=sha256(clock_path), elapsed_wall_seconds=1.)
    monkeypatch.setattr(comparison,'published_bytes',lambda path:Path(path).read_bytes())
    monkeypatch.setattr(comparison,'verify_training_source',lambda metadata,method:None)
    write_json('results/grid/manifest.json',manifest)
    assert len(list(comparison.corpus_cells('grid',method))) == 2
    if mutation == 'missing':
        manifest['cells'].pop()
    elif mutation == 'duplicate':
        manifest['cells'][1] = manifest['cells'][0]
    elif mutation == 'changed_source':
        path = Path(cells[0]['path'])/'metadata.json'
        value = json.loads(path.read_text())
        value['source_sha256'] = {'src/fltoken/ours.py':'c'*64}
        write_json(path,value)
        manifest['cells'][0]['metadata_sha256'] = sha256(path)
    else:
        clock.update(finished_utc=None,elapsed_wall_seconds=None)
        write_json(clock_path,clock)
        manifest['execution_sha256'] = sha256(clock_path)
    write_json('results/grid/manifest.json',manifest)
    with pytest.raises(ValueError):
        list(comparison.corpus_cells('grid',method))
