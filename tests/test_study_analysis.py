"""Retain complete cohorts, reused controls and honest transfer comparisons."""
from copy import deepcopy
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    return importlib.import_module('analyze_studies')


def measurements(module, seed, tau=0.):
    return {rule:{name:(.8 if name == 'final_accuracy' else .1+index*.01+(seed-42)*.001+tau)
                  for name in module.METRICS} for index, rule in enumerate(module.RULES)}


@pytest.fixture
def noisy(analysis):
    plan = analysis.read(ROOT/'configs/coordinate_noise_study.json')
    plan['statistics']['bootstrap_resamples'] = 200
    frozen = dict(clipping_bound=.7, delta=1e-5)
    def record(cell, tau):
        privacy = dict(clipping_bound=.7, delta=1e-5, noise_multiplier=tau/.7)
        account = dict(clients=[dict(client_id=i, participations=i % 4+1, finite=tau > 0,
                                    epsilon=100000. if tau else None) for i in range(25)],
                       conservative_all_rounds=dict(epsilon=200000. if tau else None), limitation='synthetic fixture',
                       noise_standard_deviation=tau)
        return dict(**cell, config=dict(seed=cell['seed'], privacy=privacy, rules=analysis.RULES,
                                       rounds=20, privacy_scope='honest'), status='complete',
                    metrics=measurements(analysis, cell['seed'], tau), accounting=account,
                    failure=None, prefix_metrics=None)
    controls = [record(dict(kind='privacy', noise_multiplier=0., seed=seed, path=f'control-{seed}'), 0.)
                for seed in plan['seeds']]
    coordinates = [record(cell, cell['coordinate_standard_deviation']) for cell in analysis.coordinate_protocol.study_cells(plan)]
    return plan, frozen, controls, coordinates


def test_coordinate_cohort_reuses_controls_and_rejects_missing_or_changed_inputs(analysis, noisy):
    plan, frozen, controls, coordinates = noisy
    result = analysis.noise_summary(controls, coordinates, plan, frozen)
    assert result['runs'] == 12 and result['new_runs'] == 9 and result['reused_control_runs'] == 3
    assert [row['coordinate_standard_deviation'] for row in result['cells']] == [0., .001, .003, .01]
    assert result['cells'][0]['per_seed_accounting'][0]['infinite_epsilon_clients'] == 25
    assert len(result['cells'][1]['per_seed_accounting'][0]['accounting']['clients']) == 25
    effect = result['cells'][1]['paired_difference_from_zero_noise']['Mstar']['honest_false_penalty_rate']['estimate']
    assert effect['mean'] == pytest.approx(.001)
    tables = []
    importlib.import_module('privacy_assets').privacy_tables(result, lambda name, columns, header, rows, context:
        tables.append((name, columns, header, rows)))
    assert len(tables) == 8
    assert all(len(columns) == len(header) and all(len(row) == len(columns) for row in rows)
               for _, columns, header, rows in tables)
    assert len(next(rows for name, _, _, rows in tables if name == 'privacy_attacker_share')) == 24
    assert next(header for name, _, header, _ in tables if name == 'privacy_attacker_share')[0] == r'$\tau$'
    with pytest.raises(ValueError, match='unique declared'):
        analysis.noise_summary(controls, coordinates[:-1], plan, frozen)
    coordinates[0]['config']['privacy']['clipping_bound'] = .8
    with pytest.raises(ValueError, match='differ beyond'):
        analysis.noise_summary(controls, coordinates, plan, frozen)


def test_failed_or_undefined_seed_never_becomes_a_survivor_only_estimate(analysis, noisy):
    plan, frozen, controls, coordinates = noisy
    coordinates[0].update(status='failed_numerical', metrics=None, failure={'completed_rounds':2})
    result = analysis.noise_summary(controls, coordinates, plan, frozen)
    cell = result['cells'][1]
    assert cell['completed_runs'] == 2 and cell['numerical_failures'] == 1
    assert cell['metrics']['B1']['attacker_share']['estimate'] is None
    assert cell['metrics']['B1']['attacker_share']['undefined'] == 1
    assert 'hypothetical' in cell['per_seed_accounting'][0]['scope']
    coordinates[0].update(status='complete', failure=None, metrics=measurements(analysis, 42, .001))
    coordinates[0]['metrics']['RFFL'] = analysis.inference_metrics(coordinates[0]['metrics']['RFFL'], 1)
    result = analysis.noise_summary(controls, coordinates, plan, frozen)
    assert result['cells'][1]['metrics']['RFFL']['attacker_share']['estimate'] is None
    assert result['cells'][1]['metrics']['RFFL']['final_accuracy']['estimate']['mean'] == .8


def test_transfer_orders_use_matching_seeds_and_shared_accuracy(analysis):
    plan = analysis.read(ROOT/'configs/transfer_study.json')
    plan['statistics']['bootstrap_resamples'] = 200
    records = []
    for seed in plan['seeds']:
        for condition in plan['fashion']['conditions']:
            records.append(dict(seed=seed, **condition, status='complete', metrics=measurements(analysis, seed),
                                history=[dict(accuracy=.8) for _ in range(20)]))
    mnist = deepcopy(records)
    for record in records:
        record['metrics']['Mstar']['attacker_share'] = .01
    result = analysis.fashion_summary(records, mnist, plan)
    assert result['nominal_cells'] == 6 and result['seed_count'] == 3
    assert len(result['accuracy_trajectories']) == 40 and len(result['contrasts']) == 10
    attacker = [row for row in result['ordering'] if row['metric'] == 'attacker_share']
    assert all(row['defined_pairs'] == 15 and row['reversed_pairs'] == 3 for row in attacker)
    assert result['statistical_plan']['minimum_two_sided_seed_p'] == .25
    tables = []
    importlib.import_module('fashion_assets').fashion_tables(result, lambda name, columns, header, rows, context:
        tables.append((name, columns, header, rows)))
    assert len(tables) == 7
    assert len(next(rows for name, _, _, rows in tables if name == 'fashion_matched_comparison')) == 24
    assert all(len(columns) == len(header) and all(len(row) == len(columns) for row in rows)
               for _, columns, header, rows in tables)
    with pytest.raises(ValueError, match='matching seeds'):
        analysis.fashion_summary(records, mnist[:-1], plan)
    records[0]['metrics']['Mstar']['final_accuracy'] = .9
    with pytest.raises(ValueError, match='one model accuracy'):
        analysis.fashion_summary(records, mnist, plan)
