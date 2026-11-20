"""Pilot display assets preserve selection identity and raw cost units."""

import copy
import importlib
import json
from pathlib import Path
import shutil

import pytest


def test_pilot_table_and_macros_follow_the_recorded_selection(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    selected = dict(candidate=1, params=dict(reference=True, ema=True, penalties=True, beta=.4,
                    threshold=.05, patience=3, initial_reputation=.5),
                    mean_attacker_share_cost=.10001234, mean_honest_false_penalty_rate=.2,
                    objective=.30001234, undefined_attacker_cells=2, empty_rounds=9, honest_clients_slashed_sum=17)
    other = copy.deepcopy(selected)
    other.update(candidate=0, mean_attacker_share_cost=.15, mean_honest_false_penalty_rate=.25, objective=.4)
    record = dict(configuration=dict(params=selected['params'], settlement=dict(vesting_rounds=3,
                  stake_tokens=100, slash_numerator=1, slash_denominator=10)),
                  selection=dict(cells=85, candidates=[other,selected], selected=selected, synthetic_fixture=True))
    original = json.loads((root/'results/summary.json').read_text(encoding='utf-8'))
    summary = dict(reproduction=original['reproduction'], method_selection=record)
    for folder in ['results/literature','configs']:
        (tmp_path/folder).mkdir(parents=True)
    shutil.copyfile(root/'configs/sendros_mix.yaml',tmp_path/'configs/sendros_mix.yaml')
    shutil.copyfile(root/'results/literature/sendros.json',tmp_path/'results/literature/sendros.json')
    monkeypatch.chdir(tmp_path)
    Path('results/summary.json').write_text(json.dumps(summary),encoding='utf-8')
    tables = importlib.import_module('make_tables')
    macros = importlib.import_module('make_macros')
    tables.main()
    macros.main()
    table = Path('paper/generated/pilot_selection.tex').read_text()
    assert table.count(r'$\star$') == 1
    assert r'1$\star$ & 0.4 & 0.05 & 3 & 0.10001234 & 0.20000000 & 0.30001234' in table
    assert '0 & 0.4 & 0.05 & 3 & 0.15000000 & 0.25000000 & 0.40000000' in table
    text = Path('paper/generated/numbers.tex').read_text()
    for name,value in [('PilotSelectedCandidate','1'),('PilotObjectiveSum','0.30001234'),
                       ('PilotAttackerCost','0.10001234'),('PilotHonestCost','0.20000000'),
                       ('PilotUndefinedCells','2'),('PilotEmptyRounds','9'),('PilotHonestSlashedSum','17')]:
        assert rf'\newcommand{{\{name}}}{{{value}}}' in text
    summary.pop('method_selection')
    Path('results/summary.json').write_text(json.dumps(summary),encoding='utf-8')
    tables.main()
    macros.main()
    assert not Path('paper/generated/pilot_selection.tex').exists()
    assert 'PilotObjectiveSum' not in Path('paper/generated/numbers.tex').read_text()


def test_method_status_separates_pilot_selection_from_evaluation(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    module = importlib.import_module('document_method')
    path = tmp_path/'README.md'
    path.write_text('Before\n<!-- generated method start -->\nold\n<!-- generated method end -->\nAfter\n',encoding='utf-8')
    module.update_readme({},path)
    assert 'Empirical pilot selection and evaluation remain pending.' in path.read_text()
    summary = dict(method_selection=dict(selection=dict(selected=dict(candidate=4),candidates=[{}, {}, {}],cells=17)))
    module.update_readme(summary,path)
    text = path.read_text()
    assert 'candidate 4 from 3 candidates over 17 cells' in text
    assert 'Main evaluation results remain pending.' in text
    assert text.startswith('Before\n') and text.endswith('\nAfter\n')
    summary['evaluation'] = dict(nominal_cells=23)
    module.update_readme(summary,path)
    assert 'complete evaluation summary covers 23 runs' in path.read_text()
    assert 'pending' not in path.read_text()
    module.update_readme({},path)
    assert 'candidate 4' not in path.read_text()
    path.write_text('Missing generated block\n',encoding='utf-8')
    with pytest.raises(ValueError, match='one ordered'):
        module.update_readme({},path)
