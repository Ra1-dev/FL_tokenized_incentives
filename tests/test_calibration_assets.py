"""Selection assets preserve costs, calibration counts and pending-result boundaries."""
import importlib
import json
from pathlib import Path
import shutil

import pytest


def records():
    def candidate(name, family, beta, attacker, honest):
        return dict(name=name, family=family, params={} if beta is None else dict(beta=beta),
                    objective=attacker+honest, mean_attacker_share_cost=attacker,
                    mean_honest_false_penalty_rate=honest, undefined_state_rounds=0)
    rows = [candidate('CGSV-A','CGSV',.75,.2,.3), candidate('CGSV-B','CGSV',1.25,.1,.25),
            candidate('RFFL-A','RFFL',None,.15,.4)]
    selected = dict(CGSV=rows[1],RFFL=rows[2])
    extension = dict(cells=37,training_device='cuda',candidates=rows,selected=selected,
                     configuration=dict(params={name:row['params'] for name,row in selected.items()}))
    evidence = dict(clipping_bound=6.5,participations=12,training_device='cuda',
                    cells=[dict(seed=1223,participations=6,minimum=1.,median=3.5,maximum=6.),
                           dict(seed=1229,participations=6,minimum=7.,median=9.5,maximum=12.)])
    return dict(extension_design=dict(selection=extension),
                privacy_calibration=dict(configuration=dict(clipping_bound=6.5),evidence=evidence),
                synthetic_fixture=True)


def test_tables_and_macros_use_the_same_recorded_calibrations(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    for name in ['configs/sendros_mix.yaml','results/literature/sendros.json']:
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    summary = records()
    summary['reproduction'] = json.loads((root/'results/summary.json').read_text())['reproduction']
    monkeypatch.chdir(tmp_path)
    Path('results/summary.json').write_text(json.dumps(summary),encoding='utf-8')
    tables, macros = [importlib.import_module(name) for name in ['make_tables','make_macros']]
    tables.main()
    macros.main()
    text = Path('paper/generated/extension_pilot_selection.tex').read_text()
    assert text.count(r'$\star$') == 2
    assert r'CGSV-B$\star$ & 1.25 & 0.10000000 & 0.25000000 & 0.35000000 & 0' in text
    assert r'RFFL-A$\star$ & -- & 0.15000000 & 0.40000000 & 0.55000000 & 0' in text
    assert 'training device: CUDA' in text
    text = Path('paper/generated/privacy_calibration.tex').read_text()
    assert r'1223 & 6 & 1 & 3.5 & 6' in text and r'1229 & 6 & 7 & 9.5 & 12' in text
    text = Path('paper/generated/numbers.tex').read_text()
    for name,value in [('ExtensionPilotCells','37'),('ExtensionPilotCandidates','3'),
                       ('ExtensionCGSVBeta','1.25'),('ExtensionCGSVPilotObjective','0.35000000'),
                       ('ExtensionRFFLPilotHonestCost','0.40000000'),('PrivacyCalibrationClipBound','6.5'),
                       ('PrivacyCalibrationParticipations','12'),('PrivacyCalibrationSeeds','2')]:
        assert rf'\newcommand{{\{name}}}{{{value}}}' in text
    summary.pop('extension_design')
    summary.pop('privacy_calibration')
    Path('results/summary.json').write_text(json.dumps(summary),encoding='utf-8')
    tables.main()
    macros.main()
    assert not Path('paper/generated/extension_pilot_selection.tex').exists()
    assert not Path('paper/generated/privacy_calibration.tex').exists()
    text = Path('paper/generated/numbers.tex').read_text()
    assert 'ExtensionCGSVBeta' not in text and 'PrivacyCalibrationClipBound' not in text


@pytest.mark.parametrize('family',['extension','privacy'])
def test_readme_status_does_not_turn_calibration_into_evaluation(family,tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module = importlib.import_module('document_extensions' if family == 'extension' else 'document_privacy')
    path = tmp_path/'README.md'
    path.write_text(f'Before\n<!-- generated {family} start -->\nold\n<!-- generated {family} end -->\nAfter\n',encoding='utf-8')
    module.update_readme({},path)
    assert 'pending' in path.read_text()
    summary = records()
    module.update_readme(summary,path)
    text = path.read_text()
    assert 'Empirical' in text and 'remain pending.' in text
    assert ('37 pilot cells' if family == 'extension' else '12 participations across 2') in text
    summary['extensions' if family == 'extension' else 'privacy'] = dict(runs=23, completed_runs=19, numerical_failures=4)
    module.update_readme(summary,path)
    text = path.read_text()
    assert ('23 runs' if family == 'extension' else '23 attempted runs: 19 complete and 4 numerical failures') in text and 'pending' not in text
    assert text.startswith('Before\n') and text.endswith('\nAfter\n')
    module.update_readme({},path)
    assert '23 ' not in path.read_text()
    path.write_text('missing block\n',encoding='utf-8')
    with pytest.raises(ValueError,match='one ordered'):
        module.update_readme(summary,path)
