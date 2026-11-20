"""Full settlement claims require exact coverage, economic units and source identity."""

import copy
import csv
import importlib
import json
from pathlib import Path
import shutil

import pytest

from fltoken.data import sample_clients
from fltoken.io import sha256, write_json
from fltoken.metrics import summarize
from fltoken.ours import OursScorer


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    module = importlib.import_module('summarize_grid_settlement')
    replay = importlib.import_module('replay_grid')
    oracle_type = importlib.import_module('chain_ledger').ChainLedger
    sources = list(replay.provenance('scripts/replay_onchain.py'))+[
        'scripts/replay_grid.py','scripts/summarize_contracts.py','scripts/rebuild_ledgers.py']
    for name in sources:
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    monkeypatch.chdir(tmp_path)
    Path('configs').mkdir()
    Path('configs/synthetic.yaml').write_text('num_clients: 5\nclients_per_round: 3\nrounds: 4\n',encoding='utf-8')
    plan = dict(dataset='mnist', config='configs/synthetic.yaml', seeds=[711,733],
                attacks=['none','zero'], fractions=[.4], rounds=4, device='cpu', workers=1,
                gpu_fraction=1., rules=['B0','B1','B2','Mstar'])
    method = dict(params=OursScorer.DEFAULTS, settlement=dict(stake_tokens=100,
                  slash_numerator=1,slash_denominator=10,vesting_rounds=2),synthetic_fixture=True)
    write_json('configs/evaluation_plan.json',plan)
    write_json('configs/method_frozen.json',method)
    monkeypatch.setattr(module,'published_bytes',lambda path:Path(path).read_bytes())
    def published_method(full_evidence=False):
        assert full_evidence
        return method
    monkeypatch.setattr(module,'published_method',published_method)
    checked = []
    monkeypatch.setattr(module,'verify_training_source',lambda metadata:checked.append(metadata['git_commit']))
    source_hashes = dict(synthetic='fixture-only')
    binding = dict(plan_sha256=sha256('configs/evaluation_plan.json'),method_sha256=sha256('configs/method_frozen.json'))
    clock = dict(**binding,training_source_sha256=source_hashes,started_utc='2026-11-20T00:00:00+00:00',
                 finished_utc='2026-11-20T00:00:01+00:00',elapsed_wall_seconds=1.)
    write_json('results/grid/execution.json',clock)
    cells = list(replay.evaluation_cells(plan))
    records, states = [], {}
    base = module.load_config(plan['config'])
    for cell in cells:
        folder = Path(cell['path'])
        folder.mkdir(parents=True)
        config = {**base,**{key:cell[key] for key in ['seed','attack','fraction','rounds']},
                  **{key:plan[key] for key in ['device','workers','gpu_fraction']},
                  'rules':plan['rules']+list(module.ABLATIONS),'ours_params':method['params']}
        metadata = dict(status='complete',config=config,source_config=plan['config'],source_git_dirty=False,
                        git_commit='a'*40,source_sha256=source_hashes,versions=dict(synthetic=True),
                        method_sha256=binding['method_sha256'],synthetic_fixture=True)
        write_json(folder/'metadata.json',metadata)
        attacks = module.expected_attacks(config)
        write_json(folder/'protocol.json',dict(adversaries=attacks))
        history = {}
        for rule in config['rules']:
            oracle = oracle_type(range(5),2)
            history[rule], states[cell['path'],rule] = [], []
            for number in range(1,5):
                ids = sample_clients(5,3,number,cell['seed']).tolist()
                penalized = rule in ['Mstar','Afull']
                shares = [10**18//3]*3 if rule != 'B2' else [10**18,0,0]
                if penalized and number == 3:
                    shares = [0,0,0]
                amounts = [1000*share for share in shares]
                item = dict(round=number,ids=ids,shares1e18=shares,payouts=amounts,
                            slash=[penalized and number in [2,4]]*3,remainder=1000*10**18-sum(amounts))
                transition = oracle.settle(item)
                if number % 2 == 0:
                    oracle.claim(0)
                history[rule].append(item)
                states[cell['path'],rule].append(json.loads(json.dumps(dict(**transition,state=oracle.snapshot()))))
        write_json(folder/'ledger.json',history)
        rows = []
        for index, item in enumerate(history['B0']):
            for position, client in enumerate(item['ids']):
                row = dict(round=item['round'],client_id=client,adversary=client in attacks,accuracy=.8)
                for rule in config['rules']:
                    row[f'{rule}_share'] = history[rule][index]['shares1e18'][position]/10**18
                    row[f'{rule}_slash'] = history[rule][index]['slash'][position]
                rows.append(row)
        with (folder/'per_round.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        write_json(folder/'metrics.json',summarize(rows,history,config,attacks))
        records.append(dict(**cell,source_commit=metadata['git_commit'],metadata_sha256=sha256(folder/'metadata.json'),
                            per_round_sha256=sha256(folder/'per_round.csv')))
    manifest = dict(**binding,cells=records,completed_cells=len(records),expected_cells=len(records),
                    execution_sha256=sha256('results/grid/execution.json'),elapsed_wall_seconds=1.)
    write_json('results/grid/manifest.json',manifest)
    jobs, cases = list(module.replay_jobs(plan)), []
    for job in jobs:
        folder = Path(job['path'])
        evidence = Path('results/grid_onchain')/(job['path'].replace('/','_')+f"-{job['rule']}.json")
        record = dict(path=job['path'],rule=job['rule'],rounds=job['rounds'],clients=5,vesting_delay=2,
                      exact=True,provenance=replay.provenance('scripts/replay_onchain.py'),
                      pool_source_sha256=sha256('contracts/src/RewardPool.sol'),
                      token_source_sha256=sha256('contracts/src/RewardToken.sol'),
                      ledger_sha256=sha256(folder/'ledger.json'),participation_sha256=sha256(folder/'per_round.csv'),
                      states=states[job['path'],job['rule']],synthetic_fixture=True)
        write_json(evidence,record)
        cases.append(dict(path=job['path'],rule=job['rule'],rounds=job['rounds'],
                          evidence=evidence.as_posix(),evidence_sha256=sha256(evidence)))
    coverage = dict(**binding,complete=True,expected_jobs=len(jobs),completed_jobs=len(jobs),cases=cases,
                    expected_round_rule_cases=sum(job['rounds'] for job in jobs),
                    round_rule_cases=sum(job['rounds'] for job in jobs),
                    grid_manifest_sha256=sha256('results/grid/manifest.json'),driver_sha256=sha256('scripts/replay_grid.py'))
    write_json('results/grid_onchain/coverage.json',coverage)
    write_json('results/summary.json',{})
    return module,plan,manifest,coverage,checked


def test_complete_settlement_keeps_minted_burned_claimed_and_locked_units_separate(corpus):
    module,_,_,_,checked = corpus
    result = module.summarize()
    assert (result['grid_runs'],result['jobs'],result['round_rule_cases']) == (4,12,48)
    assert result['by_rule']['Mstar'] == dict(runs=4,round_rule_cases=16)
    assert result['by_rule']['B0'] == dict(runs=2,round_rule_cases=8)
    assert checked == ['a'*40]
    for case in result['cases']:
        totals = case['totals']
        assert totals['retained_rewards'] == totals['claimed_rewards']+totals['claimable_rewards']+totals['unvested_rewards']
        assert totals['minted_rewards'] == totals['retained_rewards']+totals['burned_rewards']
        assert totals['remaining_stake']+totals['burned_stake'] == 500*10**18
        assert totals['minted_rewards']+case['unminted'] == 4000*10**18
        if case['rule'] == 'Mstar':
            assert totals['burned_stake'] > 0 and totals['burned_rewards'] > 0
        else:
            assert totals['burned_stake'] == totals['burned_rewards'] == 0
    assert any(case['totals']['claimable_rewards'] > 0 and case['totals']['unvested_rewards'] > 0 for case in result['cases'])
    with Path('results/grid_onchain/economic_outcomes.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 24
    original = Path('results/grid_onchain/summary.json').read_bytes()
    assert module.summarize() == result
    assert Path('results/grid_onchain/summary.json').read_bytes() == original


@pytest.mark.parametrize('change',['missing','duplicate','order','round_count'])
def test_coverage_cannot_hide_missing_or_repeated_cases(corpus,change):
    module,plan,manifest,coverage,_ = corpus
    coverage = copy.deepcopy(coverage)
    if change == 'missing':
        coverage['cases'].pop()
        coverage['completed_jobs'] -= 1
    elif change == 'duplicate':
        coverage['cases'][-1] = coverage['cases'][0]
    elif change == 'order':
        coverage['cases'].reverse()
    else:
        coverage['round_rule_cases'] += 1
    with pytest.raises(ValueError):
        module.complete_jobs(plan,manifest,coverage)


@pytest.mark.parametrize('change',['metadata','identities','state'])
def test_rehashed_tampering_does_not_make_economic_evidence_valid(corpus,change):
    module,_,manifest,coverage,_ = corpus
    cell = manifest['cells'][0]
    folder = Path(cell['path'])
    if change == 'metadata':
        metadata = module.read(folder/'metadata.json')
        metadata['config']['num_clients'] += 1
        write_json(folder/'metadata.json',metadata)
        cell['metadata_sha256'] = sha256(folder/'metadata.json')
        write_json('results/grid/manifest.json',manifest)
        coverage['grid_manifest_sha256'] = sha256('results/grid/manifest.json')
    elif change == 'identities':
        write_json(folder/'protocol.json',dict(adversaries={'0':'zero'}))
    else:
        entry = coverage['cases'][0]
        case = module.read(entry['evidence'])
        case['states'][1]['state']['accounts']['0']['wallet'] += 1
        write_json(entry['evidence'],case)
        entry['evidence_sha256'] = sha256(entry['evidence'])
    write_json('results/grid_onchain/coverage.json',coverage)
    with pytest.raises(ValueError):
        module.summarize()
    assert not Path('results/grid_onchain/summary.json').exists()


def test_allocation_log_must_reconstruct_the_replayed_integer_ledger(corpus):
    module,_,manifest,coverage,_ = corpus
    cell = manifest['cells'][0]
    path = Path(cell['path'])/'per_round.csv'
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    rows[0]['B1_share'] = '0.0'
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    cell['per_round_sha256'] = sha256(path)
    write_json('results/grid/manifest.json',manifest)
    coverage['grid_manifest_sha256'] = sha256('results/grid/manifest.json')
    for entry in coverage['cases']:
        if entry['path'] != cell['path']:
            continue
        case = module.read(entry['evidence'])
        case['participation_sha256'] = sha256(path)
        write_json(entry['evidence'],case)
        entry['evidence_sha256'] = sha256(entry['evidence'])
    write_json('results/grid_onchain/coverage.json',coverage)
    with pytest.raises(AssertionError,match='ledger or metrics mismatch'):
        module.summarize()


def test_zero_issuance_is_undefined_allocation_with_real_stake_burns(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module = importlib.import_module('summarize_grid_settlement')
    oracle = importlib.import_module('chain_ledger').ChainLedger(range(2),3)
    item = dict(round=1,ids=[0,1],shares1e18=[0,0],payouts=[0,0],slash=[True,True],remainder=1000*10**18)
    transition = oracle.settle(item)
    states = json.loads(json.dumps([dict(**transition,state=oracle.snapshot())]))
    result = module.economic_outcomes([item],states,{1:'zero'})
    assert result['minted_attacker_share'] is None
    assert result['totals']['burned_rewards'] == result['totals']['retained_rewards'] == 0
    assert result['totals']['burned_stake'] == 20*10**18
    assert result['final_supply'] == 180*10**18


def test_settlement_assets_only_present_complete_exact_coverage(corpus):
    module,_,_,_,_ = corpus
    module.summarize()
    document = importlib.import_module('document_grid_settlement')
    document.main()
    assert '12 declared run-rule jobs reproduce 48 round-rule states' in Path('docs/GRID_SETTLEMENT.md').read_text()
    table = Path('paper/generated/settlement_coverage.tex').read_text()
    assert r'B1 & 4 & 16 \\' in table and r'B0 & 2 & 8 \\' in table
    summary = module.read('results/summary.json')
    # The public entry points consume this same complete summary, including
    # the distinct baseline sample rather than implying full equal coverage.
    import shutil
    root = Path(__file__).resolve().parents[1]
    for name in ['README.md','configs/sendros_mix.yaml','results/literature/sendros.json']:
        target = Path(name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    baseline = module.read(root/'results/summary.json')
    summary.update({key:baseline[key] for key in ['contracts','gas','published']})
    write_json('results/summary.json',summary)
    contracts = importlib.import_module('document_contracts')
    macros = importlib.import_module('make_macros')
    contracts.main()
    macros.main()
    assert '48 round-rule states in 12 jobs' in Path('README.md').read_text()
    assert '4 training runs' in Path('docs/CONTRACTS.md').read_text()
    assert r'\newcommand{\SettlementGridRuns}{4}' in Path('paper/generated/numbers.tex').read_text()
    assert r'\newcommand{\SettlementEqualRuns}{2}' in Path('paper/generated/numbers.tex').read_text()
    assert r'\newcommand{\SettlementOursRuns}{4}' in Path('paper/generated/numbers.tex').read_text()
    summary['grid_settlement']['complete'] = False
    write_json('results/summary.json',summary)
    for renderer in [document,contracts,macros]:
        with pytest.raises(ValueError,match='complete exact'):
            renderer.main()


@pytest.mark.parametrize('field',['wallet','payout'])
def test_equal_floats_do_not_replace_integer_accounting_evidence(corpus,field):
    module,_,_,coverage,_ = corpus
    entry = coverage['cases'][0]
    case = module.read(entry['evidence'])
    ledger = module.read(Path(entry['path'])/'ledger.json')[entry['rule']]
    if field == 'wallet':
        case['states'][0]['state']['accounts']['0']['wallet'] = 0.0
    else:
        ledger[0]['payouts'][0] = float(ledger[0]['payouts'][0])
    with pytest.raises(ValueError,match='integer base units'):
        module.economic_outcomes(ledger,case['states'],{})


def test_recorded_chain_histories_preserve_economic_decomposition(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    monkeypatch.chdir(root)
    module = importlib.import_module('summarize_grid_settlement')
    validator = importlib.import_module('summarize_contracts').validate_states
    original = module.read('results/onchain/summary.json')
    rounds, burns = 0, 0
    for entry in original['cases']:
        folder = Path(entry['path'])
        path = Path('results/onchain')/(folder.as_posix().replace('/','_')+f"-{entry['rule']}.json")
        case = module.read(path)
        module.validate_case(case,entry,folder,case['vesting_delay'])
        ledger = module.read(folder/'ledger.json')[entry['rule']]
        attacks = {int(client):attack for client,attack in module.read(folder/'protocol.json')['adversaries'].items()}
        result = module.economic_outcomes(ledger,case['states'],attacks)
        assert result['minted_attacker_share'] == module.read(folder/'metrics.json')[entry['rule']]['attacker_share']
        assert result['totals']['retained_rewards'] == result['totals']['minted_rewards']
        rounds += case['rounds']
    stress = module.read('results/onchain/stress.json')
    for case in stress['cases']:
        ledger = [state['input'] for state in case['states']]
        validator(case['clients'],case['vesting_delay'],ledger,case['states'])
        result = module.economic_outcomes(ledger,case['states'],{})
        burns += result['totals']['burned_rewards']+result['totals']['burned_stake']
        rounds += case['rounds']
    assert rounds == original['round_rule_cases']+stress['round_cases']
    assert burns > 0
    print(json.dumps(dict(recorded_cases=len(original['cases'])+len(stress['cases']),
                          round_rule_states=rounds,stress_burned_base_units=burns),sort_keys=True))
