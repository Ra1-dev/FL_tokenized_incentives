"""Compare recorded and current reward arithmetic on the complete saved corpus."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np

from chain_ledger import ChainLedger
from evaluation import (ABLATIONS, METHOD_BINDINGS, evaluation_cells, published_bytes,
                        recorded_method, read, repository_git, verify_frozen_sources)
from fltoken.cache import load_update_rounds
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from fltoken.metrics import ROUND_POOL, payout
from fltoken.ours import OursScorer
from grid_reproduction import verify_training_source
from run_grid import checked_execution_clock
from run_pilots import cells as pilot_cells


def historical_scorer(method):
    revision = verify_frozen_sources('configs/method_frozen.json',
                                    {source: method[field] for field, source in METHOD_BINDINGS.items()})
    command, options = repository_git()
    source = subprocess.check_output([*command, 'show', f'{revision}:src/fltoken/ours.py'],
                                     **options, stderr=subprocess.PIPE)
    if hashlib.sha256(source).hexdigest() != method['scorer_sha256']:
        raise ValueError('historical scorer differs from the selected method')
    module = types.ModuleType('recorded_reward_arithmetic')
    exec(compile(source, f'{revision}:src/fltoken/ours.py', 'exec'), module.__dict__)
    return module.OursScorer, revision


def difference_counts(left, right):
    """Count changed scalar leaves exactly, including absent-client state."""
    if isinstance(left, dict) and isinstance(right, dict):
        return sum(difference_counts(left[key], right[key]) if key in left and key in right else 1
                   for key in left.keys() | right.keys())
    if isinstance(left, (list, tuple, np.ndarray)) and isinstance(right, (list, tuple, np.ndarray)):
        if len(left) != len(right):
            raise ValueError('comparison output shapes differ')
        return sum(difference_counts(a, b) for a, b in zip(left, right))
    return int(left != right)


def compare_rounds(rounds, params, settlement, clients, original, current=OursScorer):
    """Score each matrix once per implementation, then compare all component stages."""
    if (settlement['stake_tokens'], settlement['slash_numerator'], settlement['slash_denominator']) != (100, 1, 10):
        raise ValueError('selected economics differ from the integer settlement oracle')
    geometry = [implementation(dict(ema=False, penalties=False)) for implementation in [original, current]]
    states = {name: [implementation({**params, **switches}) for implementation in [original, current]]
              for name, switches in ABLATIONS.items()}
    ledgers = {name: [ChainLedger(range(clients), settlement['vesting_rounds'] if switches['penalties'] else 0)
                      for _ in range(2)] for name, switches in ABLATIONS.items()}
    fields = ['scores', 'shares', 'slash', 'selected', 'reputation', 'low_count',
              'shares1e18', 'payouts', 'remainder', 'settlement_state']
    result = dict(rounds=0, participations=0, input_rounds=[],
                  recorded_quality_entries=0, original_recorded_quality_differences=0,
                  original_recorded_output_differences=0,
                  stages={name: dict(changed={field: 0 for field in fields},
                                     absolute_payout_difference=0, maximum_payout_difference=0,
                                     maximum_score_difference=0., maximum_share_difference=0.)
                          for name in ABLATIONS})
    for number, ids, matrix, recorded in rounds:
        if number != result['rounds']+1:
            raise ValueError('comparison requires every round exactly once in order')
        result['rounds'] += 1
        result['participations'] += len(ids)
        result['input_rounds'].append(dict(round=number, ids=list(map(int, ids)),
                                           matrix_sha256=hashlib.sha256(matrix.astype('<f8').tobytes()).hexdigest()))
        quality = [scorer.score(ids, matrix, number)['scores'] for scorer in geometry]
        if recorded:
            if len(recorded) != len(ids):
                raise ValueError('recorded quality coverage differs from the sampled clients')
            key = 'calibration_quality' if 'calibration_quality' in recorded[0] else 'Mstar_score'
            stored = np.array([float(row[key]) for row in recorded])
            if not np.isfinite(stored).all() or ((stored < 0) | (stored > 1)).any():
                raise ValueError('invalid recorded quality')
            result['recorded_quality_entries'] += len(stored)
            result['original_recorded_quality_differences'] += difference_counts(quality[0], stored)
        for name, scorers in states.items():
            outputs = [scorer.allocate(ids, values, number) for scorer, values in zip(scorers, quality)]
            if recorded and 'Mstar_score' in recorded[0]:
                for rule in [name, *(['Mstar'] if name == 'Afull' else [])]:
                    for field in ['scores','shares','slash','selected']:
                        suffix = {'scores':'score','shares':'share'}.get(field,field)
                        raw = [row[f'{rule}_{suffix}'] for row in recorded]
                        if field in ['slash','selected']:
                            if any(value not in {'True','False'} for value in raw):
                                raise ValueError('recorded decision must be boolean')
                            stored = [value == 'True' for value in raw]
                        else:
                            stored = list(map(float,raw))
                            if any(not np.isfinite(value) or not 0 <= value <= 1 for value in stored):
                                raise ValueError('invalid recorded score or share')
                        result['original_recorded_output_differences'] += difference_counts(outputs[0][field],stored)
            measured = result['stages'][name]
            for field in ['scores', 'shares', 'slash', 'selected']:
                measured['changed'][field] += difference_counts(outputs[0][field], outputs[1][field])
            for field in ['scores', 'shares']:
                key = 'maximum_'+field[:-1]+'_difference'
                measured[key] = max(measured[key], float(np.max(np.abs(outputs[0][field]-outputs[1][field]))))
            for field in ['reputation', 'low_count']:
                measured['changed'][field] += difference_counts(outputs[0]['state'][field], outputs[1]['state'][field])
            histories = []
            for ledger, output in zip(ledgers[name], outputs):
                units, amounts = payout(output['shares'])
                item = dict(round=number, ids=list(map(int, ids)), shares1e18=units, payouts=amounts,
                            slash=output['slash'].tolist(), remainder=ROUND_POOL-sum(amounts))
                ledger.settle(item)
                for client in range(clients):
                    ledger.claim(client)
                histories.append(dict(**item, settlement_state=ledger.snapshot()))
            for field in ['shares1e18', 'payouts', 'remainder', 'settlement_state']:
                measured['changed'][field] += difference_counts(histories[0][field], histories[1][field])
            changes = [abs(a-b) for a,b in zip(histories[0]['payouts'], histories[1]['payouts'])]
            measured['absolute_payout_difference'] += sum(changes)
            measured['maximum_payout_difference'] = max(measured['maximum_payout_difference'], *changes)
    if not result['rounds']:
        raise ValueError('comparison requires at least one complete round')
    result['identical'] = all(not any(stage['changed'].values()) for stage in result['stages'].values())
    result['original_matches_recorded'] = (not result['original_recorded_quality_differences']
                                           and not result['original_recorded_output_differences']
                                           if result['recorded_quality_entries'] else None)
    return result


def comparison_rounds(folder, kind, method, cache_root):
    """Keep the original recorded quality as a separate reconstruction check."""
    quality = None
    if kind == 'pilots':
        output = Path('results/pilot_selection/geometry')/folder.name
        evidence = read(output/'provenance.json')
        if (evidence['geometry_source_sha256'] != method['scorer_sha256']
                or evidence['quality_sha256'] != sha256(output/'quality.csv')
                or evidence['participation_sha256'] != sha256(folder/'per_round.csv')):
            raise ValueError('pilot qualities differ from the frozen recorded geometry')
        with (output/'quality.csv').open(encoding='utf-8',newline='') as stream:
            quality = list(csv.DictReader(stream))
    offset = 0
    for number, ids, matrix, rows in load_update_rounds(folder, cache_root):
        if quality is not None:
            supplied = quality[offset:offset+len(rows)]
            fields = ['round','client_id','update_set_sha256']
            if len(supplied) != len(rows) or any(any(a[field] != b[field] for field in fields) for a,b in zip(rows,supplied)):
                raise ValueError('pilot qualities differ from the original matrix and client order')
            rows = [dict(row,calibration_quality=value['quality']) for row,value in zip(rows,supplied)]
        offset += len(rows)
        yield number, ids, matrix, rows
    if quality is not None and offset != len(quality):
        raise ValueError('pilot geometry contains extra participations')


def corpus_cells(kind, method, input_root=Path('.'), available=False):
    if available and kind != 'grid':
        raise ValueError('partial coverage is supported only for explicit grid diagnostics')
    input_root = Path(input_root)
    path = Path('configs/pilot_plan.json' if kind == 'pilots' else 'configs/evaluation_plan.json')
    plan = json.loads(published_bytes(path))
    manifest_path = input_root/'results'/kind/'manifest.json'
    manifest = read(manifest_path)
    planned = list(pilot_cells(plan) if kind == 'pilots' else evaluation_cells(plan))
    count = manifest['completed_cells']
    if (manifest['plan_sha256'] != sha256(path) or manifest['expected_cells'] != len(planned)
            or not 0 < count <= len(planned) or len(manifest['cells']) != count
            or not available and count != len(planned)
            or [{key: cell[key] for key in planned[0]} for cell in manifest['cells']] != planned[:count]):
        raise ValueError('comparison requires every declared corpus cell exactly once')
    if kind == 'grid' and manifest['method_sha256'] != sha256('configs/method_frozen.json'):
        raise ValueError('grid belongs to a different frozen method')
    if kind == 'grid':
        clock_path = input_root/'results/grid/execution.json'
        clock = checked_execution_clock(clock_path, {key:manifest[key] for key in ['plan_sha256','method_sha256']})
        if (not available and clock['finished_utc'] is None or manifest['execution_sha256'] != sha256(clock_path)
                or manifest['elapsed_wall_seconds'] != clock['elapsed_wall_seconds']):
            raise ValueError('comparison requires the completed original grid clock')
        training_sources = clock['training_source_sha256']
    else:
        training_sources = manifest['training_source_sha256']
    base, common, checked = load_config(plan['config']), None, set()
    for cell in manifest['cells']:
        folder = input_root/cell['path']
        metadata = read(folder/'metadata.json')
        keys = ['seed', 'attack', 'fraction'] + (['rounds'] if kind == 'grid' else [])
        expected = {**base, **{key:cell[key] for key in keys},
                    **{key:plan[key] for key in ['device','workers','gpu_fraction']}}
        if kind == 'grid':
            expected.update(rules=plan['rules']+list(ABLATIONS), ours_params=method['params'])
        if (metadata['status'] != 'complete' or metadata['config'] != expected
                or metadata['source_config'] != plan['config'] or metadata['git_commit'] != cell['source_commit']
                or metadata['source_sha256'] != training_sources
                or sha256(folder/'per_round.csv') != cell['per_round_sha256']):
            raise ValueError('comparison input differs from its declared training record')
        if kind == 'grid' and (metadata['method_sha256'] != manifest['method_sha256']
                               or sha256(folder/'metadata.json') != cell['metadata_sha256']):
            raise ValueError('grid training method differs from its manifest')
        implementation = (metadata['source_sha256'], metadata['versions'])
        if common is not None and implementation != common:
            raise ValueError('corpus training implementations differ')
        common = implementation
        source_key = (metadata['git_commit'], json.dumps(metadata['source_sha256'], sort_keys=True), metadata['source_git_dirty'])
        if source_key not in checked:
            verify_training_source(metadata, method=method)
            checked.add(source_key)
        yield folder, metadata


def synthetic_cases(method, original, current=OursScorer):
    """Boundary traces distinguish arithmetic repairs from empirical outcomes."""
    small = np.nextafter(0., 1.)
    matrices = [(f'aligned-{count}', np.full((count,3), small)) for count in [2,3,7,9,10,13]]
    matrices += [(f'orthogonal-{multiple}', np.array([[1.,multiple*small,multiple*small],
                                                   [-2*multiple*small,1.,1.]])) for multiple in [1,2,3,16]]
    tiny = np.finfo(float).tiny
    matrices.append(('normal-input-tiny-midpoint', np.array([[-tiny],[tiny],[-np.nextafter(tiny,np.inf)]])))
    matrices.append(('positive-quality-product-underflow', np.array([[small,1.],[1.,0.]])))
    results = []
    for name, matrix in matrices:
        rounds = [(number, np.arange(len(matrix)), matrix, []) for number in range(1,method['params']['patience']+1)]
        result = compare_rounds(rounds, method['params'], method['settlement'], len(matrix), original, current)
        result.update(synthetic_case=name, matrix=matrix.tolist())
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true', help='compare only declared arithmetic boundary traces; no empirical corpus claim')
    parser.add_argument('--cache-root', type=Path, default=Path('results/cache'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    method = recorded_method(full_evidence=not args.synthetic)
    original, revision = historical_scorer(method)
    sources = ['src/fltoken/ours.py', 'scripts/compare_method.py', 'scripts/chain_ledger.py',
               'src/fltoken/cache.py', 'src/fltoken/metrics.py', 'src/fltoken/io.py', 'scripts/evaluation.py',
               'scripts/grid_reproduction.py', 'scripts/run_grid.py', 'scripts/run_pilots.py', 'src/fltoken/config.py']
    if args.synthetic:
        results = synthetic_cases(method, original)
    else:
        for source in sources:
            published_bytes(source, reference='HEAD')
        corpora = {kind:list(corpus_cells(kind, method)) for kind in ['pilots','grid']}
        results = []
        for kind, entries in corpora.items():
            for folder, metadata in entries:
                result = compare_rounds(comparison_rounds(folder, kind, method, args.cache_root), method['params'], method['settlement'],
                                        metadata['config']['num_clients'], original)
                if (result['rounds'] != metadata['config']['rounds']
                        or result['recorded_quality_entries'] != result['participations']):
                    raise ValueError('comparison omitted a recorded round')
                result.update(corpus=kind, path=folder.as_posix(), inputs={name:sha256(folder/name) for name in
                              ['metadata.json','per_round.csv','round_metrics.csv','ledger.json','hashes.json']})
                if kind == 'pilots':
                    result['calibration_inputs'] = {name:sha256(Path('results/pilot_selection/geometry')/folder.name/name)
                                                    for name in ['quality.csv','provenance.json']}
                results.append(result)
                print(f"Compared {folder.name}: identical={result['identical']}", flush=True)
    output = dict(schema=1, complete_corpus=not args.synthetic, synthetic_only=args.synthetic,
                  identical=all(row['identical'] for row in results),
                  runs=0 if args.synthetic else len(results), synthetic_cases=len(results) if args.synthetic else 0,
                  changed_cases=sum(not row['identical'] for row in results),
                  original_matches_recorded=None if args.synthetic else all(row['original_matches_recorded'] for row in results),
                  rounds=sum(row['rounds'] for row in results), cases=results,
                  historical_commit=revision, original_scorer_sha256=method['scorer_sha256'],
                  current_scorer_sha256=sha256('src/fltoken/ours.py'), method_sha256=sha256('configs/method_frozen.json'),
                  manifest_sha256={} if args.synthetic else {kind:sha256(f'results/{kind}/manifest.json') for kind in ['pilots','grid']},
                  analysis_source_sha256={source:sha256(source) for source in sources}, numpy_version=np.__version__,
                  interpretation='Arithmetic comparison at fixed selected parameters on identical matrices; no reselection or replacement of recorded experimental outcomes. Original reconstruction versus recorded quality and grid outputs is checked separately from old-versus-current arithmetic. Full penalties equal Mstar. Each stage uses its declared vesting delay and claims all matured balances after every round. Synthetic traces do not establish empirical equivalence.')
    target = args.output or Path('results/validation/method_numerics.json' if args.synthetic else 'results/validation/method_comparison.json')
    write_json(target, output)
    print(json.dumps({key:value for key,value in output.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
