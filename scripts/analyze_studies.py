"""Reconstruct the declared noise and dataset-transfer comparisons."""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import coordinate_protocol
from evaluation import read
from extension_replay import record_geometry, replay_geometry
from fltoken.data import sample_clients
from fltoken.io import sha256, write_json
from fltoken.statistics import holm, paired_wilcoxon
from run_grid import checked_execution_clock
import study_protocol
from summarize_grid import METRICS, interval, mean_complete

RULES = ['B0', 'B1', 'B2', 'Mstar', 'CGSV', 'RFFL']
COSTS = ['attacker_share', 'honest_false_penalty_rate']
UNDEFINED_METRICS = ['attacker_share', 'honest_false_penalty_rate', 'honest_gini', 'precision', 'recall', 'f1']


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def average(values):
    value = mean_complete(values)
    return None if value is None else float(value)


def inference_metrics(metrics, undefined_rounds=0):
    result = dict(metrics, undefined_state_rounds=undefined_rounds)
    if undefined_rounds:
        result.update({name:None for name in UNDEFINED_METRICS})
    return result


def load_cohort(protocol, directory):
    plan, method, privacy, extensions, _, effective = protocol.published_study(full_evidence=True)
    planned = list(protocol.study_cells(plan))
    manifest_path = Path(directory)/'manifest.json'
    manifest = read(manifest_path)
    bound = protocol.sources(plan)
    binding = dict(plan_sha256=sha256(protocol.PLAN), training_source_sha256=bound,
                   effective_method=effective,
                   dataset_sha256={name:sha256(path) for name, path in plan['dataset_manifests'].items()})
    if (not manifest.get('complete') or manifest.get('expected_cells') != len(planned) or
            manifest.get('attempted_cells') != len(planned) or
            any(manifest.get(key) != value for key, value in binding.items()) or
            [{key:cell.get(key) for key in planned[0]} for cell in manifest['cells']] != planned):
        raise ValueError('study manifest requires every unique declared terminal cell in order')
    clock_path = Path(directory)/'execution.json'
    clock = checked_execution_clock(clock_path, binding)
    if (clock['finished_utc'] is None or manifest['execution_sha256'] != sha256(clock_path) or
            manifest['elapsed_wall_seconds'] != clock['elapsed_wall_seconds']):
        raise ValueError('study must retain its completed execution clock')
    coordinate = protocol is coordinate_protocol
    identity_key = 'coordinate_cell' if coordinate else 'study_cell'
    plan_key = 'coordinate_plan_sha256' if coordinate else 'study_plan_sha256'
    records, checked, versions = [], set(), None
    for cell, evidence in zip(planned, manifest['cells']):
        config = protocol.expected_config(plan, cell, method, privacy)
        folder = Path(cell['path'])
        metadata, inputs = study_protocol.verify_cell(folder, config, verify_cache=False)
        if (inputs != evidence['inputs'] or metadata.get(identity_key) != cell or
                metadata.get(plan_key) != sha256(protocol.PLAN) or metadata['source_config'] != plan['config'] or
                metadata['git_commit'] != evidence['source_commit'] or metadata['source_sha256'] != bound or
                metadata['source_git_dirty'] is not False or metadata['effective_method'] != effective or
                metadata['status'] != evidence['status'] or metadata['wall_seconds'] != evidence['wall_seconds']):
            raise ValueError('study cell differs from its recorded identity, producer or terminal evidence')
        if coordinate and (metadata.get('requested_coordinate_standard_deviation') != cell['coordinate_standard_deviation'] or
                metadata.get('actual_coordinate_standard_deviation') != config['privacy']['noise_multiplier']*privacy['clipping_bound']):
            raise ValueError('coordinate noise differs from the declared scale')
        if metadata['git_commit'] not in checked:
            protocol.verify_sources(metadata, plan, effective)
            checked.add(metadata['git_commit'])
        if versions is not None and versions != metadata['versions']:
            raise ValueError('study library versions differ')
        versions = metadata['versions']
        failed = metadata['status'] == 'failed_numerical'
        rows = csv_rows(folder/'per_round.csv')
        metrics = None if failed else read(folder/'metrics.json')
        if metrics is not None:
            for family in ['CGSV', 'RFFL']:
                if family in metrics:
                    count = len({row['round'] for row in rows if row[family+'_undefined_state'] == 'True'})
                    metrics[family] = inference_metrics(metrics[family], count)
        account = None
        if 'privacy' in config:
            schedule = [sample_clients(config['num_clients'], config['clients_per_round'], number, config['seed']).tolist()
                        for number in range(1, config['rounds']+1)]
            account = study_protocol.honest_accounting(config, schedule) if failed else read(folder/'privacy.json')
        records.append(dict(**cell, status=metadata['status'], config=config, metrics=metrics,
                            accounting=account, history=csv_rows(folder/'round_metrics.csv'),
                            failure=read(folder/'failure.json') if failed else None,
                            prefix_metrics=read(folder/'prefix_metrics.json') if failed else None,
                            inputs=inputs, wall_seconds=metadata['wall_seconds'], source_commit=metadata['git_commit']))
    complete = sum(row['status'] == 'complete' for row in records)
    if manifest['completed_cells'] != complete or manifest['numerical_failures'] != len(records)-complete:
        raise ValueError('study completion totals differ from terminal records')
    provenance = dict(manifest_sha256=sha256(manifest_path), plan_sha256=sha256(protocol.PLAN),
                      execution_sha256=sha256(clock_path), elapsed_wall_seconds=clock['elapsed_wall_seconds'],
                      successful_training_seconds=sum(row['wall_seconds'] for row in records if row['status'] == 'complete'),
                      versions=versions, effective_method=effective,
                      cells=[{key:row[key] for key in ['path', 'status', 'source_commit', 'inputs']} for row in records])
    return plan, privacy, extensions, records, provenance


def add_coordinate_extensions(records, parameters, recorded_only, cache_root='results/cache'):
    evidence = []
    for cell in records:
        if cell['status'] != 'complete':
            continue
        folder = Path(cell['path'])
        output = Path('results/coordinate_extensions')/folder.name
        if not recorded_only:
            record_geometry(folder, output/'geometry', parameters, cache_root=cache_root)
        outputs = {}
        for family, params in parameters.items():
            path = output/f'{family}.json'
            result = replay_geometry(folder, output/'geometry'/f'{family}.json', params)
            if path.exists() and read(path) != result:
                raise ValueError('coordinate extension replay differs from its retained output')
            if recorded_only and not path.exists():
                raise ValueError('recorded coordinate extension output is missing')
            write_json(path, result)
            cell['metrics'][family] = inference_metrics(result['metrics'], result['metrics']['undefined_state_rounds'])
            outputs[family] = dict(path=path.as_posix(), sha256=sha256(path), geometry_sha256=result['geometry_sha256'])
        evidence.append(dict(path=cell['path'], outputs=outputs))
    return evidence


def check_shared_accuracy(records):
    for cell in records:
        metrics = cell['metrics']
        if metrics is not None and (set(metrics) != set(RULES) or
                any(metrics[rule]['final_accuracy'] != metrics['B0']['final_accuracy'] for rule in RULES)):
            raise ValueError('all six shadow rules must share one model accuracy')


def noise_summary(original, coordinates, plan, frozen):
    controls = [row for row in original if row['kind'] == 'privacy' and row['noise_multiplier'] == 0]
    if [row['seed'] for row in controls] != plan['seeds'] or any(row['status'] != 'complete' for row in controls):
        raise ValueError('every completed original zero-noise control must be reused')
    planned = list(coordinate_protocol.study_cells(plan))
    if [{key:row.get(key) for key in planned[0]} for row in coordinates] != planned:
        raise ValueError('coordinate summary requires all unique declared seed/scale pairs')
    records = controls+coordinates
    check_shared_accuracy(records)
    for row in records:
        if row['status'] not in {'complete', 'failed_numerical'} or (row['metrics'] is None) != (row['status'] == 'failed_numerical'):
            raise ValueError('failed noise runs cannot supply complete-horizon metrics')
        control = next(cell for cell in controls if cell['seed'] == row['seed'])
        common = lambda cell:{key:value for key, value in cell['config'].items() if key not in {'rules', 'privacy'}}
        if (common(row) != common(control) or row['config']['privacy']['clipping_bound'] != frozen['clipping_bound'] or
                row['config']['privacy']['delta'] != frozen['delta']):
            raise ValueError('noise trajectories differ beyond declared noise and shadow-rule coverage')
    groups = []
    value = lambda cell, rule, metric:None if cell['metrics'] is None else cell['metrics'][rule][metric]
    for tau in [0., *plan['coordinate_standard_deviations']]:
        cells = controls if tau == 0 else [row for row in coordinates if row['coordinate_standard_deviation'] == tau]
        metrics = {rule:{name:interval([value(cell, rule, name) for cell in cells], plan['statistics'])
                         for name in METRICS} for rule in RULES}
        accounts = []
        for cell in cells:
            account = cell['accounting']
            clients = account['clients']
            finite = all(client['finite'] for client in clients)
            accounts.append(dict(seed=cell['seed'], path=cell['path'], all_clients_finite=finite,
                maximum_client_epsilon=max(client['epsilon'] for client in clients) if finite else None,
                infinite_epsilon_clients=sum(not client['finite'] for client in clients),
                minimum_participations=min(client['participations'] for client in clients),
                maximum_participations=max(client['participations'] for client in clients), accounting=account,
                scope='observed complete schedule' if cell['status'] == 'complete' else 'hypothetical full schedule; excludes failure signals'))
        conservative = cells[0]['accounting']['conservative_all_rounds']
        if any(cell['accounting']['conservative_all_rounds'] != conservative for cell in cells):
            raise ValueError('accounting settings differ between seeds')
        differences = {rule:interval([None if value(cell, rule, COSTS[1]) is None or value(cell, 'B1', COSTS[1]) is None
            else value(cell, rule, COSTS[1])-value(cell, 'B1', COSTS[1]) for cell in cells], plan['statistics'])
            for rule in RULES if rule != 'B1'}
        control_differences = {rule:{metric:interval([None if value(cell, rule, metric) is None or value(control, rule, metric) is None
            else value(cell, rule, metric)-value(control, rule, metric) for cell, control in zip(cells, controls)], plan['statistics'])
            for metric in [*COSTS, 'final_accuracy']} for rule in RULES}
        groups.append(dict(coordinate_standard_deviation=tau, noise_multiplier=tau/frozen['clipping_bound'],
            seeds=plan['seeds'], metrics=metrics, per_seed_accounting=accounts, conservative_all_rounds=conservative,
            paired_false_penalty_difference_from_B1=differences, paired_difference_from_zero_noise=control_differences,
            completed_runs=sum(cell['status'] == 'complete' for cell in cells),
            numerical_failures=sum(cell['status'] == 'failed_numerical' for cell in cells),
            outcomes=[dict(path=cell['path'], seed=cell['seed'], status=cell['status'],
                           completed_rounds=cell['failure']['completed_rounds'] if cell['failure'] else plan['rounds'],
                           failure=cell['failure'], prefix_metrics=cell['prefix_metrics']) for cell in cells]))
    divergence = [row for row in original if row['kind'] == 'privacy']
    return dict(schema=3, runs=len(records), new_runs=len(coordinates), reused_control_runs=len(controls),
        unique_privacy_training_runs=len(divergence)+len(coordinates), rules=RULES,
        completed_runs=sum(row['status'] == 'complete' for row in records),
        numerical_failures=sum(row['status'] == 'failed_numerical' for row in records),
        seeds=plan['seeds'], rounds=plan['rounds'], training_device=plan['device'],
        clipping_bound=frozen['clipping_bound'], delta=frozen['delta'], cells=groups, statistical_plan=plan['statistics'],
        original_multiplier_study=dict(runs=len(divergence),
            completed_runs=sum(row['status'] == 'complete' for row in divergence),
            numerical_failures=sum(row['status'] == 'failed_numerical' for row in divergence),
            outcomes=[dict(path=row['path'], seed=row['seed'], noise_multiplier=row['noise_multiplier'], status=row['status'],
                completed_rounds=row['failure']['completed_rounds'] if row['failure'] else plan['rounds'],
                failure=row['failure'], accounting=row['accounting']) for row in divergence]),
        simulation_only=True, zero_noise_control='existing honest-clipped mixture runs without additive noise',
        accounting_limitations=controls[0]['accounting']['limitation'],
        accounting_scope='Honest clients only; replacement sensitivity 2C; identified releases; no sampling amplification',
        interpretation=plan['interpretation'],
        numerical_failure_policy='All declared seeds retained; failed full-horizon metrics undefined; no survivor-only intervals',
        pairing='Same three seeds across rules and coordinate scales; percentile seed intervals are descriptive')


def fashion_summary(records, mnist, plan):
    conditions = plan['fashion']['conditions']
    expected = [(seed, condition['attack'], condition['fraction']) for seed in plan['seeds'] for condition in conditions]
    identity = lambda row:(row['seed'], row['attack'], row['fraction'])
    if ([identity(row) for row in records] != expected or [identity(row) for row in mnist] != expected or
            any(row['status'] != 'complete' for row in records)):
        raise ValueError('transfer comparison requires all matching seeds and conditions on both datasets')
    check_shared_accuracy(records)
    check_shared_accuracy(mnist)
    statistics = plan['statistics']
    intervals, trajectories, ordering = [], [], []
    for condition in conditions:
        selected = lambda rows:[row for row in rows if (row['attack'], row['fraction']) == (condition['attack'], condition['fraction'])]
        cells, reference = selected(records), selected(mnist)
        metrics = {rule:{name:interval([cell['metrics'][rule][name] for cell in cells], statistics)
                         for name in METRICS} for rule in RULES}
        intervals.append(dict(horizon='main', **condition, metrics=metrics))
        for number in range(1, plan['rounds']+1):
            trajectories.append(dict(**condition, round=number,
                accuracy=interval([float(cell['history'][number-1]['accuracy']) for cell in cells], statistics)))
        for metric in COSTS:
            means = {dataset:{rule:average([cell['metrics'][rule][metric] for cell in values]) for rule in RULES}
                     for dataset, values in [('mnist', reference), ('fashion_mnist', cells)]}
            sign = lambda value:None if value is None else (value > 0)-(value < 0)
            pairs = []
            for left, right in combinations(RULES, 2):
                directions = {dataset:sign(None if values[left] is None or values[right] is None else values[left]-values[right])
                              for dataset, values in means.items()}
                comparable = all(value is not None for value in directions.values())
                pairs.append(dict(left=left, right=right, **directions,
                                  same_order=directions['mnist'] == directions['fashion_mnist'] if comparable else None,
                                  reversed=(directions['mnist']*directions['fashion_mnist'] == -1) if comparable else None))
            effects = {rule:interval([None if a['metrics'][rule][metric] is None or b['metrics'][rule][metric] is None
                else a['metrics'][rule][metric]-b['metrics'][rule][metric] for a, b in zip(cells, reference)], statistics)
                for rule in RULES}
            ordering.append(dict(**condition, metric=metric, means=means, pairs=pairs,
                defined_pairs=sum(row['same_order'] is not None for row in pairs),
                same_order_pairs=sum(row['same_order'] is True for row in pairs),
                reversed_pairs=sum(row['reversed'] is True for row in pairs),
                fashion_minus_mnist=effects))
    pooled, contrasts = {}, []
    for metric in COSTS:
        by_seed = {rule:[average([cell['metrics'][rule][metric] for cell in records if cell['seed'] == seed])
                         for seed in plan['seeds']] for rule in RULES}
        pooled[metric] = {rule:interval(values, statistics) for rule, values in by_seed.items()}
        for baseline in [rule for rule in RULES if rule != 'Mstar']:
            left = [cell['metrics']['Mstar'][metric] for cell in records]
            right = [cell['metrics'][baseline][metric] for cell in records]
            missing = sum(a is None or b is None for a, b in zip(left, right))
            differences = [None if a is None or b is None else a-b for a, b in zip(by_seed['Mstar'], by_seed[baseline])]
            contrasts.append(dict(metric=metric, baseline=baseline,
                condition_seed_test=None if missing else paired_wilcoxon(left, right),
                seed_test=None if missing else paired_wilcoxon(by_seed['Mstar'], by_seed[baseline]),
                undefined_condition_seed_pairs=missing, planned_condition_seed_pairs=len(records),
                paired_difference=interval(differences, statistics)))
    for test in ['condition_seed_test', 'seed_test']:
        key = lambda row:row['metric']+':'+row['baseline']
        adjusted = holm({key(row):row[test]['p_value'] if row[test] else None for row in contrasts})
        for row in contrasts:
            row[test+'_holm_p'] = adjusted[key(row)]
    return dict(schema=2, nominal_cells=len(records), seed_count=len(plan['seeds']), seeds=plan['seeds'],
        distinct_main_scenarios=len(conditions), device=plan['device'], rules=RULES, intervals=intervals,
        accuracy_trajectories=trajectories, pooled=pooled, contrasts=contrasts, ordering=ordering,
        statistical_plan=dict(statistics, minimum_two_sided_seed_p=2/2**len(plan['seeds']),
            multiple_testing='Ten Mstar/baseline/cost contrasts; one Holm family, including undefined tests',
            interpretation='Three seeds; descriptive paired conditions and seed-mean sensitivity; no rule-specific accuracy gain'),
        ordering_interpretation='Lower mean costs are better; exact ties and undefined comparisons retained. Mean ordering is descriptive, not evidence of statistical equivalence.',
        mnist_reference='Same three seeds and exact attack conditions from the unclipped main MNIST grid; privacy controls are not used')


def matching_mnist(fashion, parameters):
    grid = read('results/grid/manifest.json')
    extensions = read('results/extensions/manifest.json')
    if (grid['completed_cells'] != grid['expected_cells'] or not extensions['complete'] or
            extensions['completed_cells'] != grid['expected_cells'] or
            extensions['grid_manifest_sha256'] != sha256('results/grid/manifest.json') or
            extensions['frozen_sha256'] != sha256('configs/extensions_frozen.json')):
        raise ValueError('transfer reference requires the complete frozen common-cache comparison')
    indexed = {cell['path']:cell for cell in extensions['cells']}
    records, evidence = [], []
    for target in fashion:
        candidates = [cell for cell in grid['cells'] if cell['horizon'] == 'main' and
                      all(cell[key] == target[key] for key in ['seed', 'attack', 'fraction'])]
        if len(candidates) != 1:
            raise ValueError('transfer reference must have exactly one matching MNIST cell')
        cell = candidates[0]
        folder = Path(cell['path'])
        metadata = read(folder/'metadata.json')
        if (sha256(folder/'metadata.json') != cell['metadata_sha256'] or
                sha256(folder/'per_round.csv') != cell['per_round_sha256'] or
                metadata['config']['dataset'] != 'mnist' or 'privacy' in metadata['config'] or
                metadata['config']['rounds'] != target['config']['rounds']):
            raise ValueError('transfer reference differs from its original unclipped training record')
        study_protocol.rebuild(folder)
        metrics = {name:value for name, value in read(folder/'metrics.json').items() if name in RULES}
        outputs = {}
        for family, params in parameters.items():
            binding = indexed[cell['path']]['outputs'][family]
            path = Path(binding['path'])
            geometry = path.parent/'geometry'/f'{family}.json'
            if sha256(path) != binding['sha256'] or sha256(geometry) != binding['geometry_sha256']:
                raise ValueError('matching MNIST extension evidence differs from its manifest')
            result = replay_geometry(folder, geometry, params)
            if result != read(path):
                raise ValueError('matching MNIST extension states do not reconstruct')
            metrics[family] = inference_metrics(result['metrics'], result['metrics']['undefined_state_rounds'])
            outputs[family] = binding
        records.append(dict(**cell, status='complete', metrics=metrics))
        evidence.append(dict(path=cell['path'], metadata_sha256=sha256(folder/'metadata.json'),
                             metrics_sha256=sha256(folder/'metrics.json'), outputs=outputs))
    return records, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recorded-only', action='store_true', help='reconstruct from retained geometry without reading matrices')
    parser.add_argument('--coordinate-cache', type=Path, default=Path('results/cache'), help='retained update cache from the coordinate-noise checkout')
    args = parser.parse_args()
    transfer_plan, frozen, extension, original, transfer_provenance = load_cohort(study_protocol, 'results/transfer_study')
    noise_plan, other_frozen, other_extension, coordinates, coordinate_provenance = load_cohort(coordinate_protocol, 'results/coordinate_noise')
    if frozen != other_frozen or extension != other_extension or transfer_provenance['versions'] != coordinate_provenance['versions']:
        raise ValueError('the two studies must preserve calibration, extension parameters and library versions')
    extension_evidence = add_coordinate_extensions(coordinates, extension['params'], args.recorded_only, args.coordinate_cache)
    privacy = noise_summary(original, coordinates, noise_plan, frozen)
    fashion_records = [row for row in original if row['kind'] == 'fashion']
    mnist, mnist_evidence = matching_mnist(fashion_records, extension['params'])
    fashion = fashion_summary(fashion_records, mnist, transfer_plan)
    source = {path:sha256(path) for path in ['scripts/analyze_studies.py', 'scripts/summarize_grid.py',
              'scripts/extension_replay.py', 'scripts/study_protocol.py', 'scripts/coordinate_protocol.py',
              'src/fltoken/statistics.py', 'src/fltoken/metrics.py', 'src/fltoken/privacy.py']}
    privacy['provenance'] = dict(original=transfer_provenance, coordinate=coordinate_provenance,
        extension_outputs=extension_evidence, frozen_sha256=sha256('configs/privacy_frozen.json'), source_sha256=source)
    fashion['provenance'] = dict(training=transfer_provenance, mnist_reference=mnist_evidence,
        grid_manifest_sha256=sha256('results/grid/manifest.json'), extension_manifest_sha256=sha256('results/extensions/manifest.json'),
        source_sha256=source)
    summary = read('results/summary.json')
    summary.update(privacy=privacy, fashion=fashion)
    write_json('results/summary.json', summary)
    write_json('results/coordinate_noise/summary.json', privacy)
    write_json('results/transfer_study/fashion_summary.json', fashion)
    print(f"Noise comparison: {privacy['runs']} cells, {privacy['completed_runs']} complete; "
          f"original multiplier failures: {privacy['original_multiplier_study']['numerical_failures']}")
    print(f"Fashion transfer: {fashion['nominal_cells']} cells, {len(RULES)} rules; matching MNIST cells: {len(mnist)}")


if __name__ == '__main__':
    main()
