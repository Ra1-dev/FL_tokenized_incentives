"""Measure product underflow in recorded frozen-method allocations, without matrices."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_method import corpus_cells, historical_scorer
from evaluation import ABLATIONS, recorded_method
from fltoken.data import sample_clients
from fltoken.io import sha256, write_json


def audit_rounds(rounds, params, original):
    scorers = {name:original({**params,**switches}) for name,switches in ABLATIONS.items()}
    result = dict(rounds=0,participations=0,stages={name:dict(
        positive_qualities=0,positive_stored_reputations=0,
        eligible_positive_participations=0,
        subnormal_product_participations=0,subnormal_product_rounds=0,
        erased_product_participations=0,erased_product_rounds=0,
        empty_allocation_rounds=0,minimum_positive_quality=None,
        minimum_positive_reputation=None,minimum_positive_rounded_weight=None)
        for name in ABLATIONS})
    for number,ids,quality,recorded in rounds:
        if number != result['rounds']+1:
            raise ValueError('weight audit requires every round in order')
        result['rounds'] += 1
        result['participations'] += len(ids)
        for name,scorer in scorers.items():
            output = scorer.allocate(ids,quality,number)
            if recorded is not None:
                if len(recorded) != len(ids):
                    raise ValueError('recorded allocation coverage differs')
                for rule in [name,*(['Mstar'] if name == 'Afull' else [])]:
                    for field in ['scores','shares','slash','selected']:
                        suffix = {'scores':'score','shares':'share'}.get(field,field)
                        raw = [row[f'{rule}_{suffix}'] for row in recorded]
                        if field in ['slash','selected']:
                            if any(value not in {'True','False'} for value in raw):
                                raise ValueError('recorded allocation decision must be boolean')
                            expected = np.array([value == 'True' for value in raw])
                        else:
                            expected = np.array(list(map(float,raw)))
                        if not np.array_equal(output[field],expected):
                            raise ValueError('original allocation does not reconstruct the recorded output')
            reputation = np.array([output['state']['reputation'][int(client)] for client in ids])
            factors = reputation if scorer.params['ema'] else np.ones(len(ids))
            weights = quality*factors
            eligible = (~output['slash']) & (quality>0) & (factors>0)
            subnormal = eligible & (weights<np.finfo(float).tiny)
            erased = eligible & (weights==0)
            measured = result['stages'][name]
            measured['positive_qualities'] += int(np.count_nonzero(quality>0))
            measured['positive_stored_reputations'] += int(np.count_nonzero(reputation>0))
            measured['eligible_positive_participations'] += int(np.count_nonzero(eligible))
            measured['subnormal_product_participations'] += int(np.count_nonzero(subnormal))
            measured['subnormal_product_rounds'] += int(np.any(subnormal))
            measured['erased_product_participations'] += int(np.count_nonzero(erased))
            measured['erased_product_rounds'] += int(np.any(erased))
            measured['empty_allocation_rounds'] += int(not np.any(output['shares']))
            for field,values in [('quality',quality),('reputation',reputation),('rounded_weight',weights[~output['slash']])]:
                positive = values[values>0]
                if len(positive):
                    key = 'minimum_positive_'+field
                    value = float(np.min(positive))
                    measured[key] = min(value,measured[key]) if measured[key] is not None else value
    if not result['rounds']:
        raise ValueError('weight audit requires at least one round')
    return result


def recorded_rounds(folder, config):
    with (folder/'per_round.csv').open(encoding='utf-8',newline='') as stream:
        rows = list(csv.DictReader(stream))
    count = config['clients_per_round']
    if len(rows) != config['rounds']*count:
        raise ValueError('weight audit requires every recorded participation')
    for number in range(1,config['rounds']+1):
        group = rows[(number-1)*count:number*count]
        ids = np.array([int(row['client_id']) for row in group])
        expected = sample_clients(config['num_clients'],count,number,config['seed'])
        if (not np.array_equal(ids,expected) or any(int(row['round']) != number for row in group)
                or len({row['update_set_sha256'] for row in group}) != 1):
            raise ValueError('weight audit input order or common matrix differs')
        quality = np.array([float(row['Mstar_score']) for row in group])
        yield number,ids,quality,group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,default=Path('.'),help='checkout containing the original grid artifacts')
    parser.add_argument('--available',action='store_true',help='explicit partial diagnostic; never claims complete grid coverage')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.available and args.output is None:
        parser.error('partial diagnostics require an explicit output path')
    method = recorded_method()
    original,revision = historical_scorer(method)
    manifest_path = args.input_root/'results/grid/manifest.json'
    clock_path = args.input_root/'results/grid/execution.json'
    manifest_digest,clock_digest = sha256(manifest_path),sha256(clock_path)
    entries = list(corpus_cells('grid',method,args.input_root,args.available))
    results = []
    for folder,metadata in entries:
        result = audit_rounds(recorded_rounds(folder,metadata['config']),method['params'],original)
        result.update(path=folder.relative_to(args.input_root).as_posix(),
                      inputs={name:sha256(folder/name) for name in ['metadata.json','per_round.csv']})
        results.append(result)
    if (sha256(manifest_path),sha256(clock_path)) != (manifest_digest,clock_digest):
        raise ValueError('grid coverage changed during the audit; retry the snapshot')
    counters = ['positive_qualities','positive_stored_reputations','eligible_positive_participations','subnormal_product_participations',
                'subnormal_product_rounds','erased_product_participations','erased_product_rounds','empty_allocation_rounds']
    minima = ['minimum_positive_quality','minimum_positive_reputation','minimum_positive_rounded_weight']
    totals = {}
    for name in ABLATIONS:
        totals[name] = {field:sum(row['stages'][name][field] for row in results) for field in counters}
        for field in minima:
            values = [row['stages'][name][field] for row in results if row['stages'][name][field] is not None]
            totals[name][field] = min(values) if values else None
    result = dict(schema=1,complete_grid=not args.available,runs=len(results),
                  rounds=sum(row['rounds'] for row in results),participations=sum(row['participations'] for row in results),
                  stages=totals,cases=results,original_reconstruction_matches_recorded=True,numpy_version=np.__version__,
                  original_scorer_sha256=method['scorer_sha256'],historical_commit=revision,
                  method_sha256=sha256('configs/method_frozen.json'),grid_manifest_sha256=manifest_digest,
                  execution_sha256=clock_digest,
                  source_sha256={path:sha256(path) for path in ['scripts/audit_method_weights.py','scripts/compare_method.py',
                      'scripts/evaluation.py','scripts/grid_reproduction.py','src/fltoken/data.py']},
                  interpretation='Original frozen allocations reconstructed from recorded qualities for every component stage; Afull also matches Mstar. No raw matrix is loaded, no current geometry is recomputed, and no experimental output or parameter is replaced. Counts refer to eligible positive qualities times positive stored binary64 reputations, or the unit factor in the reference-only stage. Subnormal counts include products rounded to zero. A zero count concerns this recorded corpus only, not all finite inputs.')
    write_json(args.output or Path('results/validation/method_weights.json'),result)
    print(json.dumps({key:value for key,value in result.items() if key!='cases'},indent=2))


if __name__ == '__main__':
    main()
