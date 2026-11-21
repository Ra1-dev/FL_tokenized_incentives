"""Published scope and frozen producers for the mixture-noise and transfer runs."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

import numpy as np

from compare_method import historical_scorer
from evaluation import published_bytes, read, recorded_method, repository_git
from extension_replay import read_training
from fltoken.cache import load_update_rounds
from fltoken.config import load_config, validate_config
from fltoken.extensions import CGSVScorer, RFFLScorer
from fltoken.io import sha256
from fltoken.privacy import account_schedule
from fltoken.rules import Baseline
from privacy_protocol import recorded_privacy, settings
from rebuild_ledgers import rebuild
from select_extensions import runtime_extensions
from select_method import expected_attacks

PLAN = Path('configs/transfer_study.json')
EXTRA_SOURCES = ['scripts/study_protocol.py', 'scripts/run_transfer_study.py',
                 'scripts/compare_method.py', 'scripts/privacy_protocol.py',
                 'scripts/privacy_failures.py', 'scripts/select_extensions.py',
                 'scripts/extension_replay.py', 'scripts/select_method.py',
                 'scripts/reproduction_analysis.py', 'scripts/rebuild_ledgers.py',
                 'scripts/grid_reproduction.py', 'scripts/run_pilots.py',
                 'scripts/run_grid.py', 'scripts/rescore_extensions.py',
                 'configs/transfer_study.json', 'configs/method_frozen.json',
                 'configs/extensions_frozen.json', 'configs/privacy_frozen.json']


def study_cells(plan):
    cells = []
    for seed in plan['seeds']:
        for sigma in plan['privacy']['noise_multipliers']:
            cells.append(dict(kind='privacy', dataset=plan['privacy']['dataset'], seed=seed,
                              attack=plan['privacy']['attack'], fraction=plan['privacy']['fraction'],
                              noise_multiplier=sigma))
    for seed in plan['seeds']:
        for condition in plan['fashion']['conditions']:
            cells.append(dict(kind='fashion', dataset=plan['fashion']['dataset'], seed=seed,
                              **condition, noise_multiplier=None))
    for index, cell in enumerate(cells):
        suffix = '' if cell['noise_multiplier'] is None else f"-sigma{cell['noise_multiplier']:g}"
        name = f"{cell['kind']}-{cell['dataset']}-{cell['attack']}-f{round(cell['fraction']*100):02d}-s{cell['seed']}{suffix}-{plan['device']}-t{plan['rounds']}"
        yield dict(index=index, **cell, path='results/transfer_study/'+name)


def sources(plan):
    paths = sorted(set([path.as_posix() for path in Path('src/fltoken').glob('*.py')]+
                       ['scripts/run_experiment.py', 'scripts/evaluation.py', plan['config']]+EXTRA_SOURCES+
                       list(plan['dataset_manifests'].values())))
    return {path:sha256(path) for path in paths}


def published_study(full_evidence=False):
    plan = json.loads(published_bytes(PLAN))
    if (plan['schema'] != 1 or plan['seeds'] != [42, 43, 44] or plan['rounds'] != 20 or
            plan['device'] != 'cuda' or plan['workers'] != 1 or plan['gpu_fraction'] != 1.0 or
            plan['rules'] != ['B0', 'B1', 'B2', 'Mstar', 'CGSV', 'RFFL'] or
            plan['privacy']['dataset'] != 'mnist' or plan['privacy']['attack'] != 'sendros_mix' or
            plan['privacy']['fraction'] != 1/6 or plan['privacy']['noise_multipliers'] != [0., .5, 1., 1.5] or
            plan['fashion']['dataset'] != 'fashion_mnist' or
            plan['fashion']['conditions'] != [dict(attack='sendros_mix', fraction=1/6), dict(attack='label_flip', fraction=.2)]):
        raise ValueError('study scope differs from the published 12-noise/6-transfer comparison')
    for path in sources(plan):
        published_bytes(path)
    method = recorded_method(full_evidence=full_evidence)
    privacy = recorded_privacy(full_evidence=full_evidence)
    extensions = runtime_extensions(full_evidence=full_evidence)
    # Preserve the actual clipping/noise/accounting and local training kernels.
    for path in ['src/fltoken/privacy.py', 'src/fltoken/config.py', 'src/fltoken/data.py']:
        if sha256(path) != privacy['source_sha256'][path]:
            raise ValueError('study changed a frozen privacy kernel: '+path)
    factory, revision = historical_scorer(method)
    effective = dict(path='src/fltoken/ours.py', revision=revision, sha256=method['scorer_sha256'],
                     loader='scripts/compare_method.py', loader_sha256=sha256('scripts/compare_method.py'))
    return plan, method, privacy, extensions, factory, effective


def expected_config(plan, cell, method, privacy):
    base = load_config(plan['config'])
    config = validate_config({**base, **{key:plan[key] for key in ['rounds', 'device', 'workers', 'gpu_fraction', 'rules']},
                              **{key:cell[key] for key in ['dataset', 'seed', 'attack', 'fraction']},
                              'ours_params':method['params']})
    if config['train_limit'] or config['test_limit'] or not config['save_updates']:
        raise ValueError('study requires full datasets and retained update matrices')
    if cell['kind'] == 'privacy':
        config.update(privacy=settings(privacy, cell['noise_multiplier']), privacy_scope='honest')
    return config


def verify_datasets(plan):
    """Validate both prepared datasets without importing a training framework."""
    base = load_config(plan['config'])
    checks = {}
    if set(plan['dataset_manifests']) != {'mnist', 'fashion_mnist'}:
        raise ValueError('both dataset manifests are required')
    for dataset, (directory, mean, std) in {
            'mnist':('MNIST', .1307, .3081), 'fashion_mnist':('FashionMNIST', .2860, .3530)}.items():
        path = plan['dataset_manifests'][dataset]
        record = json.loads(published_bytes(path))
        names = {f'{prefix}-{kind}-idx{dimension}-ubyte{suffix}' for prefix in ['train', 't10k']
                 for kind, dimension in [('images', 3), ('labels', 1)] for suffix in ['', '.gz']}
        if (record['dataset'] != dataset or record['normalization'] != dict(mean=mean, std=std) or
                set(record['sha256']) != names):
            raise ValueError('dataset manifest differs from the declared input protocol')
        for name, digest in record['sha256'].items():
            if sha256(Path(base['data_dir'])/directory/'raw'/name) != digest:
                raise ValueError('dataset checksum differs: '+name)
        checks[dataset] = sha256(path)
    return checks


class ShadowExtension:
    """Supply the common selection flag without changing extension arithmetic."""
    def __init__(self, scorer):
        self.scorer = scorer

    def score(self, client_ids, deltas, round):
        result = self.scorer.score(client_ids, deltas, round)
        return dict(result, selected=result['shares'] > 0)


def study_rules(config, method_factory, extension_parameters):
    rules = {name:Baseline(name, config['krum_f']) for name in ['B0', 'B1', 'B2']}
    rules['Mstar'] = method_factory(config['ours_params'])
    for name, factory in [('CGSV', CGSVScorer), ('RFFL', RFFLScorer)]:
        rules[name] = ShadowExtension(factory(extension_parameters[name]))
    if list(rules) != config['rules']:
        raise ValueError('every declared shadow rule must be scored')
    return rules


def prepare_cell(index):
    from functools import partial
    plan, method, privacy, extensions, factory, effective = published_study()
    cells = list(study_cells(plan))
    if type(index) is not int or not 0 <= index < len(cells):
        raise ValueError('study cell index is outside the declared comparison')
    cell = cells[index]
    return expected_config(plan, cell, method, privacy), dict(cell=cell, plan=plan, effective=effective), partial(
        study_rules, method_factory=factory, extension_parameters=extensions['params'])


def verify_sources(metadata, plan, effective):
    commit = metadata.get('git_commit', '')
    if (metadata.get('source_git_dirty') is not False or not re.fullmatch('[0-9a-f]{40}', commit) or
            metadata.get('effective_method') != effective or metadata.get('study_plan_sha256') != sha256(PLAN)):
        raise ValueError('study must bind clean published sources and the exact historical scorer')
    expected = sources(plan)
    if metadata.get('source_sha256') != expected:
        raise ValueError('study runtime source coverage or digest differs')
    command, options = repository_git()
    subprocess.run([*command, 'merge-base', '--is-ancestor', commit, 'origin/main'], **options, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for path, digest in expected.items():
        if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
            raise ValueError('study source must be relative')
        data = subprocess.check_output([*command, 'show', f'{commit}:{path}'], **options, stderr=subprocess.PIPE)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('recorded study source does not match published history: '+path)


def honest_accounting(config, schedule):
    account = account_schedule(config, schedule)
    attacks = expected_attacks(config)
    account['clients'] = [client for client in account['clients'] if client['client_id'] not in attacks]
    account['unprotected_client_ids'] = sorted(attacks)
    account['protected_scope'] = 'honest clients only; adversaries have no privacy guarantee'
    return account


def verify_cell(folder, config, verify_cache=True):
    folder = Path(folder)
    metadata = read(folder/'metadata.json')
    if (metadata['config'] != config or metadata.get('method_sha256') != sha256('configs/method_frozen.json') or
            metadata.get('extensions_sha256') != sha256('configs/extensions_frozen.json') or
            ('privacy' in config and metadata.get('privacy_sha256') != sha256('configs/privacy_frozen.json'))):
        raise ValueError('study cell differs from frozen settings')
    if metadata['status'] == 'failed_numerical':
        from privacy_failures import verify_failure
        return metadata, verify_failure(folder, config, verify_cache, attacks=expected_attacks(config))
    metadata, attacks, rounds, inputs = read_training(folder)
    rebuild(folder)
    if 'privacy' in config and read(folder/'privacy.json') != honest_accounting(config, [ids for _, ids, _ in rounds]):
        raise ValueError('honest privacy accounting differs from the public schedule')
    if verify_cache and sum(1 for _ in load_update_rounds(folder)) != config['rounds']:
        raise ValueError('study cache is incomplete')
    names = ['metrics.json', 'timing.json']+(['privacy.json'] if 'privacy' in config else [])
    return metadata, dict(inputs, **{name:sha256(folder/name) for name in names})
