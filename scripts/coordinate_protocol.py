"""Fixed coordinate noise scales using the original clipping calibration."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from evaluation import published_bytes, repository_git
from fltoken.io import sha256
from fltoken.privacy import validate_privacy
from fltoken.rules import Baseline
import study_protocol as original

PLAN = Path('configs/coordinate_noise_study.json')
ADDED_SOURCES = [PLAN.as_posix(), 'scripts/coordinate_protocol.py', 'scripts/run_coordinate_study.py']


def study_cells(plan):
    for index, (seed, tau) in enumerate((seed, tau) for seed in plan['seeds']
                                       for tau in plan['coordinate_standard_deviations']):
        yield dict(index=index, kind='coordinate_noise', dataset=plan['dataset'], seed=seed,
                   attack=plan['attack'], fraction=plan['fraction'], coordinate_standard_deviation=tau,
                   path=f"results/coordinate_noise/mnist-sendros_mix-s{seed}-tau{tau:g}-cuda-t{plan['rounds']}")


def sources(plan):
    return dict(original.sources(plan), **{path:sha256(path) for path in ADDED_SOURCES})


def published_study(full_evidence=False):
    plan = json.loads(published_bytes(PLAN))
    prior, method, privacy, extensions, factory, effective = original.published_study(full_evidence)
    if (plan['schema'] != 1 or plan['seeds'] != [42, 43, 44] or plan['rounds'] != 20 or
            plan['device'] != 'cuda' or plan['workers'] != 1 or plan['gpu_fraction'] != 1.0 or
            plan['rules'] != ['B0', 'B1', 'B2', 'Mstar'] or plan['dataset'] != 'mnist' or
            plan['attack'] != 'sendros_mix' or plan['fraction'] != 1/6 or
            plan['coordinate_standard_deviations'] != [.001, .003, .01] or
            plan['config'] != prior['config'] or plan['dataset_manifests'] != prior['dataset_manifests'] or
            plan['clipping_calibration'] != 'configs/privacy_frozen.json'):
        raise ValueError('coordinate comparison differs from the nine declared cells')
    for path in sources(plan):
        published_bytes(path)
    return plan, method, privacy, extensions, factory, effective


def expected_config(plan, cell, method, privacy):
    config = original.expected_config(plan, cell, method, privacy)
    tau = cell['coordinate_standard_deviation']
    if tau not in plan['coordinate_standard_deviations']:
        raise ValueError('coordinate standard deviation is not declared')
    config.update(privacy=validate_privacy(dict(clipping_bound=privacy['clipping_bound'],
                  noise_multiplier=tau/privacy['clipping_bound'], delta=privacy['delta'])), privacy_scope='honest')
    return config


def study_rules(config, method_factory):
    rules = {name:Baseline(name, config['krum_f']) for name in ['B0', 'B1', 'B2']}
    rules['Mstar'] = method_factory(config['ours_params'])
    if list(rules) != config['rules']:
        raise ValueError('every declared shadow rule must be scored')
    return rules


def verify_sources(metadata, plan, effective):
    commit = metadata.get('git_commit', '')
    if (metadata.get('source_git_dirty') is not False or not re.fullmatch('[0-9a-f]{40}', commit) or
            metadata.get('effective_method') != effective or metadata.get('coordinate_plan_sha256') != sha256(PLAN)):
        raise ValueError('coordinate study requires clean published sources and the exact historical scorer')
    expected = sources(plan)
    if metadata.get('source_sha256') != expected:
        raise ValueError('coordinate source coverage or digest differs')
    command, options = repository_git()
    subprocess.run([*command, 'merge-base', '--is-ancestor', commit, 'origin/main'], **options, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for path, digest in expected.items():
        data = subprocess.check_output([*command, 'show', f'{commit}:{path}'], **options, stderr=subprocess.PIPE)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('coordinate source differs from published history: '+path)
