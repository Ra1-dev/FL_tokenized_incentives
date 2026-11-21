"""Run the nine fixed coordinate-noise cells with frozen shadow rewards."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from evaluation import read
from fltoken.io import sha256, write_json
from run_grid import checked_execution_clock, finish_execution_clock
from coordinate_protocol import PLAN, expected_config, published_study, sources, study_cells, verify_sources
from study_protocol import verify_cell, verify_datasets


def run_cell(index):
    # Keep the driver free of training-framework imports between child runs.
    from functools import partial
    import time
    started_process = time.perf_counter()
    import run_experiment as runtime
    from coordinate_protocol import study_rules

    plan, method, privacy, _, factory, effective = published_study()
    cells = list(study_cells(plan))
    if type(index) is not int or not 0 <= index < len(cells):
        raise ValueError('coordinate cell index is outside the declared comparison')
    cell = cells[index]
    config = expected_config(plan, cell, method, privacy)
    output = Path(cell['path']).resolve()
    if output.exists():
        raise ValueError('preserve an existing coordinate cell before any retry')
    resources = runtime.ray_initialization(config['workers'], 128, 512)
    host = runtime.host_information()
    if not runtime.torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    metadata = dict(config=config.copy(), source_config=plan['config'],
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                    source_git_dirty=bool(subprocess.check_output(
                        ['git', 'status', '--porcelain', '--', 'src', 'configs', 'scripts'], text=True).strip()),
                    versions={name:runtime.importlib.metadata.version(name) for name in
                              ['flwr', 'ray', 'torch', 'torchvision', 'numpy', 'PyYAML']},
                    platform=host['platform'], python=runtime.platform.python_version(), host_information=host,
                    runtime_resources=resources, device_name=runtime.torch.cuda.get_device_name(0),
                    started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                    settlement_version=runtime.SETTLEMENT_VERSION, metrics_version=runtime.METRICS_VERSION,
                    source_sha256=sources(plan), effective_method=effective,
                    coordinate_plan_sha256=sha256(PLAN), coordinate_cell=cell,
                    requested_coordinate_standard_deviation=cell['coordinate_standard_deviation'],
                    actual_coordinate_standard_deviation=config['privacy']['noise_multiplier']*config['privacy']['clipping_bound'],
                    method_sha256=sha256('configs/method_frozen.json'),
                    extensions_sha256=sha256('configs/extensions_frozen.json'),
                    privacy_sha256=sha256('configs/privacy_frozen.json'))
    if metadata['source_git_dirty']:
        raise ValueError('commit coordinate study sources before training')
    output.mkdir(parents=True)
    write_json(output/'metadata.json', metadata)
    config['data_dir'] = str(Path(config['data_dir']).resolve())
    started = time.perf_counter()
    try:
        runtime.run_simulation(server_app=runtime.make_server(config, output, rule_factory=partial(
                                   study_rules, method_factory=factory)), client_app=runtime.client_app,
                               num_supernodes=config['num_clients'], backend_config={
                                   'init_args':resources, 'client_resources':{'num_cpus':1, 'num_gpus':config['gpu_fraction']}})
        if (output/'failure.json').exists():
            failure = read(output/'failure.json')
            metadata.update(status='failed_numerical', completed_rounds=failure['completed_rounds'],
                            failure_sha256=sha256(output/'failure.json'))
        elif not (output/'metrics.json').exists():
            raise RuntimeError('simulation returned without completed metrics')
        else:
            metadata['status'] = 'complete'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        metadata['runtime_seconds'] = time.perf_counter()-started
        metadata['wall_seconds'] = time.perf_counter()-started_process
        write_json(output/'metadata.json', metadata)
    print(f"{metadata['status']}: {output.name} in {metadata['wall_seconds']:.2f} seconds", flush=True)
    return 20 if metadata['status'] == 'failed_numerical' else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-run', action='store_true')
    parser.add_argument('--cell', type=int, help='execute one cell of the published plan')
    args = parser.parse_args()
    if args.cell is not None:
        if args.no_run:
            raise ValueError('cell execution and no-run validation are mutually exclusive')
        return run_cell(args.cell)
    lock = Path('out/coordinate-study.lock')
    lock.parent.mkdir(exist_ok=True)
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid())+'\n')
    except FileExistsError:
        raise RuntimeError('another study driver or retained interrupted-driver lock exists; inspect it first')
    try:
        plan, method, privacy, extensions, _, effective = published_study(full_evidence=True)
        planned = list(study_cells(plan))
        if len(planned) != 9 or len({cell['path'] for cell in planned}) != 9:
            raise ValueError('study requires exactly nine unique cells')
        bound = sources(plan)
        binding = dict(plan_sha256=sha256(PLAN), training_source_sha256=bound, effective_method=effective,
                       dataset_sha256=verify_datasets(plan) if not args.no_run else {
                           dataset:sha256(path) for dataset, path in plan['dataset_manifests'].items()})
        clock_path = Path('results/coordinate_noise/execution.json')
        if not clock_path.exists():
            if args.no_run or any(Path(cell['path']).exists() for cell in planned):
                raise ValueError('retain original study execution clock')
            write_json(clock_path, dict(binding, started_utc=datetime.now(timezone.utc).isoformat(),
                                       finished_utc=None, elapsed_wall_seconds=None, driver_pid=os.getpid()))
        clock = checked_execution_clock(clock_path, binding)
        completed, versions, checked = [], None, set()
        def record():
            write_json('results/coordinate_noise/manifest.json', dict(binding, schema=1, expected_cells=9,
                       attempted_cells=len(completed), completed_cells=sum(cell['status'] == 'complete' for cell in completed),
                       numerical_failures=sum(cell['status'] == 'failed_numerical' for cell in completed),
                       complete=len(completed) == 9, cells=completed,
                       execution_sha256=sha256(clock_path), elapsed_wall_seconds=clock['elapsed_wall_seconds']))
        for cell in planned:
            if sources(plan) != bound:
                raise ValueError('study sources changed during training')
            folder = Path(cell['path'])
            existing = read(folder/'metadata.json') if (folder/'metadata.json').exists() else {}
            if existing.get('status') not in {'complete', 'failed_numerical'}:
                if args.no_run or clock['finished_utc'] is not None or folder.exists():
                    raise ValueError('preserve incomplete cell as a failure before resuming')
                command = [sys.executable, 'scripts/run_coordinate_study.py', '--cell', str(cell['index'])]
                process = subprocess.run(command)
                if process.returncode not in {0, 20}:
                    raise subprocess.CalledProcessError(process.returncode, command)
                if read(folder/'metadata.json')['status'] != ('complete' if process.returncode == 0 else 'failed_numerical'):
                    raise ValueError('study exit and terminal status differ')
            config = expected_config(plan, cell, method, privacy)
            metadata, inputs = verify_cell(folder, config, verify_cache=not args.no_run)
            if metadata.get('coordinate_cell') != cell or metadata['source_config'] != plan['config'] or metadata['source_sha256'] != bound:
                raise ValueError('study cell identity or producer differs')
            if (metadata.get('requested_coordinate_standard_deviation') != cell['coordinate_standard_deviation'] or
                    metadata.get('actual_coordinate_standard_deviation') != config['privacy']['noise_multiplier']*config['privacy']['clipping_bound']):
                raise ValueError('coordinate noise scale differs from the declared release')
            key = (metadata['git_commit'], str(metadata['effective_method']), str(metadata['source_git_dirty']))
            if key not in checked:
                verify_sources(metadata, plan, effective)
                checked.add(key)
            if versions is not None and versions != metadata['versions']:
                raise ValueError('study training library versions differ')
            versions = metadata['versions']
            completed.append(dict(cell, status=metadata['status'], inputs=inputs,
                                  source_commit=metadata['git_commit'], wall_seconds=metadata['wall_seconds']))
            record()
            print(f'Validated study outcomes: {len(completed)}/9', flush=True)
        if args.no_run and clock['finished_utc'] is None:
            raise ValueError('finish original execution before validation')
        clock = finish_execution_clock(clock_path, binding)
        record()
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
