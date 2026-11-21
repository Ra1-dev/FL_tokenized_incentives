"""Run all declared mixture-noise and Fashion-MNIST cells serially."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from evaluation import read
from fltoken.io import sha256, write_json
from run_grid import checked_execution_clock, finish_execution_clock
from study_protocol import PLAN, expected_config, published_study, sources, study_cells, verify_cell, verify_datasets, verify_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-run', action='store_true')
    args = parser.parse_args()
    lock = Path('out/transfer-study.lock')
    lock.parent.mkdir(exist_ok=True)
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid())+'\n')
    except FileExistsError:
        raise RuntimeError('another study driver or retained interrupted-driver lock exists; inspect it first')
    try:
        plan, method, privacy, extensions, _, effective = published_study(full_evidence=True)
        planned = list(study_cells(plan))
        if len(planned) != 18 or len({cell['path'] for cell in planned}) != 18:
            raise ValueError('study requires exactly eighteen unique cells')
        bound = sources(plan)
        binding = dict(plan_sha256=sha256(PLAN), training_source_sha256=bound, effective_method=effective,
                       dataset_sha256=verify_datasets(plan) if not args.no_run else {
                           dataset:sha256(path) for dataset, path in plan['dataset_manifests'].items()})
        clock_path = Path('results/transfer_study/execution.json')
        if not clock_path.exists():
            if args.no_run or any(Path(cell['path']).exists() for cell in planned):
                raise ValueError('retain original study execution clock')
            write_json(clock_path, dict(binding, started_utc=datetime.now(timezone.utc).isoformat(),
                                       finished_utc=None, elapsed_wall_seconds=None, driver_pid=os.getpid()))
        clock = checked_execution_clock(clock_path, binding)
        completed, versions, checked = [], None, set()
        def record():
            write_json('results/transfer_study/manifest.json', dict(binding, schema=1, expected_cells=18,
                       attempted_cells=len(completed), completed_cells=sum(cell['status'] == 'complete' for cell in completed),
                       numerical_failures=sum(cell['status'] == 'failed_numerical' for cell in completed),
                       complete=len(completed) == 18, cells=completed,
                       execution_sha256=sha256(clock_path), elapsed_wall_seconds=clock['elapsed_wall_seconds']))
        for cell in planned:
            if sources(plan) != bound:
                raise ValueError('study sources changed during training')
            folder = Path(cell['path'])
            existing = read(folder/'metadata.json') if (folder/'metadata.json').exists() else {}
            if existing.get('status') not in {'complete', 'failed_numerical'}:
                if args.no_run or clock['finished_utc'] is not None or folder.exists():
                    raise ValueError('preserve incomplete cell as a failure before resuming')
                command = [sys.executable, 'scripts/run_experiment.py', '--study-cell', str(cell['index'])]
                process = subprocess.run(command)
                if process.returncode not in {0, 20}:
                    raise subprocess.CalledProcessError(process.returncode, command)
                if read(folder/'metadata.json')['status'] != ('complete' if process.returncode == 0 else 'failed_numerical'):
                    raise ValueError('study exit and terminal status differ')
            config = expected_config(plan, cell, method, privacy)
            metadata, inputs = verify_cell(folder, config, verify_cache=not args.no_run)
            if metadata.get('study_cell') != cell or metadata['source_config'] != plan['config'] or metadata['source_sha256'] != bound:
                raise ValueError('study cell identity or producer differs')
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
            print(f'Validated study outcomes: {len(completed)}/18', flush=True)
        if args.no_run and clock['finished_utc'] is None:
            raise ValueError('finish original execution before validation')
        clock = finish_execution_clock(clock_path, binding)
        record()
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
