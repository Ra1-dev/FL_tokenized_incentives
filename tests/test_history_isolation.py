"""Historical scientific evidence belongs to the analysis checkout's Git objects."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

import pytest


def test_training_source_reads_ignore_neighbor_repository_and_git_overrides(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    analysis = importlib.import_module('grid_reproduction')
    publication = importlib.import_module('evaluation')
    frozen, neighbor = tmp_path/'frozen', tmp_path/'neighbor'
    frozen.mkdir()
    neighbor.mkdir()
    # A shallow local clone keeps its own objects, requires no network or identity
    # configuration, and changes no commit or timestamp in the source repository.
    subprocess.run(['git','-c','pack.threads=1','clone','--bare','--depth=1','--no-local',
                    str(root),str(frozen/'.git')],check=True,capture_output=True)
    git = ['git','--git-dir',str(frozen/'.git')]
    assert not (frozen/'.git/objects/info/alternates').exists()
    commit = subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()
    subprocess.run([*git,'update-ref','refs/remotes/origin/main',commit],check=True)
    names = subprocess.check_output([*git,'ls-tree','-r','--name-only',commit,'src/fltoken'],text=True).splitlines()
    names = [name for name in names if name.endswith('.py') and len(Path(name).parts)==3]
    names += ['scripts/run_experiment.py','scripts/evaluation.py','configs/sendros_mix.yaml']
    metadata = dict(git_commit=commit,source_git_dirty=False,source_config='configs/sendros_mix.yaml',
                    source_sha256={name:hashlib.sha256(subprocess.check_output([*git,'show',f'{commit}:{name}'])).hexdigest()
                                   for name in names})
    subprocess.run(['git','init',str(neighbor)],check=True,capture_output=True)
    monkeypatch.setattr(publication,'REPOSITORY',frozen)
    monkeypatch.chdir(neighbor)
    for name in ['GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_INDEX_FILE','GIT_NAMESPACE',
                 'GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES']:
        monkeypatch.setenv(name,str(neighbor))
    analysis.verify_training_source(metadata)
    metadata['source_sha256']['src/fltoken/ours.py']='0'*64
    with pytest.raises(ValueError,match='digest differs'):
        analysis.verify_training_source(metadata)


@pytest.mark.parametrize('code',[128,3221225794])
def test_historical_git_errors_retain_diagnostics(monkeypatch,code):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    analysis = importlib.import_module('grid_reproduction')
    def failed(command,**kwargs):
        raise subprocess.CalledProcessError(code,command,stderr=b'native diagnostic')
    monkeypatch.setattr(analysis.subprocess,'run',failed)
    with pytest.raises(ValueError,match=f'Git exited {code}; native diagnostic'):
        analysis.verify_training_source(dict(git_commit='a'*40,source_git_dirty=False))


def test_frozen_producer_history_remains_verifiable_after_analysis_changes(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    publication=importlib.import_module('evaluation')
    path='configs/extensions_frozen.json'
    frozen=json.loads((root/path).read_text())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GIT_DIR',str(tmp_path))
    revision=publication.verify_frozen_sources(path,frozen['source_sha256'])
    original=subprocess.check_output(['git','-C',str(root),'--git-dir',str(root/'.git'),'show',f'{revision}:{path}'])
    assert json.loads(original)==frozen
    changed=dict(frozen['source_sha256'])
    changed['src/fltoken/extensions.py']='0'*64
    with pytest.raises(ValueError,match='historical source digest differs'):
        publication.verify_frozen_sources(path,changed)


@pytest.mark.parametrize('bindings',[{}, {'../outside':'0'*64}, {'src/fltoken/ours.py':'invalid'},
                                   {'src/fltoken/ours.py':True}])
def test_frozen_source_binding_structure_is_checked(monkeypatch,bindings):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    publication=importlib.import_module('evaluation')
    with pytest.raises(ValueError,match='source binding'):
        publication.verify_frozen_sources('configs/extensions_frozen.json',bindings)


@pytest.mark.parametrize('code',[128,3221225794])
def test_frozen_history_git_errors_retain_diagnostics(monkeypatch,code):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    publication=importlib.import_module('evaluation')
    monkeypatch.setattr(publication,'published_bytes',lambda *args:b'{}')
    def failed(command,**kwargs):
        raise subprocess.CalledProcessError(code,command,stderr=b'native diagnostic')
    monkeypatch.setattr(publication.subprocess,'check_output',failed)
    with pytest.raises(RuntimeError,match=f'Git exited {code}; native diagnostic'):
        publication.verify_frozen_sources('configs/extensions_frozen.json',{'source.py':'0'*64})
