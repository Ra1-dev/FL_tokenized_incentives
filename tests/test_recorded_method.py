"""Historical analysis must not authorize execution with a different scorer."""
import copy
import importlib
import json
from pathlib import Path
import shutil

import pytest


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    module = importlib.import_module('evaluation')
    command, options = module.repository_git()
    # Retain the real published object history, but make current local bytes
    # independently mutable without changing the actual working checkout.
    for name in ['configs/method_frozen.json', *module.METHOD_BINDINGS.values()]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    monkeypatch.setattr(module, 'REPOSITORY', tmp_path)
    monkeypatch.setattr(module, 'repository_git', lambda: (command, options))
    return module, tmp_path


@pytest.mark.parametrize('source', ['src/fltoken/ours.py', 'scripts/select_method.py', 'src/fltoken/metrics.py'])
def test_changed_local_producer_is_analysis_only(evidence, source):
    module, root = evidence
    before = module.recorded_method()
    (root / source).write_text('different producer\n', encoding='utf-8')
    assert module.recorded_method() == before
    with pytest.raises(ValueError, match='differs'):
        module.published_method()


@pytest.mark.parametrize('source', ['configs/method_frozen.json', 'configs/pilot_plan.json',
                                   'configs/method_pilot.json', 'results/pilot_selection/summary.json'])
def test_changed_record_cannot_be_rendered_as_the_published_selection(evidence, source):
    module, root = evidence
    path = root / source
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='differs'):
        module.recorded_method()


@pytest.mark.parametrize('field', ['scorer_sha256', 'selection_program_sha256', 'metrics_source_sha256'])
def test_every_historical_producer_digest_is_checked(evidence, monkeypatch, field):
    module, root = evidence
    original = module.published_bytes
    frozen = json.loads(original('configs/method_frozen.json'))
    frozen[field] = '0'*64
    monkeypatch.setattr(module, 'published_bytes', lambda path, *args:
                        json.dumps(frozen).encode() if str(path) == 'configs/method_frozen.json' else original(path, *args))
    with pytest.raises(ValueError, match='historical version|historical source digest'):
        module.recorded_method()


def test_training_commit_must_use_the_selected_scorer(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    module = importlib.import_module('grid_reproduction')
    frozen = json.loads((root / 'configs/method_frozen.json').read_text())
    # A committed pilot-training example is used only for its historical source
    # evidence; this check does not replay or select any candidate.
    manifest = json.loads((root / 'results/pilots/manifest.json').read_text())
    metadata = json.loads((root / manifest['cells'][0]['path'] / 'metadata.json').read_text())
    module.verify_training_source(metadata, method=frozen)
    for value in ['0'*64, None]:
        changed = copy.deepcopy(metadata)
        if value is None:
            del changed['source_sha256']['src/fltoken/ours.py']
        else:
            changed['source_sha256']['src/fltoken/ours.py'] = value
        with pytest.raises(ValueError, match='training scorer differs'):
            module.verify_training_source(changed, method=frozen)


@pytest.fixture
def privacy_evidence(evidence, monkeypatch):
    publication, target = evidence
    module = importlib.import_module('privacy_protocol')
    root = Path(__file__).resolve().parents[1]
    for name in [module.FROZEN, module.PLAN, module.CALIBRATION, *module.SOURCES]:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    monkeypatch.chdir(target)
    return module, target


def test_recorded_privacy_survives_producer_changes_but_execution_does_not(privacy_evidence):
    module, root = privacy_evidence
    frozen = module.recorded_privacy()
    (root / 'src/fltoken/privacy.py').write_text('changed mechanism\n',encoding='utf-8')
    assert module.recorded_privacy() == frozen
    with pytest.raises(ValueError, match='differs'):
        module.published_privacy()


@pytest.mark.parametrize('name', ['configs/privacy_plan.json','results/privacy_calibration.json'])
def test_privacy_records_cannot_change_with_producer_history(privacy_evidence, name):
    module, root = privacy_evidence
    path = root / name
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='differs'):
        module.recorded_privacy()


def test_recorded_privacy_requires_the_complete_source_map(privacy_evidence, monkeypatch):
    module, root = privacy_evidence
    original = module.published_bytes
    frozen = json.loads(original(module.FROZEN))
    frozen['source_sha256'].pop('src/fltoken/privacy.py')
    monkeypatch.setattr(module,'published_bytes',lambda path:json.dumps(frozen).encode()
                        if path == module.FROZEN else original(path))
    with pytest.raises(ValueError, match='binding is incomplete'):
        module.recorded_privacy()


@pytest.fixture
def extension_evidence(evidence, monkeypatch):
    publication, target = evidence
    module = importlib.import_module('select_extensions')
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/module.PLAN_PATH).read_text())
    names = [*module.scientific_bindings(plan), module.FROZEN_PATH, module.OUTPUT/'summary.json']
    for name in names:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/name,destination)
    monkeypatch.chdir(target)
    return module, target


def test_extension_documents_validate_history_after_producer_changes(extension_evidence):
    module, root = extension_evidence
    frozen = module.recorded_extensions()
    (root/'src/fltoken/extensions.py').write_text('different producer\n',encoding='utf-8')
    assert module.recorded_extensions() == frozen
    with pytest.raises(ValueError,match='sources differ'):
        module.published_extensions()


@pytest.mark.parametrize('source',['configs/extensions_plan.json','configs/pilot_plan.json',
                                   'configs/evaluation_plan.json','results/extension_selection/summary.json'])
def test_extension_history_cannot_hide_changed_declarations(extension_evidence, source):
    module, root = extension_evidence
    path = root/source
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='differs'):
        module.recorded_extensions()
