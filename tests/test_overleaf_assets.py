"""The final upload preserves assets and the scope of its measured claims."""

import importlib
import json
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    return root, importlib.import_module('package_overleaf'), importlib.import_module('document_paper_assets')


def test_upload_is_reproducible_and_binds_every_input(modules, tmp_path):
    root, package, _ = modules
    blobs = package.payload(root)
    first, second = tmp_path/'first.zip', tmp_path/'second.zip'
    package.assemble(blobs, first)
    package.assemble(dict(reversed(list(blobs.items()))), second)
    assert first.read_bytes() == second.read_bytes()
    assert first.stat().st_size < 5_000_000
    with zipfile.ZipFile(first) as archive:
        assert 'paper/main.tex' not in archive.namelist()
        assert 'results/summary.json' not in archive.namelist()
        manifest = json.loads(archive.read('asset-manifest.json'))
        assert manifest['summary_sha256'] == package.hashlib.sha256((root/'results/summary.json').read_bytes()).hexdigest()
        for name, entry in manifest['files'].items():
            assert archive.read(name) == (root/name).read_bytes()
            assert package.hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256']
    assert manifest['software']['version'] == '1.0.0'


def test_interrupted_upload_preserves_existing_archive(modules, tmp_path, monkeypatch):
    _, package, _ = modules
    path = tmp_path/'upload.zip'
    package.assemble({'paper/generated/numbers.tex':b'initial'}, path)
    original = path.read_bytes()
    def interrupted(*args, **kwargs):
        raise OSError('simulated disk interruption')
    monkeypatch.setattr(package.zipfile.ZipFile,'writestr',interrupted)
    with pytest.raises(OSError, match='interruption'):
        package.assemble({'paper/generated/numbers.tex':b'changed'},path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


def test_every_asset_and_valid_macro_has_a_documented_group(modules):
    root, _, guide = modules
    rows = guide.asset_inventory(root)
    assert {row['path'] for row in rows if row['kind']=='table'} == {
        path.relative_to(root).as_posix() for path in (root/'paper/generated').glob('*.tex') if path.stem != 'numbers'}
    groups = guide.macro_groups((root/'paper/generated/numbers.tex').read_text())
    assert all(row['group'] in groups for row in rows)
    with pytest.raises(ValueError, match='uniquely'):
        guide.macro_groups(r'\newcommand{\ReleaseFashionB0Mean}{1}')
    with pytest.raises(ValueError, match='group'):
        guide.macro_groups(r'\newcommand{\UnmappedValue}{1}')


def test_headlines_separate_observed_and_conservative_privacy_bounds(modules):
    root, _, _ = modules
    finding = importlib.import_module('release_findings')
    summary = json.loads((root/'results/summary.json').read_text())
    values = finding.release_values(summary)
    assert float(values['ReleaseAllRoundEpsilonHigh']) > float(values['ReleaseObservedMaxEpsilonHigh'])
    assert float(values['ReleaseAllRoundEpsilonLow']) > float(values['ReleaseObservedMaxEpsilonLow'])
    text = '\n'.join(finding.findings(summary))
    assert 'not Pareto-dominant' in text and 'not a normalized incidence ratio' in text
    assert 'every-round' in text and 'per-client' in text
    assert 'complete ordering does not' in text
