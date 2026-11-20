"""Archives contain one exact committed asset set and honest software metadata."""

import importlib
import json
from pathlib import Path
import sys
import zipfile

import pytest


@pytest.fixture
def package(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    module = importlib.import_module("package_assets")
    blobs = {name:(root/name).read_bytes() for name in module.ROOT_FILES}
    blobs.update({"paper/references.bib": b"synthetic bibliography\n",
                  "paper/generated/numbers.tex": b"synthetic macros\n",
                  "figures/example.pdf": b"synthetic figure bytes\n"})
    blobs["results/summary.json"] = b'{"reproduction":{"synthetic_fixture":true}}\n'
    return module, blobs


def test_archive_is_byte_identical_and_checksums_bind_every_asset(package, tmp_path):
    module, blobs = package
    first, second = tmp_path/"first.zip", tmp_path/"second.zip"
    result = module.assemble("a"*40, 1795161601, blobs, first)
    module.assemble("a"*40, 1795161601, dict(reversed(list(blobs.items()))), second)
    assert first.read_bytes() == second.read_bytes()
    assert result["result_sections_present"] == ["reproduction"]
    assert "evaluation" in result["result_sections_absent"] and "fashion" in result["result_sections_absent"]
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("asset-manifest.json"))
        assert manifest == result
        assert set(archive.namelist()) == set(blobs) | {"asset-manifest.json"}
        for name, content in blobs.items():
            assert archive.read(name) == content
            assert manifest["files"][name]["sha256"] == module.hashlib.sha256(content).hexdigest()
    modified = {**blobs, "paper/generated/numbers.tex": b"changed artifact\n"}
    module.assemble("b"*40, 1795161601, modified, second)
    assert first.read_bytes() != second.read_bytes()


def test_asset_selection_excludes_untracked_work_and_rejects_path_escape(package):
    module, blobs = package
    for name in ["scratch.txt", ".env", "data/train.bin", "results/cache/round.npz",
                 "docs/unpublished_notes.md", "paper/main.tex", "paper/main.pdf"]:
        assert not module.included(name)
    for name in ["../outside.txt", "/absolute.txt", "figures\\image.png"]:
        with pytest.raises(ValueError, match="relative POSIX"):
            module.included(name)
    with pytest.raises(ValueError, match="unsupported"):
        module.assemble("a"*40, 1795161601, {**blobs, ".env": b"synthetic excluded content"}, Path("unused.zip"))


def test_metadata_changes_cannot_silently_disagree(package):
    module, blobs = package
    metadata = module.validate_metadata(blobs)
    assert metadata["version"] == module.tomllib.loads(blobs["pyproject.toml"].decode())["project"]["version"]
    assert metadata["creators"] == [dict(name="Meiramuly, Rauan", affiliation="Nazarbayev University")]
    changed = json.loads(blobs[".zenodo.json"])
    changed["creators"][0]["name"] = "Different, Person"
    with pytest.raises(ValueError, match="authorship"):
        module.validate_metadata({**blobs, ".zenodo.json": json.dumps(changed).encode()})
    citation = module.yaml.safe_load(blobs["CITATION.cff"])
    citation["version"] = "999.0"
    with pytest.raises(ValueError, match="version"):
        module.validate_metadata({**blobs, "CITATION.cff": module.yaml.safe_dump(citation).encode()})
    assert "doi" not in citation and "date-released" not in citation
    assert "doi" not in changed and "publication_date" not in changed


def test_committed_blob_reader_uses_repository_objects_not_working_copies(package, monkeypatch, tmp_path):
    module, blobs = package
    monkeypatch.chdir(tmp_path)
    Path("README.md").write_text("uncommitted working text", encoding="utf-8")
    objects = {str(index).zfill(40): (name, data) for index, (name, data) in enumerate(blobs.items(), 1)}
    objects["f"*40] = ("scratch.txt", b"untracked fixture")
    commands = []
    def git(command, **kwargs):
        commands.append(command)
        if command[1] == "rev-parse":
            return "a"*40+"\n"
        if command[1] == "show":
            return "1795161601\n"
        if command[1] == "ls-tree":
            return b"".join(f"100644 blob {oid}\t{name}".encode()+b"\0" for oid, (name, _) in objects.items())
        if command[1] == "cat-file":
            return objects[command[-1]][1]
        raise AssertionError(command)
    monkeypatch.setattr(module.subprocess, "check_output", git)
    commit, stamp, saved = module.committed_assets("HEAD")
    assert commit == "a"*40 and stamp == 1795161601 and saved == blobs
    assert ["git", "cat-file", "blob", "f"*40] not in commands
    monkeypatch.setattr(sys, "argv", ["package_assets.py", "--output", "out/support.zip"])
    module.main()
    with zipfile.ZipFile("out/support.zip") as archive:
        assert archive.read("README.md") == blobs["README.md"]
    monkeypatch.setattr(sys, "argv", ["package_assets.py", "--output", "outside.zip"])
    with pytest.raises(ValueError, match="under the local out"):
        module.main()


def test_missing_committed_assets_prevent_a_partial_archive(package, tmp_path):
    module, blobs = package
    del blobs["paper/generated/numbers.tex"]
    output = tmp_path/"incomplete.zip"
    with pytest.raises(ValueError, match="missing required"):
        module.assemble("a"*40, 1795161601, blobs, output)
    assert not output.exists()


def test_interrupted_write_preserves_the_previous_archive(package, tmp_path, monkeypatch):
    module, blobs = package
    output = tmp_path/"support.zip"
    module.assemble("a"*40, 1795161601, blobs, output)
    previous = output.read_bytes()
    def failed(*args, **kwargs):
        raise OSError("synthetic interrupted write")
    monkeypatch.setattr(module.zipfile.ZipFile, "writestr", failed)
    with pytest.raises(OSError, match="interrupted"):
        module.assemble("a"*40, 1795161601, blobs, output)
    assert output.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_real_repository_reader_preserves_committed_artifacts(package, monkeypatch):
    module, _ = package
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    commit, timestamp, blobs = module.committed_assets("HEAD")
    assert commit == module.subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert timestamp > 0 and "paper/generated/numbers.tex" in blobs
    for name in ["README.md", "results/summary.json", "paper/generated/numbers.tex"]:
        expected = module.subprocess.check_output(["git", "show", f"{commit}:{name}"])
        assert blobs[name] == expected
    assert all(module.included(name) for name in blobs)
