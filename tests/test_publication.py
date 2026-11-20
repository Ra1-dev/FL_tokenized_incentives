"""A frozen checkout must not read a neighboring checkout or hide Git failures."""

import importlib
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def publication(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluation")


def test_artifact_check_uses_its_checkout_despite_cwd_and_git_environment(tmp_path, monkeypatch, publication):
    frozen = tmp_path / "frozen"
    neighbor = tmp_path / "neighbor"
    frozen.mkdir()
    neighbor.mkdir()
    (frozen / "method.json").write_bytes(b'{"selected":true}\n')
    (neighbor / "method.json").write_bytes(b'{"selected":false}\n')
    monkeypatch.setattr(publication, "REPOSITORY", frozen)
    monkeypatch.chdir(neighbor)
    for name in ["GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_NAMESPACE",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"]:
        monkeypatch.setenv(name, str(neighbor))
    calls = []
    def committed(command, **kwargs):
        calls.append((command, kwargs))
        return b'{"selected":true}\n'
    monkeypatch.setattr(publication.subprocess, "check_output", committed)
    for value in [b'{}', b'changed again']:
        (neighbor / "method.json").write_bytes(value)
        assert publication.published_bytes("method.json", reference="a"*40) == b'{"selected":true}\n'
    for command, options in calls:
        assert command == ["git", "--git-dir", str(frozen/".git"), "--work-tree", str(frozen),
                           "show", "a"*40+":method.json"]
        assert options["cwd"] == frozen
        assert not any(key in options["env"] for key in ["GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR",
                       "GIT_INDEX_FILE", "GIT_NAMESPACE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"])
    (frozen / "method.json").write_bytes(b'changed locally')
    with pytest.raises(ValueError, match="differs"):
        publication.published_bytes("method.json")


@pytest.mark.parametrize("returncode", [128, 3221225794, 3221225477])
def test_git_failure_retains_status_and_diagnostic(tmp_path, monkeypatch, publication, returncode):
    monkeypatch.setattr(publication, "REPOSITORY", tmp_path)
    def failed(command, **kwargs):
        raise subprocess.CalledProcessError(returncode, command, stderr=b"diagnostic text")
    monkeypatch.setattr(publication.subprocess, "check_output", failed)
    with pytest.raises(RuntimeError, match=f"Git exited {returncode}; diagnostic text") as error:
        publication.published_bytes("method.json")
    assert isinstance(error.value.__cause__, subprocess.CalledProcessError)
    assert "missing" not in str(error.value)


def test_parent_and_absolute_paths_remain_rejected(tmp_path, monkeypatch, publication):
    monkeypatch.setattr(publication, "REPOSITORY", tmp_path)
    for path in ["../method.json", tmp_path / "method.json"]:
        with pytest.raises(ValueError, match="relative"):
            publication.published_bytes(path)


def test_real_git_objects_remain_independent_of_neighbor_changes(tmp_path, monkeypatch, publication):
    def checkout(name, value):
        folder = tmp_path / name
        folder.mkdir()
        subprocess.run(["git", "init", "--bare", str(folder/".git")], check=True, capture_output=True)
        command = ["git", "--git-dir", str(folder/".git")]
        blob = subprocess.check_output([*command,"hash-object","-w","--stdin"], input=value).strip()
        tree = subprocess.check_output([*command,"mktree"], input=b"100644 blob "+blob+b"\tmethod.json\n").strip()
        subprocess.run([*command,"update-ref","refs/remotes/origin/main",tree.decode()], check=True)
        (folder/"method.json").write_bytes(value)
        return folder
    frozen = checkout("frozen", b'{"selected":true}\n')
    neighbor = checkout("neighbor", b'{"selected":false}\n')
    monkeypatch.setattr(publication,"REPOSITORY",frozen)
    monkeypatch.chdir(neighbor)
    monkeypatch.setenv("GIT_DIR",str(neighbor/".git"))
    monkeypatch.setenv("GIT_WORK_TREE",str(neighbor))
    assert publication.published_bytes("method.json") == b'{"selected":true}\n'
    (neighbor/"method.json").unlink()
    subprocess.run(["git","update-ref","-d","refs/remotes/origin/main"],check=True)
    assert publication.published_bytes("method.json") == b'{"selected":true}\n'
    (frozen/"method.json").write_bytes(b'changed locally')
    with pytest.raises(ValueError,match="differs"):
        publication.published_bytes("method.json")
