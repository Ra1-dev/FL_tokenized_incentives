"""The published reproduction must reject mismatched or stale evidence."""

import json
from pathlib import Path
import runpy
import shutil
import sys

import pytest


@pytest.mark.parametrize("corruption", ["stale_semantics", "unfinished", "ledger", "metrics"])
def test_reproduction_rejects_invalid_evidence(tmp_path, monkeypatch, corruption):
    root = Path(__file__).resolve().parents[1]
    folder = Path("results/runs/mnist-sendros_mix-s42-cpu-t20")
    shutil.copytree(root / folder, tmp_path / folder)
    shutil.copytree(root / "configs", tmp_path / "configs")
    (tmp_path / "results/literature").mkdir()
    shutil.copyfile(root / "results/literature/sendros.json", tmp_path / "results/literature/sendros.json")
    if corruption in {"stale_semantics", "unfinished"}:
        path = tmp_path / folder / "metadata.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if corruption == "stale_semantics":
            value["config"]["stale_mode"] = "replay_own"
        else:
            value["status"] = "running"
        path.write_text(json.dumps(value), encoding="utf-8")
    elif corruption == "ledger":
        path = tmp_path / folder / "ledger.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["B1"][0]["payouts"][0] += 1
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path = tmp_path / folder / "metrics.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["B1"]["attacker_share"] = 0.0
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(root / "scripts"))
    monkeypatch.setattr(sys, "argv", ["reproduce_sendros.py", "--no-run"])
    expected = {
        "stale_semantics": (ValueError, "stale_mode differs"),
        "unfinished": (RuntimeError, "full run missing"),
        "ledger": (AssertionError, "artifact hash mismatch"),
        "metrics": (AssertionError, "ledger or metrics mismatch"),
    }
    error, message = expected[corruption]
    with pytest.raises(error, match=message):
        runpy.run_path(str(root / "scripts/reproduce_sendros.py"), run_name="__main__")
    assert not (tmp_path / "results/reproduction").exists()
