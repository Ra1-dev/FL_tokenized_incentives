"""Elapsed grid time includes interruptions and is immutable on validation."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def clock_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return importlib.import_module("run_grid")


def test_recorded_completion_is_idempotent_and_checks_elapsed_time(clock_module, tmp_path):
    path = tmp_path/"clock.json"
    binding = dict(plan_sha256="planned", method_sha256="selected")
    record = dict(**binding, started_utc="2026-01-01T00:00:00+00:00",
                  finished_utc="2026-01-01T00:02:03+00:00", elapsed_wall_seconds=123.0)
    path.write_text(json.dumps(record), encoding="utf-8")
    before = path.read_bytes()
    assert clock_module.finish_execution_clock(path, binding) == record
    assert path.read_bytes() == before
    record["elapsed_wall_seconds"] = 100.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="elapsed time"):
        clock_module.checked_execution_clock(path, binding)


def test_execution_clock_rejects_unbound_or_incomplete_claims(clock_module, tmp_path):
    path = tmp_path/"clock.json"
    record = dict(plan_sha256="old", started_utc="2026-01-01T00:00:00+00:00",
                  finished_utc=None, elapsed_wall_seconds=None)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="different scientific inputs"):
        clock_module.checked_execution_clock(path, dict(plan_sha256="new"))
    record["elapsed_wall_seconds"] = 123.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="unfinished"):
        clock_module.checked_execution_clock(path, dict(plan_sha256="old"))
