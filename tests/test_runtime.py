"""Metadata collection must not block a run on an unavailable WMI service."""

from types import SimpleNamespace

import pytest

from fltoken import runtime


def test_windows_metadata_uses_version_api_without_optional_hardware_query(monkeypatch):
    monkeypatch.setattr(runtime.sys,"platform","win32")
    monkeypatch.setattr(runtime.sys,"getwindowsversion",lambda:SimpleNamespace(major=10,minor=0,build=12345),raising=False)
    monkeypatch.setenv("PROCESSOR_ARCHITECTURE","AMD64")
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432",raising=False)
    monkeypatch.setenv("PROCESSOR_IDENTIFIER","test processor")
    def unavailable():
        raise OSError("hardware query timed out")
    monkeypatch.setattr(runtime.platform,"platform",unavailable)
    monkeypatch.setattr(runtime.platform,"processor",unavailable)
    result=runtime.host_information()
    assert result["platform"] == "Windows-10.0.12345-AMD64"
    assert result["processor"] == "test processor"


def test_explicit_ray_memory_sizes_are_runtime_resources():
    resources=runtime.ray_initialization(1,128,512)
    assert resources["object_store_memory"] == 128*1024**2
    assert resources["_memory"] == 512*1024**2
    assert resources["num_cpus"] == 1
    with pytest.raises(ValueError):
        runtime.ray_initialization(1,0,512)
