"""Runtime provenance without optional Windows hardware-query services."""

import os
import platform
import sys


def host_information():
    if sys.platform == "win32":
        version = sys.getwindowsversion()
        architecture = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "unknown")
        return dict(platform=f"Windows-{version.major}.{version.minor}.{version.build}-{architecture}",
                    processor=os.environ.get("PROCESSOR_IDENTIFIER", "unavailable"),
                    source="Windows version API and processor environment; no WMI query")
    return dict(platform=platform.platform(), processor=platform.processor(), source="Python platform module")


def ray_initialization(workers, object_store_mb, task_memory_mb):
    if type(object_store_mb) is not int or object_store_mb < 80 or type(task_memory_mb) is not int or task_memory_mb < 1:
        raise ValueError("object store must be at least 80 MiB and task memory must be positive")
    return dict(num_cpus=workers, include_dashboard=False, log_to_driver=True, ignore_reinit_error=True,
                object_store_memory=object_store_mb*1024**2, _memory=task_memory_mb*1024**2)
