"""Compile the IEEE paper, retaining the log for reproducibility checks."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not Path("paper/main.tex").is_file():
        parser.error("Export the Overleaf source to paper/main.tex before building")
    local = Path("out/tools") / ("tectonic.exe" if os.name == "nt" else "tectonic")
    binary = str(local.resolve()) if local.exists() else shutil.which("tectonic")
    if not binary:
        raise RuntimeError("Install with python scripts/install_tectonic.py first")
    command = [binary, "main.tex", "--keep-logs", "--keep-intermediates"]
    if args.offline:
        command.append("--only-cached")
    subprocess.run(command, cwd="paper", check=True)
    log = Path("paper/main.log").read_text(encoding="utf-8", errors="replace")
    for marker in ["undefined references", "Citation `", "! LaTeX Error"]:
        if marker in log:
            raise RuntimeError(f"paper log contains {marker}")
    print("Paper compiled without unresolved references or LaTeX errors")


if __name__ == "__main__":
    main()
