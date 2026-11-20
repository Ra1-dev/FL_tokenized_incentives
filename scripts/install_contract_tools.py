"""Install checksum-pinned Foundry and Solidity binaries inside out/tools."""

import hashlib
import os
from pathlib import Path
import platform
import stat
import subprocess
import tarfile
import urllib.request
import zipfile

FOUNDRY = "v1.10.0"
SOLC = "0.8.30+commit.73712a01"
PLATFORMS = {
    "Windows": ("win32_amd64", "zip", "7228dda08fdee1d1aa0210b171285f12734e19dbbbb23611c0669e97202e880b",
                "windows-amd64", "ccbd3ed44d5fbd26fe039702d403421f1212d2e8752e3cbe3bfd074986911586", ".exe"),
    "Linux": ("linux_amd64", "tar.gz", "90729233bd9a683b872783ae2e3217fb56876fa2a3c37185f13e39dca8525041",
              "linux-amd64", "f3e987dc6ecebd4bd350c48edcbc320b46cf9e3109bd3fc3d88f1acaf4c428f7", ""),
}


def download(url, destination, expected):
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"download checksum mismatch: {url}")
        partial.replace(destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"cached checksum mismatch: {destination}")


def main():
    system = platform.system()
    if system not in PLATFORMS or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise SystemExit("This pinned installer supports Windows and Linux x86-64")
    target, extension, foundry_hash, solc_target, solc_hash, suffix = PLATFORMS[system]
    root = Path("out/tools/foundry").resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"foundry_{FOUNDRY}_{target}.{extension}"
    download(f"https://github.com/foundry-rs/foundry/releases/download/{FOUNDRY}/{archive.name}", archive, foundry_hash)
    names = [name + suffix for name in ["forge", "cast", "anvil", "chisel"]]
    if extension == "zip":
        with zipfile.ZipFile(archive) as bundle:
            for name in names:
                # Read only the expected root-level executable, never archive paths.
                (root / name).write_bytes(bundle.read(name))
    else:
        with tarfile.open(archive, "r:gz") as bundle:
            for name in names:
                member = bundle.getmember(name)
                if not member.isfile():
                    raise RuntimeError(f"expected regular executable: {name}")
                with bundle.extractfile(member) as stream:
                    (root / name).write_bytes(stream.read())
    compiler = root / ("solc" + suffix)
    download(f"https://raw.githubusercontent.com/ethereum/solc-bin/gh-pages/{solc_target}/solc-{solc_target}-v{SOLC}{suffix}",
             compiler, solc_hash)
    for name in names + [compiler.name]:
        binary = root / name
        if os.name != "nt":
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        subprocess.run([str(binary), "--version"], check=True)
    print(f"Installed pinned tools in {root}; add this directory to PATH")


if __name__ == "__main__":
    main()
