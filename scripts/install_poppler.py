"""Install a checksum-pinned portable Windows PDF renderer in out/tools."""

import hashlib
import os
from pathlib import Path
import urllib.request
import zipfile

VERSION = "26.09.0-0"
SHA256 = "7a6f256a0ddf7536182246a5733331bf4677cbcc34f4663774947ad34556c8d0"
URL = f"https://github.com/oschwartz10612/poppler-windows/releases/download/v{VERSION}/Release-{VERSION}.zip"


def main():
    if os.name != "nt":
        raise SystemExit("Install the system poppler-utils package and use pdftoppm from PATH")
    folder = Path("out/tools/poppler").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"Release-{VERSION}.zip"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Poppler archive checksum does not match the release asset")
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if not (folder / name).resolve().is_relative_to(folder):
                raise RuntimeError("archive member escapes installation directory")
        bundle.extractall(folder)
    binaries = list(folder.rglob("pdftoppm.exe"))
    if len(binaries) != 1:
        raise RuntimeError("expected one pdftoppm executable")
    print(binaries[0])


if __name__ == "__main__":
    main()
