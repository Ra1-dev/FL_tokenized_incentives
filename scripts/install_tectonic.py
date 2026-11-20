"""Install a pinned upstream Tectonic executable inside the ignored tools area."""

import hashlib
import io
import json
import platform
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from fltoken.io import write_json

VERSION = "0.17.0"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "fltoken-reproduction"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main():
    system = platform.system()
    target = "x86_64-pc-windows-msvc.zip" if system == "Windows" else "x86_64-unknown-linux-gnu.tar.gz"
    release = json.loads(fetch(f"https://api.github.com/repos/tectonic-typesetting/tectonic/releases/tags/tectonic%40{VERSION}"))
    asset = next(a for a in release["assets"] if a["name"] == f"tectonic-{VERSION}-{target}")
    payload = fetch(asset["browser_download_url"])
    digest = hashlib.sha256(payload).hexdigest()
    if asset.get("digest") and asset["digest"] != f"sha256:{digest}":
        raise ValueError("Tectonic archive does not match the upstream asset digest")
    folder = Path("out/tools")
    folder.mkdir(parents=True, exist_ok=True)
    name = "tectonic.exe" if system == "Windows" else "tectonic"
    if system == "Windows":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next(n for n in archive.namelist() if Path(n).name == name)
            executable = archive.read(member)
    else:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            member = next(m for m in archive.getmembers() if Path(m.name).name == name and m.isfile())
            executable = archive.extractfile(member).read()
    path = folder / name
    path.write_bytes(executable)
    path.chmod(0o755)
    write_json(folder / "tectonic_install.json", dict(version=VERSION, url=asset["browser_download_url"],
                                                      archive_sha256=digest, upstream_digest=asset.get("digest")))
    print(f"Installed {path}: archive SHA256 {digest}")


if __name__ == "__main__":
    main()
