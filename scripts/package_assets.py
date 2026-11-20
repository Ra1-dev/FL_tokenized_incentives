"""Build a deterministic support-asset archive from one committed repository snapshot."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import tomllib
import zipfile

import yaml


ROOT_FILES = {"README.md", "LICENSE", "CITATION.cff", ".zenodo.json", "pyproject.toml", "results/summary.json"}
DOCUMENTS = {"METHOD.md", "REPRODUCTION.md", "EVALUATION.md", "EXTENSIONS.md", "PRIVACY.md",
             "FASHION.md", "CONTRACTS.md", "ASSETS.md", "annotated_bibliography.md", "defense_notes.md", "proposal.pdf"}
SECTIONS = ["reproduction", "evaluation", "extensions", "privacy", "fashion", "contracts", "gas"]


def included(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("archive paths must be relative POSIX paths")
    return (name in ROOT_FILES or name == "paper/references.bib" or
            len(path.parts) == 2 and path.parts[0] == "figures" and path.suffix in {".png", ".pdf"} or
            len(path.parts) == 3 and path.parts[:2] == ("paper", "generated") and path.suffix == ".tex" or
            len(path.parts) == 2 and path.parts[0] == "docs" and path.name in DOCUMENTS)


def committed_assets(reference):
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", reference+"^{commit}"], text=True).strip()
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("a full committed snapshot is required")
    timestamp = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", commit], text=True))
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", "--full-tree", commit])
    blobs = {}
    for record in tree.split(b"\0"):
        if not record:
            continue
        properties, raw_name = record.split(b"\t", 1)
        mode, kind, object_id = properties.decode().split()
        name = raw_name.decode("utf-8")
        if not included(name):
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"support assets must be regular committed files: {name}")
        blobs[name] = subprocess.check_output(["git", "cat-file", "blob", object_id])
    return commit, timestamp, blobs


def validate_metadata(blobs):
    citation = yaml.safe_load(blobs["CITATION.cff"])
    zenodo = json.loads(blobs[".zenodo.json"])
    project = tomllib.loads(blobs["pyproject.toml"].decode())["project"]
    if citation["cff-version"] != "1.2.0" or citation["type"] != "software" or not citation["message"]:
        raise ValueError("software citation metadata is incomplete")
    if citation["version"] != project["version"] or citation["title"] != zenodo["title"]:
        raise ValueError("software version or title differs between metadata files")
    if citation["license"] != "MIT" or zenodo["license"] != "MIT" or not blobs["LICENSE"].startswith(b"MIT License"):
        raise ValueError("software license metadata differs")
    authors = [dict(name=f"{author['family-names']}, {author['given-names']}", affiliation=author["affiliation"])
               for author in citation["authors"]]
    if authors != zenodo["creators"] or zenodo["upload_type"] != "software":
        raise ValueError("software authorship or archive type differs")
    if citation["repository-code"] not in {row["identifier"] for row in zenodo["related_identifiers"]}:
        raise ValueError("repository identifiers differ")
    return dict(title=citation["title"], version=project["version"], creators=authors, license=citation["license"])


def assemble(commit, timestamp, blobs, destination):
    required = ROOT_FILES | {"paper/references.bib", "paper/generated/numbers.tex"}
    if required - blobs.keys():
        raise ValueError("committed support package is missing required artifacts: "+", ".join(sorted(required-blobs.keys())))
    if any(not included(name) for name in blobs):
        raise ValueError("unsupported file in the support package")
    metadata = validate_metadata(blobs)
    summary = json.loads(blobs["results/summary.json"])
    manifest = dict(schema=1, purpose="research support assets", source_commit=commit,
                    source_commit_utc=datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                    software=metadata, summary_sha256=hashlib.sha256(blobs["results/summary.json"]).hexdigest(),
                    result_sections_present=[name for name in SECTIONS if name in summary],
                    result_sections_absent=[name for name in SECTIONS if name not in summary],
                    files={name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                           for name, data in sorted(blobs.items())})
    payload = {**blobs, "asset-manifest.json": (json.dumps(manifest, indent=2, sort_keys=True)+"\n").encode()}
    date = datetime.fromtimestamp(timestamp, timezone.utc)
    if not 1980 <= date.year <= 2107:
        raise ValueError("source commit date is outside the ZIP timestamp range")
    # ZIP storage avoids compressor-version differences; timestamps are the
    # actual source commit time, never local file modification times.
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name+".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in sorted(payload.items()):
                info = zipfile.ZipInfo(name, (date.year, date.month, date.day, date.hour, date.minute, date.second))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_STORED
                archive.writestr(info, data)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != payload.keys():
                raise ValueError("archive verification failed")
            for name, data in payload.items():
                if archive.read(name) != data:
                    raise ValueError(f"archived artifact differs from its committed bytes: {name}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD", help="committed snapshot to package")
    parser.add_argument("--output", type=Path, default=Path("out/support-assets.zip"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root/"out") or output.suffix != ".zip":
        raise ValueError("support archives must be ZIP files under the local out directory")
    commit, timestamp, blobs = committed_assets(args.ref)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = assemble(commit, timestamp, blobs, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(dict(source_commit=commit, files=len(manifest["files"]), archive_sha256=digest,
                          bytes=output.stat().st_size, output=output.relative_to(root).as_posix(),
                          result_sections_absent=manifest["result_sections_absent"]), sort_keys=True))


if __name__ == "__main__":
    main()
