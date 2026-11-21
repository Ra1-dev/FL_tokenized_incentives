"""Package the current generated assets as a portable, content-bound upload."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from document_paper_assets import asset_inventory, macro_groups
from package_assets import validate_metadata


def payload(root=Path('.')):
    summary = json.loads((root/'results/summary.json').read_text(encoding='utf-8'))
    required = {'reproduction', 'evaluation', 'extensions', 'privacy', 'fashion', 'grid_settlement', 'gas'}
    if required - summary.keys():
        raise ValueError('final assets require every empirical result section')
    asset_inventory(root)
    macro_groups((root/'paper/generated/numbers.tex').read_text(encoding='utf-8'))
    paths = [*sorted((root/'figures').glob('*.pdf')), *sorted((root/'figures').glob('*.png')),
             *sorted((root/'paper/generated').glob('*.tex')),
             *[root/name for name in ['paper/README.md', 'paper/references.bib', 'docs/annotated_bibliography.md',
                                     'docs/defense_notes.md', 'docs/proposal.pdf', 'CITATION.cff', '.zenodo.json', 'LICENSE']]]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError('only existing regular assets may enter the upload')
    blobs = {path.relative_to(root).as_posix():path.read_bytes() for path in paths}
    metadata = validate_metadata({**blobs, 'pyproject.toml':(root/'pyproject.toml').read_bytes()})
    manifest = dict(schema=1, purpose='Overleaf research assets; no manuscript', software=metadata,
                    summary_sha256=hashlib.sha256((root/'results/summary.json').read_bytes()).hexdigest(),
                    files={name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(blobs.items())})
    blobs['asset-manifest.json'] = (json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    return blobs


def assemble(blobs, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if any(Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name for name in blobs):
        raise ValueError('archive paths must be relative and contained')
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, data in sorted(blobs.items()):
                # A ZIP format epoch is a serialization constant, not a publication date.
                info = zipfile.ZipInfo(name, (1980,1,1,0,0,0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info,data)
        if temporary.stat().st_size > 5_000_000:
            raise ValueError('upload exceeds the repository artifact-size limit')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(blobs):
                raise ValueError('archive integrity mismatch')
            for name, data in blobs.items():
                if archive.read(name) != data:
                    raise ValueError('archive contents differ from input assets')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='compare the saved archive with current generated assets')
    args = parser.parse_args()
    blobs = payload()
    output = Path('paper/overleaf_assets.zip')
    if args.check:
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(blobs) or any(archive.read(name) != data for name,data in blobs.items()):
                raise ValueError('saved upload differs from current assets; rebuild it')
    else:
        assemble(blobs,output)
    print(json.dumps(dict(files=len(blobs)-1, bytes=output.stat().st_size,
                          sha256=hashlib.sha256(output.read_bytes()).hexdigest(), checked=args.check),sort_keys=True))


if __name__ == '__main__':
    main()
