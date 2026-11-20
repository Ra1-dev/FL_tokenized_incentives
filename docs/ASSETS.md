# Committed research assets

The support archive contains the generated figures, tables, numerical macros,
verified bibliography, protocol notes, proposal, and the result summary from
one committed repository snapshot. It preserves the repository's directory
layout. These materials support the manuscript written in Overleaf.

Regenerate the artifacts with the README commands, commit the completed unit,
then build its archive:

```bash
python scripts/package_assets.py
python scripts/package_assets.py --ref HEAD --output out/support-assets.zip
```

The command reads Git objects. Local modifications and untracked files cannot
silently change the archive. Repeating it for the same commit produces identical
bytes on supported platforms: entries have a fixed order, stored contents,
uniform file permissions, and the actual commit timestamp. The archive is
written under the ignored `out/` directory.

`asset-manifest.json` records the source commit, summary digest, and the size
and SHA-256 of every included file. It also lists which empirical result
sections are present and absent. A support package can be useful before all
experiments finish; its manifest and protocol notes keep that incompleteness
visible. An archive does not itself establish that the research or manuscript
is complete.

For an existing Overleaf project, upload the generated tables and macros from
`paper/generated/`, the bibliography from `paper/references.bib`, and the
figures from `figures/`, retaining the paths used by the manuscript. Tables
require `booktabs`. The accompanying protocol notes identify one-column and
two-column figure sizes and table placement. Recompile the authored source
and inspect each placed asset after replacing it. Preserve the archive and its
manifest alongside the exported manuscript so its numerical inputs remain
identifiable.

## Software citation and archival metadata

`CITATION.cff` describes this software using the
[Citation File Format](https://citation-file-format.github.io/) and its
[schema guide](https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md).
Its version matches `pyproject.toml`; cite the exact commit used because
development snapshots can share a package version.

`.zenodo.json` supplies the software title, creator, affiliation, license and
repository relation following
[Zenodo's GitHub metadata documentation](https://developers.zenodo.org/#add-metadata-to-your-github-repository-release).
The package command checks consistency between both metadata files and the
project version and license. These checks cover this repository's declared
fields; they are not a replacement for an archive service's complete schema
and publication checks.

A DOI, archival publication date and final release tag are pending. Add only
identifiers actually issued for the corresponding release. The software
metadata makes no claim that a manuscript has been submitted or accepted.
