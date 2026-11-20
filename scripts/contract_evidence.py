"""Bind chain measurements to their executable sources and compiler settings."""

import hashlib
from pathlib import Path


def provenance(program):
    paths = list(Path("contracts/src").glob("*.sol"))
    paths += list(Path("third_party/openzeppelin-contracts/contracts").rglob("*.sol"))
    paths += [Path(p) for p in ["contracts/foundry.toml", "requirements-contracts.txt",
              "scripts/chain_ledger.py", "scripts/replay_onchain.py", "scripts/contract_evidence.py"]]
    paths.append(Path(program))
    return {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def validate_provenance(recorded):
    if not recorded:
        raise ValueError("missing chain measurement source provenance")
    for name, expected in recorded.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"chain measurement source changed: {name}")
