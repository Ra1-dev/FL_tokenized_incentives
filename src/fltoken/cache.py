"""Load cached update matrices only when they match their recorded participation log."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import sha256


def load_update_rounds(folder, cache_root="results/cache"):
    folder = Path(folder)
    metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("complete training run required")
    hashes = json.loads((folder/"hashes.json").read_text(encoding="utf-8"))
    for name in ["per_round.csv", "round_metrics.csv", "ledger.json"]:
        if sha256(folder/name) != hashes[name]:
            raise ValueError(f"recorded artifact hash mismatch: {name}")
    with (folder/"per_round.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    config = metadata["config"]
    if len(rows) != config["rounds"]*config["clients_per_round"]:
        raise ValueError("incomplete participation log")
    for round in range(1, config["rounds"]+1):
        selected = [row for row in rows if int(row["round"]) == round]
        with np.load(Path(cache_root)/folder.name/f"round-{round:03d}.npz", allow_pickle=False) as saved:
            ids, deltas = saved["client_ids"], saved["deltas"]
        expected = [int(row["client_id"]) for row in selected]
        if ids.tolist() != expected or len(set(expected)) != config["clients_per_round"]:
            raise ValueError("cached identities differ from recorded sample order")
        if deltas.ndim != 2 or deltas.shape[0] != len(ids) or deltas.shape[1] == 0 or not np.isfinite(deltas).all():
            raise ValueError("invalid cached update matrix")
        digest = hashlib.sha256(deltas.astype("<f8").tobytes()).hexdigest()
        if any(row["update_set_sha256"] != digest for row in selected):
            raise ValueError("cached updates differ from the recorded common matrix")
        deltas.flags.writeable = False
        yield round, ids, deltas, selected
