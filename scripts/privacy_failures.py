"""Validate numerical terminations without promoting a prefix to a full run."""

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from evaluation import read
from fltoken.data import sample_clients
from fltoken.io import sha256
from fltoken.metrics import ROUND_POOL, payout, summarize
from fltoken.privacy import account_schedule


FAILURE_FILES = ["per_round.csv", "round_metrics.csv", "ledger.json", "prefix_metrics.json", "failure.json", "timing.json"]


def planned_accounting(config):
    """The full public schedule is hypothetical when training terminates early."""
    schedule = [sample_clients(config["num_clients"], config["clients_per_round"], number, config["seed"]).tolist()
                for number in range(1, config["rounds"]+1)]
    return account_schedule(config, schedule)


def verify_failure(folder, config, verify_cache=True, attacks=None):
    folder = Path(folder)
    metadata, failure = read(folder/"metadata.json"), read(folder/"failure.json")
    number, completed = failure["failed_round"], failure["completed_rounds"]
    if (failure.get("schema") != 1 or failure.get("status") != "failed_numerical" or
            type(number) is not int or not 1 <= number <= config["rounds"] or
            type(completed) is not int or completed != number-1 or
            metadata.get("status") != "failed_numerical" or metadata.get("completed_rounds") != completed or
            metadata.get("failure_sha256") != sha256(folder/"failure.json") or
            (folder/"metrics.json").exists() or (folder/"privacy.json").exists()):
        raise ValueError("numerical termination must retain a strict prefix, without full-run metrics or transcript accounting")
    expected_ids = sample_clients(config["num_clients"], config["clients_per_round"], number, config["seed"]).tolist()
    if failure["sampled_ids"] != expected_ids or not failure["failures"]:
        raise ValueError("failure identities differ from the public sample")
    seen = set()
    for item in failure["failures"]:
        client, stage, count = item["client_id"], item["stage"], item["nonfinite_count"]
        if type(count) is not int or count <= 0 or client in seen:
            raise ValueError("invalid numerical failure evidence")
        seen.add(client)
        if stage == "local_training":
            if type(client) is not int or client not in expected_ids:
                raise ValueError("invalid failed client identity")
        elif stage in {"aggregation", "evaluation"}:
            if client is not None or len(failure["failures"]) != 1 or (stage == "evaluation" and count > 2):
                raise ValueError("invalid server numerical failure")
        else:
            raise ValueError("unrecognized numerical failure stage")
    hashes = read(folder/"hashes.json")
    if set(hashes) != set(FAILURE_FILES) or any(hashes[name] != sha256(folder/name) for name in FAILURE_FILES):
        raise ValueError("failed-run prefix differs from its recorded hashes")
    protocol = read(folder/"protocol.json")
    sizes = protocol["partition_sizes"]
    attacks = {} if attacks is None else attacks
    if ({int(key): value for key, value in protocol['adversaries'].items()} != attacks or
            len(sizes) != config["num_clients"] or any(type(n) is not int or n <= 0 for n in sizes)):
        raise ValueError("failed privacy run differs from its declared public partition and attacks")
    with (folder/"per_round.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (folder/"round_metrics.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    if len(rows) != completed*config["clients_per_round"] or len(history) != completed:
        raise ValueError("failed-run diagnostic prefix is incomplete")
    ledger = {rule: [] for rule in config["rules"]}
    typed = []
    for index in range(completed):
        number = index+1
        group = rows[index*config["clients_per_round"]:(index+1)*config["clients_per_round"]]
        ids = sample_clients(config["num_clients"], config["clients_per_round"], number, config["seed"]).tolist()
        if ([int(row["client_id"]) for row in group] != ids or
                any(int(row["round"]) != number for row in group) or int(history[index]["round"]) != number):
            raise ValueError("diagnostic prefix differs from the public schedule")
        signature = lambda row: (float(row["accuracy"]), float(row["loss"]), row["update_set_sha256"])
        for row, client in zip(group, ids):
            source = max(1, number-config['stale_lag']) if attacks.get(client) == 'stale' else number
            if (signature(row) != signature(history[index]) or row["adversary"] != str(client in attacks) or row["attack"] != attacks.get(client, 'none') or
                    int(row["source_round"]) != source or int(row["num_examples"]) != sizes[client]):
                raise ValueError("diagnostic participation differs from the shared trajectory")
            numeric = ["delta_norm", "aggregate_norm", "cosine", "predicted_bias", "accuracy", "loss"]
            numeric += [f"{rule}_{field}" for rule in ledger for field in ["score", "share"]]
            if any(not math.isfinite(float(row[key])) for key in numeric) or not 0 <= float(row["accuracy"]) <= 1 or float(row["loss"]) < 0:
                raise ValueError("completed diagnostic rounds must contain finite metrics")
            parsed = {**row, "round": number, "client_id": client, "adversary": client in attacks, "accuracy": float(row["accuracy"])}
            for rule in ledger:
                if row[f"{rule}_slash"] not in {"True", "False"}:
                    raise ValueError("invalid diagnostic slash flag")
                parsed[f"{rule}_share"] = float(row[f"{rule}_share"])
                parsed[f"{rule}_slash"] = row[f"{rule}_slash"] == "True"
            typed.append(parsed)
        for rule in ledger:
            units, amounts = payout(np.array([float(row[f"{rule}_share"]) for row in group]))
            ledger[rule].append(dict(round=number, ids=ids, shares1e18=units, payouts=amounts,
                                     slash=[row[f"{rule}_slash"] == "True" for row in group], remainder=ROUND_POOL-sum(amounts)))
        if verify_cache:
            with np.load(Path("results/cache")/folder.name/f"round-{number:03d}.npz", allow_pickle=False) as saved:
                matrix, cached_ids = saved["deltas"], saved["client_ids"]
            if (cached_ids.tolist() != ids or matrix.ndim != 2 or matrix.shape[0] != len(ids) or
                    not matrix.shape[1] or not np.isfinite(matrix).all() or
                    hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest() != group[0]["update_set_sha256"]):
                raise ValueError("diagnostic cache differs from the common matrix")
    if verify_cache:
        expected = {f"round-{number:03d}.npz" for number in range(1, completed+1)}
        if {p.name for p in (Path("results/cache")/folder.name).glob("*.npz")} != expected:
            raise ValueError("diagnostic cache must contain exactly the completed prefix")
    metrics = summarize(typed, ledger, config, attacks) if typed else None
    if read(folder/"ledger.json") != ledger or read(folder/"prefix_metrics.json") != metrics:
        raise ValueError("diagnostic ledger or prefix metrics do not reconstruct")
    timing = read(folder/"timing.json")
    if (len(timing["round_seconds"]) != completed or
            any(not math.isfinite(value) or value < 0 for value in timing["round_seconds"]) or
            timing["training_seconds"] != sum(timing["round_seconds"])):
        raise ValueError("diagnostic timing must cover exactly the completed prefix")
    names = [*FAILURE_FILES, "metadata.json", "protocol.json", "hashes.json"]
    return {name: sha256(folder/name) for name in names}
