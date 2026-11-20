"""Hash-bound offline geometry and exact allocation replay for reputation rules."""

import csv
import json
from pathlib import Path
import platform

import numpy as np

from evaluation import read
from fltoken.cache import load_update_rounds
from fltoken.data import sample_clients
from fltoken.extensions import CGSVScorer, GEOMETRY_FIELDS, RFFLScorer
from fltoken.io import sha256, write_json
from fltoken.metrics import ROUND_POOL, payout, summarize
from reproduction_analysis import exact_quantile
from select_method import expected_attacks


FACTORIES = {"CGSV": CGSVScorer, "RFFL": RFFLScorer}
SOURCE_PATHS = ["src/fltoken/extensions.py", "src/fltoken/cache.py", "src/fltoken/data.py",
                "src/fltoken/io.py", "src/fltoken/metrics.py", "scripts/extension_replay.py",
                "scripts/reproduction_analysis.py", "scripts/select_method.py"]
INPUT_NAMES = ["metadata.json", "protocol.json", "per_round.csv", "round_metrics.csv",
               "ledger.json", "hashes.json"]


def source_binding():
    return {path: sha256(path) for path in SOURCE_PATHS}


def serial(value):
    if isinstance(value, dict):
        return {key: serial(item) for key, item in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def read_training(folder):
    folder = Path(folder)
    metadata, protocol = read(folder/"metadata.json"), read(folder/"protocol.json")
    if metadata["status"] != "complete":
        raise ValueError("offline scoring requires a complete training run")
    config = metadata["config"]
    attacks = {int(key): value for key, value in protocol["adversaries"].items()}
    if attacks != expected_attacks(config):
        raise ValueError("attack identities differ from the seeded training protocol")
    hashes = read(folder/"hashes.json")
    for name in ["per_round.csv", "round_metrics.csv", "ledger.json"]:
        if hashes[name] != sha256(folder/name):
            raise ValueError("training artifact differs from its recorded hash")
    with (folder/"per_round.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != config["rounds"]*config["clients_per_round"]:
        raise ValueError("offline scoring requires every training participation")
    rounds = []
    for number in range(1, config["rounds"]+1):
        group = [row for row in rows if int(row["round"]) == number]
        ids = [int(row["client_id"]) for row in group]
        if ids != sample_clients(config["num_clients"], config["clients_per_round"], number, config["seed"]).tolist():
            raise ValueError("participation order differs from the seeded sample")
        if any(row["attack"] != attacks.get(client, "none") for client, row in zip(ids, group)):
            raise ValueError("participation attack differs from the assigned identity")
        if len({row["update_set_sha256"] for row in group}) != 1:
            raise ValueError("participations do not share one common update matrix")
        rounds.append((number, ids, group))
    inputs = {name: sha256(folder/name) for name in INPUT_NAMES}
    return metadata, attacks, rounds, inputs


def geometry_params(family, params):
    # Altruism affects quotas only; all CGSV candidates share one geometry and
    # reputation trajectory. No redundant matrix pass is needed for each beta.
    return {key: value for key, value in params.items() if family != "CGSV" or key != "beta"}


def record_geometry(folder, output, parameters, cache_root="results/cache"):
    """Read each immutable matrix once and advance both families independently."""
    folder, output = Path(folder), Path(output)
    metadata, _, _, inputs = read_training(folder)
    binding = source_binding()
    scorers = {name: FACTORIES[name](params) for name, params in parameters.items()}
    traces = {name: [] for name in scorers}
    for number, ids, matrix, original in load_update_rounds(folder, cache_root):
        for name, scorer in scorers.items():
            result = scorer.score(ids, matrix, number)
            traces[name].append(dict(round=number, client_ids=ids.tolist(),
                                     update_set_sha256=original[0]["update_set_sha256"],
                                     geometry=serial({key: result[key] for key in GEOMETRY_FIELDS})))
    paths = {}
    for name in scorers:
        path = output/f"{name}.json"
        record = dict(schema=1, family=name, params=geometry_params(name, parameters[name]),
                      training_device=metadata["config"]["device"], inputs=inputs,
                      calculation=dict(numpy=np.__version__, python=platform.python_version(), platform=platform.system()),
                      source_sha256=binding, rounds=traces[name])
        if path.exists() and read(path) != record:
            raise ValueError("existing geometry differs; preserve it before recording another implementation")
        write_json(path, record)
        paths[name] = path
    return paths


def bias_summary(rows):
    noise = [row for row in rows if row["attack"] == "noise"]
    output = {}
    for name, observations in [("all_noise", noise),
                               ("eligible_noise", [row for row in noise if row["eligible_before"]])]:
        defined = [row for row in observations if row["bias_defined"]]
        relative = [abs(row["cross_residual"])/abs(row["cosine"])
                    if row["cosine"] else float("inf") for row in defined]
        output[name] = dict(participations=len(observations), defined=len(defined),
                            undefined=len(observations)-len(defined),
                            zero_measured_cosine=sum(row["cosine"] == 0 for row in defined),
                            self_inclusion_quantiles={str(p): exact_quantile([row["self_inclusion"] for row in defined], p)
                                                      if defined else None for p in [.25, .5, .75]},
                            relative_error_quantiles={str(p): exact_quantile(relative, p) if relative else None
                                                      for p in [.25, .5, .75]},
                            positive_cosine=sum(row["cosine"] > 0 for row in observations),
                            positive_excluding_cosine=sum(row["excluding_cosine"] > 0 for row in observations))
    return output


def replay_geometry(folder, path, params):
    """Reconstruct states, shares, integer ledgers and metrics without matrices."""
    metadata, attacks, rounds, inputs = read_training(folder)
    trace = read(path)
    family = trace["family"]
    if trace.get("schema") != 1 or family not in FACTORIES:
        raise ValueError("unsupported extension geometry format")
    if trace["inputs"] != inputs or trace["source_sha256"] != source_binding():
        raise ValueError("geometry inputs or scientific sources differ")
    if trace["params"] != geometry_params(family, params) or trace["training_device"] != metadata["config"]["device"]:
        raise ValueError("geometry parameters or training device differ")
    if len(trace["rounds"]) != len(rounds):
        raise ValueError("geometry must cover every training round")
    scorer = FACTORIES[family](params)
    ledger, rows, states = [], [], []
    for (number, ids, original), recorded in zip(rounds, trace["rounds"]):
        if (recorded["round"] != number or recorded["client_ids"] != ids or
                recorded["update_set_sha256"] != original[0]["update_set_sha256"]):
            raise ValueError("geometry does not follow the common matrix and client order")
        before = scorer.active[ids].copy()
        result = scorer.score_recorded(ids, recorded["geometry"], number)
        units, amounts = payout(result["shares"])
        ledger.append(dict(round=number, ids=ids, shares1e18=units, payouts=amounts,
                           slash=result["slash"].tolist(), remainder=ROUND_POOL-sum(amounts)))
        states.append(result["state"])
        for index, (client, source) in enumerate(zip(ids, original)):
            row = dict(round=number, client_id=client, attack=source["attack"], adversary=client in attacks,
                       accuracy=float(source["accuracy"]), update_set_sha256=recorded["update_set_sha256"],
                       eligible_before=bool(before[index]), undefined_state=result["undefined_state"])
            for key in GEOMETRY_FIELDS | {"quota", "removed", "eligible", "reputation_before", "reputation_after"}:
                row[key] = serial(result[key] if key == "reference_norm" else result[key][index])
            row.update({f"{family}_share": float(result["shares"][index]), f"{family}_slash": False})
            rows.append(row)
    metrics = summarize(rows, {family: ledger}, metadata["config"], attacks)[family]
    metrics.update(undefined_state_rounds=scorer.undefined_rounds,
                   removed_clients=int((~scorer.active).sum()),
                   honest_removed_clients=int(sum(not scorer.active[client] for client in range(scorer.num_clients) if client not in attacks)))
    return dict(family=family, params=params, training_device=metadata["config"]["device"],
                geometry_sha256=sha256(path), inputs=inputs, source_sha256=source_binding(),
                metrics=metrics, bias=bias_summary(rows), rows=rows, ledger=ledger, states=states)
