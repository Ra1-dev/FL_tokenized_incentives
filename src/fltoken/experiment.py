"""Flower Message API apps for fixed-trajectory shadow reward experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp

from fltoken.attacks import gaussian_noise, sign_flip, stale_round, zero_update
from fltoken.data import dirichlet_partition, sample_clients, seed_for
from fltoken.datasets import load_tensors
from fltoken.io import sha256, write_json
from fltoken.metrics import ROUND_POOL, payout, summarize
from fltoken.models import LeNet5
from fltoken.rules import make_rules, score_round
from fltoken.scoring import cosine_to_reference
from fltoken.training import configure_torch, evaluate, flatten, local_update

client_app = ClientApp()


@client_app.train()
def train(message: Message, context: Context) -> Message:
    control = message.content["control"]
    config = json.loads(control["config"])
    initial, indices = message.content["arrays"].to_numpy_ndarrays()
    images, labels = load_tensors(config["dataset"], config["data_dir"], True)
    delta = local_update(initial, images[indices], labels[indices], config,
                         control["client_id"], control["source_round"], control["poisoned"])
    return Message(RecordDict({
        "arrays": ArrayRecord([delta]),
        "control": ConfigRecord({"client_id": control["client_id"], "num_examples": len(indices)}),
    }), reply_to=message)


def adversaries(config: dict) -> dict[int, str]:
    permutation = np.random.default_rng(config["seed"]).permutation(config["num_clients"])
    if config["attack"] == "sendros_mix":
        return dict(zip(map(int, permutation[:5]), ["zero", "zero", "noise", "noise", "stale"]))
    if config["attack"] == "none":
        return {}
    m = int(round(config["fraction"] * config["num_clients"]))
    return {int(i): config["attack"] for i in permutation[:m]}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_server(config: dict, output: Path) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        configure_torch(config["seed"], config["torch_threads"])
        train_x, train_y = load_tensors(config["dataset"], config["data_dir"], True)
        test_x, test_y = load_tensors(config["dataset"], config["data_dir"], False)
        if config["train_limit"]:
            train_x, train_y = train_x[:config["train_limit"]], train_y[:config["train_limit"]]
        if config["test_limit"]:
            test_x, test_y = test_x[:config["test_limit"]], test_y[:config["test_limit"]]
        partitions = dirichlet_partition(train_y.numpy(), config["num_clients"], config["alpha"], config["seed"])
        attack_map = adversaries(config)
        fraction = len(attack_map) / config["num_clients"]
        scoring_config = {**config, "krum_f": min(math.ceil(fraction * config["clients_per_round"]), (config["clients_per_round"] - 3) // 2)}
        rules = make_rules(scoring_config)
        global_vector = flatten(LeNet5())
        snapshots = {}
        rows, round_rows, durations = [], [], []
        ledger = {name: [] for name in rules}
        # Node ids are transport addresses only. Scientific identities are in payloads.
        deadline = time.monotonic() + 120
        nodes = []
        while len(nodes) < config["num_clients"]:
            nodes = sorted(grid.get_node_ids())
            if time.monotonic() > deadline:
                raise RuntimeError("Flower nodes did not become available")
            if len(nodes) < config["num_clients"]:
                time.sleep(.1)
        write_json(output / "protocol.json", {
            "adversaries": attack_map, "partition_sizes": list(map(len, partitions)),
            "partition_sha256": [hashlib.sha256(p.astype("<i8").tobytes()).hexdigest() for p in partitions],
            "krum_f": scoring_config["krum_f"], "aggregator": "sample-count-weighted FedAvg over all sampled updates",
            "training_runtime": "Flower ServerApp/ClientApp Message API with Ray simulation backend",
        })
        for round in range(1, config["rounds"] + 1):
            started = time.perf_counter()
            snapshots[round] = global_vector.copy()
            ids = sample_clients(config["num_clients"], config["clients_per_round"], round, config["seed"])
            messages = []
            for client in ids:
                client = int(client)
                attack = attack_map.get(client, "none")
                source = stale_round(round, config["stale_lag"]) if attack == "stale" else round
                messages.append(Message(RecordDict({
                    "arrays": ArrayRecord([snapshots[source], partitions[client]]),
                    "control": ConfigRecord({"config": json.dumps(config), "client_id": client,
                                             "source_round": source, "poisoned": attack == "label_flip"}),
                }), dst_node_id=nodes[client], message_type="train", group_id=str(round)))
            replies = list(grid.send_and_receive(messages, timeout=1800))
            if len(replies) != len(ids):
                raise RuntimeError(f"received {len(replies)} replies for {len(ids)} sampled clients")
            by_id = {}
            for reply in replies:
                if reply.has_error():
                    raise RuntimeError(f"Flower client error: {reply.error}")
                client = reply.content["control"]["client_id"]
                if client in by_id or client not in ids:
                    raise RuntimeError("duplicate or unexpected reply identity")
                by_id[client] = reply.content["arrays"].to_numpy_ndarrays()[0]
                if reply.content["control"]["num_examples"] != len(partitions[client]):
                    raise RuntimeError("client sample count mismatch")
            updates = []
            for client in ids:
                delta = by_id[int(client)]
                attack = attack_map.get(int(client), "none")
                if attack == "zero":
                    delta = zero_update(delta)
                elif attack == "noise":
                    delta = gaussian_noise(delta.shape, config["noise_sigma"], np.random.default_rng(seed_for(config["seed"], round, int(client), 3)))
                elif attack == "sign_flip":
                    delta = sign_flip(delta)
                updates.append(delta)
            deltas = np.stack(updates).astype(np.float64)
            update_hash = hashlib.sha256(deltas.astype("<f8").tobytes()).hexdigest()
            scores = score_round(rules, ids, deltas, round)
            if update_hash != hashlib.sha256(deltas.astype("<f8").tobytes()).hexdigest():
                raise RuntimeError("a scoring rule modified the common update matrix")
            counts = np.array([len(partitions[int(i)]) for i in ids], dtype=np.float64)
            global_vector = (global_vector + np.average(deltas, axis=0, weights=counts)).astype(np.float32).astype(np.float64)
            accuracy, loss = evaluate(global_vector, test_x, test_y, config["device"])
            mean = deltas.mean(axis=0)
            mean_norm = float(np.linalg.norm(mean))
            cosines = cosine_to_reference(deltas, mean)
            for name, result in scores.items():
                units, amounts = payout(result["shares"])
                ledger[name].append(dict(round=round, ids=ids.tolist(), shares1e18=units, payouts=amounts,
                                         slash=[bool(v) for v in result["slash"]], remainder=ROUND_POOL - sum(amounts)))
            for position, client in enumerate(ids):
                norm = float(np.linalg.norm(deltas[position]))
                attack = attack_map.get(int(client), "none")
                row = dict(round=round, client_id=int(client), adversary=int(client) in attack_map, attack=attack,
                           num_examples=int(counts[position]), source_round=stale_round(round, config["stale_lag"]) if attack == "stale" else round,
                           delta_norm=norm, aggregate_norm=mean_norm, cosine=float(cosines[position]),
                           predicted_bias=norm / (len(ids) * mean_norm) if mean_norm else 0.0,
                           update_set_sha256=update_hash, accuracy=accuracy, loss=loss)
                for name, result in scores.items():
                    row.update({f"{name}_score": float(result["scores"][position]), f"{name}_share": float(result["shares"][position]),
                                f"{name}_selected": bool(result["selected"][position]), f"{name}_slash": bool(result["slash"][position])})
                rows.append(row)
            round_rows.append(dict(round=round, accuracy=accuracy, loss=loss, update_set_sha256=update_hash))
            if config["save_updates"]:
                cache = Path("results/cache") / output.name
                cache.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache / f"round-{round:03d}.npz", client_ids=ids, deltas=deltas)
            durations.append(time.perf_counter() - started)
            # Flush each finished round so an interrupted run is diagnosable.
            write_csv(output / "per_round.csv", rows)
            write_csv(output / "round_metrics.csv", round_rows)
            print(f"Round {round}/{config['rounds']}: accuracy={accuracy:.4f} loss={loss:.4f} seconds={durations[-1]:.2f}", flush=True)
        write_json(output / "ledger.json", ledger)
        write_json(output / "metrics.json", summarize(rows, ledger, config, attack_map))
        write_json(output / "timing.json", {"round_seconds": durations, "training_seconds": sum(durations)})
        write_json(output / "hashes.json", {name: sha256(output / name) for name in ["per_round.csv", "round_metrics.csv", "ledger.json"]})

    return app
