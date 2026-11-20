"""Summarize complete honest-noise runs without treating analytic bounds as guarantees."""

from pathlib import Path

from evaluation import published_bytes, published_method, read
from fltoken.io import sha256, write_json
from grid_reproduction import verify_training_source
from privacy_protocol import PLAN, FROZEN, published_privacy
from run_grid import checked_execution_clock
from run_privacy import expected_config, privacy_cells, verify_cell
from summarize_grid import METRICS, interval


def summarize_records(records, plan, frozen):
    planned = list(privacy_cells(plan))
    if [{key: row[key] for key in planned[0]} for row in records] != planned:
        raise ValueError("privacy summary requires every unique declared cell in order")
    if any(row["metrics"][rule]["final_accuracy"] != row["metrics"]["B1"]["final_accuracy"]
           for row in records for rule in plan["rules"]):
        raise ValueError("privacy reward rules must share one training accuracy")
    rows = []
    for multiplier in plan["noise_multipliers"]:
        cells = [row for row in records if row["noise_multiplier"] == multiplier]
        metrics = {rule: {name: interval([row["metrics"][rule][name] for row in cells], plan["statistics"])
                          for name in METRICS} for rule in plan["rules"]}
        differences = {}
        for rule in ["Mstar", "Aref"]:
            left = [row["metrics"][rule]["honest_false_penalty_rate"] for row in cells]
            right = [row["metrics"]["B1"]["honest_false_penalty_rate"] for row in cells]
            differences[rule] = interval([None if a is None or b is None else a-b for a, b in zip(left, right)], plan["statistics"])
        accounts = [row["accounting"] for row in cells]
        per_seed = []
        for cell, account in zip(cells, accounts):
            clients = account["clients"]
            finite = all(client["finite"] for client in clients)
            per_seed.append(dict(seed=cell["seed"], all_clients_finite=finite,
                                 maximum_client_epsilon=max(client["epsilon"] for client in clients) if finite else None,
                                 infinite_epsilon_clients=sum(not client["finite"] for client in clients),
                                 minimum_participations=min(client["participations"] for client in clients),
                                 maximum_participations=max(client["participations"] for client in clients)))
        conservative = accounts[0]["conservative_all_rounds"]
        if any(account["conservative_all_rounds"] != conservative for account in accounts):
            raise ValueError("privacy accounting assumptions differ between seeds")
        rows.append(dict(noise_multiplier=multiplier, seeds=plan["seeds"], metrics=metrics,
                         paired_false_penalty_difference_from_B1=differences,
                         conservative_all_rounds=conservative, per_seed_accounting=per_seed))
    return dict(schema=1, runs=len(records), seeds=plan["seeds"], rounds=plan["rounds"],
                training_device=plan["device"], clipping_bound=frozen["clipping_bound"], delta=frozen["delta"],
                cells=rows, simulation_only=True, zero_noise_control="clipped without additive noise",
                accounting_limitations=records[0]["accounting"]["limitation"],
                comparison="Each rule scores the same released updates within a run; every multiplier trains its own perturbed Flower trajectory",
                pairing="Same seed across rules and noise multipliers; five-seed percentile intervals, no within-condition significance claim")


def main():
    published_bytes(PLAN)
    frozen, method = published_privacy(full_evidence=True), published_method()
    plan = read(PLAN)
    manifest_path = Path("results/privacy/manifest.json")
    manifest = read(manifest_path)
    planned = list(privacy_cells(plan))
    if (not manifest["complete"] or manifest["expected_cells"] != len(planned) or
            manifest["completed_cells"] != len(planned) or
            manifest["plan_sha256"] != sha256(PLAN) or manifest["frozen_sha256"] != sha256(FROZEN) or
            manifest["method_sha256"] != sha256("configs/method_frozen.json") or
            [{key: row[key] for key in planned[0]} for row in manifest["cells"]] != planned):
        raise ValueError("privacy manifest does not bind complete published evaluation coverage")
    binding = {key: manifest[key] for key in ["plan_sha256", "frozen_sha256", "method_sha256", "training_source_sha256"]}
    clock = checked_execution_clock(Path("results/privacy/execution.json"), binding)
    if (clock["finished_utc"] is None or manifest["elapsed_wall_seconds"] != clock["elapsed_wall_seconds"] or
            manifest["execution_sha256"] != sha256("results/privacy/execution.json")):
        raise ValueError("privacy execution must retain its completed elapsed clock")
    records, checked, common_versions = [], set(), None
    for cell in manifest["cells"]:
        folder = Path(cell["path"])
        metadata, inputs = verify_cell(folder, expected_config(plan, cell, frozen, method), verify_cache=False)
        if (inputs != cell["inputs"] or metadata["git_commit"] != cell["source_commit"] or
                metadata["source_sha256"] != manifest["training_source_sha256"] or
                metadata["source_config"] != plan["config"]):
            raise ValueError("privacy evidence differs from the common training record")
        if common_versions is not None and common_versions != metadata["versions"]:
            raise ValueError("privacy library versions differ between cells")
        common_versions = metadata["versions"]
        if metadata["git_commit"] not in checked:
            verify_training_source(metadata)
            checked.add(metadata["git_commit"])
        records.append(dict(**cell, metrics=read(folder/"metrics.json"), accounting=read(folder/"privacy.json")))
    result = summarize_records(records, plan, frozen)
    result["provenance"] = dict(manifest_sha256=sha256(manifest_path), frozen_sha256=sha256(FROZEN),
                                plan_sha256=sha256(PLAN), analysis_sha256=sha256(__file__),
                                elapsed_wall_seconds=clock["elapsed_wall_seconds"])
    write_json("results/privacy/summary.json", result)
    summary = read("results/summary.json")
    summary["privacy"] = result
    write_json("results/summary.json", summary)
    print(f"Summarized complete privacy runs: {len(records)}")


if __name__ == "__main__":
    main()
