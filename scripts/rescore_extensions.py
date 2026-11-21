"""Score the published reputation rules on retained common evaluation updates."""

import argparse
import json
from pathlib import Path

from evaluation import ABLATIONS, evaluation_cells, published_bytes, recorded_method, read
from extension_replay import record_geometry, replay_geometry
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from grid_reproduction import verify_training_source
from select_extensions import FROZEN_PATH, PLAN_PATH, runtime_extensions, scientific_bindings, select


def rescore(recorded_only=False, available=False):
    frozen = runtime_extensions(full_evidence=True)
    # Rebuild the complete candidate comparison before any held-out scoring.
    select(recorded_only=True)
    runtime_extensions(full_evidence=True)
    method = recorded_method()
    plan_path = Path("configs/evaluation_plan.json")
    plan = json.loads(published_bytes(plan_path))
    planned = list(evaluation_cells(plan))
    manifest_path = Path("results/grid/manifest.json")
    manifest = read(manifest_path)
    if (manifest["plan_sha256"] != sha256(plan_path) or
            manifest["method_sha256"] != sha256("configs/method_frozen.json") or
            manifest["expected_cells"] != len(planned) or
            manifest["completed_cells"] != len(manifest["cells"])):
        raise ValueError("evaluation manifest differs from the published comparison")
    if not available and manifest["completed_cells"] != len(planned):
        raise ValueError("complete evaluation coverage is required; use --available for an explicitly partial rescore")
    supplied = [{key: row[key] for key in planned[0]} for row in manifest["cells"]]
    if supplied != planned[:len(supplied)]:
        raise ValueError("completed evaluation cells must follow the declared unique order")
    base = load_config(plan["config"])
    if any(params["num_clients"] != base["num_clients"] for params in frozen["params"].values()):
        raise ValueError("registered population differs from the frozen reputation rules")
    output = Path("results/extensions")
    completed, checked, common = [], set(), None
    for cell in manifest["cells"]:
        folder = Path(cell["path"])
        metadata = read(folder/"metadata.json")
        expected = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction", "rounds"]},
                    **{key: plan[key] for key in ["device", "workers", "gpu_fraction"]},
                    "rules": plan["rules"]+list(ABLATIONS), "ours_params": method["params"]}
        if (metadata["config"] != expected or metadata["source_config"] != plan["config"] or
                metadata["method_sha256"] != sha256("configs/method_frozen.json") or
                metadata["git_commit"] != cell["source_commit"] or
                sha256(folder/"metadata.json") != cell["metadata_sha256"] or
                sha256(folder/"per_round.csv") != cell["per_round_sha256"]):
            raise ValueError("offline evaluation inputs differ from the common grid")
        implementation = (metadata["source_sha256"], metadata["versions"])
        if common is not None and common != implementation:
            raise ValueError("evaluation training implementation differs between cells")
        common = implementation
        binding = (metadata["git_commit"], json.dumps(metadata["source_sha256"], sort_keys=True), metadata["source_git_dirty"])
        if binding not in checked:
            verify_training_source(metadata, method=method)
            checked.add(binding)
        destination = output/folder.name
        if not recorded_only:
            record_geometry(folder, destination/"geometry", frozen["params"])
        results, metrics = {}, {}
        for family, params in frozen["params"].items():
            geometry = destination/"geometry"/f"{family}.json"
            result = replay_geometry(folder, geometry, params)
            path = destination/f"{family}.json"
            if path.exists() and read(path) != result:
                raise ValueError("reconstructed extension output differs; preserve the original evidence")
            write_json(path, result)
            results[family] = dict(path=path.as_posix(), sha256=sha256(path), geometry_sha256=sha256(geometry))
            metrics[family] = result["metrics"]
        completed.append(dict(**cell, outputs=results, metrics=metrics))
        print(f"Replayed common evaluation matrices: {folder.name}", flush=True)
    result = dict(schema=1, expected_cells=len(planned), completed_cells=len(completed),
                  complete=len(completed) == len(planned), training_device=plan["device"],
                  frozen_sha256=sha256(FROZEN_PATH), plan_sha256=sha256(plan_path),
                  grid_manifest_sha256=sha256(manifest_path), cells=completed,
                  validation_source_sha256=scientific_bindings(read(PLAN_PATH)),
                  undefined_state_rounds={family: sum(cell["metrics"][family]["undefined_state_rounds"] for cell in completed)
                                          for family in frozen["params"]})
    write_json(output/"manifest.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "cells"}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorded-only", action="store_true", help="reconstruct from committed geometry without raw matrices")
    parser.add_argument("--available", action="store_true", help="allow explicitly partial coverage while training continues")
    args = parser.parse_args()
    rescore(args.recorded_only, args.available)


if __name__ == "__main__":
    main()
