"""Select token-adapted reputation parameters from the complete pilot corpus."""

import argparse
import hashlib
import json
from pathlib import Path

from evaluation import published_bytes, read, verify_frozen_sources
from extension_replay import SOURCE_PATHS, record_geometry, replay_geometry
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from grid_reproduction import verify_training_source
from run_pilots import cells
from select_method import objective


PLAN_PATH = Path("configs/extensions_plan.json")
FROZEN_PATH = Path("configs/extensions_frozen.json")
OUTPUT = Path("results/extension_selection")


def candidates(plan, num_clients):
    common = dict(num_clients=num_clients, **plan["parameters"])
    if len(set(plan["cgsv"]["beta_candidates"])) != len(plan["cgsv"]["beta_candidates"]):
        raise ValueError("CGSV candidate parameters must be unique")
    for index, beta in enumerate(plan["cgsv"]["beta_candidates"]):
        yield dict(name=f"CGSV-{index:02d}", family="CGSV",
                   params=dict(**common, beta=beta, floor=plan["cgsv"]["floor"]))
    yield dict(name="RFFL-00", family="RFFL", params=dict(**common, **plan["rffl"]))


def choose(measurements, tolerance):
    """Undefined reputation histories are failures, not favorable empty costs."""
    selected = {}
    for family in ["CGSV", "RFFL"]:
        choices = [row for row in measurements if row["family"] == family]
        if any(row["undefined_state_rounds"] for row in choices):
            raise ValueError("undefined reputation histories prevent a complete candidate comparison")
        if family == "CGSV":
            values = [row["objective"] for row in choices]
            if len(values) < 2 or max(values)-min(values) <= tolerance:
                raise ValueError("CGSV pilot measurements do not discriminate the declared candidates")
            tied = [row for row in choices if row["objective"] <= min(values)+tolerance]
            winner = min(tied, key=lambda row: row["params"]["beta"])
        else:
            if len(choices) != 1:
                raise ValueError("RFFL uses one predeclared literature setting")
            winner = choices[0]
        selected[family] = {key: value for key, value in winner.items() if key != "cells"}
    return selected


def scientific_bindings(plan):
    names = sorted(set(SOURCE_PATHS + [str(PLAN_PATH.as_posix()), plan["pilot_plan"],
                                      plan["evaluation_plan"], "scripts/select_extensions.py",
                                      "scripts/run_pilots.py", "scripts/grid_reproduction.py", "scripts/rescore_extensions.py",
                                      "scripts/evaluation.py", "src/fltoken/config.py"]))
    return {name: sha256(name) for name in names}


def published_extensions(full_evidence=False):
    """Require the current implementation before scoring or reconstructing states."""
    return _validated_extensions(full_evidence, historical=False)


def recorded_extensions(full_evidence=False):
    """Validate recorded selection through the same publication boundary."""
    return _validated_extensions(full_evidence, historical=True)


def _validated_extensions(full_evidence, historical):
    frozen = json.loads(published_bytes(FROZEN_PATH))
    plan = json.loads(published_bytes(PLAN_PATH))
    current = scientific_bindings(plan)
    if frozen['source_sha256'] == current:
        for path in frozen['source_sha256']:
            published_bytes(path)
    elif historical:
        if set(frozen['source_sha256']) != set(current):
            raise ValueError('extension source binding is incomplete')
        verify_frozen_sources(FROZEN_PATH, frozen['source_sha256'])
        for path in [PLAN_PATH.as_posix(), plan['pilot_plan'], plan['evaluation_plan']]:
            published_bytes(path)
            if sha256(path) != frozen['source_sha256'][path]:
                raise ValueError('extension declaration differs from the recorded calibration')
    else:
        raise ValueError("extension sources differ from the published selection")
    summary_path = OUTPUT/"summary.json"
    summary = json.loads(published_bytes(summary_path))
    if sha256(summary_path) != frozen["selection_sha256"] or summary["corpus_sha256"] != frozen["corpus_sha256"]:
        raise ValueError("extension selection differs from the published freeze")
    if frozen["params"] != {name: row["params"] for name, row in summary["selected"].items()}:
        raise ValueError("extension parameters differ from the selected candidates")
    if full_evidence:
        manifest_path = Path("results/pilots/manifest.json")
        manifest = json.loads(published_bytes(manifest_path))
        if (sha256(manifest_path) != summary["manifest_sha256"] or
                manifest["completed_cells"] != manifest["expected_cells"] or
                manifest["completed_cells"] != summary["cells"]):
            raise ValueError("published extension pilot corpus is incomplete or different")
        for row in summary["candidates"]:
            path = OUTPUT/f"{row['name']}.json"
            published_bytes(path)
            if sha256(path) != row["sha256"]:
                raise ValueError("candidate measurements differ from the selection")
        for cell in summary["evidence"]:
            folder = Path(cell["path"])
            for name, digest in cell["inputs"].items():
                published_bytes(folder/name)
                if sha256(folder/name) != digest:
                    raise ValueError("pilot inputs differ from the extension selection")
            for name, digest in cell["geometry"].items():
                path = OUTPUT/"geometry"/folder.name/f"{name}.json"
                published_bytes(path)
                if sha256(path) != digest:
                    raise ValueError("pilot geometry differs from the extension selection")
    return frozen


def select(recorded_only=False, freeze=False):
    plan = json.loads(published_bytes(PLAN_PATH, reference="HEAD"))
    bindings = scientific_bindings(plan)
    for path in bindings:
        published_bytes(path, reference="HEAD")
    if FROZEN_PATH.exists():
        frozen = read(FROZEN_PATH)
        if frozen["source_sha256"] != bindings:
            raise ValueError("preserve the scientific sources of the existing extension freeze")
        if frozen["selection_sha256"] != sha256(OUTPUT/"summary.json"):
            raise ValueError("preserve the original extension selection summary")
        original = read(OUTPUT/"summary.json")
        if original["manifest_sha256"] != sha256("results/pilots/manifest.json"):
            raise ValueError("preserve the complete original pilot manifest")
        for row in original["candidates"]:
            if sha256(OUTPUT/f"{row['name']}.json") != row["sha256"]:
                raise ValueError("preserve the original candidate measurements")
        for cell in original["evidence"]:
            for name, digest in cell["inputs"].items():
                if sha256(Path(cell["path"])/name) != digest:
                    raise ValueError("preserve the original selection inputs")
            for name, digest in cell["geometry"].items():
                if sha256(OUTPUT/"geometry"/Path(cell["path"]).name/f"{name}.json") != digest:
                    raise ValueError("preserve the original selection geometry")
    pilot = read(plan["pilot_plan"])
    evaluation = read(plan["evaluation_plan"])
    if plan["pilot_seeds"] != pilot["seeds"] or set(pilot["seeds"]) & set(evaluation["seeds"]):
        raise ValueError("extension selection requires the complete disjoint pilot seed set")
    planned = list(cells(pilot))
    manifest_path = Path("results/pilots/manifest.json")
    manifest = read(manifest_path)
    if (manifest["plan_sha256"] != sha256(plan["pilot_plan"]) or
            manifest["completed_cells"] != len(planned) or manifest["expected_cells"] != len(planned) or
            [{key: row[key] for key in planned[0]} for row in manifest["cells"]] != planned):
        raise ValueError("extension selection requires every predeclared pilot cell")
    base = load_config(pilot["config"])
    options = list(candidates(plan, base["num_clients"]))
    values = {option["name"]: [] for option in options}
    representatives = {family: next(row["params"] for row in options if row["family"] == family)
                       for family in ["CGSV", "RFFL"]}
    checked, evidence, common_versions = set(), [], None
    for cell, recorded in zip(planned, manifest["cells"]):
        folder = Path(cell["path"])
        metadata = read(folder/"metadata.json")
        expected = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction"]},
                    **{key: pilot[key] for key in ["device", "workers", "gpu_fraction"]}}
        if (metadata["config"] != expected or metadata["source_config"] != pilot["config"] or
                metadata["git_commit"] != recorded["source_commit"] or
                metadata["source_sha256"] != manifest["training_source_sha256"] or
                sha256(folder/"per_round.csv") != recorded["per_round_sha256"]):
            raise ValueError("pilot training inputs differ from the complete selection protocol")
        if common_versions is not None and common_versions != metadata["versions"]:
            raise ValueError("pilot cells use different training library versions")
        common_versions = metadata["versions"]
        binding = (metadata["git_commit"], json.dumps(metadata["source_sha256"], sort_keys=True), metadata["source_git_dirty"])
        if binding not in checked:
            verify_training_source(metadata)
            checked.add(binding)
        destination = OUTPUT/"geometry"/folder.name
        if not recorded_only:
            record_geometry(folder, destination, representatives)
        paths = {family: destination/f"{family}.json" for family in representatives}
        inputs = None
        for option in options:
            result = replay_geometry(folder, paths[option["family"]], option["params"])
            inputs = result["inputs"]
            values[option["name"]].append(dict(**cell, **result["metrics"]))
        evidence.append(dict(**cell, inputs=inputs, geometry={name: sha256(path) for name, path in paths.items()}))
        print(f"Replayed extension candidates: {folder.name}", flush=True)
    measured = []
    for option in options:
        observations = values[option["name"]]
        measured.append(dict(**option, **objective(observations, pilot), cells=observations,
                             undefined_state_rounds=sum(row["undefined_state_rounds"] for row in observations)))
    # Keep failed comparisons as evidence, but do not emit a selection or freeze.
    for row in measured:
        write_json(OUTPUT/f"{row['name']}.json", row)
    selected = choose(measured, pilot["selection"]["tie_tolerance"])
    corpus_digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    summary = dict(schema=1, cells=len(planned), training_device=pilot["device"], source_sha256=bindings,
                   corpus_sha256=corpus_digest, manifest_sha256=sha256(manifest_path), evidence=evidence,
                   candidates=[dict(**{key: value for key, value in row.items() if key != "cells"},
                                    sha256=sha256(OUTPUT/f"{row['name']}.json")) for row in measured], selected=selected)
    write_json(OUTPUT/"summary.json", summary)
    if freeze:
        frozen = dict(schema=1, params={name: row["params"] for name, row in selected.items()},
                      source_sha256=bindings, corpus_sha256=corpus_digest,
                      selection_sha256=sha256(OUTPUT/"summary.json"))
        if FROZEN_PATH.exists() and read(FROZEN_PATH) != frozen:
            raise ValueError("a different extension freeze already exists; preserve it")
        write_json(FROZEN_PATH, frozen)
    print(json.dumps(selected, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorded-only", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    select(args.recorded_only, args.freeze)


if __name__ == "__main__":
    main()
