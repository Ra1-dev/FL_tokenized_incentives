"""Pilot-only clipping calibration and publication checks for noisy updates."""

import math
from pathlib import Path

from evaluation import published_bytes, published_method, read
from extension_replay import read_training
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from fltoken.privacy import validate_privacy
from grid_reproduction import verify_training_source
from reproduction_analysis import exact_quantile
from run_pilots import cells


PLAN = Path("configs/privacy_plan.json")
FROZEN = Path("configs/privacy_frozen.json")
CALIBRATION = Path("results/privacy_calibration.json")
SOURCES = ["src/fltoken/privacy.py", "src/fltoken/config.py", "src/fltoken/experiment.py",
           "src/fltoken/data.py", "scripts/privacy_protocol.py", "scripts/run_experiment.py",
           "scripts/extension_replay.py", "scripts/reproduction_analysis.py", "scripts/privacy_failures.py",
           "scripts/run_privacy.py", "scripts/summarize_privacy.py"]


def same_or_write(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError(f"existing privacy evidence differs; preserve its provenance: {path}")
    write_json(path, value)


def calibration(freeze=False):
    """Select one scale from all declared attack-free pilot participations."""
    for path in [PLAN, *map(Path, SOURCES)]:
        published_bytes(path, reference="HEAD")
    published_method()
    plan = read(PLAN)
    pilot_plan = read(plan["pilot_plan"])
    if (plan["calibration"]["seeds"] != pilot_plan["seeds"] or
            set(plan["seeds"]) & set(pilot_plan["seeds"]) or
            plan["calibration"]["attack"] != "none" or plan["calibration"]["quantile"] != .5):
        raise ValueError("clipping calibration requires the declared disjoint attack-free pilot seeds and median")
    planned = list(cells(pilot_plan))
    manifest = read("results/pilots/manifest.json")
    if (manifest["plan_sha256"] != sha256(plan["pilot_plan"]) or
            manifest["expected_cells"] != len(planned) or manifest["completed_cells"] != len(planned) or
            [{key: row[key] for key in planned[0]} for row in manifest["cells"]] != planned):
        raise ValueError("complete predeclared pilot coverage is required for clipping calibration")
    base, evidence, norms, common, checked = load_config(pilot_plan["config"]), [], [], None, set()
    for cell in manifest["cells"]:
        if cell["attack"] != "none":
            continue
        folder = Path(cell["path"])
        metadata, attacks, rounds, inputs = read_training(folder)
        expected = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction"]},
                    **{key: pilot_plan[key] for key in ["device", "workers", "gpu_fraction"]}}
        if (attacks or metadata.get("source_git_dirty") is not False or
                metadata["config"] != expected or metadata["source_config"] != pilot_plan["config"] or
                metadata["source_sha256"] != manifest["training_source_sha256"] or
                metadata["git_commit"] != cell["source_commit"] or
                inputs["per_round.csv"] != cell["per_round_sha256"]):
            raise ValueError("clipping observations differ from the declared common pilot training")
        implementation = (metadata["source_sha256"], metadata["versions"])
        if common is not None and common != implementation:
            raise ValueError("clipping observations must share one training implementation")
        common = implementation
        if metadata["git_commit"] not in checked:
            verify_training_source(metadata)
            checked.add(metadata["git_commit"])
        observations = [float(row["delta_norm"]) for _, _, group in rounds for row in group]
        if any(not math.isfinite(value) or value < 0 for value in observations):
            raise ValueError("clipping calibration requires finite nonnegative update norms")
        norms.extend(observations)
        evidence.append(dict(path=folder.as_posix(), seed=cell["seed"], inputs=inputs,
                             participations=len(observations), median=exact_quantile(observations, .5),
                             minimum=min(observations), maximum=max(observations)))
    if [row["seed"] for row in evidence] != plan["calibration"]["seeds"]:
        raise ValueError("clipping calibration is missing a declared seed")
    bound = exact_quantile(norms, .5)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("pilot median must be positive; a zero bound cannot define this noise comparison")
    sources = {path: sha256(path) for path in SOURCES}
    result = dict(schema=1, clipping_bound=bound, statistic=plan["calibration"]["statistic"],
                  participations=len(norms), cells=evidence, training_device=pilot_plan["device"],
                  plan_sha256=sha256(PLAN), pilot_manifest_sha256=sha256("results/pilots/manifest.json"),
                  method_sha256=sha256("configs/method_frozen.json"), source_sha256=sources)
    same_or_write(CALIBRATION, result)
    if freeze:
        same_or_write(FROZEN, dict(schema=1, clipping_bound=bound, delta=plan["delta"],
                                  noise_multipliers=plan["noise_multipliers"], plan_sha256=sha256(PLAN),
                                  calibration_sha256=sha256(CALIBRATION),
                                  method_sha256=result["method_sha256"], source_sha256=sources))
    return result


def published_privacy(path=FROZEN, full_evidence=False):
    import json
    frozen = json.loads(published_bytes(path))
    for source, digest in frozen["source_sha256"].items():
        published_bytes(source)
        if sha256(source) != digest:
            raise ValueError("privacy source differs from the published calibration")
    if set(frozen["source_sha256"]) != set(SOURCES):
        raise ValueError("privacy source binding is incomplete")
    for source, field in [(PLAN, "plan_sha256"), (CALIBRATION, "calibration_sha256"),
                          (Path("configs/method_frozen.json"), "method_sha256")]:
        published_bytes(source)
        if sha256(source) != frozen[field]:
            raise ValueError("privacy declaration, calibration or frozen reward method differs")
    published_method()
    plan, calibrated = read(PLAN), read(CALIBRATION)
    if (frozen["clipping_bound"] != calibrated["clipping_bound"] or frozen["delta"] != plan["delta"] or
            frozen["noise_multipliers"] != plan["noise_multipliers"] or
            calibrated["source_sha256"] != frozen["source_sha256"] or
            calibrated["plan_sha256"] != frozen["plan_sha256"] or
            calibrated["method_sha256"] != frozen["method_sha256"]):
        raise ValueError("privacy parameters differ from the pilot calibration")
    if full_evidence:
        published_bytes("results/pilots/manifest.json")
        if sha256("results/pilots/manifest.json") != calibrated["pilot_manifest_sha256"]:
            raise ValueError("privacy pilot manifest differs")
        for cell in calibrated["cells"]:
            for name, digest in cell["inputs"].items():
                source = Path(cell["path"])/name
                published_bytes(source)
                if sha256(source) != digest:
                    raise ValueError("privacy calibration evidence differs")
    return frozen


def settings(frozen, multiplier):
    if multiplier not in frozen["noise_multipliers"]:
        raise ValueError("noise multiplier must be predeclared before evaluation")
    return validate_privacy(dict(clipping_bound=frozen["clipping_bound"], noise_multiplier=multiplier,
                                 delta=frozen["delta"]))


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    result = calibration(args.freeze)
    print(json.dumps({key: value for key, value in result.items() if key != "cells"}, indent=2))
