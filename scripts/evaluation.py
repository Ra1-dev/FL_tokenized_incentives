"""Common evaluation cells and publication checks for the selected method."""

import itertools
import json
from pathlib import Path
import subprocess

from fltoken.io import sha256
from fltoken.ours import OursScorer


ABLATIONS = {"Aref": dict(reference=True, ema=False, penalties=False),
             "Aema": dict(reference=True, ema=True, penalties=False),
             "Afull": dict(reference=True, ema=True, penalties=True)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conditions(plan):
    for attack in plan["attacks"]:
        fractions = [0.] if attack == "none" else [1/6] if attack == "sendros_mix" else plan["fractions"]
        for fraction in fractions:
            yield attack, fraction


def evaluation_cells(plan):
    for seed, (attack, fraction) in itertools.product(plan["seeds"], conditions(plan)):
        name = f"{plan['dataset']}-{attack}-f{round(fraction*100):02d}-s{seed}-{plan['device']}-t{plan['rounds']}"
        yield dict(seed=seed, attack=attack, fraction=fraction, rounds=plan["rounds"],
                   horizon="main", path=f"results/grid/{name}")
    for seed in plan["seeds"]:
        name = f"{plan['dataset']}-{plan['long_run']['attack']}-s{seed}-{plan['device']}-t{plan['long_run']['rounds']}"
        yield dict(seed=seed, attack=plan["long_run"]["attack"], fraction=1/6,
                   rounds=plan["long_run"]["rounds"], horizon="long", path=f"results/grid/{name}")


def published_bytes(path, reference="origin/main"):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("publication paths must stay relative to the repository")
    try:
        published = subprocess.check_output(["git", "show", f"{reference}:{path.as_posix()}"], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"required published artifact is missing: {path}") from error
    if not path.is_file() or path.read_bytes() != published:
        raise ValueError(f"local artifact differs from the published version: {path}")
    return published


def published_method(path="configs/method_frozen.json", full_evidence=False):
    """Require the freeze, scientific sources and complete selection to be public."""
    frozen = json.loads(published_bytes(path))
    bindings = {
        "pilot_plan_sha256": "configs/pilot_plan.json",
        "design_prior_sha256": "configs/method_pilot.json",
        "scorer_sha256": "src/fltoken/ours.py",
        "selection_program_sha256": "scripts/select_method.py",
        "metrics_source_sha256": "src/fltoken/metrics.py",
        "selection_summary_sha256": "results/pilot_selection/summary.json",
    }
    for field, source in bindings.items():
        published_bytes(source)
        if sha256(source) != frozen[field]:
            raise ValueError(f"frozen source digest differs: {source}")
    selection = read(bindings["selection_summary_sha256"])
    if selection["corpus_sha256"] != frozen["corpus_sha256"] or selection["selected"]["params"] != frozen["params"]:
        raise ValueError("selected parameters or corpus differ from the published freeze")
    OursScorer(frozen["params"])
    if not all(frozen["params"][component] for component in ["reference", "ema", "penalties"]):
        raise ValueError("the frozen full method must enable every component")
    if not full_evidence:
        return frozen
    manifest = json.loads(published_bytes("results/pilots/manifest.json"))
    if manifest["completed_cells"] != manifest["expected_cells"] or manifest["completed_cells"] != selection["cells"]:
        raise ValueError("published pilot corpus is incomplete")
    for cell in manifest["cells"]:
        folder = Path(cell["path"])
        for filename in ["metadata.json", "per_round.csv", "protocol.json", "metrics.json", "ledger.json", "round_metrics.csv", "hashes.json"]:
            published_bytes(folder / filename)
        for filename in ["quality.csv", "provenance.json"]:
            published_bytes(Path("results/pilot_selection/geometry") / folder.name / filename)
    for candidate in selection["candidates"]:
        published_bytes(f"results/pilot_selection/candidate-{candidate['candidate']:02d}.json")
    return frozen


def scenario_key(cell):
    # These labels do not change the identities or scientific configuration.
    fraction = None if cell["attack"] in {"none", "sendros_mix"} else cell["fraction"]
    return cell["attack"], fraction
