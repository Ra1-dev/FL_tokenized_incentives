"""Common evaluation cells and publication checks for the selected method."""

import itertools
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from fltoken.io import sha256
from fltoken.ours import OursScorer


ABLATIONS = {"Aref": dict(reference=True, ema=False, penalties=False),
             "Aema": dict(reference=True, ema=True, penalties=False),
             "Afull": dict(reference=True, ema=True, penalties=True)}
REPOSITORY = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def conditions(plan):
    for attack in plan["attacks"]:
        fractions = [0.] if attack == "none" else [1/6] if attack == "sendros_mix" else plan["fractions"]
        for fraction in fractions:
            yield attack, fraction


def evaluation_cells(plan):
    output = comparison_output(plan)
    for seed, (attack, fraction) in itertools.product(plan["seeds"], conditions(plan)):
        name = f"{plan['dataset']}-{attack}-f{round(fraction*100):02d}-s{seed}-{plan['device']}-t{plan['rounds']}"
        yield dict(seed=seed, attack=attack, fraction=fraction, rounds=plan["rounds"],
                   horizon="main", path=f"{output.as_posix()}/{name}")
    for seed in plan["seeds"] if plan.get("long_run") else []:
        name = f"{plan['dataset']}-{plan['long_run']['attack']}-s{seed}-{plan['device']}-t{plan['long_run']['rounds']}"
        yield dict(seed=seed, attack=plan["long_run"]["attack"], fraction=1/6,
                   rounds=plan["long_run"]["rounds"], horizon="long", path=f"{output.as_posix()}/{name}")


def comparison_output(plan):
    path = Path(plan.get("output_root", "results/grid"))
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2 or path.parts[0] != "results":
        raise ValueError("comparison output must be a named folder directly under results")
    if path.name not in {"grid", "fashion"}:
        raise ValueError("unsupported comparison output folder")
    return path


def repository_git():
    """Bind Git reads to this checkout, independent of cwd or inherited overrides."""
    environment = {key: value for key, value in os.environ.items() if key not in {
        "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES"}}
    command = ["git", "--git-dir", str(REPOSITORY / ".git"), "--work-tree", str(REPOSITORY)]
    return command, dict(cwd=REPOSITORY, env=environment)


def published_bytes(path, reference="origin/main"):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("publication paths must stay relative to the repository")
    local = REPOSITORY / path
    if not local.resolve().is_relative_to(REPOSITORY.resolve()):
        raise ValueError("publication paths must stay relative to the repository")
    command, options = repository_git()
    try:
        published = subprocess.check_output(
            [*command, "show", f"{reference}:{path.as_posix()}"], **options, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot verify published artifact {path}: Git exited {error.returncode}; {detail}") from error
    if not local.is_file() or local.read_bytes() != published:
        raise ValueError(f"local artifact differs from the published version: {path}")
    return published


METHOD_BINDINGS = {
    "pilot_plan_sha256": "configs/pilot_plan.json",
    "design_prior_sha256": "configs/method_pilot.json",
    "scorer_sha256": "src/fltoken/ours.py",
    "selection_program_sha256": "scripts/select_method.py",
    "metrics_source_sha256": "src/fltoken/metrics.py",
    "selection_summary_sha256": "results/pilot_selection/summary.json",
}
METHOD_RECORDS = {"pilot_plan_sha256", "design_prior_sha256", "selection_summary_sha256"}


def published_method(path="configs/method_frozen.json", full_evidence=False):
    """Require current sources to match the published freeze before execution."""
    return _validated_method(path, full_evidence, historical=False)


def recorded_method(path="configs/method_frozen.json", full_evidence=False):
    """Validate historical selection evidence without executing its old scorer."""
    return _validated_method(path, full_evidence, historical=True)


def _validated_method(path, full_evidence, historical):
    frozen = json.loads(published_bytes(path))
    if historical:
        verify_frozen_sources(path, {source: frozen[field] for field, source in METHOD_BINDINGS.items()})
    for field, source in METHOD_BINDINGS.items():
        if historical and field not in METHOD_RECORDS:
            continue
        published_bytes(source)
        if sha256(REPOSITORY / source) != frozen[field]:
            raise ValueError(f"frozen source digest differs: {source}")
    selection = read(REPOSITORY / METHOD_BINDINGS["selection_summary_sha256"])
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


def verify_frozen_sources(path, bindings, reference="origin/main"):
    """Check recorded producer bytes at the published freeze, for analysis only.

    Execution still uses the current-source gates. Rendering recorded evidence
    must not silently relabel it as output from a later producer implementation.
    """
    frozen = published_bytes(path, reference)
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("frozen source bindings must be nonempty")
    for name, digest in bindings.items():
        source = Path(name)
        if source.is_absolute() or ".." in source.parts or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid frozen source binding")
    command, options = repository_git()
    try:
        revision = subprocess.check_output(
            [*command, "log", "-1", "--format=%H", reference, "--", Path(path).as_posix()],
            **options, stderr=subprocess.PIPE).decode().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("published freeze history is unavailable")
        original = subprocess.check_output([*command, "show", f"{revision}:{Path(path).as_posix()}"],
                                           **options, stderr=subprocess.PIPE)
        if original != frozen:
            raise ValueError("published freeze differs from its historical version")
        for name, digest in bindings.items():
            original = subprocess.check_output([*command, "show", f"{revision}:{Path(name).as_posix()}"],
                                               **options, stderr=subprocess.PIPE)
            if hashlib.sha256(original).hexdigest() != digest:
                raise ValueError(f"frozen historical source digest differs: {name}")
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"cannot verify frozen source history: Git exited {error.returncode}; {detail}") from error
    return revision


def scenario_key(cell):
    # These labels do not change the identities or scientific configuration.
    fraction = None if cell["attack"] in {"none", "sendros_mix"} else cell["fraction"]
    return cell["attack"], fraction
