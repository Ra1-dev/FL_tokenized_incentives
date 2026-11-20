"""Refresh source-comparison evidence from published evaluation training commits."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from evaluation import ABLATIONS, evaluation_cells, published_bytes, published_method, read, repository_git
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from fltoken.statistics import sampled_bootstrap_mean
from reproduction_analysis import collect_run, summarize_runs, write_csv


def verify_training_source(metadata, reference="origin/main"):
    commit = metadata.get("git_commit", "")
    if (metadata.get("source_git_dirty") is not False or not isinstance(commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("training must identify a clean source commit")
    command, options = repository_git()
    checking_ancestry = True
    try:
        subprocess.run([*command, "merge-base", "--is-ancestor", commit, reference], check=True,
                       **options, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        checking_ancestry = False
        names = subprocess.check_output([*command, "ls-tree", "-r", "--name-only", commit, "src/fltoken"],
                                        **options, text=True, stderr=subprocess.PIPE).splitlines()
        expected = {name for name in names if name.endswith(".py") and len(PurePosixPath(name).parts) == 3}
        expected.update(["scripts/run_experiment.py", "scripts/evaluation.py", metadata["source_config"]])
        sources = metadata["source_sha256"]
        if set(sources) != expected:
            raise ValueError("recorded training source coverage differs from the source commit")
        for name, digest in sources.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("training source paths must stay inside the repository")
            data = subprocess.check_output([*command, "show", f"{commit}:{name}"], **options, stderr=subprocess.PIPE)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"training source digest differs: {name}")
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or b"").decode("utf-8", errors="replace").strip()
        reason = ("training commit must be reachable from the published branch" if checking_ancestry and error.returncode == 1
                  else "cannot verify published training history")
        raise ValueError(f"{reason}; Git exited {error.returncode}; {detail}") from error


def mixture_cells(manifest, plan):
    expected = [cell for cell in evaluation_cells(plan) if cell["horizon"] == "main" and cell["attack"] == "sendros_mix"]
    supplied = [cell for cell in manifest["cells"] if cell["horizon"] == "main" and cell["attack"] == "sendros_mix"]
    indexed = {cell["seed"]: cell for cell in supplied}
    if len(indexed) != len(supplied) or set(indexed) != {cell["seed"] for cell in expected}:
        raise ValueError("every evaluation seed needs one complete main mixture run")
    ordered = []
    for cell in expected:
        recorded = indexed[cell["seed"]]
        if any(recorded.get(key) != value for key, value in cell.items()):
            raise ValueError("mixture run differs from the declared main evaluation cell")
        ordered.append(recorded)
    return ordered


def comparison_groups(runs, source_seeds, evaluation_seeds):
    indexed = {run["share"]["seed"]: run for run in runs}
    if len(indexed) != len(runs) or set(indexed) != set(evaluation_seeds):
        raise ValueError("overall reproduction requires each evaluation seed exactly once")
    if len(set(source_seeds)) != len(source_seeds) or not set(source_seeds) <= indexed.keys():
        raise ValueError("source comparison seeds must be a unique subset of the evaluation seeds")
    return [indexed[seed] for seed in source_seeds], [indexed[seed] for seed in evaluation_seeds]


def refresh_grid_reproduction(requested_device=None):
    plan_path = Path("configs/evaluation_plan.json")
    plan = json.loads(published_bytes(plan_path))
    method = published_method()
    device = plan["device"]
    if requested_device is not None and requested_device != device:
        raise ValueError("the grid reproduction must use the grid's common training device")
    source = read("results/literature/sendros.json")
    manifest_path = Path("results/grid/manifest.json")
    manifest = read(manifest_path)
    if manifest["plan_sha256"] != sha256(plan_path) or manifest["method_sha256"] != sha256("configs/method_frozen.json"):
        raise ValueError("mixture manifest differs from the published plan or method")
    runs, evidence = [], []
    checked_commits = set()
    common_implementation = None
    for cell in mixture_cells(manifest, plan):
        folder = Path(cell["path"])
        metadata = read(folder/"metadata.json")
        if metadata["status"] != "complete" or sha256(folder/"metadata.json") != cell["metadata_sha256"]:
            raise ValueError("mixture metadata differs from the completed grid manifest")
        if sha256(folder/"per_round.csv") != cell["per_round_sha256"] or metadata["git_commit"] != cell["source_commit"]:
            raise ValueError("mixture training provenance differs from the grid manifest")
        expected = {**load_config(plan["config"]), **{key:cell[key] for key in ["seed","attack","fraction","rounds"]},
                    **{key:plan[key] for key in ["device","workers","gpu_fraction"]},
                    "rules":plan["rules"]+list(ABLATIONS), "ours_params":method["params"]}
        if metadata["config"] != expected or metadata["source_config"] != plan["config"]:
            raise ValueError("mixture scientific configuration differs from the complete evaluation protocol")
        if metadata.get("method_sha256") != sha256("configs/method_frozen.json"):
            raise ValueError("mixture run did not bind the published method")
        implementation = (metadata["source_sha256"], metadata["versions"])
        if common_implementation is not None and implementation != common_implementation:
            raise ValueError("mixture runs used different training sources or library versions")
        common_implementation = implementation
        # The complete source record, not just its commit label, is part of the cache key.
        binding = (metadata["git_commit"], json.dumps(metadata["source_sha256"], sort_keys=True), metadata["source_git_dirty"])
        if binding not in checked_commits:
            verify_training_source(metadata)
            checked_commits.add(binding)
        runs.append(collect_run(folder, cell["seed"]))
        evidence.append(dict(path=folder.as_posix(), metadata_sha256=sha256(folder/"metadata.json"),
                             participation_sha256=sha256(folder/"per_round.csv")))
    primary, overall = comparison_groups(runs, source["setup"]["seeds"], plan["seeds"])
    statistics = plan["statistics"]
    provenance = dict(manifest_sha256=sha256(manifest_path), plan_sha256=sha256(plan_path),
                      method_sha256=sha256("configs/method_frozen.json"),
                      analysis_source_sha256={name:sha256(name) for name in
                          ["scripts/reproduce_sendros.py", "scripts/grid_reproduction.py", "scripts/reproduction_analysis.py",
                           "scripts/rebuild_ledgers.py", "src/fltoken/metrics.py", "src/fltoken/statistics.py"]},
                      inputs=evidence)
    summary = read("results/summary.json")
    output = Path("results/reproduction")
    output.mkdir(exist_ok=True, parents=True)
    for key, records, prefix in [("reproduction", primary, ""), ("reproduction_overall", overall, "overall_")]:
        result = summarize_runs(records, source, device)
        result.update(basis="evaluation_grid", seeds=[run["share"]["seed"] for run in records], provenance=provenance,
                      attacker_share_bootstrap=sampled_bootstrap_mean([run["share"]["attacker_share"] for run in records],
                          statistics["confidence"], statistics["bootstrap_resamples"], statistics["bootstrap_seed"]))
        for name, field in [("noise_participations", "noise_observations"), ("quality_trajectory", "quality_trajectory"), ("per_seed", "per_seed")]:
            write_csv(output/f"{prefix}{name}.csv", result[field])
        write_json(output/f"{prefix}summary.json", result)
        summary[key] = result
        print(json.dumps(dict(group=key, seeds=result["seeds"], device=device,
                              noise_participations=result["noise_participations"],
                              median_relative_error=result["relative_error_quantiles"]["0.5"],
                              attacker_share_mean=result["attacker_share_mean"],
                              attacker_share_bootstrap=result["attacker_share_bootstrap"]), sort_keys=True))
    write_json("results/summary.json", summary)
    if not summary["reproduction"]["median_error_pass"]:
        raise SystemExit(1)
