"""Offline rules must consume the same matrices and reconstruct integer payouts."""

import csv
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from fltoken.data import sample_clients
from fltoken.io import sha256, write_json


def synthetic_run(module, folder, config):
    cache = Path("results/cache")/folder.name
    folder.mkdir(parents=True)
    cache.mkdir(parents=True)
    attacks = module.expected_attacks(config)
    rng = np.random.default_rng(827)
    rows = []
    for number in range(1, config["rounds"]+1):
        ids = sample_clients(config["num_clients"], config["clients_per_round"], number, config["seed"])
        matrix = rng.normal(.3, .2, (6, 23))
        for index, client in enumerate(ids):
            if client in attacks:
                matrix[index] = rng.normal(size=23)
        digest = hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest()
        np.savez(cache/f"round-{number:03d}.npz", client_ids=ids, deltas=matrix)
        rows.extend(dict(round=number, client_id=int(client), attack=attacks.get(int(client), "none"),
                         accuracy=.8, update_set_sha256=digest) for client in ids)
    with (folder/"per_round.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (folder/"round_metrics.csv").write_text("round,accuracy\n", encoding="utf-8")
    write_json(folder/"ledger.json", {})
    write_json(folder/"metadata.json", dict(status="complete", synthetic_fixture=True, config=config))
    write_json(folder/"protocol.json", dict(adversaries=attacks))
    write_json(folder/"hashes.json", {name: sha256(folder/name) for name in
                                    ["per_round.csv", "round_metrics.csv", "ledger.json"]})


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    module = importlib.import_module("extension_replay")
    binding = module.source_binding()
    monkeypatch.setattr(module, "source_binding", lambda: binding)
    monkeypatch.chdir(tmp_path)
    folder = Path("results/pilots/synthetic")
    config = dict(num_clients=9, clients_per_round=6, rounds=5, seed=821,
                  attack="noise", fraction=1/3, device="cpu")
    synthetic_run(module, folder, config)
    parameters = dict(CGSV=dict(num_clients=9, alpha=.95, gamma=.5, beta=1., floor=.001),
                      RFFL=dict(num_clients=9, alpha=.95, gamma=.5, threshold_divisor=3))
    return module, folder, parameters


def test_geometry_replay_covers_every_participation_and_integer_payout(corpus):
    module, folder, parameters = corpus
    paths = module.record_geometry(folder, "out/geometry", parameters)
    first = {name: path.read_bytes() for name, path in paths.items()}
    module.record_geometry(folder, "out/geometry", parameters)
    assert first == {name: path.read_bytes() for name, path in paths.items()}
    for name, path in paths.items():
        result = module.replay_geometry(folder, path, parameters[name])
        assert len(result["rows"]) == 30
        assert len(result["ledger"]) == len(result["states"]) == 5
        assert result["bias"]["all_noise"]["participations"] == sum(row["attack"] == "noise" for row in result["rows"])
        direct = module.FACTORIES[name](parameters[name])
        for (number, ids, matrix, _), ledger, state in zip(module.load_update_rounds(folder), result["ledger"], result["states"]):
            output = direct.score(ids, matrix, number)
            units, amounts = module.payout(output["shares"])
            assert ledger["shares1e18"] == units
            assert ledger["payouts"] == amounts
            assert state == output["state"]
        write_json(Path("out")/f"{name}.json", result)
        assert json.loads((Path("out")/f"{name}.json").read_text()) == result


def test_beta_candidates_share_geometry_but_discriminate_token_quotas(corpus):
    module, folder, parameters = corpus
    paths = module.record_geometry(folder, "out/geometry", parameters)
    low = module.replay_geometry(folder, paths["CGSV"], parameters["CGSV"])
    high = module.replay_geometry(folder, paths["CGSV"], {**parameters["CGSV"], "beta": 2.})
    assert low["states"] == high["states"]
    assert low["ledger"] != high["ledger"]
    assert low["metrics"]["attacker_share"] != high["metrics"]["attacker_share"]
    assert low["bias"] == high["bias"]
    with pytest.raises(ValueError, match="parameters"):
        module.replay_geometry(folder, paths["CGSV"], {**parameters["CGSV"], "alpha": .8})


def test_corrupted_matrix_fails_before_any_offline_output_is_written(corpus):
    module, folder, parameters = corpus
    cache = Path("results/cache/synthetic/round-001.npz")
    with np.load(cache) as saved:
        ids, matrix = saved["client_ids"], saved["deltas"]
    matrix[0, 0] += 1
    np.savez(cache, client_ids=ids, deltas=matrix)
    with pytest.raises(ValueError, match="common matrix"):
        module.record_geometry(folder, "out/geometry", parameters)
    assert not Path("out/geometry/CGSV.json").exists()


def test_recorded_reconstruction_rejects_missing_rounds_order_and_source_drift(corpus, monkeypatch):
    module, folder, parameters = corpus
    paths = module.record_geometry(folder, "out/geometry", parameters)
    path = paths["RFFL"]
    original = json.loads(path.read_text())
    changed = {**original, "rounds": original["rounds"][:-1]}
    write_json(path, changed)
    with pytest.raises(ValueError, match="every training round"):
        module.replay_geometry(folder, path, parameters["RFFL"])
    changed = json.loads(json.dumps(original))
    changed["rounds"][0]["client_ids"].reverse()
    write_json(path, changed)
    with pytest.raises(ValueError, match="client order"):
        module.replay_geometry(folder, path, parameters["RFFL"])
    write_json(path, original)
    monkeypatch.setattr(module, "source_binding", lambda: {})
    with pytest.raises(ValueError, match="scientific sources"):
        module.replay_geometry(folder, path, parameters["RFFL"])


def test_complete_pilot_selection_freeze_and_replay_are_identical(corpus, monkeypatch):
    geometry, _, _ = corpus
    module = importlib.import_module("select_extensions")
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/module.PLAN_PATH).read_text())
    pilot = json.loads((root/plan["pilot_plan"]).read_text())
    plan["pilot_seeds"] = [821, 823]
    pilot.update(seeds=plan["pilot_seeds"], attacks=["none", "noise"], fractions=[1/3], rounds=5, device="cpu")
    sources = set(geometry.SOURCE_PATHS+["scripts/select_extensions.py", "scripts/run_pilots.py",
                  "scripts/grid_reproduction.py", "scripts/rescore_extensions.py", "scripts/evaluation.py",
                  "src/fltoken/config.py", plan["evaluation_plan"], "scripts/summarize_extensions.py",
                  "scripts/summarize_grid.py", "scripts/rebuild_ledgers.py", "src/fltoken/statistics.py",
                  "results/literature/extensions.json"])
    for path in sources:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/path, path)
    write_json(module.PLAN_PATH, plan)
    write_json(plan["pilot_plan"], pilot)
    evaluation = json.loads(Path(plan["evaluation_plan"]).read_text())
    evaluation.update(seeds=[901, 903], attacks=["none", "noise", "sendros_mix"], fractions=[1/3],
                      rounds=5, device="cpu", config=pilot["config"])
    evaluation["long_run"]["rounds"] = 6
    write_json(plan["evaluation_plan"], evaluation)
    Path(pilot["config"]).write_text("num_clients: 9\nclients_per_round: 6\nrounds: 5\n", encoding="utf-8")
    base = module.load_config(pilot["config"])
    cells = list(module.cells(pilot))
    records = []
    for cell in cells:
        folder = Path(cell["path"])
        config = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction"]},
                  **{key: pilot[key] for key in ["device", "workers", "gpu_fraction"]}}
        synthetic_run(geometry, folder, config)
        metadata = geometry.read(folder/"metadata.json")
        metadata.update(source_config=pilot["config"], source_sha256={"synthetic": "bound"},
                        git_commit="a"*40, source_git_dirty=False, versions={"synthetic": True})
        write_json(folder/"metadata.json", metadata)
        records.append(dict(**cell, source_commit="a"*40, per_round_sha256=sha256(folder/"per_round.csv")))
    manifest = dict(plan_sha256=sha256(plan["pilot_plan"]), expected_cells=4, completed_cells=4,
                    cells=records, training_source_sha256={"synthetic": "bound"})
    write_json("results/pilots/manifest.json", manifest)
    publication_calls = []
    def committed(path, reference="origin/main"):
        publication_calls.append((str(path), reference))
        return Path(path).read_bytes()
    checked = []
    monkeypatch.setattr(module, "published_bytes", committed)
    monkeypatch.setattr(module, "verify_training_source", lambda metadata: checked.append(metadata["git_commit"]))
    first = module.select(freeze=True)
    frozen = module.FROZEN_PATH.read_bytes()
    assert checked == ["a"*40]
    assert first["cells"] == 4 and len(first["candidates"]) == 5
    assert all(reference == "HEAD" for _, reference in publication_calls)
    def unexpected_write(*args, **kwargs):
        raise AssertionError("recorded reconstruction must not write")
    with monkeypatch.context() as patch:
        patch.setattr(module, "write_json", unexpected_write)
        assert module.select(recorded_only=True) == first
        replay = module.replay_geometry
        def changed_measurement(*args, **kwargs):
            result = replay(*args, **kwargs)
            result["metrics"]["honest_gini"] += .01
            return result
        patch.setattr(module, "replay_geometry", changed_measurement)
        with pytest.raises(ValueError, match="reconstructed extension candidate differs"):
            module.select(recorded_only=True)
    assert module.FROZEN_PATH.read_bytes() == frozen
    assert module.published_extensions(full_evidence=True)["params"].keys() == {"CGSV", "RFFL"}
    assert any(reference == "origin/main" for _, reference in publication_calls)
    write_json("results/summary.json", dict(synthetic_fixture=True))
    Path("docs").mkdir(exist_ok=True)
    shutil.copyfile(root/'README.md', 'README.md')
    shutil.copyfile(root/'docs/EXPERIMENTS.md', 'docs/EXPERIMENTS.md')
    documentation = importlib.import_module("document_extensions")
    documentation.main()
    documented = Path("docs/EXTENSIONS.md").read_text()
    assert "Measured pilot selection" in documented and "empirical selection remains pending" not in documented
    assert geometry.read("results/summary.json")["extension_design"]["selection"]["cells"] == 4
    assert 'Extension calibration covers 4 pilot cells' in Path('docs/EXPERIMENTS.md').read_text()
    assert 'Empirical extension results remain pending.' in Path('docs/EXPERIMENTS.md').read_text()

    rescorer = importlib.import_module("rescore_extensions")
    monkeypatch.setattr(rescorer, "published_bytes", committed)
    monkeypatch.setattr(rescorer, "recorded_method", lambda: dict(params={}))
    method_checks = []
    monkeypatch.setattr(rescorer, "verify_training_source", lambda metadata, method: method_checks.append(method))
    write_json("configs/method_frozen.json", dict(params={}))
    evaluation_records = []
    for cell in rescorer.evaluation_cells(evaluation):
        folder = Path(cell["path"])
        config = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction", "rounds"]},
                  **{key: evaluation[key] for key in ["device", "workers", "gpu_fraction"]},
                  "rules": evaluation["rules"]+list(rescorer.ABLATIONS), "ours_params": {}}
        synthetic_run(geometry, folder, config)
        metadata = geometry.read(folder/"metadata.json")
        metadata.update(source_config=evaluation["config"], source_sha256={"synthetic": "bound"},
                        git_commit="b"*40, source_git_dirty=False, versions={"synthetic": True},
                        method_sha256=sha256("configs/method_frozen.json"))
        write_json(folder/"metadata.json", metadata)
        evaluation_records.append(dict(**cell, source_commit="b"*40,
                                       metadata_sha256=sha256(folder/"metadata.json"),
                                       per_round_sha256=sha256(folder/"per_round.csv")))
    grid = dict(plan_sha256=sha256(plan["evaluation_plan"]), method_sha256=sha256("configs/method_frozen.json"),
                expected_cells=8, completed_cells=1, cells=evaluation_records[:1])
    write_json("results/grid/manifest.json", grid)
    with pytest.raises(ValueError, match="complete evaluation coverage"):
        rescorer.rescore()
    partial = rescorer.rescore(available=True)
    assert not partial["complete"] and partial["completed_cells"] == 1
    write_json("results/grid/manifest.json", {**grid, "completed_cells": 8, "cells": evaluation_records})
    complete = rescorer.rescore()
    assert complete["complete"] and complete["completed_cells"] == 8
    assert complete["undefined_state_rounds"] == {"CGSV": 0, "RFFL": 0}
    assert rescorer.rescore(recorded_only=True) == complete
    assert method_checks and all(method == dict(params={}) for method in method_checks)
    analysis = importlib.import_module("summarize_extensions")
    analysis_records = []
    for cell in complete["cells"]:
        results = {}
        for family in ["CGSV", "RFFL"]:
            output = cell["outputs"][family]
            assert sha256(output["path"]) == output["sha256"]
            result = geometry.read(output["path"])
            assert len(result["rows"]) == cell["rounds"]*6
            assert result["training_device"] == evaluation["device"]
            results[family] = result
        baseline = {name: {**results["CGSV"]["metrics"], "attacker_share": .1 if name == "B1" else .05}
                    for name in ["B1", "Mstar"]}
        write_json(Path(cell["path"])/"metrics.json", baseline)
        analysis_records.append(dict(**cell, extensions=results, baseline_metrics=baseline))
    summarized = analysis.summarize_records(analysis_records, evaluation, [901])
    assert summarized["runs"] == 8
    assert summarized["source_seeds"] == [901]
    assert len(summarized["reward_intervals"]) == 4
    assert summarized["reward_intervals"][0]["metrics"]["B1"]["attacker_share"]["estimate"]["mean"] == .1
    assert summarized["reward_intervals"][0]["metrics"]["Mstar"]["attacker_share"]["estimate"]["mean"] == .05
    assert len(summarized["contrasts"]) == 8
    assert summarized["pooled"]["attacker_share"]["scenarios"] == 2
    assert summarized["pooled"]["honest_false_penalty_rate"]["scenarios"] == 3
    for contrast in summarized["contrasts"]:
        assert contrast["condition_seed_test"]["pairs"] == 4
        assert contrast["seed_test"]["pairs"] == 2
        assert contrast["condition_seed_test_holm_p"] >= contrast["condition_seed_test"]["p_value"]
    assert len(summarized["noise_observations"]) == sum(row["attack"] == "noise" for cell in analysis_records
                                                     for result in cell["extensions"].values() for row in result["rows"])
    write_json("out/synthetic-extension-summary.json", summarized)
    tables = []
    assets = importlib.import_module("extension_assets")
    assets.extension_tables(summarized, lambda name, columns, header, rows, context: tables.append((name, columns, header, rows, context)))
    assert len(tables) == 13
    assert {'extensions_bias_main_eligible', 'extensions_bias_long_eligible'} <= {row[0] for row in tables}
    assert all(len(header) == len(columns) and all(len(row) == len(columns) for row in rows)
               for _, columns, header, rows, _ in tables)
    assert all("CPU" in context for *_, context in tables)
    monkeypatch.setattr(analysis, "published_bytes", committed)
    monkeypatch.setattr(analysis, "recorded_method", lambda: dict(params={}))
    monkeypatch.setattr(analysis, "verify_training_source", lambda metadata, method: None)
    monkeypatch.setattr(analysis, "rebuild", lambda folder: None)
    literature = json.loads((root/"results/literature/sendros.json").read_text())
    literature["setup"]["seeds"] = [901]
    write_json("results/literature/sendros.json", literature)
    write_json("results/summary.json", dict(synthetic_fixture=True))
    analysis.main()
    actual = geometry.read("results/summary.json")["extensions"]
    assert {key: actual[key] for key in summarized if key != 'noise_observations'} == {
        key:value for key, value in summarized.items() if key != 'noise_observations'}
    assert actual['noise_observations'] == [row for row in summarized['noise_observations']
        if row['horizon'] == 'main' and row['attack'] == 'sendros_mix' and row['seed'] in summarized['source_seeds']]
    assert actual['noise_observations_total'] == len(summarized['noise_observations'])
    assert actual['full_summary']['sha256'] == sha256('results/extensions/summary.json')
    macros = importlib.import_module("make_macros")
    macro_input = json.loads((root/"results/summary.json").read_text())
    macro_input["extensions"] = actual
    write_json("results/summary.json", macro_input)
    macros.main()
    numbers = Path("paper/generated/numbers.tex").read_text()
    assert r"\newcommand{\ExtensionRuns}{8}" in numbers
    assert r"\newcommand{\ExtensionCGSVInclusiveMean}" in numbers
    assert r"\newcommand{\ExtensionRFFLNoiseParticipations}" in numbers
    assert r"\newcommand{\ExtensionCGSVVsCosineAttackerShareCellHolmP}" in numbers
    previous = Path("results/extensions/summary.json").read_bytes()
    analysis.main()
    assert Path("results/extensions/summary.json").read_bytes() == previous
    altered_path = Path(complete["cells"][0]["outputs"]["CGSV"]["path"])
    original_bytes = altered_path.read_bytes()
    altered = geometry.read(altered_path)
    altered["ledger"][0]["payouts"][0] += 1
    write_json(altered_path, altered)
    with pytest.raises(ValueError, match="output differs"):
        analysis.main()
    altered_path.write_bytes(original_bytes)
    with pytest.raises(ValueError, match="every unique"):
        analysis.summarize_records(analysis_records[:-1], evaluation, [901])
    damaged = json.loads(json.dumps(analysis_records))
    damaged[0]["extensions"]["RFFL"]["metrics"]["undefined_state_rounds"] = 1
    undefined = analysis.summarize_records(damaged, evaluation, [901])
    assert undefined["reward_intervals"][0]["metrics"]["RFFL"]["attacker_share"]["estimate"] is None
    assert undefined["undefined_state_rounds"]["RFFL"] == 1
    assert len(undefined["contrasts"]) == 8

    write_json("results/pilots/manifest.json", {**manifest, "completed_cells": 3})
    with pytest.raises(ValueError, match="original pilot manifest"):
        module.select(recorded_only=True)


def test_selection_keeps_undefined_and_indistinguishable_candidates_visible(corpus):
    module = importlib.import_module("select_extensions")
    choices = [dict(family="CGSV", params=dict(beta=beta), objective=cost, undefined_state_rounds=0)
               for beta, cost in [(1., .2), (2., .3)]]
    choices.append(dict(family="RFFL", params={}, objective=.4, undefined_state_rounds=0))
    assert module.choose(choices, 1e-12)["CGSV"]["params"]["beta"] == 1.
    choices[1]["objective"] = .2
    with pytest.raises(ValueError, match="discriminate"):
        module.choose(choices, 1e-12)
    choices[1]["objective"] = .3
    choices[-1]["undefined_state_rounds"] = 1
    with pytest.raises(ValueError, match="undefined reputation"):
        module.choose(choices, 1e-12)


def test_bias_summary_never_replaces_undefined_geometry_with_a_zero(corpus):
    module = importlib.import_module("summarize_extensions")
    row = dict(attack="noise", eligible_before=True, bias_defined=True, cosine=.3,
               self_inclusion=.2, cross_residual=.1, excluding_cosine=.05)
    result = module.noise_statistics([row])
    assert result["mean_cosine"] == .3
    assert result["median_relative_error"] == pytest.approx(1/3)
    assert result["mean_inclusive_minus_excluding"] == .25
    undefined = module.noise_statistics([row, {**row, "bias_defined": False}])
    assert undefined["undefined_geometry"] == 1
    assert undefined["mean_self_inclusion"] is None and undefined["median_relative_error"] is None
    zero = module.noise_statistics([{**row, "cosine": 0., "cross_residual": -.2}])
    assert zero["zero_measured_cosine"] == 1 and zero["median_relative_error"] is None
    empty = module.noise_statistics([])
    assert empty["participations"] == 0 and empty["mean_cosine"] is None


def test_evaluation_cannot_begin_without_a_published_extension_freeze(corpus, monkeypatch):
    module = importlib.import_module("rescore_extensions")
    def reject(**kwargs):
        raise ValueError("required published artifact is missing")
    monkeypatch.setattr(module, "runtime_extensions", reject)
    with pytest.raises(ValueError, match="published artifact"):
        module.rescore(available=True)
    assert not Path("results/extensions").exists()


def test_missing_extension_contrasts_keep_all_eight_holm_slots(corpus):
    module = importlib.import_module("summarize_extensions")
    records = []
    for seed in range(5):
        metrics = dict(attacker_share=0., honest_false_penalty_rate=0., undefined_state_rounds=0)
        records.append(dict(seed=seed, horizon="main", attack="noise",
            extensions={family: dict(metrics={**metrics, "undefined_state_rounds": int(family == "RFFL" and seed == 0)})
                        for family in ["CGSV", "RFFL"]},
            baseline_metrics={rule: dict(attacker_share=value, honest_false_penalty_rate=value)
                              for rule, value in [("B1", .5), ("Mstar", .4)]}))
    plan = dict(seeds=list(range(5)), statistics=dict(confidence=.95, bootstrap_resamples=20000, bootstrap_seed=123))
    _, contrasts = module.comparison_statistics(records, plan)
    assert len(contrasts) == 8
    for row in contrasts:
        if row["family"] == "RFFL":
            assert row["undefined_condition_seed_pairs"] == 1
            assert row["condition_seed_test"] is None and row["condition_seed_test_holm_p"] is None
            assert row["paired_difference"]["estimate"] is None
        else:
            # Five same-sign pairs: exact two-sided p=2/32; retain all eight family slots.
            assert row["condition_seed_test"]["p_value"] == .0625
            assert row["condition_seed_test_holm_p"] == .5
