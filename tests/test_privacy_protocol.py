"""Pilot scale selection must be complete, immutable and published before use."""

import csv
import importlib
from pathlib import Path
import shutil

import pytest

from fltoken.data import sample_clients
from fltoken.io import sha256, write_json


@pytest.fixture
def calibration_corpus(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    module = importlib.import_module("privacy_protocol")
    monkeypatch.chdir(tmp_path)
    for source in module.SOURCES:
        Path(source).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/source, source)
    plan = module.read(root/module.PLAN)
    pilot = module.read(root/plan["pilot_plan"])
    pilot.update(seeds=[1223, 1229], attacks=["none"], rounds=2, device="cpu")
    plan["calibration"]["seeds"] = pilot["seeds"]
    plan["seeds"] = [1231, 1237]
    write_json(module.PLAN, plan)
    write_json(plan["pilot_plan"], pilot)
    Path(pilot["config"]).write_text("num_clients: 5\nclients_per_round: 3\nrounds: 2\n", encoding="utf-8")
    write_json("configs/method_frozen.json", dict(synthetic_fixture=True, params={}))
    base = module.load_config(pilot["config"])
    records = []
    for offset, cell in enumerate(module.cells(pilot)):
        folder = Path(cell["path"])
        folder.mkdir(parents=True)
        config = {**base, **{key: cell[key] for key in ["seed", "attack", "fraction"]},
                  **{key: pilot[key] for key in ["device", "workers", "gpu_fraction"]}}
        rows = []
        for number in [1, 2]:
            ids = sample_clients(5, 3, number, cell["seed"])
            rows.extend(dict(round=number, client_id=int(client), attack="none",
                             delta_norm=offset*6+(number-1)*3+position+1.,
                             update_set_sha256=str(number)*64) for position, client in enumerate(ids))
        with (folder/"per_round.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        write_json(folder/"metadata.json", dict(status="complete", synthetic_fixture=True, config=config,
                   source_config=pilot["config"], git_commit="a"*40, source_git_dirty=False,
                   source_sha256={"synthetic": "bound"}, versions={"synthetic": True}))
        write_json(folder/"protocol.json", dict(adversaries={}))
        write_json(folder/"ledger.json", {})
        (folder/"round_metrics.csv").write_text("round,accuracy\n", encoding="utf-8")
        write_json(folder/"hashes.json", {name: sha256(folder/name) for name in
                                        ["per_round.csv", "round_metrics.csv", "ledger.json"]})
        records.append(dict(**cell, source_commit="a"*40, per_round_sha256=sha256(folder/"per_round.csv")))
    write_json("results/pilots/manifest.json", dict(plan_sha256=sha256(plan["pilot_plan"]),
               expected_cells=2, completed_cells=2, cells=records, training_source_sha256={"synthetic": "bound"}))
    published = []
    def committed(path, reference="origin/main"):
        published.append((str(path), reference))
        return Path(path).read_bytes()
    monkeypatch.setattr(module, "published_bytes", committed)
    monkeypatch.setattr(module, "published_method", lambda: dict(params={}))
    checked = []
    monkeypatch.setattr(module, "verify_training_source", lambda metadata: checked.append(metadata["git_commit"]))
    return module, published, checked


def test_clipping_uses_every_pilot_participation_and_reconstructs_identically(calibration_corpus):
    module, published, checked = calibration_corpus
    result = module.calibration(freeze=True)
    assert result["participations"] == 12 and result["clipping_bound"] == 6.5
    assert [cell["median"] for cell in result["cells"]] == [3.5, 9.5]
    assert checked == ["a"*40]
    frozen = module.FROZEN.read_bytes()
    calibration = module.CALIBRATION.read_bytes()
    assert module.calibration() == result
    assert module.FROZEN.read_bytes() == frozen and module.CALIBRATION.read_bytes() == calibration
    assert all(reference == "HEAD" for _, reference in published)
    parameters = module.published_privacy(full_evidence=True)
    assert any(reference == "origin/main" for _, reference in published)
    assert module.settings(parameters, 1.) == dict(clipping_bound=6.5, noise_multiplier=1., delta=1e-5)
    with pytest.raises(ValueError, match="predeclared"):
        module.settings(parameters, .75)
    changed = module.read(module.FROZEN)
    changed["clipping_bound"] += 1
    write_json(module.FROZEN, changed)
    with pytest.raises(ValueError, match="parameters differ"):
        module.published_privacy()


def test_calibration_rejects_incomplete_tampered_and_overlapping_evidence(calibration_corpus):
    module, _, _ = calibration_corpus
    path = Path("results/pilots/manifest.json")
    manifest = module.read(path)
    write_json(path, {**manifest, "completed_cells": 1})
    with pytest.raises(ValueError, match="complete predeclared"):
        module.calibration()
    write_json(path, manifest)
    folder = Path(manifest["cells"][0]["path"])
    original = (folder/"per_round.csv").read_bytes()
    (folder/"per_round.csv").write_bytes(original+b"\n")
    with pytest.raises(ValueError, match="recorded hash"):
        module.calibration()
    (folder/"per_round.csv").write_bytes(original)
    plan = module.read(module.PLAN)
    plan["seeds"].append(plan["calibration"]["seeds"][0])
    write_json(module.PLAN, plan)
    with pytest.raises(ValueError, match="disjoint"):
        module.calibration()
    assert not module.FROZEN.exists() and not module.CALIBRATION.exists()


def test_calibration_requires_publication_before_exposing_evaluation_settings(calibration_corpus, monkeypatch):
    module, _, _ = calibration_corpus
    module.calibration(freeze=True)
    def unpublished(path, reference="origin/main"):
        raise ValueError("required published artifact is missing")
    monkeypatch.setattr(module, "published_bytes", unpublished)
    with pytest.raises(ValueError, match="published artifact"):
        module.published_privacy()


def test_noise_summary_retains_nonprivate_controls_and_seed_paired_costs(calibration_corpus):
    protocol, _, _ = calibration_corpus
    runner = importlib.import_module("run_privacy")
    analysis = importlib.import_module("summarize_privacy")
    from fltoken.privacy import account_schedule
    from fltoken.data import sample_clients
    plan = protocol.read(protocol.PLAN)
    plan.update(rounds=2, device="cpu")
    frozen = dict(clipping_bound=.7, delta=1e-5, noise_multipliers=plan["noise_multipliers"])
    records = []
    for cell in runner.privacy_cells(plan):
        config = runner.expected_config(plan, cell, frozen, dict(params={}))
        schedule = [sample_clients(5, 3, number, cell["seed"]).tolist() for number in [1, 2]]
        accounting = account_schedule(config, schedule)
        metrics = {}
        for rule in plan["rules"]:
            values = dict.fromkeys(analysis.METRICS, 0.)
            values["attacker_share"] = None
            values["honest_false_penalty_rate"] = .1+.1*cell["noise_multiplier"]+.02*plan["seeds"].index(cell["seed"])
            values["final_accuracy"] = .95-.4*cell["noise_multiplier"]-.01*plan["seeds"].index(cell["seed"])
            if rule == "Mstar":
                values["honest_false_penalty_rate"] += .125
            metrics[rule] = values
        records.append(dict(**cell, metrics=metrics, accounting=accounting))
    result = analysis.summarize_records(records, plan, frozen)
    assert result["runs"] == 8 and result["simulation_only"]
    assert result["cells"][0]["conservative_all_rounds"]["epsilon"] is None
    assert all(row["infinite_epsilon_clients"] > 0 for row in result["cells"][0]["per_seed_accounting"])
    for row in result["cells"]:
        assert row["metrics"]["B1"]["attacker_share"]["estimate"] is None
        paired = row["paired_false_penalty_difference_from_B1"]["Mstar"]["estimate"]
        assert paired["mean"] == pytest.approx(.125)
        assert paired["observations"] == 2
    assert result["cells"][1]["conservative_all_rounds"]["epsilon"] > result["cells"][3]["conservative_all_rounds"]["epsilon"]
    assert result["cells"][0]["metrics"]["B1"]["honest_false_penalty_rate"]["estimate"]["mean"] < result["cells"][3]["metrics"]["B1"]["honest_false_penalty_rate"]["estimate"]["mean"]
    with pytest.raises(ValueError, match="every unique"):
        analysis.summarize_records(records[:-1], plan, frozen)
    tables = []
    assets = importlib.import_module("privacy_assets")
    assets.privacy_tables(result, lambda name, columns, header, rows, context:
                          tables.append((name, columns, header, rows, context)))
    assert len(tables) == 4
    assert all(len(columns) == len(header) and all(len(row) == len(columns) for row in rows)
               for _, columns, header, rows, _ in tables)
    accounting_rows = next(rows for name, _, _, rows, _ in tables if name == "privacy_accounting")
    assert all(row[-1] == r"$\infty$" for row in accounting_rows[:2])
    root = Path(__file__).resolve().parents[1]
    summary = protocol.read(root/"results/summary.json")
    summary["privacy"] = result
    write_json("results/summary.json", summary)
    write_json("results/literature/sendros.json", protocol.read(root/"results/literature/sendros.json"))
    macros = importlib.import_module("make_macros")
    macros.main()
    numbers = Path("paper/generated/numbers.tex").read_text()
    assert r"\newcommand{\PrivacyRuns}{8}" in numbers
    assert r"\newcommand{\PrivacyAConservativeEpsilon}{\ensuremath{\infty}}" in numbers
    # Retain a clearly synthetic layout fixture outside tracked result folders.
    write_json("out/synthetic-privacy-summary.json", result)
