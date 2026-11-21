"""An absent empirical section cannot leave a misleading comparison export."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return importlib.import_module("asset_scope")


def put(name):
    path = Path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic artifact\n")
    return path


def test_absent_sections_remove_only_their_comparison_outputs(scope, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    missing = ["figures/privacy_false_penalty.png", "figures/privacy_false_penalty.pdf",
               "paper/generated/method_numerical_comparison.tex",
               "paper/generated/privacy_outcomes.tex", "paper/generated/extensions_bias_main.tex",
               "paper/generated/fashion_accuracy.tex"]
    retained = ["figures/evaluation_ablation.pdf", "paper/generated/evaluation_paired.tex",
                "paper/generated/privacy_calibration.tex", "paper/generated/extension_pilot_selection.tex",
                "figures/architecture.png", "paper/generated/numbers.tex", "paper/generated/notes.tex",
                "figures/privacy_readme.txt", "paper/privacy_outcomes.tex",
                "figures/nested/privacy_false_penalty.png"]
    for name in missing+retained:
        put(name)
    removed = scope.remove_absent_comparisons({"evaluation": {}}, "figures")
    removed += scope.remove_absent_comparisons({"evaluation": {}}, "paper/generated")
    assert sorted(removed) == sorted(missing)
    assert all(not Path(name).exists() for name in missing)
    assert all(Path(name).read_bytes() == b"synthetic artifact\n" for name in retained)


def test_table_regeneration_removes_an_old_paired_table_when_results_are_absent(scope, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root/"results/summary.json").read_text(encoding="utf-8"))
    for section in ["evaluation", "extensions", "privacy", "fashion"]:
        summary.pop(section, None)
    monkeypatch.chdir(tmp_path)
    Path("results").mkdir()
    Path("results/summary.json").write_text(json.dumps(summary), encoding="utf-8")
    old = put("paper/generated/evaluation_paired_seed.tex")
    importlib.import_module("make_tables").main()
    assert not old.exists()
    assert Path("paper/generated/reproduction_runs.tex").exists()


@pytest.mark.parametrize("name", ["figures/evaluation_ablation.png", "figures/extensions_bias.pdf",
                                  "paper/generated/method_numerical_comparison.tex", "docs/NUMERICAL_COMPARISON.md",
                                  "paper/generated/privacy_outcomes.tex", "paper/generated/fashion_accuracy.tex"])
def test_archive_rejects_stale_comparison_without_replacing_previous_file(scope, tmp_path, name):
    root = Path(__file__).resolve().parents[1]
    module = importlib.import_module("package_assets")
    blobs = {path:(root/path).read_bytes() for path in module.ROOT_FILES}
    blobs.update({"paper/references.bib": b"synthetic bibliography\n",
                  "paper/generated/numbers.tex": b"synthetic macros\n"})
    blobs["results/summary.json"] = b'{}\n'
    destination = tmp_path/"support.zip"
    destination.write_bytes(b"previous archive")
    with pytest.raises(ValueError, match="lack their empirical summary section"):
        module.assemble("a"*40, 1795161601, {**blobs, name:b"stale result"}, destination)
    assert destination.read_bytes() == b"previous archive"
    assert list(tmp_path.glob("*.tmp")) == []
    section = scope.comparison_section(name)
    blobs["results/summary.json"] = json.dumps({section:dict(synthetic_fixture=True)}).encode()
    manifest = module.assemble("a"*40, 1795161601, {**blobs, name:b"current result"}, destination)
    assert name in manifest["files"] and section in manifest["result_sections_present"]


def test_calibration_is_not_mislabeled_as_an_empirical_comparison(scope):
    for name in ["paper/generated/privacy_calibration.tex", "paper/generated/extension_pilot_selection.tex",
                 "paper/generated/pilot_selection.tex", "docs/PRIVACY.md", "docs/EVALUATION.md"]:
        assert scope.comparison_section(name) is None
