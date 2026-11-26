"""The parameter replay reproduces the frozen rule before scoring any other setting."""

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"scripts"))
    monkeypatch.chdir(ROOT)
    return importlib.import_module("sensitivity_sweep")


def frozen_params():
    return json.loads((ROOT/"configs/method_frozen.json").read_text(encoding="utf-8"))["params"]


def test_declared_grid_contains_the_frozen_setting_once(sweep):
    settings = sweep.settings(frozen_params())
    assert settings.count(frozen_params()) == 1
    assert len(settings) == len(sweep.BETAS)*len(sweep.THRESHOLDS)*len(sweep.PATIENCES) + len(sweep.BETAS) + 1


@pytest.mark.parametrize("path", ["results/grid/mnist-noise-f30-s42-cuda-t20", "results/grid/mnist-stale-f10-s44-cuda-t20"])
def test_frozen_replay_matches_every_recorded_share_and_metric(sweep, path):
    manifest = json.loads((ROOT/"results/grid/manifest.json").read_text(encoding="utf-8"))
    cell = next(cell for cell in manifest["cells"] if cell["path"] == path)
    replayed = sweep.replay(*sweep.load_cell(cell), frozen_params(), check=True)
    recorded = json.loads((ROOT/path/"metrics.json").read_text(encoding="utf-8"))["Mstar"]
    for metric in sweep.METRICS:
        assert replayed[metric] == recorded[metric]


def test_replay_rejects_a_setting_that_differs_from_the_recorded_allocation(sweep):
    manifest = json.loads((ROOT/"results/grid/manifest.json").read_text(encoding="utf-8"))
    cell = next(cell for cell in manifest["cells"] if cell["attack"] == "noise")
    with pytest.raises(ValueError, match="does not reproduce"):
        sweep.replay(*sweep.load_cell(cell), {**frozen_params(), "beta": 0.5}, check=True)


def test_pareto_keeps_only_undominated_points(sweep):
    point = lambda a, f: {"attacker_share": {"mean": a}, "honest_false_penalty_rate": {"mean": f}}
    points = [point(.05, .15), point(.04, .20), point(.06, .16), point(.05, .15)]
    assert sweep.pareto(points) == [0, 1, 3]


def test_recorded_sweep_reproduces_the_published_frozen_estimates():
    result = json.loads((ROOT/"results/sensitivity_sweep/summary.json").read_text(encoding="utf-8"))
    published = json.loads((ROOT/"results/summary.json").read_text(encoding="utf-8"))["evaluation"]["pooled"]
    frozen = [row for row in result["settings"] if row["frozen"]]
    assert len(frozen) == 1 and frozen[0]["params"] == frozen_params()
    for metric in ["attacker_share", "honest_false_penalty_rate"]:
        assert frozen[0][metric] == published[metric]["rules"]["Mstar"]["estimate"]
    assert result["exploratory"] is True
