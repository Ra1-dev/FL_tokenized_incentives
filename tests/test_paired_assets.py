"""Paper exports must preserve pairing units, signed effects and missing seeds."""

import copy
import importlib
from pathlib import Path

import pytest

from fltoken.statistics import holm, paired_wilcoxon, sampled_bootstrap_mean


@pytest.fixture
def exports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return (importlib.import_module("evaluation_assets"), importlib.import_module("make_macros"),
            importlib.import_module("make_tables"))


@pytest.fixture
def contrasts():
    rows = []
    for metric, sign in [("attacker_share", -1), ("honest_false_penalty_rate", 1)]:
        for baseline, magnitude in [("B0", .1), ("B1", .2), ("B2", .3)]:
            values = [sign*magnitude]*5
            rows.append(dict(metric=metric, baseline=baseline,
                             condition_seed_test=paired_wilcoxon(values*16, [0]*80),
                             seed_test=paired_wilcoxon(values, [0]*5),
                             paired_difference=dict(defined=5, undefined=0, estimate=sampled_bootstrap_mean(values))))
    for key in ["condition_seed_test", "seed_test"]:
        adjusted = holm({index: row[key]["p_value"] for index, row in enumerate(rows)})
        for index, row in enumerate(rows):
            row[key+"_holm_p"] = adjusted[index]
    return dict(device="cuda", contrasts=rows)


def rendered(exports, result):
    records = {}
    exports[0].paired_tables(result, lambda name, columns, header, rows, context:
                            records.update({name: dict(columns=columns, header=header, rows=rows, context=context)}))
    return records


def test_tests_and_intervals_keep_their_distinct_sampling_units(exports, contrasts):
    before = copy.deepcopy(contrasts)
    tables = rendered(exports, contrasts)
    primary = tables["evaluation_paired"]
    seed = tables["evaluation_paired_seed"]
    intervals = tables["evaluation_paired_intervals"]
    assert "Condition--seed" in primary["context"] and "Seed-pooled" in seed["context"]
    assert primary["rows"][0][2] == 80 and seed["rows"][0][2] == 5
    assert float(primary["rows"][0][-2]) < 1e-20
    assert seed["rows"][0][-2:] == ["0.0625", "0.375"]
    assert primary["rows"][0][3:5] == ["-10.000", "-1.000"]
    assert intervals["rows"][0] == ["Attacker", "B0", "-10.000", "0.000", "-10.000", "-10.000", 5, 0]
    assert intervals["rows"][3][2:6] == ["10.000", "0.000", "10.000", "10.000"]
    assert contrasts == before


def test_contrast_macros_preserve_units_family_and_baseline(exports, contrasts):
    values = exports[1].paired_macros(contrasts)
    assert len(values) == 84
    assert values["EvaluationAttackerShareEqualDifferenceMean"] == "-10.000"
    assert values["EvaluationHonestPenaltyKrumDifferenceLow"] == "30.000"
    assert values["EvaluationAttackerShareCosineConditionSeedPairs"] == 80
    assert values["EvaluationAttackerShareCosineSeedPairs"] == 5
    assert values["EvaluationAttackerShareCosineSeedP"] == "0.0625"
    assert values["EvaluationAttackerShareCosineSeedHolmP"] == "0.375"
    assert values["EvaluationAttackerShareCosineSeedRankEffect"] == "-1"
    assert values["EvaluationHonestPenaltyKrumConditionSeedRankEffect"] == "1"


def test_missing_seed_exports_undefined_claims_and_retains_counts(exports, contrasts):
    row = contrasts["contrasts"][0]
    row["paired_difference"] = dict(defined=4, undefined=1, estimate=None)
    for key in ["condition_seed_test", "seed_test"]:
        row[key] = row[key+"_holm_p"] = None
    tables = rendered(exports, contrasts)
    for name in ["evaluation_paired", "evaluation_paired_seed"]:
        assert tables[name]["rows"][0][2:] == ["Undefined"]*5
    assert tables["evaluation_paired_intervals"]["rows"][0][2:] == ["Undefined"]*4+[4, 1]
    values = exports[1].paired_macros(contrasts)
    prefix = "EvaluationAttackerShareEqual"
    assert values[prefix+"PairedDefined"] == 4 and values[prefix+"PairedMissing"] == 1
    assert all(value == "undefined" for name, value in values.items()
               if name.startswith(prefix) and not name.endswith(("PairedDefined", "PairedMissing")))


def test_booktabs_exports_keep_each_cell_in_its_declared_column(exports, contrasts, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exports[0].paired_tables(contrasts, exports[2].table)
    paths = sorted(Path("paper/generated").glob("*.tex"))
    assert len(paths) == 3
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "CUDA" in text and "Mstar minus baseline" in text
        width = 8 if "intervals" in path.stem else 7
        data = [line for line in text.splitlines() if line.startswith(("Attacker &", "Honest &"))]
        assert len(data) == 6 and all(line.count("&") == width-1 for line in data)
        assert b"\r" not in path.read_bytes()
