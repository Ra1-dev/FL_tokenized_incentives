"""A completed result label requires complete, plan-bound evaluation evidence."""
import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("coverage", ["pending", "complete", "partial", "different_plan"])
def test_rendered_status_preserves_actual_evaluation_coverage(monkeypatch, tmp_path, coverage):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    module = importlib.import_module("document_evaluation")
    monkeypatch.chdir(tmp_path)
    for directory in ["configs", "results", "docs"]:
        Path(directory).mkdir()
    Path("configs/evaluation_plan.json").write_bytes((root/"configs/evaluation_plan.json").read_bytes())
    plan = module.read("configs/evaluation_plan.json")
    summary = {}
    if coverage != "pending":
        summary["evaluation"] = dict(
            nominal_cells=len(list(module.evaluation_cells(plan))) - (coverage == "partial"),
            provenance=dict(plan_sha256="0"*64 if coverage == "different_plan" else
                            module.sha256("configs/evaluation_plan.json")),
            inference_status="complete fixed-design comparison")
    Path("results/summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if coverage in {"partial", "different_plan"}:
        with pytest.raises(ValueError, match="complete declared comparison"):
            module.main()
        assert not Path("docs/EVALUATION.md").exists()
    else:
        module.main()
        document = Path("docs/EVALUATION.md").read_text(encoding="utf-8")
        assert ("Status: **predeclared; evaluation results pending**" in document) == (coverage == "pending")
        assert ("Status: **complete fixed-design comparison; frozen-method outcomes retained**" in document) == (coverage == "complete")
