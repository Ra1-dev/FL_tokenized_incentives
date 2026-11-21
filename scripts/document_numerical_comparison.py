"""Publish a separate numerical variant comparison without replacing frozen results."""
from pathlib import Path

from evaluation import ABLATIONS, evaluation_cells, read, recorded_method
from fltoken.io import sha256, write_json


def comparison_summary(weights, comparison, expected, frozen_sha):
    cases = [case for case in comparison["cases"] if case.get("corpus") == "grid"]
    indexed = {case["path"]: case for case in cases}
    if (weights["complete_grid"] is not True or comparison["complete_corpus"] is not True
            or comparison["synthetic_only"] is not False or len(indexed) != len(cases)
            or indexed.keys() != expected.keys() or weights["runs"] != len(expected)
            or weights["rounds"] != sum(expected.values())
            or any(case["rounds"] != expected[path] for path, case in indexed.items())):
        raise ValueError("numerical comparison requires every complete evaluation run")
    if any(record["original_scorer_sha256"] != frozen_sha for record in [weights, comparison]):
        raise ValueError("numerical comparison must identify the frozen scorer")
    if (weights["original_reconstruction_matches_recorded"] is not True
            or comparison["original_matches_recorded"] is not True
            or any(case["original_matches_recorded"] is not True for case in cases)):
        raise ValueError("the original scorer must reconstruct the recorded evaluation")
    for case in cases:
        if set(case["stages"]) != set(ABLATIONS):
            raise ValueError("numerical comparison requires every component stage")
        counts = [count for stage in case["stages"].values() for count in stage["changed"].values()]
        if (any(type(count) is not int or count < 0 for count in counts)
                or case["identical"] != (not any(counts))):
            raise ValueError("numerical identity differs from the measured changes")
    changed = sum(not case["identical"] for case in cases)
    return dict(grid_runs=len(cases), grid_rounds=sum(expected.values()),
                changed_grid_runs=changed, identical_on_grid=changed == 0,
                original_scorer_sha256=frozen_sha,
                variant_scorer_sha256=comparison["current_scorer_sha256"],
                frozen_product_counts=weights["stages"],
                maximum_share_difference=max(case["stages"]["Afull"]["maximum_share_difference"] for case in cases),
                maximum_payout_difference=max(case["stages"]["Afull"]["maximum_payout_difference"] for case in cases),
                implementation="separately labelled post-freeze numerical variant",
                evaluation_source="original frozen training outputs; never replaced by this comparison")


def main():
    method = recorded_method()
    weights_path = Path("results/validation/method_weights.json")
    comparison_path = Path("results/validation/method_comparison.json")
    weights, comparison = read(weights_path), read(comparison_path)
    if comparison["current_scorer_sha256"] != sha256("src/fltoken/ours.py"):
        raise ValueError("numerical variant source differs from its comparison")
    for record, field in [(weights, "source_sha256"), (comparison, "analysis_source_sha256")]:
        if record["method_sha256"] != sha256("configs/method_frozen.json"):
            raise ValueError("comparison belongs to a different frozen method")
        for name, digest in record[field].items():
            if sha256(name) != digest:
                raise ValueError(f"comparison producer changed: {name}")
    if (weights["grid_manifest_sha256"] != sha256("results/grid/manifest.json")
            or comparison["manifest_sha256"]["grid"] != sha256("results/grid/manifest.json")
            or weights["execution_sha256"] != sha256("results/grid/execution.json")):
        raise ValueError("comparison belongs to a different completed grid")
    for case in comparison["cases"]:
        for name, digest in case["inputs"].items():
            if sha256(Path(case["path"])/name) != digest:
                raise ValueError("comparison input differs from its recorded evidence")
    expected = {cell["path"]: cell["rounds"] for cell in evaluation_cells(read("configs/evaluation_plan.json"))}
    result = comparison_summary(weights, comparison, expected, method["scorer_sha256"])
    result["provenance"] = dict(weights_sha256=sha256(weights_path), comparison_sha256=sha256(comparison_path),
                                generator_sha256=sha256(__file__))
    summary = read("results/summary.json")
    summary["method_numerical_comparison"] = result
    write_json("results/summary.json", summary)
    result = read("results/summary.json")["method_numerical_comparison"]
    counts = result["frozen_product_counts"]["Afull"]
    equality = ("The frozen and numerical-variant implementations are identical on the evaluation grid."
                if result["identical_on_grid"] else
                "The numerical variant differs on the evaluation grid; the frozen outcomes remain the reported results.")
    text = f"""# Frozen method and numerical variant

Generated by `python scripts/document_numerical_comparison.py` from `results/summary.json`.

Every headline Mstar result uses the original frozen scorer executed during training.
The current library is a separately labelled post-freeze numerical variant.

The complete comparison covers {result['grid_runs']} evaluation runs and {result['grid_rounds']} rounds.
The original scorer reconstructs the recorded outputs exactly. Its full-method
allocation has {counts['subnormal_product_rounds']} rounds with eligible subnormal
products, including {counts['erased_product_rounds']} rounds with products rounded to zero.
Legitimate empty allocations are counted separately: {counts['empty_allocation_rounds']} rounds.

| Implementation | Compared runs | Runs differing from the frozen result | Maximum share difference | Maximum payout difference (base units) |
|---|---:|---:|---:|---:|
| Frozen Mstar | {result['grid_runs']} | Reference | Reference | Reference |
| Post-freeze numerical variant | {result['grid_runs']} | {result['changed_grid_runs']} | {result['maximum_share_difference']:.17g} | {result['maximum_payout_difference']} |

{equality} This claim includes all component stages, stored state, integer payouts
and vesting wealth on the same cached matrices; zero underflow incidence alone
is not used as a substitute for that comparison. No parameters were selected again.
Agreement on this corpus does not remove the documented finite-range limitations.

Frozen scorer SHA-256: `{result['original_scorer_sha256']}`.
Numerical variant SHA-256: `{result['variant_scorer_sha256']}`.
"""
    Path("docs/NUMERICAL_COMPARISON.md").write_text(text, encoding="utf-8", newline="\n")
    table = f"""% Generated from results/summary.json by document_numerical_comparison.py.
\\begin{{tabular}}{{lrrr}}
\\toprule
Implementation & Runs & Changed runs & Max payout change \\\\
\\midrule
Frozen Mstar & {result['grid_runs']} & Reference & Reference \\\\
Post-freeze numerical variant & {result['grid_runs']} & {result['changed_grid_runs']} & {result['maximum_payout_difference']} \\\\
\\bottomrule
\\end{{tabular}}
% Payout changes are integer token base units; comparison uses the same cached updates.
"""
    Path("paper/generated/method_numerical_comparison.tex").write_text(table, encoding="utf-8", newline="\n")
    print(f"Numerical variant: {result['grid_runs']} grid runs, {result['changed_grid_runs']} changed; "
          f"frozen erased-product rounds: {counts['erased_product_rounds']}")


if __name__ == "__main__":
    main()
