"""Publish only matched cross-platform derivation evidence and its definitions."""

import json
from pathlib import Path

from fltoken.io import sha256, write_json


def main():
    reports = [json.loads(Path(f"results/validation/derived-{system}.json").read_text(encoding="utf-8"))
               for system in ["windows", "linux"]]
    for report in reports:
        if not report["passed"] or report["changed_artifacts"]:
            raise ValueError("portable reconstruction has unresolved changes")
        for name, digest in {**report["source_sha256"], **report["artifact_sha256"]}.items():
            if sha256(name) != digest:
                raise ValueError(f"portable evidence changed: {name}")
    if reports[0]["artifact_sha256"] != reports[1]["artifact_sha256"] or reports[0]["source_sha256"] != reports[1]["source_sha256"]:
        raise ValueError("Windows and Linux did not reconstruct identical evidence")
    record = dict(runs=reports[0]["runs"], artifacts=reports[0]["artifacts"],
                  migration_runs=reports[0]["migration_runs"], migrated_rule_fields=reports[0]["migrated_rule_fields"],
                  byte_identical=True, environments=[{key:r[key] for key in ["platform","python","versions"]} for r in reports])
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    summary["metric_portability"] = record
    write_json("results/summary.json",summary)
    record = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))["metric_portability"]
    lines = ["# Portable reward metrics", "", "Generated from `results/summary.json` by `python scripts/document_metric_portability.py`.", "",
        "## Exact definitions", "",
        "For sorted nonnegative integer client payouts x_i, Gini is sum((2*i-n-1)*x_i)/(n*sum(x_i)), with rank i starting at one. Both sums and the quotient use Python integers and Fraction; only the final value is converted to a float. Empty and all-zero populations have undefined Gini. No payout is first converted to floating point.", "",
        "Attacker share, registered headcount share, honest false-penalty rate, precision and recall are exact ratios of integer totals or counts, converted once. Detection F1 is 2*TP/(2*TP+FP+FN), with zero when its denominator is zero. It is not reconstructed through already-rounded precision and recall. Flagging compares the exact binary-float value of a recorded share against the rational threshold 1/(2*K).",
        "", "Reproduction quality sums are integers. Per-round and pooled means use rational division. Linear quantiles interpolate the two adjacent sorted observations with Fraction arithmetic; undefined observations remain explicit. Seed means and standard deviations use Python's statistics module, whose accumulation is exact rational arithmetic, rather than vectorized floating reductions. Per-observation subtraction and division follow a fixed scalar order.", "",
        "## Declared migration", "",
        "The previous Gini implementation converted large integer payouts to float and used a vector dot product. Platform-dependent reduction order changed its final bits. Stored metrics were regenerated from the same participation logs and integer ledgers, without retraining or changing payouts. Each changed run retains `metrics_provenance.json`: previous and new metric hashes, changed fields, training-log hash, original training commit and derivation-source hashes.", "",
        f"Migrated runs: {record['migration_runs']}. Changed rule-level fields: "+", ".join(f"{key}={value}" for key,value in record["migrated_rule_fields"].items())+".", "",
        f"Both environments reconstructed {record['artifacts']} artifacts across {record['runs']} runs without any byte changes. Artifact and derivation-source hashes agree between environments.", "",
        "| Environment | Python | NumPy | PyYAML |", "|---|---|---|---|"]
    for environment in record["environments"]:
        lines.append(f"| {environment['platform']} | {environment['python']} | {environment['versions']['numpy']} | {environment['versions']['PyYAML']} |")
    lines += ["", "## Reproduce", "", "In each prepared environment, run the same command with a platform-specific output name:", "", "```bash",
              "python scripts/check_derived_metrics.py --output results/validation/derived-linux.json", "```", "",
              "Use `derived-windows.json` for the Windows execution. Then run `python scripts/document_metric_portability.py` to verify that both reports match the current artifact bytes. The check invokes ledger verification, reproduction analysis and stale-sensitivity analysis without training or network access.", "",
              "A deliberate metric-definition migration uses `python scripts/rebuild_ledgers.py --rewrite`, followed by both analysis commands and the platform checks. Normal verification omits `--rewrite`; mismatches fail without an approximate comparison. This evidence concerns reconstruction of fixed logs. It does not claim cross-platform bit identity of newly trained neural networks.", ""]
    Path("docs/METRICS.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    print(f"Portable derivation: {record['runs']} runs, {record['artifacts']} identical artifacts")


if __name__ == "__main__":
    main()
