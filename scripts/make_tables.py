"""Generate booktabs fragments and an artifact manifest from the shared summary."""

import json
from pathlib import Path


def table(name, columns, header, rows):
    lines = ["% Generated from results/summary.json; requires booktabs.",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = Path("paper/generated") / (name + ".tex")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    r = summary.get("reproduction")
    if not r:
        raise SystemExit("Full reproduction summary is required before generating result tables")
    table("reproduction_runs", "rrrr", ["Seed", r"Attacker (\%)", r"Test acc. (\%)", r["device"].upper()+" (s)"],
          [[row["seed"], f"{100*row['attacker_share']:.3f}", f"{100*row['final_accuracy']:.2f}",
            f"{run['wall_seconds']:.1f}"] for row, run in zip(r["per_seed"], r["runs"])])
    table("reproduction_comparison", "lrr", ["Source", r"Mean (\%)", r"SD (pp)"],
          [["Sendros et al.", f"{100*r['published']['attacker_share_mean']:.2f}", f"{100*r['published']['attacker_share_sd']:.2f}"],
           ["Reproduction (sample)", f"{100*r['attacker_share_mean']:.2f}", f"{100*r['attacker_share_sample_sd']:.2f}"],
           ["Reproduction (population)", f"{100*r['attacker_share_mean']:.2f}", f"{100*r['attacker_share_population_sd']:.2f}"]])
    table("reproduction_error", "rr", ["Percentile", r"Relative error (\%)"],
          [[f"{100*float(p):g}", f"{100*value:.4f}" if value is not None else "Undefined"]
           for p, value in r["relative_error_quantiles"].items()])
    print("Generated reproduction run, source comparison, and error-distribution tables")


if __name__ == "__main__":
    main()
