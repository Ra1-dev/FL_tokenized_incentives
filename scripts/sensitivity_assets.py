"""Figure and table for the exploratory Mstar parameter sensitivity."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation_assets import COLORS, MARKERS
from make_figures import save

SWEEP = Path("results/sensitivity_sweep/summary.json")
OTHER_COLORS = {"CGSV": "#2E8B57", "RFFL": "#B8860B"}
OTHER_MARKERS = {"CGSV": "P", "RFFL": "X"}


def comparison_points(summary):
    """Published pooled estimates of the comparison rules on the same grid."""
    evaluation, extensions = summary["evaluation"]["pooled"], summary["extensions"]["pooled"]
    points = {}
    for rule in ["B0", "B1", "B2"]:
        points[rule] = {metric: evaluation[metric]["rules"][rule]["estimate"]
                        for metric in ["attacker_share", "honest_false_penalty_rate"]}
    for rule in ["CGSV", "RFFL"]:
        points[rule] = {metric: extensions[metric]["rules"][rule]["estimate"]
                        for metric in ["attacker_share", "honest_false_penalty_rate"]}
    return points


def representative(settings):
    """Reference-only, every reputation weight with inert penalties, and two active penalty settings."""
    chosen = [row for row in settings if not row["params"]["ema"]]
    chosen += [row for row in settings if row["params"]["ema"] and not row["params"]["penalties"]]
    for threshold, patience in [(0.05, 3), (0.2, 1)]:
        chosen += [row for row in settings if row["params"]["ema"] and row["params"]["penalties"] and
                   row["params"]["beta"] == 0.9 and row["params"]["threshold"] == threshold and
                   row["params"]["patience"] == patience]
    return chosen


def figure(sweep, points):
    settings = sweep["settings"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    x = [100*row["honest_false_penalty_rate"]["mean"] for row in settings]
    y = [100*row["attacker_share"]["mean"] for row in settings]
    ax.scatter(x, y, s=7, color="#B8BEC8", linewidths=0, label="Mstar settings", zorder=2)
    front = sorted((a, b) for a, b, row in zip(x, y, settings) if row["pareto"])
    ax.plot([a for a, _ in front], [b for _, b in front], color=COLORS["Mstar"], lw=1, zorder=3)
    frozen = next(row for row in settings if row["frozen"])
    ax.scatter([100*frozen["honest_false_penalty_rate"]["mean"]], [100*frozen["attacker_share"]["mean"]],
               marker=MARKERS["Mstar"], s=26, color=COLORS["Mstar"], label="Mstar (frozen)", zorder=5)
    for rule, estimate in points.items():
        color = COLORS.get(rule, OTHER_COLORS.get(rule))
        marker = MARKERS.get(rule, OTHER_MARKERS.get(rule))
        fp, share = estimate["honest_false_penalty_rate"], estimate["attacker_share"]
        ax.errorbar(100*fp["mean"], 100*share["mean"],
                    xerr=[[100*(fp["mean"]-fp["lower"])], [100*(fp["upper"]-fp["mean"])]],
                    yerr=[[100*(share["mean"]-share["lower"])], [100*(share["upper"]-share["mean"])]],
                    fmt=marker, ms=4.5, color=color, elinewidth=.7, capsize=1.5, label=rule, zorder=4)
    floor = min(x)
    ax.axvline(floor, color=COLORS["Mstar"], lw=.6, ls=":", zorder=1)
    ax.text(floor+1, 1, "reference-only floor", fontsize=6.5, color=COLORS["Mstar"], va="bottom")
    ax.set_xlabel("Honest low-reward participations (%)")
    ax.set_ylabel("Attacker reward share (%)")
    ax.set_xlim(-2, max(x)+3)
    ax.set_ylim(0, max(100*p["attacker_share"]["upper"] for p in points.values())+1.5)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right")
    save(fig, "sensitivity_frontier")


def table(sweep, points):
    def cell(estimate):
        return (r"\shortstack{"+f"{100*estimate['mean']:.2f}"+r"\\{}"
                +f"[{100*estimate['lower']:.2f}, {100*estimate['upper']:.2f}]"+"}")

    def setting(row):
        p = row["params"]
        if not p["ema"]:
            return "Reference only", "--", "--", "--"
        if not p["penalties"]:
            return "Reputation", f"{p['beta']:g}", "off", "--"
        name = "Frozen" if row["frozen"] else "Penalties"
        return name, f"{p['beta']:g}", f"{p['threshold']:g}", str(p["patience"])

    lines = ["% Generated from results/sensitivity_sweep/summary.json; requires booktabs.",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"\multicolumn{6}{l}{Exploratory replay on the evaluation grid; means and 95\% seed CIs (\%)} \\",
             r"Setting & $\beta$ & Threshold & Patience & Attacker share & Honest low reward \\", r"\midrule"]
    rows = representative(sweep["settings"])
    frozen = next(row for row in sweep["settings"] if row["frozen"])
    if frozen not in rows:
        rows.append(frozen)
    for row in rows:
        lines.append(" & ".join([*setting(row), cell(row["attacker_share"]), cell(row["honest_false_penalty_rate"])])+r" \\")
    lines.append(r"\midrule")
    for rule, estimate in points.items():
        lines.append(" & ".join([rule, "--", "--", "--", cell(estimate["attacker_share"]),
                                 cell(estimate["honest_false_penalty_rate"])])+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = Path("paper/generated/sensitivity_frontier.tex")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def document(sweep, points):
    settings = sweep["settings"]
    pct = lambda row, metric: 100*row[metric]["mean"]
    reference = next(row for row in settings if not row["params"]["ema"])
    frozen = next(row for row in settings if row["frozen"])
    lowest = min(settings, key=lambda row: row["attacker_share"]["mean"])
    floor = min(pct(row, "honest_false_penalty_rate") for row in settings)
    inert = [row for row in settings if row["params"]["ema"] and row["params"]["penalties"] and row["params"]["threshold"] == 0]
    def unpenalized(beta):
        return next(r for r in settings if r["params"]["ema"] and not r["params"]["penalties"] and r["params"]["beta"] == beta)
    same = all(row[metric] == unpenalized(row["params"]["beta"])[metric]
               for row in inert for metric in ["attacker_share", "honest_false_penalty_rate"])
    dominating = [row for row in settings if row["attacker_share"]["mean"] <= frozen["attacker_share"]["mean"]
                  and row["honest_false_penalty_rate"]["mean"] <= frozen["honest_false_penalty_rate"]["mean"] and row is not frozen
                  and (row["attacker_share"]["mean"], row["honest_false_penalty_rate"]["mean"]) !=
                      (frozen["attacker_share"]["mean"], frozen["honest_false_penalty_rate"]["mean"])]
    if frozen["pareto"]:
        position = "lies on this frontier"
    else:
        gain = max(pct(frozen, "honest_false_penalty_rate")-pct(row, "honest_false_penalty_rate") for row in dominating)
        position = f"lies just inside it; the best dominating setting lowers honest cost by only {gain:.2f} percentage points"
    grid = sweep["grid"]
    lines = ["# Parameter sensitivity of the proposed rule", "",
             "Generated by `python scripts/sensitivity_assets.py` from `results/sensitivity_sweep/summary.json`.", "",
             "## What was replayed", "",
             "Mstar computes one leave-one-out median quality per participation; reputation, the low-quality threshold and "
             "penalties act only on those recorded qualities, and the shared global model never depends on rewards. "
             f"`python scripts/sensitivity_sweep.py` therefore replays {len(settings)} settings over the "
             f"{sweep['scenarios']} main-horizon scenarios and {len(sweep['seeds'])} seeds of the recorded grid without retraining: "
             f"beta in {grid['beta']}, threshold in {grid['threshold']} and patience in {grid['patience']} with penalties, "
             "every beta without penalties, and the reference alone. Before any other setting is scored, the frozen setting must reproduce "
             "every recorded share and slash and the published pooled estimates exactly; the script stops otherwise.", "",
             "Pooling and intervals follow the published comparison: scenarios are averaged within each seed and whole seeds are resampled. "
             "Attacker share pools attacked scenarios; the honest low-reward rate also includes attack-free runs.", "",
             "## Findings", "",
             f"- No setting lowers the honest low-reward rate below {floor:.2f}%, the value of the reference alone "
             f"({pct(reference, 'attacker_share'):.2f}% attacker share). Reputation and penalties can only trade along a narrow "
             f"frontier: attacker share ranges from {pct(lowest, 'attacker_share'):.2f}% (at {pct(lowest, 'honest_false_penalty_rate'):.2f}% honest cost) "
             f"to {max(pct(row, 'attacker_share') for row in settings):.2f}%.",
             f"- The frozen setting ({pct(frozen, 'attacker_share'):.2f}%, {pct(frozen, 'honest_false_penalty_rate'):.2f}%) {position}. "
             "The honest cost is set by the leave-one-out median reference, not by the selected parameters.",
             f"- With a zero threshold, every penalty setting {'matches' if same else 'differs from'} reputation without penalties "
             "at the same beta on both pooled metrics: a participation slashed for zero quality already earns nothing. "
             "This agrees with the identical Aema and Afull ablation outcomes; the slash events still burn stake in settlement.",
             f"- CGSV ({100*points['CGSV']['attacker_share']['mean']:.2f}%, {100*points['CGSV']['honest_false_penalty_rate']['mean']:.2f}%) and "
             f"RFFL ({100*points['RFFL']['attacker_share']['mean']:.2f}%, {100*points['RFFL']['honest_false_penalty_rate']['mean']:.2f}%) "
             "lie outside the region any Mstar setting reaches: lower honest cost at a higher attacker share.", "",
             "## Scope", "",
             "This is an exploratory, post-hoc replay on the evaluation grid itself. The frozen parameters were selected earlier on disjoint pilot seeds "
             "([method](METHOD.md)); no setting here replaces them, and choosing the best-looking setting from this table would be optimistic. "
             "Only parameters that act after the reference were varied; the reference construction and the initial reputation are unchanged.", "",
             "Assets: `figures/sensitivity_frontier.pdf` and `paper/generated/sensitivity_frontier.tex`; macros with the `Sensitivity` prefix "
             "are generated by `python scripts/make_macros.py`.", ""]
    Path("docs/SENSITIVITY.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    points = comparison_points(summary)
    figure(sweep, points)
    table(sweep, points)
    document(sweep, points)
    print("Generated the sensitivity figure and table")


if __name__ == "__main__":
    main()
