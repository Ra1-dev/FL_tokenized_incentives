"""Seed-level honest-cost and analytic-accounting assets from complete noisy runs."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from decimal import Decimal
from evaluation_assets import interval_cells


DISPLAY = [("B1", "Inclusive cosine", "#176B87", "o"),
           ("Mstar", "Full excluding rule", "#B4473D", "s"),
           ("Aref", "Excluding reference only", "#6D56A0", "^")]


def privacy_figures(result, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    def draw(ax, x, estimate, color, marker):
        low, high = 100*estimate["lower"], 100*estimate["upper"]
        ax.plot([x, x], [low, high], color=color, lw=1)
        for value in [low, high]:
            ax.plot([x-.015, x+.015], [value, value], color=color, lw=1)
        ax.plot(x, 100*estimate["mean"], marker, color=color, ms=4, mfc="white")
    undefined = False
    for offset, (rule, label, color, marker) in zip([-.035, 0., .035], DISPLAY):
        for row in result["cells"]:
            x = row["noise_multiplier"]+offset
            estimate = row["metrics"][rule]["honest_false_penalty_rate"]["estimate"]
            if estimate is None:
                undefined = True
                axes[0].plot(x, 1.02, "x", transform=axes[0].get_xaxis_transform(), color=color, clip_on=False)
            else:
                draw(axes[0], x, estimate, color, marker)
    # Every rule shares one model trajectory; display its accuracy only once.
    for row in result["cells"]:
        estimate = row["metrics"]["B1"]["final_accuracy"]["estimate"]
        if estimate is None:
            undefined = True
            axes[1].plot(row["noise_multiplier"], 1.02, "x", transform=axes[1].get_xaxis_transform(), color="#64748B", clip_on=False)
        else:
            draw(axes[1], row["noise_multiplier"], estimate, "#334155", "o")
    axes[0].set_ylabel("Honest false penalties (%)")
    axes[1].set_ylabel("Final test accuracy (%)")
    for ax in axes:
        ax.set(xlabel="Per-coordinate noise multiplier", ylim=(-2, 102),
               xticks=[row["noise_multiplier"] for row in result["cells"]])
        ax.grid(alpha=.15)
    handles = [Line2D([], [], color=color, marker=marker, ls="", ms=4, label=label) for _, label, color, marker in DISPLAY]
    if undefined:
        handles.append(Line2D([], [], color="#64748B", marker="x", ls="", ms=4, label="Undefined; all seeds retained"))
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=7, ncol=2 if undefined else 3)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save(fig, "privacy_false_penalty")


def privacy_tables(result, table):
    context = result["training_device"].upper()+"; seed CIs; failed seeds retained"
    table("privacy_honest_false_penalty_rate", "rlrrrr", [r"$\sigma$", "Rule", r"False penalty (\%)", "SD (pp)", "CI low", "CI high"],
          [[f"{row['noise_multiplier']:g}", rule, *interval_cells(row["metrics"][rule]["honest_false_penalty_rate"])]
           for row in result["cells"] for rule in ["B1", "Mstar", "Aref"]], context=context)
    table("privacy_final_accuracy", "rrrrr", [r"$\sigma$", r"Accuracy (\%)", "SD (pp)", "CI low", "CI high"],
          [[f"{row['noise_multiplier']:g}", *interval_cells(row["metrics"]["B1"]["final_accuracy"])]
           for row in result["cells"]], context="Shared trajectory; "+context)
    table("privacy_paired_penalties", "rlrrrr", [r"$\sigma$", "Rule minus B1", "Mean (pp)", "SD (pp)", "CI low", "CI high"],
          [[f"{row['noise_multiplier']:g}", rule, *interval_cells(row["paired_false_penalty_difference_from_B1"][rule])]
           for row in result["cells"] for rule in ["Mstar", "Aref"]], context=context)
    def epsilon(value):
        return r"$\infty$" if value is None else f"{value:.3f}"
    table("privacy_accounting", "rrrrrr", [r"$\sigma$", "Seed", "Min planned", "Max planned", r"Max client $\epsilon$", r"All-round $\epsilon$"],
          [[f"{row['noise_multiplier']:g}", seed["seed"], seed["minimum_participations"], seed["maximum_participations"],
            epsilon(seed["maximum_client_epsilon"]), epsilon(row["conservative_all_rounds"]["epsilon"])]
           for row in result["cells"] for seed in row["per_seed_accounting"]],
          context=rf"Ideal full-schedule bounds; $\delta={format(Decimal(str(result['delta'])), 'f')}$; excludes numerical failure signals")
    if all("outcomes" in row for row in result["cells"]):
        table("privacy_outcomes", "rrrll", [r"$\sigma$", "Seed", "Rounds", "Outcome", "Failure stage"],
              [[f"{row['noise_multiplier']:g}", cell["seed"], cell["completed_rounds"],
                "Complete" if cell["status"] == "complete" else "Numerical failure",
                ", ".join(sorted({item["stage"].replace("_", " ") for item in cell["failure"]["failures"]})) if cell["failure"] else "---"]
               for row in result["cells"] for cell in row["outcomes"]],
              context="Failed-run rounds are diagnostic prefixes; every seed is retained")
