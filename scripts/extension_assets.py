"""Figures and tables for the complete offline reputation comparison."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evaluation_assets import ATTACKS, condition_label, interval_cells


def extension_figures(result, save):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    common_low, common_high = -.02, .1
    for ax, family in zip(axes, ["CGSV", "RFFL"]):
        rows = [row for row in result["noise_observations"] if row["family"] == family
                and row["horizon"] == "main" and row["attack"] == "sendros_mix" and row["seed"] in result["source_seeds"]]
        defined = [row for row in rows if row["bias_defined"]]
        for eligible, marker, color, label in [(True, "o", "#176B87", "Active before round"), (False, "x", "#B4473D", "Previously excluded")]:
            selected = [row for row in defined if row["eligible_before"] == eligible]
            if selected:
                ax.scatter([row["self_inclusion"] for row in selected], [row["cosine"] for row in selected],
                           marker=marker, color=color, s=13, alpha=.7, linewidths=.6, label=label)
        if defined:
            high = max([row["self_inclusion"] for row in defined]+[row["cosine"] for row in defined]+[.1])*1.05
            low = min([row["cosine"] for row in defined]+[0.])-.02
            common_low, common_high = min(common_low, low), max(common_high, high)
            ax.plot([0, high], [0, high], "--", color="#64748B", lw=.8)
            ax.set(xlim=(-.02, high), ylim=(low, high))
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.text(.5, .5, "No defined observations", ha="center", transform=ax.transAxes, fontsize=8)
        ax.set(title=family, xlabel="Weighted self-inclusion term")
        ax.text(.02, .98, f"Noise participations: {len(rows)}\nUndefined geometry: {len(rows)-len(defined)}",
                ha="left", va="top", transform=ax.transAxes, fontsize=6)
        ax.grid(alpha=.15)
    axes[0].set_ylim(common_low, common_high)
    axes[0].set_ylabel("Measured inclusive cosine")
    fig.suptitle("Fixed mixture: source-comparison seeds", fontsize=9)
    fig.tight_layout()
    save(fig, "extensions_bias")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharex=True)
    fields = [("mean_self_inclusion", "Self term", "#6D56A0", "^"),
              ("mean_cosine", "Inclusive", "#176B87", "o"),
              ("mean_excluding_cosine", "Excluding", "#B4473D", "s")]
    any_missing = False
    for ax, family in zip(axes, ["CGSV", "RFFL"]):
        rows = [row for row in result["bias_intervals"] if row["family"] == family
                and row["horizon"] == "main" and row["subset"] == "all_noise"]
        missing = False
        for index, row in enumerate(rows):
            for offset, (field, label, color, marker) in zip([-.22, 0, .22], fields):
                estimate = row["metrics"][field]["estimate"]
                if estimate is None:
                    missing = True
                    ax.plot(1.02, index+offset, "x", transform=ax.get_yaxis_transform(), color=color, ms=3, clip_on=False)
                else:
                    ax.plot([estimate["lower"], estimate["upper"]], [index+offset]*2, color=color, lw=1)
                    ax.plot(estimate["mean"], index+offset, marker, color=color, ms=3)
        any_missing |= missing
        ax.set(yticks=range(len(rows)), yticklabels=[condition_label(row) for row in rows],
               xlabel="Mean across seeds with bootstrap CI", title=family)
        ax.tick_params(axis="y", labelsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="#94A3B8", lw=.7)
        ax.grid(axis="x", alpha=.15)
    handles = [Line2D([], [], color=color, marker=marker, ms=3, label=label) for _, label, color, marker in fields]
    if any_missing:
        handles.append(Line2D([], [], color="#64748B", marker="x", ls="", ms=3, label="Undefined"))
    fig.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save(fig, "extensions_bias_contrasts")


def extension_tables(result, table):
    context = "Device: "+result["device"].upper()+"; uncertainty across complete seeds"
    def number(value, scale=1):
        return "Undefined" if value is None else f"{scale*value:.4g}"
    contrasts = []
    for row in result["contrasts"]:
        estimate = row["paired_difference"]["estimate"]
        contrasts.append([row["family"], row["baseline"],
            "Attackers" if row["metric"] == "attacker_share" else "Honest cost",
            *[number(None if estimate is None else estimate[field], 100) for field in ["mean", "lower", "upper"]],
            number(row["condition_seed_test_holm_p"]), number(row["seed_test_holm_p"])])
    table("extensions_paired_contrasts", "lllrrrrr",
          ["Rule", "Base", "Outcome", "Diff (pp)", "CI low", "CI high", "Cell p", "Seed p"], contrasts,
          context=result["device"].upper()+"; extension minus baseline; seed CI; Holm p; dependent cells")
    for metric in ["attacker_share", "honest_false_penalty_rate"]:
        for attack in ATTACKS:
            records = [row for row in result["reward_intervals"] if row["horizon"] == "main" and row["attack"] == attack]
            if records:
                table("extensions_"+metric+"_"+attack, "llrrrr",
                      ["Condition", "Rule", r"Mean (\%)", "SD (pp)", "CI low", "CI high"],
                      [[condition_label(row, tex=True), rule, *interval_cells(row["metrics"][rule][metric])]
                       for row in records for rule in result["rules"]], context=context)
        records = [row for row in result["reward_intervals"] if row["horizon"] == "long"]
        if records:
            table("extensions_long_"+metric, "lrrrr", ["Rule", r"Mean (\%)", "SD (pp)", "CI low", "CI high"],
                  [[rule, *interval_cells(row["metrics"][rule][metric])] for row in records for rule in result["rules"]], context=context)
    for horizon in ["main", "long"]:
        def value(row, field, key="mean"):
            estimate = row["metrics"][field]["estimate"]
            return "Undefined" if estimate is None else f"{estimate[key]:.4f}"
        for subset in ['all_noise', 'eligible_noise']:
            rows = [row for row in result['bias_intervals'] if row['horizon'] == horizon and row['subset'] == subset]
            if rows:
                table("extensions_bias_"+horizon+('_eligible' if subset == 'eligible_noise' else ''), "llrrrrr",
                      ["Condition", "Rule", "Noise count", "Inclusive", "Excluding", "Difference CI low", "CI high"],
                      [[condition_label(row, tex=True), row["family"], row["participations"], value(row, "mean_cosine"),
                        value(row, "mean_excluding_cosine"), value(row, "mean_inclusive_minus_excluding", "lower"),
                        value(row, "mean_inclusive_minus_excluding", "upper")] for row in rows],
                      context=context+('; eligible noise only' if subset == 'eligible_noise' else '; all noise'))
