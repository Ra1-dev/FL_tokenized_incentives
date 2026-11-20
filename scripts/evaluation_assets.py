"""Plot and typeset the complete comparison without inventing missing estimates."""

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ATTACKS = {"none": "No attackers", "zero": "Zero", "noise": "Noise", "stale": "Stale",
           "label_flip": "Label flip", "sign_flip": "Sign flip", "sendros_mix": "Fixed mixture"}
COLORS = {"B0": "#697386", "B1": "#B4473D", "B2": "#6D56A0", "Mstar": "#176B87",
          "Aref": "#668D55", "Aema": "#C17F25", "Afull": "#176B87"}
MARKERS = {"B0": "o", "B1": "s", "B2": "^", "Mstar": "D", "Aref": "o", "Aema": "s", "Afull": "D"}


def condition_label(row, tex=False):
    attack = row["attack"]
    label = ATTACKS[attack]
    if attack not in {"none", "sendros_mix"}:
        label += f" {100*row['fraction']:g}" + (r"\%" if tex else "%")
    return label


def point_interval(ax, estimate, y, color, label=None, marker="o"):
    if estimate is None:
        ax.plot(1.02, y, "x", transform=ax.get_yaxis_transform(), ms=3, color=color, clip_on=False)
        return
    lo, hi, mean = [100*estimate[key] for key in ["lower", "upper", "mean"]]
    ax.plot([lo, hi], [y, y], color=color, lw=1)
    ax.plot([lo, hi], [y, y], "|", color=color, ms=4)
    ax.plot(mean, y, marker, color=color, ms=3, label=label)


def evaluation_figures(result, save):
    main = [row for row in result["intervals"] if row["horizon"] == "main"]
    rules = ["B0", "B1", "B2", "Mstar"]
    for metric, title in [("attacker_share", "Attacker share of minted rewards (%)"),
                          ("honest_false_penalty_rate", "Honest false-penalty rate (%)")]:
        fig, ax = plt.subplots(figsize=(3.5, 5.4))
        offsets = np.linspace(-.27, .27, len(rules))
        upper = 0
        for index, row in enumerate(main):
            for offset, rule in zip(offsets, rules):
                estimate = row["metrics"][rule][metric]["estimate"]
                point_interval(ax, estimate, index+offset, COLORS[rule], rule if index == 0 else None, MARKERS[rule])
                if estimate is not None:
                    upper = max(upper, 100*estimate["upper"])
        ax.set(yticks=range(len(main)), yticklabels=[condition_label(row) for row in main],
               xlabel=title, xlim=(-1, max(30, math.ceil(upper/10)*10)+1))
        ax.tick_params(axis="y", labelsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        handles = [Line2D([],[],color=COLORS[rule],marker=MARKERS[rule],ms=3,label=rule) for rule in rules]
        if any(row["metrics"][rule][metric]["estimate"] is None for row in main for rule in rules):
            handles.append(Line2D([],[],color="#475569",marker="x",ls="",ms=3,label="Undefined"))
        ax.legend(handles=handles,
                  loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        save(fig, "evaluation_"+metric)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    stages = ["Aref", "Aema", "Afull"]
    for ax, metric, title in zip(axes, ["attacker_share", "honest_false_penalty_rate"],
                                ["Attacked conditions", "All conditions"]):
        for index, stage in enumerate(stages):
            point_interval(ax, result["pooled"][metric]["rules"][stage]["estimate"], index, COLORS[stage])
        ax.set(yticks=range(3), yticklabels=["Reference", "+ EMA", "+ Penalties = Mstar"],
               title=title, xlabel=("Attacker share (%)" if metric=="attacker_share" else "Honest false penalties (%)"))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        if any(result["pooled"][metric]["rules"][stage]["estimate"] is None for stage in stages):
            ax.legend(handles=[Line2D([],[],color="#475569",marker="x",ls="",ms=3,label="Undefined")],
                      frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    save(fig, "evaluation_ablation")

    trajectories = result.get("accuracy_trajectories", [])
    if trajectories:
        fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.8), sharey=True)
        palette = ["#176B87", "#668D55", "#B4473D"]
        for ax, attack in zip(axes.flat, ATTACKS):
            rows = [row for row in trajectories if row["horizon"] == "main" and row["attack"] == attack]
            for color, fraction in zip(palette, sorted({row["fraction"] for row in rows})):
                values = sorted((row for row in rows if row["fraction"] == fraction), key=lambda row: row["round"])
                estimates = [row["accuracy"]["estimate"] for row in values]
                if any(value is None for value in estimates):
                    raise ValueError("accuracy curves require all seed observations")
                rounds = [row["round"] for row in values]
                ax.plot(rounds, [100*value["mean"] for value in estimates], color=color, lw=1, label=f"{100*fraction:g}%")
                ax.fill_between(rounds, [100*value["lower"] for value in estimates],
                                [100*value["upper"] for value in estimates], color=color, alpha=.12, linewidth=0)
            ax.set(title=ATTACKS[attack], xlabel="Round", ylim=(0, 100))
            ax.tick_params(labelsize=6)
            ax.grid(alpha=.15)
            if attack not in {"none", "sendros_mix"}:
                ax.legend(frameon=False, fontsize=6, loc="lower right")
        axes[0, 0].set_ylabel("Test accuracy (%)")
        axes[1, 0].set_ylabel("Test accuracy (%)")
        axes[-1, -1].axis("off")
        axes[-1, -1].text(0, .5, "Shared FedAvg trajectory\nMean and seed bootstrap CI\nReward rules do not change training", fontsize=7)
        fig.tight_layout()
        save(fig, "evaluation_accuracy")
        long_rows = sorted((row for row in trajectories if row["horizon"]=="long"), key=lambda row:row["round"])
        if long_rows:
            fig, ax = plt.subplots(figsize=(3.5, 2.6))
            estimates=[row["accuracy"]["estimate"] for row in long_rows]
            if any(value is None for value in estimates):
                raise ValueError("long accuracy curve requires all seeds")
            rounds=[row["round"] for row in long_rows]
            ax.plot(rounds,[100*v["mean"] for v in estimates],color=COLORS["Mstar"],lw=1.2)
            ax.fill_between(rounds,[100*v["lower"] for v in estimates],[100*v["upper"] for v in estimates],
                            color=COLORS["Mstar"],alpha=.15,linewidth=0)
            ax.set(title="Fixed mixture, long horizon",xlabel="FL round",ylabel="Shared test accuracy (%)",ylim=(0,100))
            ax.grid(alpha=.15)
            fig.tight_layout()
            save(fig,"evaluation_accuracy_long")


def interval_cells(record, scale=100):
    estimate = record["estimate"]
    if estimate is None:
        return ["Undefined"]*4
    return [f"{scale*estimate[key]:.3f}" for key in ["mean", "sample_sd", "lower", "upper"]]


def evaluation_tables(result, table):
    emit = lambda name, columns, header, rows: table(name,columns,header,rows,context="Device: "+result["device"].upper())
    for metric in ["attacker_share", "honest_false_penalty_rate"]:
        emit("evaluation_pooled_"+metric, "lrrrr", ["Rule", r"Mean (\%)", "SD (pp)", "CI low", "CI high"],
              [[rule, *interval_cells(result["pooled"][metric]["rules"][rule])]
               for rule in ["B0", "B1", "B2", "Mstar", "Aref", "Aema", "Afull"]])
        for attack in ATTACKS:
            rows = [row for row in result["intervals"] if row["horizon"] == "main" and row["attack"] == attack]
            emit("evaluation_"+metric+"_"+attack, "llrrrr",
                  ["Condition", "Rule", r"Mean (\%)", "SD (pp)", "CI low", "CI high"],
                  [[condition_label(row,tex=True), rule, *interval_cells(row["metrics"][rule][metric])]
                   for row in rows for rule in ["B0", "B1", "B2", "Mstar"]])
        for row in (row for row in result["intervals"] if row["horizon"]=="long"):
            emit("evaluation_long_"+metric,"lrrrr",["Rule",r"Mean (\%)","SD (pp)","CI low","CI high"],
                 [[rule,*interval_cells(row["metrics"][rule][metric])] for rule in result["rules"]])
    emit("evaluation_paired", "llrrrr", ["Cost", "Baseline", "Difference (pp)", "Rank effect", "p", "Holm p"],
          [["Attacker" if row["metric"]=="attacker_share" else "Honest", row["baseline"],
            *([f"{100*row['condition_seed_test']['mean_difference']:.3f}",
               f"{row['condition_seed_test']['rank_biserial']:.3f}",
               f"{row['condition_seed_test']['p_value']:.4g}", f"{row['condition_seed_test_holm_p']:.4g}"]
              if row["condition_seed_test"] else ["Undefined"]*4)] for row in result["contrasts"]])
