"""Transfer comparison assets from complete seed summaries."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from math import ceil

from evaluation_assets import COLORS, MARKERS, condition_label, interval_cells, point_interval

COLORS = dict(COLORS, CGSV='#668D55', RFFL='#C17F25')
MARKERS = dict(MARKERS, CGSV='v', RFFL='P')


def displayed_rules(result):
    return [rule for rule in result.get('rules', ['B0', 'B1', 'B2', 'Mstar'])
            if rule in {'B0', 'B1', 'B2', 'Mstar', 'CGSV', 'RFFL'}]


def fashion_figures(result, save):
    rows = result["intervals"]
    rules = displayed_rules(result)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0))
    for ax, metric, label in zip(axes, ["attacker_share", "honest_false_penalty_rate"],
                                 ["Attacker share (%)", "Honest false penalties (%)"]):
        for index, row in enumerate(rows):
            for offset, rule in zip([-.32+.64*i/(len(rules)-1) for i in range(len(rules))], rules):
                point_interval(ax, row["metrics"][rule][metric]["estimate"], index+offset, COLORS[rule], marker=MARKERS[rule])
        ax.set(yticks=range(len(rows)), yticklabels=[condition_label(row) for row in rows],
               xlabel=label, xlim=(-2, 102), ylim=(len(rows)-.5, -.5))
        ax.tick_params(axis="y", labelsize=7)
        if 'ordering' in result:
            estimates = [row['metrics'][rule][metric]['estimate'] for row in rows for rule in rules]
            ax.set_xlim(-1, max(5, 5*ceil(max((value['upper']*100 for value in estimates if value), default=0)/5)))
        ax.grid(axis="x", alpha=.15)
    handles = [Line2D([], [], color=COLORS[rule], marker=MARKERS[rule], ls="", ms=4, label=rule) for rule in rules]
    if any(row["metrics"][rule][metric]["estimate"] is None for row in rows for rule in rules
           for metric in ["attacker_share", "honest_false_penalty_rate"]):
        handles.append(Line2D([], [], color="#64748B", marker="x", ls="", ms=4, label="Undefined"))
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save(fig, "fashion_comparison")

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for color, style, row in zip(["#176B87", "#B4473D"], ["-", "--"], rows):
        values = [item for item in result["accuracy_trajectories"] if item["attack"] == row["attack"] and item["fraction"] == row["fraction"]]
        values.sort(key=lambda item:item["round"])
        estimates = [item["accuracy"]["estimate"] for item in values]
        if any(value is None for value in estimates):
            raise ValueError("transfer accuracy curves require every seed")
        rounds = [item["round"] for item in values]
        ax.plot(rounds, [100*value["mean"] for value in estimates], label=condition_label(row), color=color, lw=1, ls=style)
        ax.fill_between(rounds, [100*value["lower"] for value in estimates],
                        [100*value["upper"] for value in estimates], color=color, alpha=.13, linewidth=0)
    ax.set(xlabel="FL round", ylabel="Shared test accuracy (%)", ylim=(0, 100))
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, "fashion_accuracy")


def fashion_tables(result, table):
    rules = displayed_rules(result)
    context = f"Fashion-MNIST; {result['device'].upper()}; {100*result['statistical_plan']['confidence']:g}\\% seed CI"
    for metric in ["attacker_share", "honest_false_penalty_rate", "honest_gini"]:
        scale = 1 if metric == "honest_gini" else 100
        table("fashion_"+metric, "llrrrr", ["Condition", "Rule", "Mean Gini" if scale == 1 else r"Mean (\%)",
                                           "SD" if scale == 1 else "SD (pp)", "CI low", "CI high"],
              [[condition_label(row, tex=True), rule, *interval_cells(row["metrics"][rule][metric], scale)]
               for row in result["intervals"] for rule in rules], context=context)
    table("fashion_accuracy", "lrrrr", ["Condition", r"Accuracy (\%)", "SD (pp)", "CI low", "CI high"],
          [[condition_label(row, tex=True), *interval_cells(row["metrics"]["B0"]["final_accuracy"])]
           for row in result["intervals"]], context="Shared model; "+context)
    table("fashion_paired", "llrrrr", ["Cost", "Baseline", "Mean (pp)", "SD (pp)", "CI low", "CI high"],
          [["Attacker" if row["metric"] == "attacker_share" else "Honest", row["baseline"], *interval_cells(row["paired_difference"])]
           for row in result["contrasts"]], context="Mstar minus baseline; "+result["device"].upper()+"; seed CI")
    if 'ordering' in result:
        table('fashion_transfer_ordering', 'llrrrr', ['Condition', 'Cost', 'Same', 'Reversed', 'Other changes', 'Undefined'],
              [[condition_label(row, tex=True), 'Attacker' if row['metric'] == 'attacker_share' else 'Honest',
                row['same_order_pairs'], row['reversed_pairs'],
                row['defined_pairs']-row['same_order_pairs']-row['reversed_pairs'], len(row['pairs'])-row['defined_pairs']]
               for row in result['ordering']], context='Matched-seed mean ordering; ties retained; descriptive')
        def number(value):
            return 'Undefined' if value is None else f'{100*value:.3f}'
        table('fashion_matched_comparison', 'lllrrrr', ['Condition', 'Cost', 'Rule', r'MNIST (\%)', r'Fashion (\%)', 'Diff CI low', 'CI high'],
              [[condition_label(row, tex=True), 'Attacker' if row['metric'] == 'attacker_share' else 'Honest', rule,
                number(row['means']['mnist'][rule]), number(row['means']['fashion_mnist'][rule]),
                number(row['fashion_minus_mnist'][rule]['estimate']['lower']) if row['fashion_minus_mnist'][rule]['estimate'] else 'Undefined',
                number(row['fashion_minus_mnist'][rule]['estimate']['upper']) if row['fashion_minus_mnist'][rule]['estimate'] else 'Undefined']
               for row in result['ordering'] for rule in rules], context='Same three seeds and conditions; Fashion minus MNIST; seed CI in pp')
