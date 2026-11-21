"""Seed-level honest-cost and analytic-accounting assets from complete noisy runs."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from decimal import Decimal
from math import ceil, floor
from evaluation_assets import interval_cells


DISPLAY = [("B1", "Inclusive cosine", "#176B87", "o"),
           ("Mstar", "Full excluding rule", "#B4473D", "s"),
           ("Aref", "Excluding reference only", "#6D56A0", "^")]


def display(result):
    if 'rules' not in result:
        return DISPLAY
    from evaluation_assets import COLORS, MARKERS
    colors = dict(COLORS, CGSV='#668D55', RFFL='#C17F25')
    markers = dict(MARKERS, CGSV='v', RFFL='P')
    return [(rule, rule, colors[rule], markers[rule]) for rule in result['rules']]


def privacy_figures(result, save):
    selected = display(result)
    coordinate = 'coordinate_standard_deviation' in result['cells'][0]
    positions = list(range(len(result['cells']))) if coordinate else [row['noise_multiplier'] for row in result['cells']]
    offsets = [(.6*index/(len(selected)-1)-.3) if coordinate else (.07*index/(len(selected)-1)-.035)
               for index in range(len(selected))]
    xlabel = 'Noise SD per coordinate' if coordinate else 'Per-coordinate noise multiplier'
    labels = [f"{row['coordinate_standard_deviation'] if coordinate else row['noise_multiplier']:g}" for row in result['cells']]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    def draw(ax, x, estimate, color, marker):
        low, high = 100*estimate["lower"], 100*estimate["upper"]
        ax.plot([x, x], [low, high], color=color, lw=1)
        for value in [low, high]:
            ax.plot([x-.015, x+.015], [value, value], color=color, lw=1)
        ax.plot(x, 100*estimate["mean"], marker, color=color, ms=4, mfc="white")
    undefined = False
    for offset, (rule, label, color, marker) in zip(offsets, selected):
        for position, row in zip(positions, result["cells"]):
            x = position+offset
            estimate = row["metrics"][rule]["honest_false_penalty_rate"]["estimate"]
            if estimate is None:
                undefined = True
                axes[0].plot(x, 1.02, "x", transform=axes[0].get_xaxis_transform(), color=color, clip_on=False)
            else:
                draw(axes[0], x, estimate, color, marker)
    # Every rule shares one model trajectory; display its accuracy only once.
    for position, row in zip(positions, result["cells"]):
        estimate = row["metrics"]["B1"]["final_accuracy"]["estimate"]
        if estimate is None:
            undefined = True
            axes[1].plot(position, 1.02, "x", transform=axes[1].get_xaxis_transform(), color="#64748B", clip_on=False)
        else:
            draw(axes[1], position, estimate, "#334155", "o")
    axes[0].set_ylabel("Honest false penalties (%)")
    axes[1].set_ylabel("Final test accuracy (%)")
    for ax in axes:
        ax.set(xlabel=xlabel, ylim=(-2, 102), xticks=positions, xticklabels=labels)
        ax.grid(alpha=.15)
    if coordinate:
        costs = [row['metrics'][rule]['honest_false_penalty_rate']['estimate']
                 for row in result['cells'] for rule, *_ in selected]
        accuracy = [row['metrics']['B1']['final_accuracy']['estimate'] for row in result['cells']]
        axes[0].set_ylim(-1, max(5, 5*ceil(max((value['upper']*100 for value in costs if value), default=0)/5)))
        axes[1].set_ylim(min(95, 5*floor(min((value['lower']*100 for value in accuracy if value), default=0)/5)), 100)
    handles = [Line2D([], [], color=color, marker=marker, ls="", ms=4, label=label) for _, label, color, marker in selected]
    if undefined:
        handles.append(Line2D([], [], color="#64748B", marker="x", ls="", ms=4, label="Undefined; all seeds retained"))
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=7, ncol=2 if undefined else 3)
    fig.tight_layout(rect=(0, 0, 1, .88))
    save(fig, "privacy_false_penalty")
    if coordinate:
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        for offset, (rule, label, color, marker) in zip(offsets, selected):
            for position, row in zip(positions, result['cells']):
                estimate = row['metrics'][rule]['attacker_share']['estimate']
                if estimate is None:
                    ax.plot(position+offset, 1.02, 'x', transform=ax.get_xaxis_transform(), color=color, clip_on=False)
                else:
                    draw(ax, position+offset, estimate, color, marker)
        ax.set(xlabel=xlabel, ylabel='Attacker share (%)', ylim=(-2, 102), xticks=positions, xticklabels=labels)
        shares = [row['metrics'][rule]['attacker_share']['estimate']
                  for row in result['cells'] for rule, *_ in selected]
        ax.set_ylim(-1, max(5, 5*ceil(max((value['upper']*100 for value in shares if value), default=0)/5)))
        ax.grid(alpha=.15)
        fig.legend(handles=handles, loc='upper center', frameon=False, fontsize=7, ncol=3)
        fig.tight_layout(rect=(0, 0, 1, .80))
        save(fig, 'privacy_attacker_share')


def privacy_tables(result, table):
    coordinate = 'coordinate_standard_deviation' in result['cells'][0]
    variable = r'$\tau$' if coordinate else r'$\sigma$'
    scale = lambda row:f"{row['coordinate_standard_deviation'] if coordinate else row['noise_multiplier']:g}"
    rules = [row[0] for row in display(result)]
    comparisons = [rule for rule in rules if rule != 'B1']
    context = result["training_device"].upper()+"; seed CIs; failed seeds retained"
    table("privacy_honest_false_penalty_rate", "rlrrrr", [variable, "Rule", r"False penalty (\%)", "SD (pp)", "CI low", "CI high"],
          [[scale(row), rule, *interval_cells(row["metrics"][rule]["honest_false_penalty_rate"])]
           for row in result["cells"] for rule in rules], context=context)
    table("privacy_final_accuracy", "rrrrr", [variable, r"Accuracy (\%)", "SD (pp)", "CI low", "CI high"],
          [[scale(row), *interval_cells(row["metrics"]["B1"]["final_accuracy"])]
           for row in result["cells"]], context="Shared trajectory; "+context)
    table("privacy_paired_penalties", "rlrrrr", [variable, "Rule minus B1", "Mean (pp)", "SD (pp)", "CI low", "CI high"],
          [[scale(row), rule, *interval_cells(row["paired_false_penalty_difference_from_B1"][rule])]
           for row in result["cells"] for rule in comparisons], context=context)
    def epsilon(value):
        return r"$\infty$" if value is None else f"{value:.4g}"
    table("privacy_accounting", "rrrrrr", [variable, "Seed", "Min releases", "Max releases", r"Max client $\epsilon$", r"All-round $\epsilon$"],
          [[scale(row), seed["seed"], seed["minimum_participations"], seed["maximum_participations"],
            epsilon(seed["maximum_client_epsilon"]), epsilon(row["conservative_all_rounds"]["epsilon"])]
           for row in result["cells"] for seed in row["per_seed_accounting"]],
          context=rf"Ideal full-schedule bounds; $\delta={format(Decimal(str(result['delta'])), 'f')}$; excludes numerical failure signals")
    if all("outcomes" in row for row in result["cells"]):
        table("privacy_outcomes", "rrrll", [variable, "Seed", "Rounds", "Outcome", "Failure stage"],
              [[scale(row), cell["seed"], cell["completed_rounds"],
                "Complete" if cell["status"] == "complete" else "Numerical failure",
                ", ".join(sorted({item["stage"].replace("_", " ") for item in cell["failure"]["failures"]})) if cell["failure"] else "---"]
               for row in result["cells"] for cell in row["outcomes"]],
              context="Failed-run rounds are diagnostic prefixes; every seed is retained")
    if coordinate:
        table('privacy_attacker_share', 'rlrrrr', [variable, 'Rule', r'Share (\%)', 'SD (pp)', 'CI low', 'CI high'],
              [[scale(row), rule, *interval_cells(row['metrics'][rule]['attacker_share'])]
               for row in result['cells'] for rule in rules], context=context)
        table('privacy_noise_scale', 'rrr', [variable, r'$z=\tau/C$', 'Actual SD'],
              [[scale(row), f"{row['noise_multiplier']:.7g}",
                f"{row['per_seed_accounting'][0]['accounting']['noise_standard_deviation']:.7g}"] for row in result['cells']],
              context='Unchanged clipping bound; no dimension rescaling')
        table('privacy_divergence', 'rrrll', [r'$\sigma$', 'Seed', 'Rounds', 'Outcome', 'Failure stage'],
              [[f"{row['noise_multiplier']:g}", row['seed'], row['completed_rounds'],
                'Complete' if row['status'] == 'complete' else 'Numerical failure',
                ', '.join(sorted({item['stage'].replace('_', ' ') for item in row['failure']['failures']})) if row['failure'] else '---']
               for row in result['original_multiplier_study']['outcomes']],
              context='Original multipliers; failed prefixes only; controls reused')
