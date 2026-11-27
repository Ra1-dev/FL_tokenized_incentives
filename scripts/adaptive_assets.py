"""Summary, figure, tables and note for the adaptive-attacker study.

Only seeds whose every declared cell completed are analyzed; scenarios are
averaged within each seed and whole seeds are resampled, as in the published
comparison.
"""

import csv
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation_assets import COLORS
from fltoken.io import sha256, write_json
from fltoken.statistics import rational, sampled_bootstrap_mean
from make_figures import save
from run_adaptive_study import ATTACKS, PLAN, study_cells, read_plan
from sensitivity_assets import OTHER_COLORS

SUMMARY = Path("results/adaptive_study/summary.json")
METRICS = ["attacker_share", "honest_false_penalty_rate", "final_accuracy"]
NAMES = {"alie": "ALIE (full knowledge)", "mimic": "Mimic (previous aggregate)"}


def recorded(plan):
    """Metrics of completed cells, grouped by seed; verifies each cell against the plan."""
    by_seed = {}
    for cell in study_cells(plan):
        folder = Path(cell["path"])
        if not (folder/"metadata.json").exists():
            continue
        metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        if metadata["status"] != "complete":
            continue
        config = metadata["config"]
        if (metadata.get("study_cell") != cell or metadata.get("plan_sha256") != sha256(PLAN) or metadata.get("smoke") or
                config["attack"] != cell["attack"] or config["fraction"] != cell["fraction"] or config["seed"] != cell["seed"] or
                config["rounds"] != plan["rounds"] or config["rules"] != plan["rules"]):
            raise ValueError(f"recorded cell differs from the declared plan: {folder}")
        metrics = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        log = json.loads((folder/"attack_log.json").read_text(encoding="utf-8"))
        if len(log) != plan["rounds"]:
            raise ValueError(f"attack log is incomplete: {folder}")
        by_seed.setdefault(cell["seed"], {})[(cell["attack"], cell["fraction"])] = dict(
            metrics={rule: {metric: metrics[rule][metric] for metric in METRICS} for rule in plan["rules"]},
            headcount=metrics["B0"]["headcount_share"], attacked_rounds=sum(entry["attackers"] > 0 for entry in log),
            git_commit=metadata["git_commit"], source_git_dirty=metadata["source_git_dirty"], device=config["device"],
            wall_seconds=metadata["wall_seconds"])
    expected = len(ATTACKS)*len(plan["fractions"])
    return {seed: cells for seed, cells in sorted(by_seed.items()) if len(cells) == expected}


def estimate(values, statistics):
    return sampled_bootstrap_mean(values, statistics["confidence"], statistics["bootstrap_resamples"], statistics["bootstrap_seed"])


def summarize(plan, complete):
    if len(complete) < 2:
        raise ValueError("at least two complete seeds are required")
    stats = plan["statistics"]
    seeds = list(complete)
    cells, pooled = [], {}
    for attack in ATTACKS:
        for fraction in plan["fractions"]:
            row = dict(attack=attack, fraction=fraction,
                       headcount=complete[seeds[0]][(attack, fraction)]["headcount"], rules={})
            for rule in plan["rules"]:
                row["rules"][rule] = {metric: estimate([rational(complete[s][(attack, fraction)]["metrics"][rule][metric]) for s in seeds], stats)
                                      for metric in METRICS}
            cells.append(row)
        pooled[attack] = {}
        for rule in plan["rules"]:
            pooled[attack][rule] = {}
            for metric in METRICS:
                values = [sum((rational(complete[s][(attack, f)]["metrics"][rule][metric]) for f in plan["fractions"]), Fraction(0))
                          / len(plan["fractions"]) for s in seeds]
                pooled[attack][rule][metric] = estimate(values, stats)
    quality = {}
    for attack in ATTACKS:
        values = {True: [], False: []}
        for cell in study_cells(plan):
            if cell["attack"] == attack and cell["seed"] in complete:
                with (Path(cell["path"])/"per_round.csv").open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        values[row["adversary"] == "True"].append(float(row["Mstar_score"]))
        quality[attack] = dict(attacker=sum(values[True])/len(values[True]), honest=sum(values[False])/len(values[False]),
                               attacker_participations=len(values[True]), honest_participations=len(values[False]))
    clean = [json.loads((Path(cell["path"])/"metrics.json").read_text(encoding="utf-8"))["B0"]["final_accuracy"]
             for cell in json.loads(Path("results/grid/manifest.json").read_text(encoding="utf-8"))["cells"]
             if cell["horizon"] == "main" and cell["attack"] == "none" and cell["seed"] in complete]
    records = [record for cells_ in complete.values() for record in cells_.values()]
    return dict(schema=1, plan_sha256=sha256(PLAN), seeds=seeds, complete_cells=len(records),
                declared_cells=len(study_cells(plan)), devices=sorted({r["device"] for r in records}),
                source_commits=sorted({r["git_commit"] for r in records}),
                any_dirty_source=any(r["source_git_dirty"] for r in records),
                wall_seconds=sum(r["wall_seconds"] for r in records), cells=cells, pooled=pooled,
                mstar_quality=quality, clean_accuracy=dict(seeds=len(clean), mean=sum(clean)/len(clean)))


def figure(plan, summary):
    rules = plan["rules"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    width = .8/len(rules)
    top = 0
    for ax, attack in zip(axes, ATTACKS):
        rows = [row for row in summary["cells"] if row["attack"] == attack]
        for offset, rule in enumerate(rules):
            color = COLORS.get(rule, OTHER_COLORS.get(rule))
            for x, row in enumerate(rows):
                e = row["rules"][rule]["attacker_share"]
                position = x - .4 + width*(offset+.5)
                ax.bar(position, 100*e["mean"], width, color=color, label=rule if x == 0 else None, zorder=2)
                ax.errorbar(position, 100*e["mean"], yerr=[[100*(e["mean"]-e["lower"])], [100*(e["upper"]-e["mean"])]],
                            fmt="none", ecolor="#333333", elinewidth=.6, capsize=1.2, zorder=3)
                top = max(top, 100*e["upper"])
        for x, row in enumerate(rows):
            ax.hlines(100*row["headcount"], x-.45, x+.45, colors="#333333", linestyles="--", lw=.8, zorder=4)
        ax.set_xticks(range(len(rows)), [f"{round(100*row['fraction'])}%" for row in rows])
        ax.set_xlabel("Attacker fraction")
        ax.set_title(NAMES[attack], fontsize=9)
    axes[0].set_ylabel("Attacker reward share (%)")
    axes[0].set_ylim(0, top*1.12)
    axes[1].legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")
    save(fig, "adaptive_attacker_share")


def interval(e, scale=100):
    return r"\shortstack{"+f"{scale*e['mean']:.2f}"+r"\\{}"+f"[{scale*e['lower']:.2f}, {scale*e['upper']:.2f}]"+"}"


def tables(plan, summary):
    rules = plan["rules"]
    note = r"\multicolumn{%d}{l}{Means and 95\%% seed CIs (\%%)} \\" % (len(rules)+2)
    lines = ["% Generated from results/adaptive_study/summary.json; requires booktabs.",
             r"\begin{tabular}{ll"+"r"*len(rules)+"}", r"\toprule", note,
             "Attack & Fraction & "+" & ".join(rules)+r" \\", r"\midrule"]
    for row in summary["cells"]:
        lines.append(" & ".join([NAMES[row["attack"]].split(" ")[0], f"{round(100*row['fraction'])}\\%"] +
                                [interval(row["rules"][rule]["attacker_share"]) for rule in rules])+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    folder = Path("paper/generated")
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"adaptive_attacker_share.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    lines = ["% Generated from results/adaptive_study/summary.json; requires booktabs.",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{ALIE (full knowledge)} & \multicolumn{2}{c}{Mimic (previous aggregate)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Rule & Attacker share & Honest low reward & Attacker share & Honest low reward \\", r"\midrule"]
    for rule in rules:
        lines.append(" & ".join([rule] + [interval(summary["pooled"][attack][rule][metric]) for attack in ATTACKS
                                          for metric in ["attacker_share", "honest_false_penalty_rate"]])+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (folder/"adaptive_pooled.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def document(plan, summary):
    pct = lambda attack, rule, metric="attacker_share": 100*summary["pooled"][attack][rule][metric]["mean"]
    fair = 100*sum(row["headcount"] for row in summary["cells"] if row["attack"] == "alie")/len(plan["fractions"])
    lines = ["# Adaptive attackers", "",
             "Generated by `python scripts/adaptive_assets.py` from the runs declared in `configs/adaptive_study.json`.", "",
             "## Design", "",
             "The main comparison uses fixed attacks. This study asks what happens when attackers adapt to similarity-based rewards. "
             "It reuses the unchanged server protocol, client training, aggregation, frozen Mstar scorer and published CGSV/RFFL "
             "parameters; `scripts/run_adaptive_study.py` replaces only the reported rows of attackers sampled in a round.", "",
             "- **ALIE (full knowledge)** after Baruch et al.: every sampled attacker reports the coordinate-wise mean minus z standard "
             "deviations of the honest updates of the same round, with z from the published formula for n sampled clients and m sampled "
             "attackers. With one attacker in a round z = 0, so the attacker reports the honest mean. Knowing other clients' current "
             "updates is an upper bound: commit-reveal is designed to prevent exactly this.",
             "- **Mimic (public information)**: a free-rider reports the previous round's aggregate update without training. It needs "
             "only public history, so commit-reveal does not prevent it.", "",
             f"Attackers are {', '.join(f'{round(100*f)}%' for f in plan['fractions'])} of {30} clients; "
             f"{len(summary['seeds'])} seeds ({', '.join(map(str, summary['seeds']))}) with all {len(ATTACKS)*len(plan['fractions'])} cells "
             f"complete are analyzed ({summary['complete_cells']} of {summary['declared_cells']} declared runs). "
             "Pooling averages the three fractions within each seed and resamples seeds.", "",
             "## Findings", ""]
    for attack in ATTACKS:
        ranked = sorted(plan["rules"], key=lambda rule: pct(attack, rule))
        lines.append(f"- **{NAMES[attack]}**: pooled attacker share Mstar {pct(attack, 'Mstar'):.2f}%, "
                     f"B1 {pct(attack, 'B1'):.2f}%, B2 {pct(attack, 'B2'):.2f}%, CGSV {pct(attack, 'CGSV'):.2f}%, "
                     f"RFFL {pct(attack, 'RFFL'):.2f}% against an equal-split share of {pct(attack, 'B0'):.2f}% "
                     f"(average headcount {fair:.2f}%). Lowest: {ranked[0]}; highest: {ranked[-1]}. "
                     f"Final model accuracy, identical for every rule: {pct(attack, 'B0', 'final_accuracy'):.2f}%.")
    q = summary["mstar_quality"]
    lines += [f"- Honest low-reward rate under Mstar rises to {pct('alie', 'Mstar', 'honest_false_penalty_rate'):.2f}% (ALIE) and "
              f"{pct('mimic', 'Mstar', 'honest_false_penalty_rate'):.2f}% (mimic), against "
              f"{pct('alie', 'CGSV', 'honest_false_penalty_rate'):.2f}% and {pct('mimic', 'CGSV', 'honest_false_penalty_rate'):.2f}% for CGSV.",
              f"- Mechanism: the leave-one-out median sits near the centre of the honest updates. Its mean quality for attackers is "
              f"{q['alie']['attacker']:.3f} under ALIE and {q['mimic']['attacker']:.3f} under mimic, against {q['alie']['honest']:.3f} and "
              f"{q['mimic']['honest']:.3f} for honest non-IID clients in the same runs. Updates placed at the consensus therefore outscore "
              "genuine but heterogeneous training. All attackers in a round report the same vector, so the other attackers also pull an "
              "attacker's leave-one-out median toward it.",
              f"- Neither attack noticeably harms the model here: final accuracy is {pct('alie', 'B0', 'final_accuracy'):.2f}% and "
              f"{pct('mimic', 'B0', 'final_accuracy'):.2f}% against {100*summary['clean_accuracy']['mean']:.2f}% without attackers "
              f"({summary['clean_accuracy']['seeds']} matching seeds of the main grid). The attacks steal rewards rather than damage training.",
              "", "A share above the equal split means the rule pays the attackers more than their headcount; the rule then rewards "
              "the adaptive behavior instead of resisting it. The noise loophole closed by self-exclusion is replaced by a consensus-mimicry "
              "loophole that commit-reveal cannot prevent for the public-information attack.", "",
              "## Scope", "",
              "All rules score one common update matrix per round, so the training trajectory and accuracy are identical across rules "
              "within a run. No rule was tuned for these attacks, and both attacks are deliberately simple; stronger adaptive attacks remain possible. "
              f"Runs: devices {', '.join(summary['devices'])}; source commits {', '.join(c[:7] for c in summary['source_commits'])}"
              f"{'; some runs had uncommitted sources' if summary['any_dirty_source'] else ''}.", ""]
    Path("docs/ADAPTIVE.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    plan = read_plan()
    summary = summarize(plan, recorded(plan))
    write_json(SUMMARY, summary)
    figure(plan, summary)
    tables(plan, summary)
    document(plan, summary)
    print(json.dumps(dict(seeds=summary["seeds"], complete_cells=summary["complete_cells"],
                          pooled={a: {r: round(100*summary["pooled"][a][r]["attacker_share"]["mean"], 2) for r in plan["rules"]}
                                  for a in ATTACKS}), indent=1))


if __name__ == "__main__":
    main()
