"""Publication figures generated exclusively from the shared result summary."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

BLUE, RED, GRAY = "#176B87", "#B4473D", "#475569"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})


def save(fig, name):
    folder = Path("figures")
    folder.mkdir(exist_ok=True)
    for extension in ["png", "pdf"]:
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {}
        fig.savefig(folder / f"{name}.{extension}", bbox_inches="tight", facecolor="white", metadata=metadata)
    plt.close(fig)


def architecture(summary):
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    boxes = [(.2, 1.2, 2.3, 1.6, "Sampled clients\nLocal SGD + attacks"),
             (3.1, 2.2, 2.8, 1.3, "Fixed FedAvg\nGlobal model\nCentral evaluation"),
             (3.1, .3, 2.8, 1.3, "Common updates\nShadow reward rules"),
             (7.0, 1.2, 2.8, 1.6, "Logged scores + shares\nInteger payout ledger")]
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1,
                                  edgecolor=BLUE, facecolor="#F0F7FA"))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", color="#162737")
    for start, end in [((2.6, 2.3), (3.0, 2.8)), ((2.6, 1.6), (3.0, 1.0)), ((6.0, 1.0), (6.9, 1.6))]:
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))
    ax.text(6.6, 3.3, "Rewards do not alter\nthe training trajectory", ha="left", va="center", color=GRAY)
    save(fig, "architecture")


def reproduction(result):
    seeds = [r["seed"] for r in result["per_seed"]]
    fig, axes = plt.subplots(1, len(seeds), figsize=(7.1, 2.6), sharey=True, squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        rows = [r for r in result["quality_trajectory"] if r["seed"] == seed]
        rounds = [r["round"] for r in rows]
        ax.plot(rounds, [r["honest_mean_q"] for r in rows], "o-", ms=2.5, lw=1.2, color=BLUE, label="Honest")
        ax.plot(rounds, [r["noise_mean_q"] if r["noise_mean_q"] is not None else np.nan for r in rows], "s-", ms=2.5, lw=1.2, color=RED, label="Noise")
        ax.set(title=f"Seed {seed}", xlabel="FL round", ylim=(0, 100))
        ax.grid(alpha=.15)
    axes[0, 0].set_ylabel("Mean cosine quality score")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "reproduction_trajectory")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    rows = result["noise_observations"]
    predicted = np.array([r["predicted"] for r in rows])
    measured = np.array([r["measured"] for r in rows])
    for seed, marker in zip(seeds, ["o", "s", "^"]):
        mask = np.array([r["seed"] == seed for r in rows])
        axes[0].scatter(predicted[mask], measured[mask], s=18, alpha=.75, marker=marker, label=f"Seed {seed}")
    maximum = max(predicted.max(), measured.max()) * 1.05
    axes[0].plot([0, maximum], [0, maximum], "--", color=GRAY, lw=1)
    axes[0].set(xlabel="Predicted self-inclusion cosine", ylabel="Measured cosine", xlim=(0, maximum), ylim=(0, maximum))
    axes[0].legend(frameon=False, fontsize=7)
    errors = sorted(100*r["relative_error"] for r in rows if r["relative_error"] is not None)
    axes[1].step(errors, np.arange(1, len(errors)+1)/len(errors), where="post", color=BLUE)
    axes[1].set(xlabel="Absolute relative error (%)", ylabel="Cumulative participation fraction", ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig, "reproduction_bias")
    rules = list(result["per_seed"][0]["baseline_attacker_shares"])
    values = np.array([[row["baseline_attacker_shares"][rule] for rule in rules]
                       for row in result["per_seed"]]) * 100
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x = np.arange(len(rules))
    ax.bar(x, values.mean(axis=0), color=[GRAY, RED, BLUE], alpha=.75,
           yerr=values.std(axis=0, ddof=1), capsize=4)
    for offset, row in zip(np.linspace(-.09, .09, len(values)), values):
        ax.scatter(x+offset, row, s=15, color="#111827", zorder=3)
    ax.axhline(100*result["published"]["attacker_share_mean"], color=RED, ls="--", lw=1,
               label="Published cosine mean")
    ax.set(xticks=x, xticklabels=rules, ylabel="Attacker share of minted rewards (%)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    save(fig, "reproduction_attacker_share")


def main():
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    architecture(summary)
    if "reproduction" in summary:
        reproduction(summary["reproduction"])
    print("Figures generated from results/summary.json")


if __name__ == "__main__":
    main()
