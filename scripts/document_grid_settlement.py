"""Render complete settlement coverage and definitions from the shared summary."""

from pathlib import Path

from evaluation import read


def checked_coverage(result):
    if (result.get("complete") is not True or result.get("exact") is not True
            or sum(row["runs"] for row in result["by_rule"].values()) != result["jobs"]
            or sum(row["round_rule_cases"] for row in result["by_rule"].values()) != result["round_rule_cases"]):
        raise ValueError("complete exact settlement coverage is required for presentation")
    return result


def main():
    result = checked_coverage(read("results/summary.json")["grid_settlement"])
    lines = ["# Grid settlement and retained balances", "",
             "Generated from `results/summary.json` by `python scripts/document_grid_settlement.py`.", "",
             f"All {result['jobs']} declared run-rule jobs reproduce {result['round_rule_cases']} round-rule states exactly, "
             f"using {result['grid_runs']} common {result['training_device'].upper()} training runs. "
             "The chain replay itself uses a local CPU process. It does not train another model.", "",
             "## Coverage", "", "| Rule | Replayed runs | Round-rule states |", "|---|---:|---:|"]
    table = [r"\begin{tabular}{lrr}", r"\toprule", r"Rule & Runs & Round-rule states \\", r"\midrule"]
    for rule, row in result["by_rule"].items():
        lines.append(f"| {rule} | {row['runs']} | {row['round_rule_cases']} |")
        table.append(f"{rule} & {row['runs']} & {row['round_rule_cases']} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    lines += ["", "B1 and Mstar include every planned main and long-horizon run. B0 and B2 use the predeclared sample. "
              "Their smaller coverage is explicit; aggregate balances across these unequal samples are not comparable performance estimates.", "",
              f"All replayed contracts use the frozen vesting delay of {result['vesting_rounds']} rounds and the same initial economic settings. "
              "The replay does not include the reference-only and EMA-only economic ablation states. "
              "The ablation's allocation metrics remain in the common grid summary.", "",
              "## Allocation, burns and retained wealth", "",
              "[Per-run economic records](../results/grid_onchain/economic_outcomes.csv) retain separate honest and adversary groups, "
              "condition, seed, horizon, rule and original run path. Every amount is an integer token base-unit value. "
              "They come from the validated on-chain terminal state and original integer payout ledger.", "",
              "- `minted_rewards`: rewards issued during the run, before later burns. The required attacker-share metric uses these historical amounts.",
              "- `claimed_rewards`: reward tokens already transferred into client wallets by the recorded claim schedule.",
              "- `claimable_rewards`: matured rewards still held by the pool and available to claim.",
              "- `unvested_rewards`: surviving reward tranches whose unlock rounds have not yet settled.",
              "- `retained_rewards`: claimed plus claimable plus unvested rewards. This includes locked balances, not only immediately spendable rewards.",
              "- `burned_rewards`: historical minted rewards minus retained rewards.",
              "- `remaining_stake` and `burned_stake`: surviving initial collateral and its loss, kept separate from reward burns.", "",
              "The decomposition conserves issuance exactly: minted rewards equal retained rewards plus reward burns; "
              "initial stake equals remaining stake plus stake burns. Reward and stake burns together equal recorded contract burns. "
              "Remaining stake plus retained rewards equals final token supply. "
              "Claimable rewards, unvested rewards and stake equal the pool's token balance.", "",
              "A run with no minted rewards retains an undefined minted attacker share. Slashing can reduce retained wealth "
              "without changing historical minted allocation. Claim timing can change the split between wallets and claimable balances "
              "without changing their sum. Final unvested balances are retained liabilities, not claims that a withdrawal has occurred; "
              "later unlock rounds are outside the recorded horizon.", "",
              "## Reconstruction", "", "```bash", "python scripts/replay_grid.py --no-run",
              "python scripts/summarize_grid_settlement.py", "python scripts/document_grid_settlement.py", "```", "",
              "These commands validate existing evidence without launching chains or downloading data. Full unique coverage, "
              "the completed training clock, published method, original source/configuration bindings and seeded attack identities "
              "are required. Floating participation shares reconstruct the integer ledger before every recorded state is compared "
              "with the independent settlement oracle. Refreshing hashes cannot excuse a different configuration, incorrect balance, "
              "duplicated job or missing run.", "",
              "The compact [coverage table](../paper/generated/settlement_coverage.tex) requires `booktabs` and fits a single column. "
              "Exact monetary records remain in the CSV and shared JSON summary. Coverage counts establish successful reconstruction "
              "of these recorded histories; they do not prove honest off-chain scores, private data, unique identities, adaptive-attack "
              "security or a general smart-contract safety theorem. See [contract assumptions](CONTRACTS.md).", ""]
    Path("docs").mkdir(exist_ok=True)
    Path("paper/generated").mkdir(parents=True,exist_ok=True)
    Path("docs/GRID_SETTLEMENT.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    Path("paper/generated/settlement_coverage.tex").write_text("\n".join(table)+"\n",encoding="utf-8",newline="\n")
    print(f"Generated exact settlement coverage: {result['jobs']} jobs, {result['round_rule_cases']} states")


if __name__ == "__main__":
    main()
