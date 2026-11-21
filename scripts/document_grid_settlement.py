"""Render complete settlement coverage and definitions from the shared summary."""

from pathlib import Path
from decimal import Decimal

from evaluation import read


def checked_coverage(result):
    if (result.get("complete") is not True or result.get("exact") is not True
            or sum(row["runs"] for row in result["by_rule"].values()) != result["jobs"]
            or sum(row["round_rule_cases"] for row in result["by_rule"].values()) != result["round_rule_cases"]):
        raise ValueError("complete exact settlement coverage is required for presentation")
    component = result["penalty_ablation"]
    expected = {case["path"] for case in result["cases"] if case["rule"] == "Mstar"}
    if (component["runs"] != result["grid_runs"] or len(component["cases"]) != result["grid_runs"]
            or {case["path"] for case in component["cases"]} != expected):
        raise ValueError("complete component settlement coverage is required for presentation")
    return result


def reward_forfeiture_note(result):
    component = result["penalty_ablation"]
    cases = component["cases"]
    total = sum(case["stages"]["Afull"]["totals"]["burned_rewards"] for case in cases)
    affected = sum(case["stages"]["Afull"]["totals"]["burned_rewards"] > 0 for case in cases)
    text = (f"Forfeited unvested rewards total {total} base units across {len(cases)} full-component histories; "
            f"{affected} histories have a positive reward burn.")
    if total == 0 and component["quality_threshold"] == 0 and component["patience"] >= result["vesting_rounds"]:
        text += (f" At the frozen zero quality threshold, {component['patience']} bad participations are needed "
                 f"before a slash, while rewards vest after {result['vesting_rounds']} rounds. Each bad participation "
                 "earns no new reward; the preceding positive reward has matured by the slash, even with gaps "
                 "between participations. The measured penalty effect is therefore entirely stake loss. "
                 "Reward forfeiture does not contribute to this grid's outcomes.")
    return text


def component_assets(result):
    from make_tables import table
    component = result["penalty_ablation"]
    lines = ["## What the penalty component changes", "",
             f"Aema and Afull have identical integer share allocations in {component['identical_allocation_runs']} of "
             f"{component['runs']} runs. At the frozen zero threshold, the flagged current quality and weight are already "
             "zero; additional weight zeroing therefore cannot change minted allocation. Penalties instead burn 10% of "
             "remaining stake and each still-unvested prior reward tranche, rounded down separately. Maturity occurs before "
             "slashing, and the current round's new rewards are created afterward.", "",
             "The EMA-only ledger is settled off-chain with the same contract economics and claims after each round. "
             "The full component uses its recorded ledger and its schedule-independent balances are checked against the "
             "validated Mstar Anvil states. The EMA-only results are a settlement ablation, not additional chain replays.", "",
             reward_forfeiture_note(result), "",
             "[Every component/run/group](../results/grid_onchain/penalty_ablation.csv) retains exact base-unit losses, "
             "net outcomes and event counts. `net_token_outcome` is minted rewards minus forfeited unvested rewards "
             "minus slashed stake: final wealth minus initial collateral. It may be negative. `ever_slashed_clients` "
             "counts distinct flagged clients within a run; `slash_events` counts repeated flags; `clients_with_balance_loss` "
             "counts clients with a positive stake or reward burn.", "",
             "| Horizon | Runs | Honest client-run identities | Honest ever slashed | Attacker client-run identities | Attackers ever slashed |",
             "|---|---:|---:|---:|---:|---:|"]
    count_rows = []
    for horizon, record in component["aggregates"].items():
        groups = record["groups"]["Afull"]
        values = [horizon,record["runs"],groups["honest"]["clients"],groups["honest"]["ever_slashed_clients"],
                  groups["adversary"]["clients"],groups["adversary"]["ever_slashed_clients"]]
        lines.append("| "+" | ".join(map(str,values))+" |")
        count_rows.append(values)
    lines += ["", "Counts sum distinct experimental histories, not unique people across runs. "
              "The main horizon includes attack-free runs; the long horizon contains only the fixed mixture.", "",
              "The following estimates average scenario totals within each seed, then bootstrap the complete seed means. "
              "Monetary estimates are tokens per run for the entire group, not tokens per client. "
              "The CSV preserves integer base units. All estimates include both losses and retained gains.", "",
              "| Horizon / stage / group | Slashed stake, mean [CI] | Forfeited rewards, mean [CI] | Net outcome, mean [CI] |",
              "|---|---:|---:|---:|"]
    fields = ["burned_stake", "burned_rewards", "net_token_outcome", "ever_slashed_clients"]
    for horizon in component["aggregates"]:
        rows = []
        for row in (row for row in component["intervals"] if row["horizon"] == horizon):
            values = [f"{row['metrics'][field]['mean']:.3f} [{row['metrics'][field]['lower']:.3f}, "
                      f"{row['metrics'][field]['upper']:.3f}]" for field in fields]
            lines.append("| "+" / ".join([horizon,row["stage"],row["group"]])+" | "+" | ".join(values[:3])+" |")
            cells = [r"\shortstack{"+f"{row['metrics'][field]['mean']:.3f}"+r"\\{}"+
                     f"[{row['metrics'][field]['lower']:.3f}, {row['metrics'][field]['upper']:.3f}]"+"}" for field in fields]
            rows.append([row["stage"],row["group"],*cells])
        table("settlement_penalties_"+horizon,"llrrrr",
              ["Stage","Group","Stake loss","Reward loss","Net outcome","Ever slashed"],rows,
              context=horizon+r"; tokens or clients per run; mean and 95\% seed CI")
    table("settlement_slashed_clients","lrrrrr",
          ["Horizon","Runs","Honest","Slashed","Attackers","Slashed"],count_rows,
          context="Client-run identities; distinct within each run")
    groups = component["aggregates"]["main"]["groups"]["Afull"]
    incidence_rows = []
    lines += ["", "## Main-horizon stake losses", "",
              "| Group | Client-run identities | Ever slashed | Slash events | Stake burned (tokens) | Reward forfeiture (tokens) |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in ["adversary", "honest"]:
        measured = groups[group]
        row = [group, measured["clients"], measured["ever_slashed_clients"], measured["slash_events"],
               format(Decimal(measured["burned_stake"])/10**18, "f"),
               format(Decimal(measured["burned_rewards"])/10**18, "f")]
        incidence_rows.append(row)
        lines.append("| "+" | ".join(map(str, row))+" |")
    table("settlement_main_incidence", "lrrrrr",
          ["Group", "Client-runs", "Slashed", "Events", "Stake burned", "Rewards burned"], incidence_rows,
          context="Main horizon; exact totals; amounts in tokens")
    attacker, honest = groups["adversary"], groups["honest"]
    if honest["ever_slashed_clients"] and attacker["clients"] and honest["clients"]:
        count_ratio = Decimal(attacker["ever_slashed_clients"])/honest["ever_slashed_clients"]
        rate_ratio = count_ratio*honest["clients"]/attacker["clients"]
        lines += ["", f"The raw number of ever-slashed attacker client-runs is {count_ratio:.2f} times the honest count. "
                  f"After dividing each count by its own client-run population, the incidence ratio is {rate_ratio:.2f}. "
                  "These ratios have different denominators and are not interchangeable. "
                  "Slashing supplies an economic deterrent through stake loss, not a share-allocation component. "
                  "Attack behavior is fixed in these experiments, so a behavioral response to deterrence was not measured."]
    return lines+[""]


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
              "Only the listed rules were executed on-chain; the component analysis below explicitly identifies its off-chain scope.", "",
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
    lines += component_assets(result)
    Path("docs").mkdir(exist_ok=True)
    Path("paper/generated").mkdir(parents=True,exist_ok=True)
    Path("docs/GRID_SETTLEMENT.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    Path("paper/generated/settlement_coverage.tex").write_text("\n".join(table)+"\n",encoding="utf-8",newline="\n")
    print(f"Generated exact settlement coverage: {result['jobs']} jobs, {result['round_rule_cases']} states")


if __name__ == "__main__":
    main()
