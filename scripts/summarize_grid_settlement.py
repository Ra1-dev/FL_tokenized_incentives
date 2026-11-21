"""Reconstruct complete grid settlement and separate allocation from retained wealth."""

import csv
from fractions import Fraction
from pathlib import Path

from chain_ledger import ChainLedger, INITIAL_STAKE, SCALE
from evaluation import ABLATIONS, published_bytes, recorded_method, read
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from fltoken.statistics import sampled_bootstrap_mean
from grid_reproduction import verify_training_source
from replay_grid import covered_cells, replay_jobs, validate_case
from rebuild_ledgers import rebuild
from run_grid import checked_execution_clock
from select_method import expected_attacks


FIELDS = ["clients", "minted_rewards", "claimed_rewards", "claimable_rewards", "unvested_rewards",
          "retained_rewards", "burned_rewards", "remaining_stake", "burned_stake",
          "net_token_outcome", "slash_events", "ever_slashed_clients", "clients_with_balance_loss"]


def economic_outcomes(ledger, states, adversaries):
    """Decompose already validated integer state histories without rounding."""
    amounts = [amount for item in ledger for amount in item["payouts"]]
    for recorded in states:
        state, transition = recorded["state"], recorded.get("transition", recorded)
        amounts.extend(state[key] for key in ["supply", "pool_balance", "unminted"])
        amounts.extend(transition[key] for key in ["minted", "burned", "remainder"])
        for account in state["accounts"].values():
            amounts.extend(account[key] for key in ["stake", "wallet", "claimable", "unvested"])
            amounts.extend(account["tranches"].values())
    if any(type(amount) is not int or amount < 0 for amount in amounts):
        raise ValueError("settlement amounts must be nonnegative integer base units")
    terminal = states[-1]["state"]
    accounts = {int(client): account for client, account in terminal["accounts"].items()}
    if not set(adversaries) <= accounts.keys():
        raise ValueError("adversary identities must belong to the registered population")
    minted = dict.fromkeys(accounts, 0)
    slash_events = dict.fromkeys(accounts, 0)
    for item in ledger:
        if len(item["ids"]) != len(item["payouts"]) or len(item["ids"]) != len(item["slash"]):
            raise ValueError("settlement participation arrays differ")
        for client, amount, flag in zip(item["ids"], item["payouts"], item["slash"]):
            if type(flag) is not bool:
                raise ValueError("settlement slash flags must be booleans")
            minted[client] += amount
            slash_events[client] += int(flag)
    result = {group: dict.fromkeys(FIELDS, 0) for group in ["honest", "adversary"]}
    for client, account in accounts.items():
        group = result["adversary" if client in adversaries else "honest"]
        retained = account["wallet"]+account["claimable"]+account["unvested"]
        burned_rewards, burned_stake = minted[client]-retained, INITIAL_STAKE-account["stake"]
        if burned_rewards < 0 or burned_stake < 0:
            raise ValueError("terminal wealth exceeds its recorded issuance or initial stake")
        values = dict(clients=1, minted_rewards=minted[client], claimed_rewards=account["wallet"],
                      claimable_rewards=account["claimable"], unvested_rewards=account["unvested"],
                      retained_rewards=retained, burned_rewards=burned_rewards,
                      remaining_stake=account["stake"], burned_stake=burned_stake,
                      net_token_outcome=retained-burned_stake,
                      slash_events=slash_events[client], ever_slashed_clients=int(slash_events[client] > 0),
                      clients_with_balance_loss=int(burned_rewards+burned_stake > 0))
        for name, value in values.items():
            group[name] += value
    totals = {name: sum(group[name] for group in result.values()) for name in FIELDS}
    transitions = [state.get("transition", state) for state in states]
    if (totals["burned_rewards"]+totals["burned_stake"] != sum(row["burned"] for row in transitions)
            or totals["minted_rewards"] != sum(row["minted"] for row in transitions)
            or totals["remaining_stake"]+totals["retained_rewards"] != terminal["supply"]
            or totals["remaining_stake"]+totals["claimable_rewards"]+totals["unvested_rewards"] != terminal["pool_balance"]):
        raise ValueError("terminal economic decomposition does not conserve issuance and burns")
    share = (float(Fraction(result["adversary"]["minted_rewards"],totals["minted_rewards"]))
             if totals["minted_rewards"] else None)
    return dict(groups=result, totals=totals, minted_attacker_share=share,
                final_supply=terminal["supply"], final_pool_balance=terminal["pool_balance"],
                unminted=terminal["unminted"])


def penalty_ablation(ledgers, onchain_outcome, clients, delay, adversaries):
    """Settle recorded component ledgers; never recompute qualities or shares."""
    if ledgers["Afull"] != ledgers["Mstar"]:
        raise ValueError("fully enabled component ledger differs from Mstar")
    allocation = lambda rows: [{key:value for key,value in row.items() if key != "slash"} for row in rows]
    stages = {}
    for stage in ["Aema", "Afull"]:
        ledger = ledgers[stage]
        if stage == "Aema" and any(any(item["slash"]) for item in ledger):
            raise ValueError("EMA-only component must not request penalties")
        oracle = ChainLedger(range(clients), delay)
        states = []
        for item in ledger:
            transition = oracle.settle(item)
            # Claims cannot change slashable tranches or total retained wealth.
            for client in oracle.accounts:
                oracle.claim(client)
            states.append(dict(**transition, state=oracle.snapshot()))
        stages[stage] = economic_outcomes(ledger, states, adversaries)
    # Claim schedules can differ in other valid histories; compare all
    # schedule-independent balances and event counts against the Anvil result.
    compared = [field for field in FIELDS if field not in ["claimed_rewards", "claimable_rewards"]]
    for group in ["honest", "adversary"]:
        if any(stages["Afull"]["groups"][group][field] != onchain_outcome["groups"][group][field]
               for field in compared):
            raise ValueError("component settlement differs from validated full-method chain balances")
    return dict(allocations_identical=allocation(ledgers["Aema"]) == allocation(ledgers["Afull"]),
                source="off-chain settlement of original component ledgers; Afull checked against Anvil",
                stages=stages)


def aggregate_components(cases):
    """Descriptive sums over distinct experimental histories, never unique people."""
    result = {}
    for horizon in sorted({case["horizon"] for case in cases}):
        selected = [case for case in cases if case["horizon"] == horizon]
        result[horizon] = dict(runs=len(selected),
            groups={stage:{group:{field:sum(case["stages"][stage]["groups"][group][field] for case in selected)
                                  for field in FIELDS} for group in ["honest", "adversary"]}
                    for stage in ["Aema", "Afull"]})
    return result


def component_intervals(cases, plan):
    """Average scenarios within seed before estimating economic uncertainty."""
    result = []
    statistics = plan["statistics"]
    for horizon in sorted({case["horizon"] for case in cases}):
        by_seed = {seed:[case for case in cases if case["horizon"] == horizon and case["seed"] == seed]
                   for seed in plan["seeds"]}
        if not all(by_seed.values()) or len({len(group) for group in by_seed.values()}) != 1:
            raise ValueError("economic uncertainty requires equal complete scenario coverage per seed")
        for stage in ["Aema", "Afull"]:
            for group in ["honest", "adversary"]:
                metrics = {}
                for field in ["burned_stake", "burned_rewards", "net_token_outcome", "ever_slashed_clients"]:
                    factor = 1 if field == "ever_slashed_clients" else SCALE
                    values = [Fraction(sum(case["stages"][stage]["groups"][group][field] for case in subset),
                                       len(subset)*factor) for subset in by_seed.values()]
                    metrics[field] = sampled_bootstrap_mean(values, statistics["confidence"],
                        statistics["bootstrap_resamples"], statistics["bootstrap_seed"])
                result.append(dict(horizon=horizon, stage=stage, group=group, seeds=plan["seeds"],
                                   scenarios_per_seed=len(next(iter(by_seed.values()))), metrics=metrics))
    return result


def complete_jobs(plan, manifest, coverage):
    supplied = covered_cells(manifest, plan, False)
    jobs = list(replay_jobs(plan))
    if (coverage.get("complete") is not True or coverage["expected_jobs"] != len(jobs)
            or coverage["completed_jobs"] != len(jobs) or len(coverage["cases"]) != len(jobs)
            or coverage["expected_round_rule_cases"] != sum(job["rounds"] for job in jobs)
            or coverage["round_rule_cases"] != sum(job["rounds"] for job in jobs)):
        raise ValueError("complete declared settlement coverage is required")
    for job, case in zip(jobs, coverage["cases"]):
        expected = dict(path=job["path"], rule=job["rule"], rounds=job["rounds"],
                        evidence="results/grid_onchain/"+job["path"].replace("/", "_")+f"-{job['rule']}.json")
        if any(case.get(key) != value for key, value in expected.items()):
            raise ValueError("settlement cases must follow every unique declared run and rule")
    return supplied, jobs


def summarize():
    plan_path = Path("configs/evaluation_plan.json")
    published_bytes(plan_path)
    plan = read(plan_path)
    method = recorded_method(full_evidence=True)
    manifest_path, coverage_path = Path("results/grid/manifest.json"), Path("results/grid_onchain/coverage.json")
    manifest, coverage = read(manifest_path), read(coverage_path)
    binding = dict(plan_sha256=sha256(plan_path), method_sha256=sha256("configs/method_frozen.json"))
    if any(record.get(key) != value for record in [manifest, coverage] for key, value in binding.items()):
        raise ValueError("settlement evidence belongs to a different published plan or method")
    if (coverage["grid_manifest_sha256"] != sha256(manifest_path)
            or coverage["driver_sha256"] != sha256("scripts/replay_grid.py")):
        raise ValueError("settlement coverage differs from its grid or replay implementation")
    clock_path = Path("results/grid/execution.json")
    clock = checked_execution_clock(clock_path, binding)
    if (clock["finished_utc"] is None or manifest["execution_sha256"] != sha256(clock_path)
            or manifest["elapsed_wall_seconds"] != clock["elapsed_wall_seconds"]):
        raise ValueError("settlement analysis requires the completed original training clock")
    supplied, jobs = complete_jobs(plan, manifest, coverage)
    settlement = method["settlement"]
    if (settlement["stake_tokens"], settlement["slash_numerator"], settlement["slash_denominator"]) != (100, 1, 10):
        raise ValueError("frozen economics differ from the replayed contract")
    base = load_config(plan["config"])
    if base["dataset"] != plan["dataset"] or base["train_limit"] or base["test_limit"] or not base["save_updates"]:
        raise ValueError("settlement requires the declared complete training dataset and retained updates")
    checked, common, adversaries = set(), None, {}
    for path, cell in supplied.items():
        folder = Path(path)
        metadata = read(folder/"metadata.json")
        expected = {**base, **{key:cell[key] for key in ["seed", "attack", "fraction", "rounds"]},
                    **{key:plan[key] for key in ["device", "workers", "gpu_fraction"]},
                    "rules":plan["rules"]+list(ABLATIONS), "ours_params":method["params"]}
        if (metadata["status"] != "complete" or metadata["config"] != expected
                or metadata["source_config"] != plan["config"] or metadata["source_git_dirty"]
                or metadata["git_commit"] != cell["source_commit"]
                or metadata["method_sha256"] != binding["method_sha256"]
                or metadata["source_sha256"] != clock["training_source_sha256"]
                or sha256(folder/"metadata.json") != cell["metadata_sha256"]
                or sha256(folder/"per_round.csv") != cell["per_round_sha256"]):
            raise ValueError("settlement training inputs differ from the declared common grid")
        implementation = (metadata["source_sha256"], metadata["versions"])
        if common is not None and common != implementation:
            raise ValueError("settlement cells use different training implementations")
        common = implementation
        if metadata["git_commit"] not in checked:
            verify_training_source(metadata, method=method)
            checked.add(metadata["git_commit"])
        attacks = {int(client): attack for client, attack in read(folder/"protocol.json")["adversaries"].items()}
        if attacks != expected_attacks(metadata["config"]):
            raise ValueError("economic groups differ from the seeded attack assignment")
        rebuild(folder)
        adversaries[path] = attacks
    cases, ablations = [], []
    for job, recorded in zip(jobs, coverage["cases"]):
        folder, evidence = Path(job["path"]), Path(recorded["evidence"])
        if sha256(evidence) != recorded["evidence_sha256"]:
            raise ValueError("chain state evidence differs from its coverage digest")
        case = read(evidence)
        validate_case(case, job, folder, settlement["vesting_rounds"])
        ledger = read(folder/"ledger.json")[job["rule"]]
        outcome = economic_outcomes(ledger,case["states"],adversaries[job["path"]])
        if outcome["groups"]["honest"]["ever_slashed_clients"] != read(folder/"metrics.json")[job["rule"]]["honest_ever_slashed"]:
            raise ValueError("settlement honest slash count differs from the original participation metric")
        cases.append(dict(**job, **outcome, evidence_sha256=sha256(evidence),
                          ledger_sha256=sha256(folder/"ledger.json"), protocol_sha256=sha256(folder/"protocol.json")))
        if job["rule"] == "Mstar":
            component = penalty_ablation(read(folder/"ledger.json"), outcome, case["clients"],
                                         settlement["vesting_rounds"], adversaries[job["path"]])
            ablations.append(dict(**{key:value for key,value in job.items() if key != "rule"}, **component,
                                  ledger_sha256=sha256(folder/"ledger.json")))
    by_rule = {rule:dict(runs=sum(case["rule"] == rule for case in cases),
                        round_rule_cases=sum(case["rounds"] for case in cases if case["rule"] == rule))
               for rule in plan["rules"]}
    result = dict(schema=2, exact=True, complete=True, training_device=plan["device"],
                  grid_runs=len(supplied), jobs=len(cases), round_rule_cases=sum(case["rounds"] for case in cases),
                  vesting_rounds=settlement["vesting_rounds"], units="token base units", by_rule=by_rule, cases=cases,
                  penalty_ablation=dict(runs=len(ablations), cases=ablations,
                      quality_threshold=method["params"]["threshold"], patience=method["params"]["patience"],
                      identical_allocation_runs=sum(case["allocations_identical"] for case in ablations),
                      aggregates=aggregate_components(ablations),
                      intervals=component_intervals(ablations, plan),
                      net_definition="minted rewards minus forfeited unvested rewards minus slashed stake; final wealth minus initial stake",
                      count_definition="unique clients within each run; aggregated counts are client-run identities, not unique people"),
                  provenance=dict(**binding, grid_manifest_sha256=sha256(manifest_path),
                                  coverage_sha256=sha256(coverage_path), analysis_sha256=sha256(__file__),
                                  validation_source_sha256={name:sha256(name) for name in
                                      ["scripts/replay_grid.py", "scripts/summarize_contracts.py", "scripts/rebuild_ledgers.py",
                                       "src/fltoken/statistics.py"]}))
    write_json("results/grid_onchain/summary.json", result)
    summary = read("results/summary.json")
    summary["grid_settlement"] = result
    write_json("results/summary.json", summary)
    rows = []
    for case in cases:
        for group, values in case["groups"].items():
            rows.append({**{name:case[name] for name in ["path", "rule", "seed", "attack", "fraction", "horizon", "rounds"]},
                         "group":group, **values})
    with Path("results/grid_onchain/economic_outcomes.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    component_rows = [{**{name:case[name] for name in ["path", "seed", "attack", "fraction", "horizon", "rounds"]},
                       "stage":stage, "group":group, **outcome["groups"][group]}
                      for case in ablations for stage,outcome in case["stages"].items() for group in ["honest", "adversary"]]
    with Path("results/grid_onchain/penalty_ablation.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(component_rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(component_rows)
    print(f"Exact complete settlement: {len(cases)} jobs, {result['round_rule_cases']} round-rule cases, {len(rows)} economic group records")
    return result


if __name__ == "__main__":
    summarize()
