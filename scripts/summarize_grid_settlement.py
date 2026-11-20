"""Reconstruct complete grid settlement and separate allocation from retained wealth."""

import csv
from fractions import Fraction
from pathlib import Path

from chain_ledger import INITIAL_STAKE
from evaluation import ABLATIONS, published_bytes, published_method, read
from fltoken.config import load_config
from fltoken.io import sha256, write_json
from grid_reproduction import verify_training_source
from replay_grid import covered_cells, replay_jobs, validate_case
from rebuild_ledgers import rebuild
from run_grid import checked_execution_clock
from select_method import expected_attacks


FIELDS = ["clients", "minted_rewards", "claimed_rewards", "claimable_rewards", "unvested_rewards",
          "retained_rewards", "burned_rewards", "remaining_stake", "burned_stake"]


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
    for item in ledger:
        for client, amount in zip(item["ids"], item["payouts"]):
            minted[client] += amount
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
                      remaining_stake=account["stake"], burned_stake=burned_stake)
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
    method = published_method(full_evidence=True)
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
            verify_training_source(metadata)
            checked.add(metadata["git_commit"])
        attacks = {int(client): attack for client, attack in read(folder/"protocol.json")["adversaries"].items()}
        if attacks != expected_attacks(metadata["config"]):
            raise ValueError("economic groups differ from the seeded attack assignment")
        rebuild(folder)
        adversaries[path] = attacks
    cases = []
    for job, recorded in zip(jobs, coverage["cases"]):
        folder, evidence = Path(job["path"]), Path(recorded["evidence"])
        if sha256(evidence) != recorded["evidence_sha256"]:
            raise ValueError("chain state evidence differs from its coverage digest")
        case = read(evidence)
        validate_case(case, job, folder, settlement["vesting_rounds"])
        ledger = read(folder/"ledger.json")[job["rule"]]
        outcome = economic_outcomes(ledger,case["states"],adversaries[job["path"]])
        cases.append(dict(**job, **outcome, evidence_sha256=sha256(evidence),
                          ledger_sha256=sha256(folder/"ledger.json"), protocol_sha256=sha256(folder/"protocol.json")))
    by_rule = {rule:dict(runs=sum(case["rule"] == rule for case in cases),
                        round_rule_cases=sum(case["rounds"] for case in cases if case["rule"] == rule))
               for rule in plan["rules"]}
    result = dict(schema=1, exact=True, complete=True, training_device=plan["device"],
                  grid_runs=len(supplied), jobs=len(cases), round_rule_cases=sum(case["rounds"] for case in cases),
                  vesting_rounds=settlement["vesting_rounds"], units="token base units", by_rule=by_rule, cases=cases,
                  provenance=dict(**binding, grid_manifest_sha256=sha256(manifest_path),
                                  coverage_sha256=sha256(coverage_path), analysis_sha256=sha256(__file__),
                                  validation_source_sha256={name:sha256(name) for name in
                                      ["scripts/replay_grid.py", "scripts/summarize_contracts.py", "scripts/rebuild_ledgers.py"]}))
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
    print(f"Exact complete settlement: {len(cases)} jobs, {result['round_rule_cases']} round-rule cases, {len(rows)} economic group records")
    return result


if __name__ == "__main__":
    summarize()
