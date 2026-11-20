"""Validate recorded chain states and summarize reproducible settlement evidence."""

import json
from pathlib import Path

from chain_ledger import ChainLedger
from contract_evidence import validate_provenance
from gas_fit import gas_fits
from fltoken.io import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def validate_states(clients, delay, inputs, states):
    """Rebuild settlement and infer only legal all-or-nothing claims."""
    if len(inputs) != len(states):
        raise ValueError("recorded state count differs from the input ledger")
    oracle = ChainLedger(range(clients), delay)
    for item, recorded in zip(inputs, states):
        transition = oracle.settle(item)
        target = recorded["state"]
        actual_transition = recorded["transition"] if "transition" in recorded else {k: recorded[k] for k in transition}
        if transition != actual_transition:
            raise ValueError("recorded mint, burn or remainder differs from the ledger")
        for client, account in oracle.accounts.items():
            wallet = target["accounts"][str(client)]["wallet"]
            claimed = wallet-account["wallet"]
            if claimed:
                if claimed != account["claimable"]:
                    raise ValueError("recorded wallet cannot follow a valid claim")
                oracle.claim(client)
        if canonical(oracle.snapshot()) != target:
            raise ValueError("recorded stake, vesting, wallet or supply differs from the ledger")
    return len(states)


def main():
    checks = read("results/contracts/checks.json")
    validate_provenance(checks["provenance"])
    source = sha256("contracts/src/RewardPool.sol")
    for name, digest in checks["source_sha256"].items():
        if sha256(name) != digest:
            raise ValueError("contract checks used different source bytes")
    if checks["tests_failed"] or any(checks["static_counts"][level] for level in ["High", "Medium"]):
        raise ValueError("contract checks contain unresolved failures")
    replay = read("results/onchain/summary.json")
    case_summaries = []
    for entry in replay["cases"]:
        folder = Path(entry["path"])
        filename = folder.as_posix().replace("/", "_").replace(":", "_")+f"-{entry['rule']}.json"
        case = read(Path("results/onchain")/filename)
        validate_provenance(case["provenance"])
        if not case["exact"] or case["pool_source_sha256"] != source:
            raise ValueError("recorded replay does not match the contract")
        if case["ledger_sha256"] != sha256(folder/"ledger.json") or case["participation_sha256"] != sha256(folder/"per_round.csv"):
            raise ValueError("recorded replay does not match the training artifacts")
        ledger = read(folder/"ledger.json")[entry["rule"]]
        count = validate_states(case["clients"], case["vesting_delay"], ledger, case["states"])
        if count != entry["rounds"] or any(case[key] != entry[key] for key in ["path", "rule", "clients", "rounds"]):
            raise ValueError("replay manifest differs from its case")
        case_summaries.append(dict(path=entry["path"], rule=entry["rule"], rounds=count,
                                   minted=sum(s["minted"] for s in case["states"]),
                                   burned=sum(s["burned"] for s in case["states"]),
                                   unminted=case["states"][-1]["state"]["unminted"]))
    if sum(c["rounds"] for c in case_summaries) != replay["round_rule_cases"]:
        raise ValueError("replay round count differs")
    stress = read("results/onchain/stress.json")
    validate_provenance(stress["provenance"])
    if not stress["exact"] or stress["pool_source_sha256"] != source:
        raise ValueError("stress evidence does not match the contract")
    for case in stress["cases"]:
        count = validate_states(case["clients"], case["vesting_delay"], [s["input"] for s in case["states"]], case["states"])
        if count != case["rounds"]:
            raise ValueError("stress case has an incomplete state history")
    if sum(c["rounds"] for c in stress["cases"]) != stress["round_cases"]:
        raise ValueError("stress manifest count differs")
    gas = read("results/gas/summary.json")
    boundary = read("results/gas/boundary/boundary.json")
    validate_provenance(gas["provenance"])
    validate_provenance(boundary["provenance"])
    if gas["pool_source_sha256"] != source or gas["cap"] != boundary["cap"]:
        raise ValueError("gas evidence differs from contract or cap")
    if [c["clients"] for c in gas["cases"]] != [10, 30, 50, 100, 200, 300]:
        raise ValueError("incomplete gas scaling coverage")
    if gas_fits(gas["cases"],gas["rounds_per_case"],gas["cap"]) != gas["fits"]:
        raise ValueError("reported gas fits differ from the integer measurements")
    operations = []
    for case in gas["cases"]:
        if not case["passed"] or len(case["rounds"]) != gas["rounds_per_case"]:
            raise ValueError("incomplete gas state history")
        for round in case["rounds"]:
            receipts = [g for g in case["gas"] if g["round"] == round["round"]]
            if sum(g["gas_used"] for g in receipts) != round["total_gas"]:
                raise ValueError("round gas total differs from receipts")
        for name in sorted({g["operation"] for g in case["gas"]}):
            values = [g["gas_used"] for g in case["gas"] if g["operation"] == name]
            operations.append(dict(clients=case["clients"], operation=name, count=len(values),
                                   minimum=min(values), maximum=max(values), mean=sum(values)/len(values)))
    successes = [c["clients"] for c in boundary["cases"] if c["passed"]]
    failures = [c["clients"] for c in boundary["cases"] if not c["passed"]]
    if not successes or not failures or min(failures) != max(successes)+1:
        raise ValueError("adjacent executed transaction-cap boundary required")
    for case in boundary["cases"]:
        settlements = [g for g in case["gas"] if g["operation"] == "settle"]
        if any(g["gas_limit"] != boundary["cap"] for g in settlements):
            raise ValueError("boundary did not use the declared transaction gas limit")
        if case["passed"]:
            if len(case["rounds"]) != gas["rounds_per_case"] or any(g["status"] != 1 for g in settlements):
                raise ValueError("incomplete successful boundary case")
        elif settlements[-1]["status"] != 0 or settlements[-1]["gas_used"] != boundary["cap"]:
            raise ValueError("failed boundary did not exhaust the gas limit")
    summary = read("results/summary.json")
    summary["contracts"] = dict(checks=checks, replay_cases=case_summaries,
        replay_rounds=replay["round_rule_cases"], stress_rounds=stress["round_cases"],
        stress_slashes=sum(c["slash_participations"] for c in stress["cases"]),
        stress_delays=[c["vesting_delay"] for c in stress["cases"]], exact=True,
        pool_source_sha256=source)
    portability_path = Path("results/contracts/portability-linux.json")
    if portability_path.exists():
        portability = read(portability_path)
        if not portability["passed"] or not all(portability["checks"].values()):
            raise ValueError("contract platform comparison failed")
        if portability["comparison_source_sha256"] != sha256("scripts/check_contract_portability.py"):
            raise ValueError("contract platform comparison source differs")
        validate_provenance(portability["canonical_input_sha256"])
        for case in portability["replay_cases"]:
            if sha256(Path("results/onchain")/case["case"]) != case["canonical_sha256"]:
                raise ValueError("contract platform comparison inputs differ")
        summary["contracts"]["portability"] = portability
    summary["gas"] = dict(**{k:v for k,v in gas.items() if k!="cases"},
        operations=operations, rounds=[dict(clients=c["clients"], **r) for c in gas["cases"] for r in c["rounds"]],
        executed_success_n=max(successes), executed_failure_n=min(failures),
        boundary_failure_round=min(c["failure_round"] for c in boundary["cases"] if not c["passed"]))
    write_json("results/summary.json", summary)
    print(f"Exact replay: {replay['round_rule_cases']} round-rule cases; stress: {stress['round_cases']} rounds; "
          f"gas: {len(gas['cases'])} sizes; executed cap boundary: {max(successes)}/{min(failures)}")


if __name__ == "__main__":
    main()
