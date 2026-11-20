"""Replay recorded integer rewards on a temporary, loopback-only Anvil chain."""

import argparse
from contextlib import contextmanager
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from web3 import Web3

from chain_ledger import ChainLedger
from contract_evidence import provenance


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def local_chain(clients, block_gas_limit=30_000_000, enforce_transaction_cap=False):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    suffix = ".exe" if os.name == "nt" else ""
    binary = Path("out/tools/foundry") / ("anvil" + suffix)
    process = subprocess.Popen([str(binary.resolve()), "--host", "127.0.0.1", "--port", str(port),
                                "--accounts", str(clients + 1), "--hardfork", "cancun",
                                "--gas-limit", str(block_gas_limit), "--silent",
                                *(["--enable-tx-gas-limit"] if enforce_transaction_cap else [])],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        web3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{port}", request_kwargs={"timeout": 15}))
        deadline = time.monotonic() + 20
        while not web3.is_connected():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("local Anvil failed to start")
            time.sleep(.1)
        yield web3
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def transaction(web3, function, sender, operation, gas_log, round=0):
    receipt = web3.eth.wait_for_transaction_receipt(function.transact({"from": sender}))
    if receipt.status != 1:
        raise AssertionError(f"transaction failed: {operation}, round {round}")
    gas_log.append(dict(operation=operation, round=round, gas_used=receipt.gasUsed))
    return receipt


def advance(web3, timestamp):
    result = web3.provider.make_request("evm_setNextBlockTimestamp", [timestamp])
    if "error" in result:
        raise RuntimeError(result["error"])
    web3.provider.make_request("evm_mine", [])


def replay(folder, rule, output, vesting_delay=3):
    sources = provenance("scripts/replay_onchain.py")
    metadata, ledger = read(folder / "metadata.json"), read(folder / "ledger.json")[rule]
    if metadata["status"] != "complete":
        raise ValueError("only complete runs can be replayed")
    hashes = read(folder / "hashes.json")
    for name in ["per_round.csv", "round_metrics.csv", "ledger.json"]:
        if hashes.get(name) != digest(folder / name):
            raise ValueError(f"recorded artifact hash mismatch: {name}")
    with (folder / "per_round.csv").open(encoding="utf-8", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    rows = {(int(row["round"]), int(row["client_id"])): row for row in recorded}
    if len(rows) != len(recorded) or len(rows) != metadata["config"]["rounds"]*metadata["config"]["clients_per_round"]:
        raise ValueError("incomplete or duplicated participation log")
    clients = metadata["config"]["num_clients"]
    if len(ledger) != metadata["config"]["rounds"]:
        raise ValueError("incomplete round ledger")
    for round, item in enumerate(ledger, 1):
        participants = [row for row in recorded if int(row["round"]) == round]
        if item["round"] != round or item["ids"] != [int(row["client_id"]) for row in participants]:
            raise ValueError("ledger sample order differs from recorded participations")
        floating = [float(row[f"{rule}_share"]) for row in participants]
        if any(not math.isfinite(value) or value < 0 for value in floating):
            raise ValueError("invalid recorded floating share")
        fractions = [Fraction(value) for value in floating]
        total = sum(fractions, Fraction(0))
        units = [int(value*10**18//total) for value in fractions] if total else [0]*len(fractions)
        if units != item["shares1e18"] or item["slash"] != [row[f"{rule}_slash"] == "True" for row in participants]:
            raise ValueError("ledger shares or penalties differ from recorded scores")
    pool_artifact = read("contracts/out/RewardPool.sol/RewardPool.json")
    token_artifact = read("contracts/out/RewardToken.sol/RewardToken.json")
    build = pool_artifact["metadata"]
    if build["compiler"]["version"] != "0.8.30+commit.73712a01":
        raise ValueError("unexpected Solidity compiler")
    if build["settings"]["optimizer"] != {"enabled": True, "runs": 200} or build["settings"]["evmVersion"] != "cancun":
        raise ValueError("unexpected compiler settings")
    for name, source in build["sources"].items():
        if "0x" + Web3.keccak((Path("contracts") / name).read_bytes()).hex() != source["keccak256"]:
            raise ValueError(f"compiled source differs from working source: {name}")
    gas_log, rounds = [], []
    oracle = ChainLedger(range(clients), vesting_delay)
    started = time.perf_counter()
    with local_chain(clients) as web3:
        accounts = web3.eth.accounts
        coordinator, addresses = accounts[0], accounts[1:]
        factory = web3.eth.contract(abi=pool_artifact["abi"], bytecode=pool_artifact["bytecode"]["object"])
        receipt = transaction(web3, factory.constructor(vesting_delay, 3600, 3600), coordinator, "deploy", gas_log)
        pool = web3.eth.contract(address=receipt.contractAddress, abi=pool_artifact["abi"])
        token = web3.eth.contract(address=pool.functions.token().call(), abi=token_artifact["abi"])
        for address in addresses:
            transaction(web3, pool.functions.grantInitial(address), coordinator, "grantInitial", gas_log)
            transaction(web3, token.functions.approve(pool.address, 100 * 10**18), address, "approve", gas_log)
            transaction(web3, pool.functions.register(), address, "register", gas_log)
        for item in ledger:
            round = item["round"]
            ids = [addresses[i] for i in item["ids"]]
            transaction(web3, pool.functions.openRound(round, ids), coordinator, "openRound", gas_log, round)
            reports = []
            for client, address in zip(item["ids"], ids):
                row = rows[(round, client)]
                # Historical logs contain the common update-set hash. The replay
                # commits its participation record; it does not prove raw-update quality.
                update_digest = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).digest()
                salt = hashlib.sha256(f"local-replay:{rule}:{round}:{client}".encode()).digest()
                commitment = Web3.keccak(web3.codec.encode(["bytes32", "bytes32", "uint256", "address"],
                                                          [update_digest, salt, round, address]))
                transaction(web3, pool.functions.commit(round, commitment), address, "commit", gas_log, round)
                reports.append((address, update_digest, salt))
            commit_deadline, reveal_deadline, *_ = pool.functions.roundInfo(round).call()
            advance(web3, commit_deadline)
            for address, update_digest, salt in reports:
                transaction(web3, pool.functions.reveal(round, update_digest, salt), address, "reveal", gas_log, round)
            advance(web3, reveal_deadline)
            transaction(web3, pool.functions.postScores(round, ids, item["shares1e18"], item["slash"]),
                        coordinator, "postScores", gas_log, round)
            transaction(web3, pool.functions.settle(round), coordinator, "settle", gas_log, round)
            transition = oracle.settle(item)
            before_claim = oracle.snapshot()
            for client, address in enumerate(addresses):
                expected = before_claim["accounts"][client]
                actual = dict(stake=pool.functions.stake(address).call(), claimable=pool.functions.claimable(address).call(),
                              unvested=pool.functions.unvested(address).call(), wallet=token.functions.balanceOf(address).call())
                if any(actual[key] != expected[key] for key in actual):
                    raise AssertionError((rule, round, client, actual, expected))
                actual_tranches = {}
                for slot in range(vesting_delay+1):
                    amount, unlock = pool.functions.vestingTranche(address, slot).call()
                    if amount:
                        if unlock in actual_tranches:
                            raise AssertionError("duplicate unlock round in vesting slots")
                        actual_tranches[unlock] = amount
                if actual_tranches != expected["tranches"]:
                    raise AssertionError(("tranche mismatch", round, client, actual_tranches, expected["tranches"]))
                if expected["claimable"]:
                    transaction(web3, pool.functions.claim(), address, "claim", gas_log, round)
                    oracle.claim(client)
            expected = oracle.snapshot()
            if token.functions.totalSupply().call() != expected["supply"]:
                raise AssertionError("token supply mismatch")
            if token.functions.balanceOf(pool.address).call() != expected["pool_balance"]:
                raise AssertionError("pool token balance mismatch")
            if pool.functions.totalUnminted().call() != expected["unminted"]:
                raise AssertionError("rounding remainder mismatch")
            for client, address in enumerate(addresses):
                if token.functions.balanceOf(address).call() != expected["accounts"][client]["wallet"]:
                    raise AssertionError("post-claim wallet mismatch")
                if pool.functions.claimable(address).call() != expected["accounts"][client]["claimable"]:
                    raise AssertionError("post-claim vesting balance mismatch")
            rounds.append(dict(**transition, state=expected))
            print(f"{folder.name} {rule}: round {round}/{len(ledger)} exact", flush=True)
    if sources != provenance("scripts/replay_onchain.py"):
        raise ValueError("chain sources changed during replay")
    result = dict(path=folder.as_posix(), rule=rule, rounds=len(rounds), clients=clients, exact=True, provenance=sources,
                  ledger_sha256=digest(folder / "ledger.json"), participation_sha256=digest(folder / "per_round.csv"),
                  pool_source_sha256=digest("contracts/src/RewardPool.sol"), token_source_sha256=digest("contracts/src/RewardToken.sol"),
                  vesting_delay=vesting_delay, slash_fraction=[1, 10], commit_window=3600, reveal_window=3600,
                  compiler="0.8.30", optimizer_runs=200, evm="cancun", wall_seconds=time.perf_counter()-started,
                  bytecode_sha256=hashlib.sha256(bytes.fromhex(pool_artifact["bytecode"]["object"].removeprefix("0x"))).hexdigest(),
                  commitment_semantics="SHA-256 of canonical recorded participation including the common update-set hash; deterministic test salts; not raw-update validation.",
                  gas=gas_log, states=rounds)
    case = folder.as_posix().replace("/", "_").replace(":", "_")
    write(output / f"{case}-{rule}.json", result)
    return {key: result[key] for key in ["path", "rule", "rounds", "clients", "exact", "wall_seconds"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--rule", action="append")
    parser.add_argument("--vesting-delay", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/onchain"))
    args = parser.parse_args()
    build_contracts()
    results = []
    for folder in args.runs:
        for rule in args.rule or list(read(folder / "ledger.json")):
            results.append(replay(folder, rule, args.output, args.vesting_delay))
    write(args.output / "summary.json", dict(cases=results, exact=all(r["exact"] for r in results),
                                           round_rule_cases=sum(r["rounds"] for r in results)))
    print(json.dumps(results, indent=2))


def build_contracts():
    tools = Path("out/tools/foundry").resolve()
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")}
    compiler_run = subprocess.run([str(tools / ("forge.exe" if os.name == "nt" else "forge")), "build",
                                   "--root", "contracts", "--offline"], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    if compiler_run.returncode:
        raise RuntimeError(compiler_run.stdout)


if __name__ == "__main__":
    main()
