"""Measure full-participation transactions with fixed compiler and state histories."""

import argparse
import hashlib
from pathlib import Path

from chain_ledger import ChainLedger, ROUND_POOL, SCALE
from contract_evidence import provenance
from gas_fit import gas_fits
from replay_onchain import advance, build_contracts, digest, local_chain, read, transaction, write

CAP = 2**24


def measure(clients, capped=False):
    pool_artifact = read("contracts/out/RewardPool.sol/RewardPool.json")
    token_artifact = read("contracts/out/RewardToken.sol/RewardToken.json")
    gas, rounds = [], []
    ledger = ChainLedger(range(clients))
    # Cancun permits measuring operations beyond the later per-transaction cap.
    # Eligibility under that cap is computed separately; block capacity differs.
    with local_chain(clients, block_gas_limit=100_000_000, enforce_transaction_cap=capped) as web3:
        coordinator, *addresses = web3.eth.accounts
        factory = web3.eth.contract(abi=pool_artifact["abi"], bytecode=pool_artifact["bytecode"]["object"])
        receipt = transaction(web3, factory.constructor(3, 3600, 3600), coordinator, "deploy", gas)
        pool = web3.eth.contract(address=receipt.contractAddress, abi=pool_artifact["abi"])
        token = web3.eth.contract(address=pool.functions.token().call(), abi=token_artifact["abi"])
        for address in addresses:
            transaction(web3, pool.functions.grantInitial(address), coordinator, "grantInitial", gas)
            transaction(web3, token.functions.approve(pool.address, 100*SCALE), address, "approve", gas)
            transaction(web3, pool.functions.register(), address, "register", gas)
        for round in range(1, 6):
            transaction(web3, pool.functions.openRound(round, addresses), coordinator, "openRound", gas, round)
            reports = []
            for client, address in enumerate(addresses):
                update = hashlib.sha256(f"gas:{round}:{client}".encode()).digest()
                salt = hashlib.sha256(f"gas-salt:{round}:{client}".encode()).digest()
                commitment = web3.keccak(web3.codec.encode(["bytes32", "bytes32", "uint256", "address"],
                                                          [update, salt, round, address]))
                transaction(web3, pool.functions.commit(round, commitment), address, "commit", gas, round)
                reports.append((address, update, salt))
            commit_end, reveal_end, *_ = pool.functions.roundInfo(round).call()
            advance(web3, commit_end)
            for address, update, salt in reports:
                transaction(web3, pool.functions.reveal(round, update, salt), address, "reveal", gas, round)
            advance(web3, reveal_end)
            shares = [SCALE//clients]*clients
            flags = [round == 4]*clients
            transaction(web3, pool.functions.postScores(round, addresses, shares, flags), coordinator, "postScores", gas, round)
            if capped:
                receipt = web3.eth.wait_for_transaction_receipt(pool.functions.settle(round).transact({"from":coordinator,"gas":CAP}))
                gas.append(dict(operation="settle",round=round,gas_used=receipt.gasUsed,gas_limit=CAP,status=receipt.status))
                if not receipt.status:
                    assert receipt.gasUsed == CAP, "failure was not exhaustion of the provided gas"
                    assert pool.functions.lastSettledRound().call() == round-1, "failed settlement mutated state"
                    return dict(clients=clients,sampled=clients,capped=True,passed=False,
                                failure_round=round,failure="out of gas",rounds=rounds,gas=gas)
                estimate = None
            else:
                estimate = pool.functions.settle(round).estimate_gas({"from":coordinator})
                transaction(web3, pool.functions.settle(round), coordinator, "settle", gas, round)
            payouts = [ROUND_POOL*s//SCALE for s in shares]
            ledger.settle(dict(round=round, ids=list(range(clients)), shares1e18=shares, slash=flags,
                               payouts=payouts, remainder=ROUND_POOL-sum(payouts)))
            for client, address in enumerate(addresses):
                expected = ledger.accounts[client]
                assert pool.functions.stake(address).call() == expected["stake"]
                assert pool.functions.unvested(address).call() == sum(expected["tranches"].values())
                assert pool.functions.claimable(address).call() == expected["claimable"]
                if expected["claimable"]:
                    transaction(web3, pool.functions.claim(), address, "claim", gas, round)
                    ledger.claim(client)
            expected = ledger.snapshot()
            assert token.functions.totalSupply().call() == expected["supply"]
            assert token.functions.balanceOf(pool.address).call() == expected["pool_balance"]
            entries = [g for g in gas if g["round"] == round]
            rounds.append(dict(round=round, total_gas=sum(g["gas_used"] for g in entries),
                               settle_estimated_limit=estimate, settle_limit_within_cap=True if capped else estimate<=CAP,
                               largest_receipt_gas=max(g["gas_used"] for g in entries)))
            print(f"N={clients}, round={round}, total_gas={rounds[-1]['total_gas']}", flush=True)
    return dict(clients=clients, sampled=clients, capped=capped, passed=True, rounds=rounds, gas=gas)


def sources():
    return {**provenance("scripts/measure_gas.py"), "scripts/gas_fit.py": digest("scripts/gas_fit.py")}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("results/gas"))
    parser.add_argument("--sizes",type=int,nargs="+",default=[10,30,50,100,200,300])
    parser.add_argument("--capped",action="store_true")
    args=parser.parse_args()
    measured_sources=sources()
    build_contracts()
    cases=[]
    for n in args.sizes:
        case=measure(n,args.capped)
        write(args.output/f"n{n}.json",case)
        cases.append(case)
    if measured_sources != sources():
        raise ValueError("chain sources changed during gas measurement")
    if args.capped:
        write(args.output/"boundary.json",dict(cap=CAP,cases=cases,provenance=measured_sources,cap_source="https://eips.ethereum.org/EIPS/eip-7825"))
        return
    fits=gas_fits(cases,5,CAP)
    write(args.output/"summary.json",dict(cases=cases,fits=fits,cap=CAP,provenance=measured_sources,
          cap_source="https://eips.ethereum.org/EIPS/eip-7825",
          cap_semantics="Sender-specified transaction gas limit, independent of block gas limit. Receipt gas after refunds alone is insufficient; fitted cap values require boundary execution checks.",
          pool_source_sha256=digest("contracts/src/RewardPool.sol"),compiler="0.8.30",optimizer_runs=200,
          evm="cancun",vesting_delay=3,slash_rounds=[4],rounds_per_case=5))


if __name__ == "__main__":
    main()
