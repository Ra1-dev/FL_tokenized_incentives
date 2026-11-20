"""Exercise randomized slash, vesting, absence and claim histories on local Anvil."""

import argparse
import hashlib
import random
from pathlib import Path

from chain_ledger import ChainLedger, ROUND_POOL, SCALE
from contract_evidence import provenance
from replay_onchain import advance, build_contracts, digest, local_chain, read, transaction, write


def check_case(clients, delay, seed, count):
    rng = random.Random(seed)
    oracle = ChainLedger(range(clients), delay)
    pool_artifact = read("contracts/out/RewardPool.sol/RewardPool.json")
    token_artifact = read("contracts/out/RewardToken.sol/RewardToken.json")
    gas, states = [], []
    with local_chain(clients) as web3:
        coordinator, *addresses = web3.eth.accounts
        factory = web3.eth.contract(abi=pool_artifact["abi"], bytecode=pool_artifact["bytecode"]["object"])
        receipt = transaction(web3, factory.constructor(delay, 3600, 3600), coordinator, "deploy", gas)
        pool = web3.eth.contract(address=receipt.contractAddress, abi=pool_artifact["abi"])
        token = web3.eth.contract(address=pool.functions.token().call(), abi=token_artifact["abi"])
        for address in addresses:
            transaction(web3, pool.functions.grantInitial(address), coordinator, "grantInitial", gas)
            transaction(web3, token.functions.approve(pool.address, 100*SCALE), address, "approve", gas)
            transaction(web3, pool.functions.register(), address, "register", gas)
        for round in range(1, count+1):
            ids = rng.sample(range(clients), rng.randint(1, clients))
            selected = [addresses[i] for i in ids]
            weights = [rng.randint(0, SCALE) for _ in ids]
            shares = [w*SCALE//sum(weights) for w in weights]
            if round % 5 == 0:
                shares = [0]*len(ids)
            # Repeated penalties and delayed claims distinguish stake, unvested
            # tranches and matured balances. This is a contract stress trace,
            # independent of any learned scoring parameters.
            flags = [rng.random() < .7 for _ in ids]
            payouts = [ROUND_POOL*s//SCALE for s in shares]
            item = dict(round=round, ids=ids, shares1e18=shares, slash=flags,
                        payouts=payouts, remainder=ROUND_POOL-sum(payouts))
            transaction(web3, pool.functions.openRound(round, selected), coordinator, "openRound", gas, round)
            reports = []
            for client, address in zip(ids, selected):
                digest = hashlib.sha256(f"synthetic:{seed}:{round}:{client}".encode()).digest()
                salt = hashlib.sha256(f"test-salt:{seed}:{round}:{client}".encode()).digest()
                commitment = web3.keccak(web3.codec.encode(["bytes32", "bytes32", "uint256", "address"],
                                                          [digest, salt, round, address]))
                transaction(web3, pool.functions.commit(round, commitment), address, "commit", gas, round)
                reports.append((address, digest, salt))
            commit_end, reveal_end, *_ = pool.functions.roundInfo(round).call()
            advance(web3, commit_end)
            for address, digest, salt in reports:
                transaction(web3, pool.functions.reveal(round, digest, salt), address, "reveal", gas, round)
            advance(web3, reveal_end)
            transaction(web3, pool.functions.postScores(round, selected, shares, flags), coordinator, "postScores", gas, round)
            transaction(web3, pool.functions.settle(round), coordinator, "settle", gas, round)
            transition = oracle.settle(item)
            for client, address in enumerate(addresses):
                account = oracle.accounts[client]
                if account["claimable"] and rng.random() < .35:
                    transaction(web3, pool.functions.claim(), address, "claim", gas, round)
                    oracle.claim(client)
            expected = oracle.snapshot()
            for client, address in enumerate(addresses):
                account = expected["accounts"][client]
                actual = dict(stake=pool.functions.stake(address).call(),
                              claimable=pool.functions.claimable(address).call(),
                              unvested=pool.functions.unvested(address).call(),
                              wallet=token.functions.balanceOf(address).call())
                assert all(actual[k] == account[k] for k in actual), (seed, round, client, actual, account)
                tranches = {}
                for slot in range(delay+1):
                    amount, unlock = pool.functions.vestingTranche(address, slot).call()
                    if amount:
                        assert unlock not in tranches
                        tranches[unlock] = amount
                assert tranches == account["tranches"], (seed, round, client, tranches, account["tranches"])
            assert token.functions.totalSupply().call() == expected["supply"]
            assert token.functions.balanceOf(pool.address).call() == expected["pool_balance"]
            assert pool.functions.totalUnminted().call() == expected["unminted"]
            states.append(dict(input=item, transition=transition, state=expected))
    return dict(clients=clients, vesting_delay=delay, seed=seed, rounds=count, exact=True,
                slash_participations=sum(sum(s["input"]["slash"]) for s in states),
                burned=sum(s["transition"]["burned"] for s in states), states=states, gas=gas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/onchain/stress.json"))
    args = parser.parse_args()
    sources = provenance("scripts/check_chain_states.py")
    build_contracts()
    cases = [check_case(n, delay, seed, 24) for n, delay, seed in [(2, 0, 631), (2, 1, 814), (9, 3, 990), (11, 7, 1729)]]
    if sources != provenance("scripts/check_chain_states.py"):
        raise ValueError("chain sources changed during state checks")
    write(args.output, dict(exact=True, provenance=sources, round_cases=sum(c["rounds"] for c in cases), cases=cases,
                           pool_source_sha256=digest("contracts/src/RewardPool.sol")))
    for case in cases:
        print({k: v for k, v in case.items() if k not in {"states", "gas"}})


if __name__ == "__main__":
    main()
