"""Public-testnet demonstration of the reward pool protocol on Sepolia.

Deploys the unchanged RewardPool (and its token) with demonstration timing,
registers two honest clients and one zero-update free-rider, and runs three
complete rounds: commit, reveal, coordinator score posting, settlement,
vesting maturity, claims, and a slash of the free-rider after the frozen
patience of three low-quality participations. Every expected integer state is
recomputed independently and compared with the chain after each round.

Scores are illustrative protocol inputs (honest clients split the pool, the
zero update earns nothing); the experiment's allocations are verified
separately by the local replay of the recorded grid. The demonstration shows
that the same bytecode settles correctly on a public network, with receipts
anyone can inspect.

Usage (Sepolia):  set SEPOLIA_PRIVATE_KEY, then python scripts/sepolia_demo.py
Usage (local):    python scripts/sepolia_demo.py --local   (same protocol on Anvil/Cancun; gas reference)
Usage (document): python scripts/sepolia_demo.py --document-only
Run in the contract environment (requirements-contracts.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eth_account import Account
from web3 import Web3

SEPOLIA = 11155111
EXPLORER = "https://sepolia.etherscan.io"
DEFAULT_RPC = "https://ethereum-sepolia-rpc.publicnode.com"
SCALE = 10**18
ROUND_POOL = 1000 * SCALE
INITIAL_STAKE = 100 * SCALE
SLASH_DENOMINATOR = 10
PATIENCE = 3
# Public networks may price state creation far above the local Cancun schedule (deployment
# alone used about 16.7 million gas on Sepolia), so the budgets are deliberately generous.
CLIENT_GAS_BUDGET = 6_000_000
COORDINATOR_GAS_BUDGET = 40_000_000
PLACEHOLDER_GAS = 2**24  # skips estimation while building; replaced by a fresh estimate before signing


def artifacts():
    tools = Path("out/tools/foundry").resolve()
    forge = tools / ("forge.exe" if os.name == "nt" else "forge")
    env = {**os.environ, "PATH": str(tools) + os.pathsep + os.environ.get("PATH", "")}
    build = subprocess.run([str(forge), "build", "--root", "contracts", "--offline"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    if build.returncode:
        raise RuntimeError(build.stdout)
    pool = json.loads(Path("contracts/out/RewardPool.sol/RewardPool.json").read_text(encoding="utf-8"))
    token = json.loads(Path("contracts/out/RewardToken.sol/RewardToken.json").read_text(encoding="utf-8"))
    return pool, token


class Chain:
    """Signs locally, sends sequentially per sender and records every receipt."""

    def __init__(self, web3: Web3, local: bool):
        self.web3, self.local, self.log, self.nonces, self.block = web3, local, [], {}, 0

    def retry(self, action, what, attempts=40):
        """Public RPC endpoints are load-balanced; a node can briefly lag the receipt it just served."""
        for attempt in range(attempts):
            try:
                return action()
            except Exception as error:  # noqa: BLE001 - every RPC failure is retried, then reported
                if attempt == attempts - 1:
                    raise RuntimeError(f"{what} kept failing: {error}") from error
                time.sleep(3)

    def read(self, function):
        """Read contract state at (or after) the last mined transaction of this demonstration."""
        return self.retry(lambda: function.call(block_identifier=self.block or "latest"), "contract read")

    def fees(self):
        base = self.web3.eth.get_block("latest")["baseFeePerGas"]
        tip = max(self.web3.eth.max_priority_fee, Web3.to_wei(0.1, "gwei"))
        return {"maxFeePerGas": 2*base + tip, "maxPriorityFeePerGas": tip}

    def send(self, account, transaction, operation, round=0):
        if account.address not in self.nonces:
            self.nonces[account.address] = self.web3.eth.get_transaction_count(account.address, "pending")
        transaction = {**transaction, "from": account.address, "chainId": self.web3.eth.chain_id,
                       "nonce": self.nonces[account.address], **self.fees()}
        transaction.pop("gasPrice", None)
        if "gas" not in transaction:
            # Estimate against the block of our last mined transaction: a lagging node rejects that block
            # (and is retried) instead of silently estimating against older state.
            estimate = self.retry(lambda: self.web3.eth.estimate_gas(transaction, block_identifier=self.block or "latest"),
                                  f"gas estimate for {operation}")
            # The network enforces its own per-transaction limit; a rejected limit surfaces as an RPC error.
            transaction["gas"] = int(estimate*1.2) + 10_000
        signed = account.sign_transaction(transaction)
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        self.nonces[account.address] += 1
        return tx_hash, operation, round, account.address

    def call(self, account, function, operation, round=0, **extra):
        built = self.retry(lambda: function.build_transaction({"from": account.address, "gas": PLACEHOLDER_GAS, **extra}),
                           f"building {operation}")
        built.pop("gas")
        return self.send(account, built, operation, round)

    def wait(self, *pending):
        receipts = []
        for tx_hash, operation, round, sender in pending:
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=600, poll_latency=2)
            if receipt.status != 1:
                raise RuntimeError(f"transaction failed: {operation} (round {round}) {tx_hash.hex()}")
            self.log.append(dict(operation=operation, round=round, sender=sender, hash="0x"+bytes(tx_hash).hex().removeprefix("0x"),
                                 block=receipt.blockNumber, gas_used=receipt.gasUsed,
                                 effective_gas_price=receipt.get("effectiveGasPrice", 0)))
            receipts.append(receipt)
            self.block = max(self.block, receipt.blockNumber)
            print(f"  {operation:<14} round {round}: block {receipt.blockNumber}, gas {receipt.gasUsed}", flush=True)
        return receipts

    def until(self, timestamp):
        """Return once the chain has produced a block at or after timestamp."""
        if self.local:
            self.web3.provider.make_request("evm_setNextBlockTimestamp", [timestamp])
            self.web3.provider.make_request("evm_mine", [])
        while True:
            block = self.retry(lambda: self.web3.eth.get_block("latest"), "latest block")
            if block["timestamp"] >= timestamp and block["number"] >= self.block:
                self.block = block["number"]
                return
            time.sleep(4)


def local_chain():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    anvil = Path("out/tools/foundry").resolve()/("anvil.exe" if os.name == "nt" else "anvil")
    process = subprocess.Popen([str(anvil), "--host", "127.0.0.1", "--port", str(port), "--hardfork", "cancun", "--silent"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    web3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{port}", request_kwargs={"timeout": 30}))
    deadline = time.monotonic() + 20
    while not web3.is_connected():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("local Anvil failed to start")
        time.sleep(.1)
    return web3, process


def scores(round, clients):
    """Illustrative posted scores: honest clients split the pool; the zero update earns nothing.

    The free-rider's quality is zero in every round, so with the frozen threshold 0 and
    patience 3 it is flagged in round 3, as Mstar's penalty rule would do.
    """
    shares = [SCALE//2, SCALE//2, 0]
    flags = [False, False, round % PATIENCE == 0]
    return shares, flags


class Expected:
    """Independent integer model of the pool's settlement order."""

    def __init__(self, clients, vesting):
        self.vesting = vesting
        self.stake = {c: INITIAL_STAKE for c in clients}
        self.claimable = {c: 0 for c in clients}
        self.tranches = {c: [] for c in clients}
        self.claimed = {c: 0 for c in clients}
        self.minted = self.burned = 0

    def settle(self, round, clients, shares, flags):
        for c in self.tranches:
            mature = [t for t in self.tranches[c] if t[1] <= round]
            self.claimable[c] += sum(amount for amount, _ in mature)
            self.tranches[c] = [t for t in self.tranches[c] if t[1] > round]
        for c, share, flag in zip(clients, shares, flags):
            if flag:
                burned = self.stake[c]//SLASH_DENOMINATOR
                self.stake[c] -= burned
                kept = []
                for amount, unlock in self.tranches[c]:
                    cut = amount//SLASH_DENOMINATOR
                    burned += cut
                    kept.append((amount - cut, unlock))
                self.tranches[c] = kept
                self.burned += burned
            amount = ROUND_POOL*share//SCALE
            if amount:
                if self.vesting == 0:
                    self.claimable[c] += amount
                else:
                    self.tranches[c].append((amount, round + self.vesting))
                self.minted += amount

    def claim(self, c):
        self.claimed[c] += self.claimable[c]
        self.claimable[c] = 0


def check(chain, pool, token, expected, clients):
    problems = []
    for c in clients:
        if chain.read(pool.functions.stake(c)) != expected.stake[c]:
            problems.append(f"stake {c}")
        if chain.read(pool.functions.claimable(c)) != expected.claimable[c]:
            problems.append(f"claimable {c}")
        if chain.read(pool.functions.unvested(c)) != sum(a for a, _ in expected.tranches[c]):
            problems.append(f"unvested {c}")
        if chain.read(token.functions.balanceOf(c)) != expected.claimed[c]:
            problems.append(f"wallet {c}")
    if problems:
        raise AssertionError("on-chain state differs from the integer model: " + ", ".join(problems))


def run(args):
    pool_artifact, token_artifact = artifacts()
    process = None
    if args.local:
        web3, process = local_chain()
        key = Account.create().key.hex()
        web3.provider.make_request("anvil_setBalance", [Account.from_key(key).address, hex(10**20)])
    else:
        key = os.environ.get("SEPOLIA_PRIVATE_KEY", "").strip()
        if not key:
            raise SystemExit("set SEPOLIA_PRIVATE_KEY to the coordinator's testnet-only key")
        web3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={"timeout": 60}))
        if web3.eth.chain_id != SEPOLIA:
            raise SystemExit(f"RPC chain id {web3.eth.chain_id} is not Sepolia")
    try:
        chain = Chain(web3, args.local)
        coordinator = Account.from_key(key)
        keys_path = Path(args.keys)
        if keys_path.exists() and not args.local:
            raise SystemExit(f"{keys_path} exists from an earlier attempt: recover its funds or move it before a new run")
        clients = [Account.create() for _ in range(3)]
        names = ["honest-A", "honest-B", "free-rider"]
        if not args.local:
            keys_path.parent.mkdir(parents=True, exist_ok=True)
            keys_path.write_text(json.dumps({n: c.key.hex() for n, c in zip(names, clients)}, indent=2), encoding="utf-8")
        price = chain.fees()["maxFeePerGas"]
        fund = CLIENT_GAS_BUDGET*price
        need = COORDINATOR_GAS_BUDGET*price + len(clients)*fund
        balance = web3.eth.get_balance(coordinator.address)
        print(f"Coordinator {coordinator.address}: balance {Web3.from_wei(balance, 'ether')} ETH; "
              f"budget {Web3.from_wei(need, 'ether')} ETH at {Web3.from_wei(price, 'gwei')} gwei max fee", flush=True)
        if balance < need:
            raise SystemExit("insufficient testnet ETH: request more from a Sepolia faucet and rerun (nothing was spent)")

        print("Deploying", flush=True)
        factory = web3.eth.contract(abi=pool_artifact["abi"], bytecode=pool_artifact["bytecode"]["object"])
        receipt, = chain.wait(chain.call(coordinator, factory.constructor(args.vesting, args.window, args.window), "deploy"))
        pool = web3.eth.contract(address=receipt.contractAddress, abi=pool_artifact["abi"])
        token = web3.eth.contract(address=chain.read(pool.functions.token()), abi=token_artifact["abi"])

        print("Funding and registering clients", flush=True)
        # No fixed 21,000 gas: a network may charge more for a transfer that creates a new account.
        chain.wait(*[chain.send(coordinator, {"to": c.address, "value": fund}, "fund") for c in clients])
        chain.wait(*[chain.call(coordinator, pool.functions.grantInitial(c.address), "grant") for c in clients])
        chain.wait(*[chain.call(c, token.functions.approve(pool.address, INITIAL_STAKE), "approve") for c in clients])
        chain.wait(*[chain.call(c, pool.functions.register(), "register") for c in clients])

        addresses = [c.address for c in clients]
        expected = Expected(addresses, args.vesting)
        check(chain, pool, token, expected, addresses)
        rounds = []
        for round in range(1, args.rounds + 1):
            print(f"Round {round}", flush=True)
            chain.wait(chain.call(coordinator, pool.functions.openRound(round, addresses), "open", round))
            commit_deadline, reveal_deadline = chain.read(pool.functions.roundInfo(round))[:2]
            digests, salts = [], []
            for c in clients:
                update = b"zero update" if c is clients[2] else os.urandom(32)
                digests.append(hashlib.sha256(update + round.to_bytes(4, "big")).digest())
                salts.append(os.urandom(32))
            commitments = [Web3.keccak(web3.codec.encode(["bytes32", "bytes32", "uint256", "address"], [d, s, round, c.address]))
                           for d, s, c in zip(digests, salts, clients)]
            chain.wait(*[chain.call(c, pool.functions.commit(round, h), "commit", round) for c, h in zip(clients, commitments)])
            chain.until(commit_deadline)
            chain.wait(*[chain.call(c, pool.functions.reveal(round, d, s), "reveal", round)
                         for c, d, s in zip(clients, digests, salts)])
            chain.until(reveal_deadline)
            shares, flags = scores(round, clients)
            chain.wait(chain.call(coordinator, pool.functions.postScores(round, addresses, shares, flags), "post_scores", round))
            chain.wait(chain.call(coordinator, pool.functions.settle(round), "settle", round))
            expected.settle(round, addresses, shares, flags)
            check(chain, pool, token, expected, addresses)
            claimers = [c for c in clients if expected.claimable[c.address]]
            if claimers:
                chain.wait(*[chain.call(c, pool.functions.claim(), "claim", round) for c in claimers])
                for c in claimers:
                    expected.claim(c.address)
                check(chain, pool, token, expected, addresses)
            rounds.append(dict(round=round, shares1e18=[str(s) for s in shares], slash=flags,
                               commit_deadline=commit_deadline, reveal_deadline=reveal_deadline))

        state = {name: dict(address=c.address, stake=str(expected.stake[c.address]), wallet=str(expected.claimed[c.address]),
                            claimable=str(expected.claimable[c.address]),
                            unvested=str(sum(a for a, _ in expected.tranches[c.address])))
                 for name, c in zip(names, clients)}
        if not args.local:
            print("Returning unused client ETH", flush=True)
            sweeps = []
            for c in clients:
                left = chain.retry(lambda: web3.eth.get_balance(c.address, block_identifier=chain.block), "balance")
                gas = int(chain.retry(lambda: web3.eth.estimate_gas({"from": c.address, "to": coordinator.address, "value": 1}),
                                      "sweep estimate")*1.2) + 1_000
                cost = gas*chain.fees()["maxFeePerGas"]
                if left > cost:
                    sweeps.append(chain.send(c, {"to": coordinator.address, "value": left - cost, "gas": gas}, "sweep"))
            chain.wait(*sweeps)
        bytecode = bytes.fromhex(pool_artifact["bytecode"]["object"].removeprefix("0x"))
        record = dict(schema=1, network="local" if args.local else "sepolia", chain_id=web3.eth.chain_id,
                      explorer=None if args.local else EXPLORER, completed_utc=datetime.now(timezone.utc).isoformat(),
                      pool=pool.address, token=token.address, coordinator=coordinator.address,
                      constructor=dict(vesting_delay=args.vesting, commit_window=args.window, reveal_window=args.window),
                      policy=dict(round_pool=str(ROUND_POOL), initial_stake=str(INITIAL_STAKE), slash_denominator=SLASH_DENOMINATOR,
                                  patience=PATIENCE),
                      pool_source_sha256=hashlib.sha256(Path("contracts/src/RewardPool.sol").read_bytes()).hexdigest(),
                      token_source_sha256=hashlib.sha256(Path("contracts/src/RewardToken.sol").read_bytes()).hexdigest(),
                      creation_bytecode_sha256=hashlib.sha256(bytecode).hexdigest(),
                      git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      rounds=rounds, final_state=state, minted=str(expected.minted), burned=str(expected.burned),
                      state_checks="every round matched the independent integer model", transactions=chain.log,
                      total_gas_used=sum(t["gas_used"] for t in chain.log))
        return record
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


def median(values):
    values = sorted(values)
    return values[len(values)//2] if len(values) % 2 else (values[len(values)//2 - 1] + values[len(values)//2])//2


def gas_table(record, reference):
    """Median gas per operation on the public network against the same script on the local Cancun chain."""
    rows = ["", "## Gas against the local Cancun schedule", "",
            "Medians per operation for the same script, contracts and protocol; the local run is "
            "`python scripts/sepolia_demo.py --local` (`results/sepolia/local_reference.json`).", "",
            "| Operation | Local Cancun | Sepolia | Ratio |", "|---|---:|---:|---:|"]
    for operation in ["deploy", "fund", "grant", "approve", "register", "open", "commit", "reveal", "post_scores", "settle", "claim"]:
        here = [x["gas_used"] for x in record["transactions"] if x["operation"] == operation]
        local = [x["gas_used"] for x in reference["transactions"] if x["operation"] == operation]
        if here and local:
            rows.append(f"| {operation} | {median(local):,} | {median(here):,} | {median(here)/median(local):.2f}x |")
    rows.append(f"| total (without sweeps) | {sum(x['gas_used'] for x in reference['transactions']):,} | "
                f"{sum(x['gas_used'] for x in record['transactions'] if x['operation'] != 'sweep'):,} | "
                f"{sum(x['gas_used'] for x in record['transactions'] if x['operation'] != 'sweep')/sum(x['gas_used'] for x in reference['transactions']):.2f}x |")
    return rows


def document(record, path, reference=None):
    url = lambda kind, value: f"[{value[:10]}…]({record['explorer']}/{kind}/{value})" if record["explorer"] else f"`{value}`"
    lines = ["# Public-testnet demonstration", "",
             "Generated by `python scripts/sepolia_demo.py` from `results/sepolia/demo.json`.", "",
             f"The unchanged reward pool ran three complete rounds on {record['network'].capitalize()} "
             f"(chain id {record['chain_id']}) on {record['completed_utc'][:10]}: two honest clients and one zero-update "
             "free-rider registered with stake, committed and revealed update digests, received coordinator-posted scores, "
             "settled with vesting, and claimed matured rewards. The free-rider was slashed in round 3 after three zero-quality "
             "participations (threshold 0, patience 3). After every round the pool's stakes, claimable balances, unvested "
             "tranches and wallets matched an independent integer model of the settlement order.", "",
             f"- Pool: {url('address', record['pool'])}; token: {url('address', record['token'])}; "
             f"coordinator: {url('address', record['coordinator'])}",
             f"- Constructor: vesting delay {record['constructor']['vesting_delay']} round(s), commit and reveal windows "
             f"{record['constructor']['commit_window']} s each (demonstration timing; the measured local policy uses 3 rounds and 3600 s). "
             "Pool size, stake and slash fraction are compile-time constants and unchanged.",
             f"- Source commit `{record['git_commit'][:7]}`; RewardPool source SHA-256 `{record['pool_source_sha256'][:16]}…`; "
             f"creation bytecode SHA-256 `{record['creation_bytecode_sha256'][:16]}…`.",
             f"- {len(record['transactions'])} transactions, {record['total_gas_used']:,} gas in total.",
             f"- Deployment used {next(x['gas_used'] for x in record['transactions'] if x['operation'] == 'deploy'):,} gas here, "
             "against about 2.39 million for the same contracts in the local Cancun measurements ([contracts](CONTRACTS.md)). "
             "The recorded gas figures therefore describe the Cancun schedule; a network that prices state creation higher "
             "charges more and fits fewer participants under the per-transaction cap.", "",
             "Scores are illustrative protocol inputs: the honest clients split each round's pool and the zero update earns "
             "nothing. The experiment's allocations are verified separately by replaying the recorded grid on a local chain "
             "([contracts](CONTRACTS.md)). The demonstration shows the same bytecode settling on a public network; the "
             "coordinator remains a trusted score source, as in the threat model.", "",
             "## Final state (tokens)", "", "| Client | Stake | Wallet | Claimable | Unvested |", "|---|---:|---:|---:|---:|"]
    for name, s in record["final_state"].items():
        lines.append(f"| {name} | {int(s['stake'])/SCALE:g} | {int(s['wallet'])/SCALE:g} | {int(s['claimable'])/SCALE:g} | "
                     f"{int(s['unvested'])/SCALE:g} |")
    if reference is not None:
        lines += gas_table(record, reference)
    lines += ["", "## Transactions", "", "| Round | Operation | Transaction | Gas |", "|---:|---|---|---:|"]
    for t in record["transactions"]:
        lines.append(f"| {t['round']} | {t['operation']} | {url('tx', t['hash'])} | {t['gas_used']:,} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--local", action="store_true", help="run against a fresh local Anvil chain instead of Sepolia")
    parser.add_argument("--rpc", default=DEFAULT_RPC)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--window", type=int, default=90, help="commit and reveal windows in seconds")
    parser.add_argument("--vesting", type=int, default=1, help="vesting delay in rounds")
    parser.add_argument("--keys", default="out/sepolia/client_keys.json", help="where throwaway client keys are kept")
    parser.add_argument("--output", default="results/sepolia/demo.json")
    parser.add_argument("--reference", default="results/sepolia/local_reference.json")
    parser.add_argument("--document-only", action="store_true", help="rewrite docs/SEPOLIA.md from the recorded runs")
    args = parser.parse_args()
    reference_path = Path(args.reference)
    if args.document_only:
        record = json.loads(Path(args.output).read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else None
        document(record, "docs/SEPOLIA.md", reference)
        print("Rewrote docs/SEPOLIA.md")
        return 0
    if args.rounds < PATIENCE:
        raise SystemExit(f"at least {PATIENCE} rounds are needed to show the slash")
    record = run(args)
    output = reference_path if args.local else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    if not args.local:
        reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else None
        document(record, "docs/SEPOLIA.md", reference)
    print(f"Completed {record['network']} demonstration: {len(record['transactions'])} transactions, "
          f"{record['total_gas_used']:,} gas; record {output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
