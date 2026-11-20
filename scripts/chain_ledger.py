"""Independent integer-state oracle for reward-pool settlement and claims."""

from copy import deepcopy

SCALE = 10**18
ROUND_POOL = 1000 * SCALE
INITIAL_STAKE = 100 * SCALE


class ChainLedger:
    def __init__(self, clients, vesting_delay=3):
        if type(vesting_delay) is not int or not 0 <= vesting_delay <= 50:
            raise ValueError("vesting delay must be an integer from 0 through 50")
        clients = list(clients)
        if not clients or len(set(clients)) != len(clients):
            raise ValueError("registered clients must be nonempty and unique")
        self.delay = vesting_delay
        self.round = 0
        self.unminted = 0
        self.accounts = {client: dict(stake=INITIAL_STAKE, claimable=0, wallet=0, tranches={}) for client in clients}
        self.supply = len(self.accounts) * INITIAL_STAKE

    def settle(self, item):
        round = item["round"]
        ids, shares, flags = item["ids"], item["shares1e18"], item["slash"]
        if round != self.round + 1 or not ids or len(set(ids)) != len(ids):
            raise ValueError("round order or participant uniqueness violated")
        if not (len(ids) == len(shares) == len(flags)) or any(i not in self.accounts for i in ids):
            raise ValueError("invalid participants")
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("slash flags must be booleans")
        if any(type(s) is not int or s < 0 for s in shares) or sum(shares) > SCALE:
            raise ValueError("invalid integer shares")
        amounts = [ROUND_POOL * s // SCALE for s in shares]
        if amounts != item["payouts"] or item["remainder"] != ROUND_POOL - sum(amounts):
            raise ValueError("recorded payouts or remainder differ from integer shares")
        for account in self.accounts.values():
            matured = [unlock for unlock in account["tranches"] if unlock <= round]
            for unlock in matured:
                account["claimable"] += account["tranches"].pop(unlock)
        burned = 0
        for client, amount, slash in zip(ids, amounts, flags):
            account = self.accounts[client]
            if slash:
                stake_burn = account["stake"] // 10
                account["stake"] -= stake_burn
                burned += stake_burn
                for unlock, pending in account["tranches"].items():
                    reward_burn = pending // 10
                    account["tranches"][unlock] -= reward_burn
                    burned += reward_burn
            if amount:
                if self.delay:
                    account["tranches"][round + self.delay] = amount
                else:
                    account["claimable"] += amount
        minted = sum(amounts)
        self.supply += minted - burned
        self.unminted += ROUND_POOL - minted
        self.round = round
        return dict(round=round, minted=minted, burned=burned, remainder=ROUND_POOL - minted)

    def claim(self, client):
        account = self.accounts[client]
        amount = account["claimable"]
        account["wallet"] += amount
        account["claimable"] = 0
        return amount

    def snapshot(self):
        accounts = deepcopy(self.accounts)
        for account in accounts.values():
            account["unvested"] = sum(account["tranches"].values())
        pool_balance = sum(a["stake"] + a["claimable"] + a["unvested"] for a in accounts.values())
        if self.supply != pool_balance + sum(a["wallet"] for a in accounts.values()):
            raise AssertionError("token liabilities do not conserve supply")
        return dict(round=self.round, accounts=accounts, pool_balance=pool_balance,
                    supply=self.supply, unminted=self.unminted)
