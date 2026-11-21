# Grid settlement and retained balances

Generated from `results/summary.json` by `python scripts/document_grid_settlement.py`.

All 196 declared run-rule jobs reproduce 4280 round-rule states exactly, using 90 common CUDA training runs. The chain replay itself uses a local CPU process. It does not train another model.

## Coverage

| Rule | Replayed runs | Round-rule states |
|---|---:|---:|
| B0 | 8 | 190 |
| B1 | 90 | 1950 |
| B2 | 8 | 190 |
| Mstar | 90 | 1950 |

B1 and Mstar include every planned main and long-horizon run. B0 and B2 use the predeclared sample. Their smaller coverage is explicit; aggregate balances across these unequal samples are not comparable performance estimates.

All replayed contracts use the frozen vesting delay of 3 rounds and the same initial economic settings. Only the listed rules were executed on-chain; the component analysis below explicitly identifies its off-chain scope.

## Allocation, burns and retained wealth

[Per-run economic records](../results/grid_onchain/economic_outcomes.csv) retain separate honest and adversary groups, condition, seed, horizon, rule and original run path. Every amount is an integer token base-unit value. They come from the validated on-chain terminal state and original integer payout ledger.

- `minted_rewards`: rewards issued during the run, before later burns. The required attacker-share metric uses these historical amounts.
- `claimed_rewards`: reward tokens already transferred into client wallets by the recorded claim schedule.
- `claimable_rewards`: matured rewards still held by the pool and available to claim.
- `unvested_rewards`: surviving reward tranches whose unlock rounds have not yet settled.
- `retained_rewards`: claimed plus claimable plus unvested rewards. This includes locked balances, not only immediately spendable rewards.
- `burned_rewards`: historical minted rewards minus retained rewards.
- `remaining_stake` and `burned_stake`: surviving initial collateral and its loss, kept separate from reward burns.

The decomposition conserves issuance exactly: minted rewards equal retained rewards plus reward burns; initial stake equals remaining stake plus stake burns. Reward and stake burns together equal recorded contract burns. Remaining stake plus retained rewards equals final token supply. Claimable rewards, unvested rewards and stake equal the pool's token balance.

A run with no minted rewards retains an undefined minted attacker share. Slashing can reduce retained wealth without changing historical minted allocation. Claim timing can change the split between wallets and claimable balances without changing their sum. Final unvested balances are retained liabilities, not claims that a withdrawal has occurred; later unlock rounds are outside the recorded horizon.

## Reconstruction

```bash
python scripts/replay_grid.py --no-run
python scripts/summarize_grid_settlement.py
python scripts/document_grid_settlement.py
```

These commands validate existing evidence without launching chains or downloading data. Full unique coverage, the completed training clock, published method, original source/configuration bindings and seeded attack identities are required. Floating participation shares reconstruct the integer ledger before every recorded state is compared with the independent settlement oracle. Refreshing hashes cannot excuse a different configuration, incorrect balance, duplicated job or missing run.

The compact [coverage table](../paper/generated/settlement_coverage.tex) requires `booktabs` and fits a single column. Exact monetary records remain in the CSV and shared JSON summary. Coverage counts establish successful reconstruction of these recorded histories; they do not prove honest off-chain scores, private data, unique identities, adaptive-attack security or a general smart-contract safety theorem. See [contract assumptions](CONTRACTS.md).

## What the penalty component changes

Aema and Afull have identical integer share allocations in 90 of 90 runs. At the frozen zero threshold, the flagged current quality and weight are already zero; additional weight zeroing therefore cannot change minted allocation. Penalties instead burn 10% of remaining stake and each still-unvested prior reward tranche, rounded down separately. Maturity occurs before slashing, and the current round's new rewards are created afterward.

The EMA-only ledger is settled off-chain with the same contract economics and claims after each round. The full component uses its recorded ledger and its schedule-independent balances are checked against the validated Mstar Anvil states. The EMA-only results are a settlement ablation, not additional chain replays.

Forfeited unvested rewards total 0 base units across 90 full-component histories; 0 histories have a positive reward burn. At the frozen zero quality threshold, 3 bad participations are needed before a slash, while rewards vest after 3 rounds. Each bad participation earns no new reward; the preceding positive reward has matured by the slash, even with gaps between participations. The measured penalty effect is therefore entirely stake loss. Reward forfeiture does not contribute to this grid's outcomes.

[Every component/run/group](../results/grid_onchain/penalty_ablation.csv) retains exact base-unit losses, net outcomes and event counts. `net_token_outcome` is minted rewards minus forfeited unvested rewards minus slashed stake: final wealth minus initial collateral. It may be negative. `ever_slashed_clients` counts distinct flagged clients within a run; `slash_events` counts repeated flags; `clients_with_balance_loss` counts clients with a positive stake or reward burn.

| Horizon | Runs | Honest client-run identities | Honest ever slashed | Attacker client-run identities | Attackers ever slashed |
|---|---:|---:|---:|---:|---:|
| long | 5 | 125 | 44 | 25 | 21 |
| main | 85 | 2075 | 29 | 475 | 296 |

Counts sum distinct experimental histories, not unique people across runs. The main horizon includes attack-free runs; the long horizon contains only the fixed mixture.

The following estimates average scenario totals within each seed, then bootstrap the complete seed means. Monetary estimates are tokens per run for the entire group, not tokens per client. The CSV preserves integer base units. All estimates include both losses and retained gains.

| Horizon / stage / group | Slashed stake, mean [CI] | Forfeited rewards, mean [CI] | Net outcome, mean [CI] |
|---|---:|---:|---:|
| long / Aema / honest | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 47576.544 [47016.365, 48192.264] |
| long / Aema / adversary | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 2223.456 [1473.171, 2973.742] |
| long / Afull / honest | 111.040 [74.620, 151.280] | 0.000 [0.000, 0.000] | 47465.504 [46907.531, 48050.584] |
| long / Afull / adversary | 118.123 [98.118, 137.303] | 0.000 [0.000, 0.000] | 2105.333 [1343.642, 2867.023] |
| main / Aema / honest | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 18800.676 [18574.051, 19015.344] |
| main / Aema / adversary | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 964.029 [855.244, 1073.007] |
| main / Afull / honest | 3.412 [1.529, 5.294] | 0.000 [0.000, 0.000] | 18797.265 [18569.110, 19013.815] |
| main / Afull / adversary | 55.566 [53.462, 57.632] | 0.000 [0.000, 0.000] | 908.464 [799.763, 1017.165] |
