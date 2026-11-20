# Proposed reward method

Status: **implemented with synthetic contract checks; pilot selection and empirical evaluation pending**. This note is generated from the recorded design prior by `python scripts/document_method.py`. No parameter is selected from reproduction or evaluation-seed outcomes. A later frozen configuration must identify the pilot measurements and the selection commit before this rule scores evaluation updates.

## Objective and threat model

Reduce rewards to the declared free-rider and poisoning identities while measuring the cost to heterogeneous honest clients. The coordinator sees individual updates and computes scores honestly. Clients can submit the fixed attack transformations in the experiment. A ledger enforces the coordinator's posted decisions; it cannot prove that a cosine score reflects useful private data. Adaptive contribution attacks, colluding-majority updates and dishonest coordination are not covered by a security theorem.

## Reference and immediate quality

For a sampled client i, construct each coordinate of r_i as the median of that coordinate among all sampled updates except delta_i. Compute c_i = cos(delta_i, r_i), with zero cosine when either norm is zero, and u_i = max(0, c_i). The reference does not contain delta_i. A common positive rescaling of all updates scales r_i equally and leaves c_i unchanged. Reordering clients reorders the outputs; reputation and counters are keyed by client id.

The coordinate median is inexpensive at the sampled-set size and limits an extreme coordinate's influence when the other clients have an honest majority. It is basis dependent and can suppress legitimate minority data. Exclusion removes the direct self inner-product term; it does not certify honesty or remove correlations among clients. No norm threshold or additive epsilon is introduced into the cosine, preserving the stated scale invariance.

## Participation-based reputation

Maintain rho_i by scientific client id. For a sampled client, rho_i' = beta*rho_i + (1-beta)*u_i; absent clients retain their state. When reputation is enabled, raw reward weight is a_i = u_i*rho_i'. When disabled, a_i = u_i. Multiplication by current quality prevents a previously reputable zero update from earning solely on its history. Initialization is shared across clients, so an identity receives no personalized prior advantage.

Normalize the nonnegative weights over this sampled set; if all weights vanish, pay nobody. The ablation uses the same updates and separate state instances for reference-only, reference plus reputation, and reference plus reputation plus penalties. Disabling the reference mitigation uses the self-inclusive mean explicitly as a diagnostic; it is not the full proposed rule.

## Persistent weak contributions and settlement

Maintain a consecutive-low-participation counter b_i. A participation is low if u_i <= tau; otherwise reset its counter. Absence neither increments nor resets the counter. When b_i reaches the selected patience, emit a slash flag and reset the counter. Set that participation's raw reward weight to zero when penalties are enabled, then normalize remaining weights. A client can recover on a later useful participation; no permanent ban is proposed.

The penalty component includes deferred settlement: when disabled, minted rewards are immediately claimable and slash flags remain false; when enabled, the fixed vesting delay applies. This permits a reference-only, reference-plus-EMA and full economic ablation. Allocation metrics are always computed from minted rewards; delayed claims and subsequent burns are separate outcomes.

Settlement first matures previously vested tranches for the current round, then applies a slash to the remaining stake and still-unvested tranches using integer floor arithmetic, then records the current round's minted rewards with their unlock rounds. Matured claimable rewards are not retroactively slashed. Claims transfer only matured balances. The implementation and independent integer-state replay pin this order; see [contract evidence](CONTRACTS.md).

Minted attacker share remains the required allocation metric even if rewards are later burned. Net claimed balance, unvested balance and stake loss must be reported separately; slashing cannot be presented as reducing a historical minted-share metric. The baseline ledger now converts floating shares through exact rational normalization and floors integer base units, leaving rounding remainders unminted. The recorded baseline replays establish exact on-chain settlement for their inputs. Stateful random histories
exercise slashing and vesting; empirical selected-method replay remains a separate required measurement.

## Proposed parameters and selection

| Parameter | Proposed value or candidates | Reason |
|---|---|---|
| Pilot seeds | [100, 101, 102, 103, 104] | Disjoint from evaluation; every candidate uses common cached pilot updates. |
| Initial reputation | 0.5 | Symmetric midrange prior; no client-specific information. |
| EMA decay | [0.0, 0.5, 0.9] | Includes memoryless, moderate and persistent memory. |
| Low-quality threshold | [0.0, 0.025, 0.05] | Includes nonpositive-only and small positive alignment thresholds. |
| Patience | [2, 3] | Requires repeated weak participations instead of a single noisy event. |
| Initial stake (tokens) | 100 | Equal endowments isolate update quality from wealth. |
| Slash fraction | 1/10 | Fixed economic design prior; not optimized for attack suppression. |
| Vesting delay (rounds) | 3 | Short delayed claim window for persistent evidence, fixed before pilots. |

Selection objective: minimize mean attacker share plus mean honest false-penalty rate among fully enabled candidates; ties prefer lower threshold, longer patience, lower EMA decay.

The objective gives equal weight to reducing attacker allocation and limiting honest false penalties, avoiding an unreported one-sided optimum. Compute means over complete seed/attack/fraction cells, not over the number of participations or only successful runs. The attack-free cells contribute honest-cost measurements, while attacker allocation is averaged over attack-present cells. A cell with no minted rewards receives unit attacker-allocation cost instead of disappearing from the average. Report all candidates and the tradeoff, not just the selected scalar objective. Freeze the measured configuration and a digest of the pilot corpus before scoring evaluation seeds.

Candidate reference geometry is identical, so the pilot program computes quality once per common update matrix, then applies each candidate's independent state transitions. Cached matrices must match the client order and SHA-256 in the original training log. The compact recorded quality tables allow the allocation comparison to be regenerated without model checkpoints; recomputing those qualities requires the training update cache.

## Pilot execution and freeze

Commit the plan, scorer and selection code before empirical execution. The programs reject uncommitted changes to their scientific inputs. Run:

```bash
python scripts/run_pilots.py
python scripts/select_method.py --freeze
```

Commit and publish the resulting `configs/method_frozen.json`, pilot evidence and selection summary before evaluation scoring. The frozen file binds parameters to the corpus, plan, selection summary, scorer and metric-implementation digests. To regenerate candidate allocation measurements from recorded qualities, use `python scripts/select_method.py --recorded-only`.

The fixed Sendros mixture always has the same attack counts, and attack-free cells always have no attackers. Each occurs once per seed. Every single attack occurs at each declared fraction, with one common update corpus for every candidate.

The selected Mstar configuration always enables the reference, EMA and penalties. Only their numerical parameters are selected from pilots. The component ablation retains those numerical parameters and forces the switches through reference-only, reference plus EMA, and reference plus EMA plus penalties. The final stage equals Mstar. No stage gets a separate parameter search. A zero selected EMA decay means the EMA-enabled stage is memoryless; it must not be described as evidence for a benefit from temporal memory.

## Required implementation checks

- Perturb a client's update and verify its own reference is unchanged.
- Permute client ids and updates together; compare shares, counters and reputations by id.
- Apply a common positive scale; compare quality, rewards and flags across extreme but representable scales.
- Repeat a stateful sequence, including absent clients, duplicate updates and all-zero rounds.
- Disable each component and verify the specified ablation behavior with separate state.
- Replay maturity, slash, new reward and claim boundaries with exact integer balances.

No empirical improvement, pilot winner, contract safety or theorem is claimed in this design note.

The predeclared corpus has 85 cells and 18 distinct candidate configurations.

The callable development defaults below exercise every component. They are not selected empirical parameters; evaluation must use the committed pilot selection instead.

```json
{
  "beta": 0.5,
  "ema": true,
  "initial_reputation": 0.5,
  "patience": 3,
  "penalties": true,
  "reference": true,
  "threshold": 0.0
}
```
