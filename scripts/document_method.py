"""Render the proposed method and pilot plan without claiming a frozen rule."""

import json
from pathlib import Path

from fltoken.io import write_json
from fltoken.ours import OursScorer
from run_pilots import cells
from select_method import candidates


def main():
    p = json.loads(Path("configs/method_pilot.json").read_text(encoding="utf-8"))
    plan = json.loads(Path("configs/pilot_plan.json").read_text(encoding="utf-8"))
    p["pilot_plan"] = plan
    p["pilot_cells"] = sum(1 for _ in cells(plan))
    p["pilot_candidates"] = sum(1 for _ in candidates(plan))
    p["development_defaults"] = OursScorer.DEFAULTS
    write_json("results/method/proposed.json", p)
    p = json.loads(Path("results/method/proposed.json").read_text(encoding="utf-8"))
    text = r"""# Proposed reward method

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

Settlement first matures previously vested tranches for the current round, then applies a slash to the remaining stake and still-unvested tranches using integer floor arithmetic, then records the current round's minted rewards with their unlock rounds. Matured claimable rewards are not retroactively slashed. Claims transfer only matured balances. The implementation and replay tests must pin this order; it is presently a specification, not a working contract.

Minted attacker share remains the required allocation metric even if rewards are later burned. Net claimed balance, unvested balance and stake loss must be reported separately; slashing cannot be presented as reducing a historical minted-share metric. The baseline ledger now converts floating shares through exact rational normalization and floors integer base units, leaving rounding remainders unminted. This resolves off-chain budget conservation; contract replay is still required before claiming on-chain equality.

## Proposed parameters and selection

"""
    text += "| Parameter | Proposed value or candidates | Reason |\n|---|---|---|\n"
    rows = [
        ("Pilot seeds", p["pilot_seeds"], "Disjoint from evaluation; every candidate uses common cached pilot updates."),
        ("Initial reputation", p["ema_initial"], "Symmetric midrange prior; no client-specific information."),
        ("EMA decay", p["ema_decay_candidates"], "Includes memoryless, moderate and persistent memory."),
        ("Low-quality threshold", p["low_quality_candidates"], "Includes nonpositive-only and small positive alignment thresholds."),
        ("Patience", p["patience_candidates"], "Requires repeated weak participations instead of a single noisy event."),
        ("Initial stake (tokens)", p["stake_tokens"], "Equal endowments isolate update quality from wealth."),
        ("Slash fraction", f"{p['slash_numerator']}/{p['slash_denominator']}", "Fixed economic design prior; not optimized for attack suppression."),
        ("Vesting delay (rounds)", p["vesting_rounds"], "Short delayed claim window for persistent evidence, fixed before pilots."),
    ]
    text += "\n".join(f"| {key} | {value} | {reason} |" for key, value, reason in rows)
    text += "\n\nSelection objective: " + p["selection"] + ".\n\n"
    text += r"""The objective gives equal weight to reducing attacker allocation and limiting honest false penalties, avoiding an unreported one-sided optimum. Compute means over complete seed/attack/fraction cells, not over the number of participations or only successful runs. The attack-free cells contribute honest-cost measurements, while attacker allocation is averaged over attack-present cells. A cell with no minted rewards receives unit attacker-allocation cost instead of disappearing from the average. Report all candidates and the tradeoff, not just the selected scalar objective. Freeze the measured configuration and a digest of the pilot corpus before scoring evaluation seeds.

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
"""
    text += f"\nThe predeclared corpus has {p['pilot_cells']} cells and {p['pilot_candidates']} distinct candidate configurations.\n"
    text += "\nThe callable development defaults below exercise every component. They are not selected empirical parameters; evaluation must use the committed pilot selection instead.\n\n```json\n"
    text += json.dumps(p["development_defaults"],sort_keys=True,indent=2)+"\n```\n"
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    if "contracts" in summary:
        text = text.replace("implementation and replay tests must pin this order; it is presently a specification, not a working contract.",
                            "implementation and independent integer-state replay pin this order; see [contract evidence](CONTRACTS.md).")
        text = text.replace("This resolves off-chain budget conservation; contract replay is still required before claiming on-chain equality.",
                            "The recorded baseline replays establish exact on-chain settlement for their inputs. Stateful random histories\nexercise slashing and vesting; empirical selected-method replay remains a separate required measurement.")
    Path("docs/METHOD.md").write_text(text, encoding="utf-8", newline="\n")
    print("Generated pre-pilot method specification; no selected configuration")


if __name__ == "__main__":
    main()
