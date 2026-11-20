"""Render the proposed method and pilot plan without claiming a frozen rule."""

import json
from pathlib import Path

from fltoken.io import write_json


def main():
    p = json.loads(Path("configs/method_pilot.json").read_text(encoding="utf-8"))
    write_json("results/method/proposed.json", p)
    p = json.loads(Path("results/method/proposed.json").read_text(encoding="utf-8"))
    text = r"""# Proposed reward method

Status: **pre-pilot design, not an implemented or evaluated defense**. This note is generated from the recorded design prior by `python scripts/document_method.py`. No parameter is selected from reproduction or evaluation-seed outcomes. A later frozen configuration must identify the pilot measurements and the selection commit before this rule scores evaluation updates.

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

Settlement first matures previously vested tranches for the current round, then applies a slash to the remaining stake and still-unvested tranches using integer floor arithmetic, then records the current round's minted rewards with their unlock rounds. Matured claimable rewards are not retroactively slashed. Claims transfer only matured balances. The implementation and replay tests must pin this order; it is presently a specification, not a working contract.

Minted attacker share remains the required allocation metric even if rewards are later burned. Net claimed balance, unvested balance and stake loss must be reported separately; slashing cannot be presented as reducing a historical minted-share metric. Rounding of float shares still has an exposed budget issue in the baseline ledger and must be resolved before settlement exactness is claimed.

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
    text += r"""The objective gives equal weight to reducing attacker allocation and limiting honest false penalties, avoiding an unreported one-sided optimum. Compute means over complete seed/attack/fraction cells, not over the number of participations or only successful runs. The attack-free cells contribute honest-cost measurements, while attacker allocation is averaged over attack-present cells. Report all candidates and the tradeoff, not just the selected scalar objective. Freeze the measured configuration and a digest of the pilot corpus before scoring evaluation seeds.

## Required implementation checks

- Perturb a client's update and verify its own reference is unchanged.
- Permute client ids and updates together; compare shares, counters and reputations by id.
- Apply a common positive scale; compare quality, rewards and flags across extreme but representable scales.
- Repeat a stateful sequence, including absent clients, duplicate updates and all-zero rounds.
- Disable each component and verify the specified ablation behavior with separate state.
- Replay maturity, slash, new reward and claim boundaries with exact integer balances.

No empirical improvement, pilot winner, contract safety or theorem is claimed in this design note.
"""
    Path("docs/METHOD.md").write_text(text, encoding="utf-8", newline="\n")
    print("Generated pre-pilot method specification; no selected configuration")


if __name__ == "__main__":
    main()
