"""Render the design prior and, when available, the hash-bound pilot selection."""

import json
from pathlib import Path

from evaluation import recorded_method
from fltoken.io import sha256, write_json
from fltoken.ours import OursScorer
from run_pilots import cells
from select_method import candidates


def frozen_selection():
    target = Path("configs/method_frozen.json")
    if not target.exists():
        return None
    frozen = recorded_method()
    selection_path = Path("results/pilot_selection/summary.json")
    if sha256(selection_path) != frozen["selection_summary_sha256"]:
        raise ValueError("frozen method evidence differs: selection summary")
    result = json.loads(selection_path.read_text(encoding="utf-8"))
    if result["selected"]["params"] != frozen["params"] or result["corpus_sha256"] != frozen["corpus_sha256"]:
        raise ValueError("selected parameters or pilot corpus differ from the frozen method")
    return {"configuration": frozen, "selection": result}


def selection_note(record):
    frozen, selection = record["configuration"], record["selection"]
    selected = selection["selected"]
    reasons = {
        "reference": "Predeclared exclusion of the scored client's update.",
        "ema": "Enabled in every candidate; ablation alone disables this component.",
        "penalties": "Enabled in every candidate; ablation alone disables this component.",
        "beta": "Selected jointly by the predeclared pilot objective and tie order.",
        "threshold": "Selected jointly by the predeclared pilot objective and tie order.",
        "patience": "Selected jointly by the predeclared pilot objective and tie order.",
        "initial_reputation": "Fixed symmetric prior, not optimized.",
        "vesting_rounds": "Fixed economic design prior, not optimized.",
        "slash_numerator": "Fixed economic design prior, not optimized.",
        "slash_denominator": "Fixed economic design prior, not optimized.",
        "stake_tokens": "Fixed equal endowment, not optimized.",
    }
    text = "\n## Frozen pilot selection\n\n"
    text += f"Selected candidate {selected['candidate']} from {len(selection['candidates'])} candidates on {selection['cells']} pilot cells. "
    text += "Selection is joint: a winning combination does not identify the independent causal benefit of each parameter. "
    text += "The grid launcher verifies publication of the frozen configuration and its evidence before evaluation scoring.\n\n"
    text += ("Historical analysis verifies the producer sources at the published freeze and requires every run's "
             "recorded scorer digest to match that freeze. It reads the recorded qualities, shares and ledgers; "
             "it does not substitute the current scorer. Execution and selection reconstruction use the "
             "[pinned calibration checkout](../README.md#pinned-calibration-checkout).\n\n")
    text += "| Parameter | Frozen value | Reason |\n|---|---|---|\n"
    for name, value in {**frozen["params"], **frozen["settlement"]}.items():
        text += f"| {name} | {value} | {reasons[name]} |\n"
    text += "\nAll candidate costs below are proportions, not percentages. Lower objective is preferred; undefined attacker cells retain the predeclared unit cost.\n\n"
    text += "| Candidate | EMA decay | Threshold | Patience | Attacker cost | Honest false-penalty cost | Sum |\n|---|---|---|---|---|---|---|\n"
    for row in selection["candidates"]:
        p = row["params"]
        text += (f"| {row['candidate']} | {p['beta']} | {p['threshold']} | {p['patience']} | "
                 f"{row['mean_attacker_share_cost']:.8f} | {row['mean_honest_false_penalty_rate']:.8f} | {row['objective']:.8f} |\n")
    text += f"\nPilot corpus SHA-256: `{frozen['corpus_sha256']}`. Selection summary SHA-256: `{frozen['selection_summary_sha256']}`.\n"
    if frozen["params"]["beta"] == 0:
        text += "\nThe selected EMA decay is zero: its state has no temporal memory. Any observed benefit of the EMA-enabled stage cannot be attributed to persistence across participations.\n"
    if frozen["params"]["threshold"] == 0:
        text += ("\nThe selected threshold is zero, so a penalty flag can only occur when current quality is already zero. "
                 "Zeroing that participation's weight does not change its allocation. Consequently the full method and "
                 "the reference-plus-EMA stage have identical minted shares on the same updates; penalties can still "
                 "change stake losses, reward burns and vesting. The declared tie order chooses patience where these "
                 "allocation costs tie; the selection objective does not establish a benefit from that penalty timing.\n")
    text += "\nThese are pilot selection measurements. Evaluation performance and uncertainty must be reported separately from the complete evaluation corpus.\n"
    text += ("\nThe generated [pilot comparison table](../paper/generated/pilot_selection.tex) uses double-column width; "
             "a star marks the recorded selected combination. Costs are proportions, and patience counts consecutive "
             "low-quality participations rather than global rounds. The stable PilotSelectedCandidate, PilotObjectiveSum, "
             "PilotAttackerCost and PilotHonestCost macros reproduce the chosen calibration costs. "
             "PilotHonestSlashedSum sums per-run counts; it is not a count of distinct people across the corpus.\n")
    return text


def update_readme(summary, path=Path("README.md")):
    start, end = "<!-- generated method start -->", "<!-- generated method end -->"
    text = path.read_text(encoding="utf-8")
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) > text.index(end):
        raise ValueError("README requires one ordered method-status block")
    record = summary.get("method_selection")
    if record is None:
        status = "Empirical pilot selection and evaluation remain pending."
    else:
        selection = record["selection"]
        status = (f"Pilot selection is complete: candidate {selection['selected']['candidate']} "
                  f"from {len(selection['candidates'])} candidates over {selection['cells']} cells. "
                  "The frozen configuration and every candidate cost are recorded in the "
                  "[method specification](docs/METHOD.md). ")
        evaluation = summary.get("evaluation")
        status += (f"The complete evaluation summary covers {evaluation['nominal_cells']} runs."
                   if evaluation else "Main evaluation results remain pending.")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(before+start+"\n"+status+"\n"+end+after, encoding="utf-8", newline="\n")


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

The coordinate median is inexpensive at the sampled-set size and limits an extreme coordinate's influence when the other clients have an honest majority. It is basis dependent and can suppress legitimate minority data. Exclusion removes the direct self inner-product term; it does not certify honesty or remove correlations among clients. No norm threshold or additive epsilon is introduced into the cosine, avoiding an explicit scale-dependent cutoff.

The scale-invariance statement assumes the input transformation preserves geometry. Floating-point rescaling can itself round, erase or overflow coordinates; no implementation can reconstruct information already lost from its inputs. The current library retains exact dyadic integer directions when a reference averages coordinates. It also recomputes near-cancelling inner products exactly, with an absolute subnormal rounding allowance as well as a relative error bound. Squared integer midpoint comparisons round the exact positive cosine once, avoiding a double-rounding zero at the smallest magnitudes. These numerical branches do not depend on reward thresholds or reputation parameters, so cached geometry remains shared across candidates. Ordinary non-averaging geometry retains stable floating-point normalization.

The `references()` method returns diagnostic floating-point coordinates. A mathematically nonzero midpoint may be too small to represent there; scoring retains its exact direction separately. A zero reference still gives zero quality, while an exact orthogonal direction cannot become a positive rounding residue that avoids a penalty.

The frozen experiments retain their original scorer digest and execution checkout. That implementation predates this correction and can lose a subnormal averaged reference or retain a tiny positive cancellation residue. The declared experiment uses odd-sized excluded reference sets and avoids the averaging branch, but that alone does not establish empirical equivalence. Historical analysis uses the recorded scores and shares and never relabels them as results of the current library. `python scripts/compare_method.py --synthetic` records discriminating boundary traces. After both complete corpora and their cached matrices are available, `python scripts/compare_method.py` compares the original and current arithmetic at the fixed selected parameters, including component states and integer settlement. No selection or experimental outcome is replaced by either command.

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
    frozen = frozen_selection()
    if frozen is None and "method_selection" in summary:
        raise ValueError("the method selection summary requires its frozen configuration")
    if frozen is not None:
        text = text.replace("# Proposed reward method", "# Reward method and frozen pilot selection", 1)
        text = text.replace("Status: **implemented with synthetic contract checks; pilot selection and empirical evaluation pending**.",
                            "Status: **pilot-selected parameters frozen; evaluation evidence is reported separately**.", 1)
        text = text.replace("A later frozen configuration must identify the pilot measurements and the selection commit before this rule scores evaluation updates.",
                            "The frozen configuration binds the pilot measurements; publication is checked before evaluation scoring.", 1)
        text = text.replace("No empirical improvement, pilot winner, contract safety or theorem is claimed in this design note.",
                            "The pilot winner is reported below. No evaluation improvement, contract safety theorem or general security theorem is inferred from pilot selection.", 1)
        text += selection_note(frozen)
        summary["method_selection"] = frozen
        write_json("results/summary.json", summary)
    if "contracts" in summary:
        text = text.replace("implementation and replay tests must pin this order; it is presently a specification, not a working contract.",
                            "implementation and independent integer-state replay pin this order; see [contract evidence](CONTRACTS.md).")
        text = text.replace("This resolves off-chain budget conservation; contract replay is still required before claiming on-chain equality.",
                            "The recorded baseline replays establish exact on-chain settlement for their inputs. Stateful random histories\nexercise slashing and vesting; empirical selected-method replay remains a separate required measurement.")
    Path("docs/METHOD.md").write_text(text, encoding="utf-8", newline="\n")
    update_readme(summary)
    print("Generated frozen method specification" if frozen else "Generated pre-pilot method specification; no selected configuration")


if __name__ == "__main__":
    main()
