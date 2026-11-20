"""Publish the declared offline adaptations and their primary-source differences."""

import json
from pathlib import Path

from fltoken.io import sha256, write_json


def main():
    plan_path, sources_path = Path("configs/extensions_plan.json"), Path("results/literature/extensions.json")
    record = dict(plan=json.loads(plan_path.read_text(encoding="utf-8")),
                  literature=json.loads(sources_path.read_text(encoding="utf-8")),
                  plan_sha256=sha256(plan_path), literature_sha256=sha256(sources_path))
    frozen_path = Path("configs/extensions_frozen.json")
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        selection_path = Path("results/extension_selection/summary.json")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if (sha256(selection_path) != frozen["selection_sha256"] or
                any(sha256(path) != digest for path, digest in frozen["source_sha256"].items()) or
                frozen["params"] != {name: row["params"] for name, row in selection["selected"].items()}):
            raise ValueError("extension freeze and complete selection evidence differ")
        record["selection"] = dict(configuration=frozen, cells=selection["cells"],
                                   selected=selection["selected"], candidates=selection["candidates"],
                                   training_device=selection["training_device"])
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    summary["extension_design"] = record
    write_json("results/summary.json",summary)
    plan, sources = summary["extension_design"]["plan"], summary["extension_design"]["literature"]
    lines = ["# Offline reputation rules", "", "Generated from `results/summary.json` by `python scripts/document_extensions.py`.",
             "", f"Status: **{'parameters selected on the complete pilot corpus' if 'selection' in record else plan['status']}**.", "", "## Source mechanisms", "",
             f"[CGSV]({sources['cgsv']['paper']}) normalizes updates to a common magnitude, forms a prior-importance-weighted inclusive reference, approximates contribution by cosine, and updates normalized importance with an EMA. A tanh transform sets a gradient-download budget. This is the cosine approximation, not exact coalition Shapley valuation.", "",
             f"[RFFL]({sources['rffl']['paper']}) combines an inclusive normalized reference with EMA reputation. Low-reputation identities are permanently excluded. Relative reputation determines a gradient-download budget. Its original mechanism also changes aggregation and personalized model updates; those training effects are absent from a shadow comparison.", "",
             "## Declared token and participation adaptations", "",
             "Let N be the registered population and S the current sample. Initialize every registered reputation to r_i = 1/N. Normalize the nonnegative prior reputations of active sampled identities into weights w_i. Set u_i = gamma * delta_i / norm(delta_i), with a zero vector remaining zero, and reference v = sum_i w_i*u_i. Compute c_i = cosine(u_i,v), using zero when either vector is zero.", "",
             "For active sampled identities, update a_i = alpha*r_i + (1-alpha)*c_i. Unobserved identities receive no observation update. Global normalization can rescale their importance units; absence is not itself a removal event. CGSV floors the updated observed values, then normalizes over the registered population. RFFL normalizes signed EMA values, removes observed identities below its threshold, sets their importance to zero, and renormalizes survivors. Removal is permanent and affects rewards and the scoring reference only.", "",
             "The uniform registered prior is our partial-participation adaptation. It avoids starting later-sampled identities with zero history while earlier identities already hold normalized reputation. It is not the source's zero-history/data-weighted startup. This choice was made from the participation contract before empirical extension measurements.", "",
             "CGSV quotas are tanh(beta*r_i) divided by the largest transformed reputation; RFFL quotas are r_i divided by the largest active reputation. Token shares normalize these continuous quotas over sampled eligible identities. The integer gradient-mask budget and personalized downloads are not reproduced. Quotas are not multiplied by current cosine: accumulated reputation can pay a current zero contribution. A zero common reference pays nothing because the corresponding gradient reward contains no information. Neither rule emits stake-slash flags.", "",
             "A nonpositive signed EMA normalization mass is an undefined state: retain prior history, pay nothing, and count the event. Such a candidate cannot support a complete comparison or be selected by silently benefiting from empty payouts. Empty eligible samples are recorded separately. These boundaries do not establish the original mechanisms' fairness or robustness guarantees for tokens.", "",
             "## Parameters and protocol differences", "",
             "| Parameter | Declared value | Reason |", "|---|---|---|",
             f"| EMA alpha | {plan['parameters']['alpha']} | Published setting for both mechanisms |",
             f"| Normalization magnitude | {plan['parameters']['gamma']} | Published MNIST setting; common magnitude cancels in cosine |",
             f"| CGSV beta candidates | {plan['cgsv']['beta_candidates']} | Published altruism range; select on pilots only |",
             f"| CGSV importance floor | {plan['cgsv']['floor']} | Released implementation's nonnegative importance safeguard |",
             f"| RFFL threshold | 1 / ({plan['rffl']['threshold_divisor']} * N) | Published fixed-population threshold |",
             f"| Pilot seeds | {plan['pilot_seeds']} | Disjoint from evaluation |", "",
             f"The [CGSV implementation]({sources['cgsv']['implementation']}) supplies the importance floor and a nonzero first reference; its zero history and data-size startup differ from our declared prior. The [RFFL implementation]({sources['rffl']['implementation']}) also differs from its paper: it clips updates, delays removal for {sources['rffl']['implementation_warmup_rounds']} rounds, and uses the active population for its threshold. Here we use full normalization, immediate eligibility checks and the registered population. Signed normalization before removal and survivor renormalization are explicit, rather than left implicit in the algorithm's normalized-reputation notation.", "",
             "## Bias measurements and numerical reconstruction", "",
             "For a nonzero reference and update, the exact decomposition is c_i = w_i*norm(u_i)/norm(v) + cross_i. Record the self term, cross residual, cosine after removing w_i*u_i from the reference, eligibility and all weights. Zero references and self terms outside finite binary64 range retain explicit undefined flags. Analyze every noise participation, including excluded identities, with eligible observations also reported separately. The source reproduction's small residual criterion is not assumed for these differently normalized references.", "",
             "Raw matrix hashes and seeded client order bind geometry to the common training trace. Recorded geometry replay verifies the current reputation weights and the self/cross identities before reconstructing state and shares. Importance and quota sums use scalar accumulation. CGSV tanh quotas use Decimal precision before one float conversion, avoiding platform-specific vector transcendental rounding. Raw geometry recomputation is a separate numerical check; replay does not claim newly computed matrix products are bit-identical across platforms. Bias quantiles condition on defined geometry and retain zero measured cosines as undefined relative errors; the observation counts report both exclusions explicitly.", "",
             "## Selection and execution boundary", "", plan["selection_objective"]+".",
             plan["tie_break"]+".", plan["publication_gate"]+".", "",
             "All compared rules consume the same retained updates from the declared training device; the offline calculation device does not create a new training trajectory.", "",
             "## Commands and retained evidence", "",
             "The complete pilot corpus must already exist. Commit the declaration and scientific sources before running selection. Selection refuses incomplete coverage, source drift, undefined histories and an indistinguishable candidate search. It writes each candidate's per-cell metrics, shared geometry and a digest-bound selection summary.", "",
             "```bash", "python scripts/select_extensions.py --freeze",
             "python scripts/select_extensions.py --recorded-only",
             "python scripts/document_extensions.py", "```", "",
             "Commit and push the frozen configuration and complete selection evidence before evaluation scoring. The evaluation command checks published bytes and reconstructs the pilot selection first. It never trains a model or modifies the original run folder.", "",
             "```bash", "python scripts/rescore_extensions.py",
             "python scripts/rescore_extensions.py --recorded-only",
             "python scripts/summarize_extensions.py",
             "python scripts/make_figures.py", "python scripts/make_tables.py", "python scripts/make_macros.py", "```", "",
             "The first command reads and verifies every retained matrix. The second reconstructs exact allocations from the recorded geometry and does not require raw matrices. `--available` permits an explicitly partial pass while the evaluation runner is active; the manifest keeps expected and completed coverage separate. Keep `results/cache/` until all desired raw-matrix calculations are complete.", "",
             "Each output retains the common input digests, raw and excluding cosine, weighted self term, cross residual, every registered reputation/eligibility state, per-participation shares, integer ledger and the standard reward metrics. CGSV altruism changes quotas, so its candidates reuse a single geometry history. RFFL uses an independent state and reference on the same updates.", "",
             "Summarization requires complete coverage and reconstructs every state and integer payout before generating seed-level intervals. Missing or undefined seed observations invalidate their interval rather than disappearing from it. If an evaluation history encounters undefined signed normalization, its inferential reward and bias quantities remain undefined, while accounting totals and event counts remain available. This is an honest failure case, not evidence of successful mitigation.", "",
             "The scatter figure uses the original source-comparison seeds in the main fixed mixture; the contrast figure uses all evaluation seeds and reports means of within-seed noise cosines, self terms and excluding-reference cosines. The arithmetic excluding-reference contrast holds the observed updates and prior weights fixed; it is not the outcome of separately training a leave-one-out mechanism. Long-horizon results have separate tables. Both figure pairs use double-column width with PNG and PDF output; the bias table also needs a wide table environment.", ""]
    if "selection" in record:
        selected = record["selection"]
        lines += ["## Measured pilot selection", "",
                  f"Complete cells: {selected['cells']}; training device: {selected['training_device']}.", "",
                  "| Candidate | Objective | Attacker cost | Honest false penalties | Selected |",
                  "|---|---|---|---|---|"]
        for row in selected["candidates"]:
            winner = row["name"] == selected["selected"][row["family"]]["name"]
            lines.append(f"| {row['name']} | {row['objective']:.9g} | {row['mean_attacker_share_cost']:.9g} | {row['mean_honest_false_penalty_rate']:.9g} | {winner} |")
        lines += ["", "Frozen parameters:", "", "```json", json.dumps(selected["configuration"]["params"], indent=2), "```", ""]
    else:
        lines += ["No empirical extension result or chosen beta is available yet.", ""]
    Path("docs/EXTENSIONS.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    print("Generated offline extension design; selection " + ("complete" if "selection" in record else "pending"))


if __name__ == "__main__":
    main()
