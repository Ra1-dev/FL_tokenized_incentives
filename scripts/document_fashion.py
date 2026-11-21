"""Document the unchanged-method dataset transfer from its executable declaration."""

from fractions import Fraction
from pathlib import Path

from evaluation import evaluation_cells, read
from fltoken.config import load_config
from fltoken.io import sha256, write_json


def main():
    summary = read('results/summary.json')
    if 'ordering' in summary.get('fashion', {}):
        from study_documents import fashion_document
        plan = read('configs/transfer_study.json')
        summary['fashion_design'] = dict(plan=plan, dataset=read(plan['dataset_manifests']['fashion_mnist']),
            config={**load_config(plan['config']), 'dataset':'fashion_mnist'}, plan_sha256=sha256('configs/transfer_study.json'),
            runs=summary['fashion']['nominal_cells'], minimum_seed_p=summary['fashion']['statistical_plan']['minimum_two_sided_seed_p'])
        write_json('results/summary.json', summary)
        fashion_document(summary)
        print('Documented complete six-rule transfer with matching MNIST seeds')
        return
    plan = read("configs/fashion_plan.json")
    config = load_config(plan["config"])
    dataset = read(plan["dataset_manifest"])
    design = dict(plan=plan, config=config, dataset=dataset,
                  plan_sha256=sha256("configs/fashion_plan.json"),
                  runs=len(list(evaluation_cells(plan))),
                  minimum_seed_p=float(Fraction(2, 2**len(plan["seeds"]))))
    summary = read("results/summary.json")
    summary["fashion_design"] = design
    write_json("results/summary.json", summary)
    design = read("results/summary.json")["fashion_design"]
    lines = ["# Fashion-MNIST transfer", "",
             "This comparison tests whether the reward outcomes transfer to a second image dataset without selecting new reward parameters.", "",
             "## Fixed experiment", "",
             f"The declared cohort contains {design['runs']} complete Flower runs: seeds {', '.join(map(str, plan['seeds']))}, "
             f"the fixed Sendros mixture and label flipping at {100*plan['fractions'][0]:g}% attackers, each for {plan['rounds']} rounds. "
             f"Every compared run uses {plan['device'].upper()}. "+plan["condition_reason"]+".", "",
             plan["method_policy"]+". No Fashion-MNIST result changes the clipping-free training protocol, thresholds, reputation decay or penalty patience. "
             "The reference-only and reference-plus-reputation stages use those same frozen parameters and independent state.", "",
             f"The model, optimizer and partition protocol match the MNIST experiment: {config['num_clients']} registered clients, "
             f"{config['clients_per_round']} sampled per round, Dirichlet concentration {config['alpha']}, "
             f"{config['local_epochs']} local epochs, batch size {config['batch_size']}, learning rate {config['lr']} and momentum {config['momentum']}. "
             f"Fashion-MNIST normalization uses mean {dataset['normalization']['mean']} and standard deviation {dataset['normalization']['std']}. "
             "Both full official splits are used. Label flipping maps class y to 9-y. The class semantics differ from digits, so an identical poisoning transformation need not cause the same accuracy loss.", "",
             "All reward rules consume the same submitted updates and fixed FedAvg trajectory in each run. "
             "Shadow rewards cannot improve the measured model accuracy. The figures and accuracy table therefore display one shared model result per condition.", "",
             "## Reproduction", "", "```bash",
             "python scripts/download_data.py --dataset fashion_mnist",
             "# Publish the experiment declaration and selected method before training.",
             "python scripts/run_grid.py --plan configs/fashion_plan.json",
             "python scripts/run_grid.py --plan configs/fashion_plan.json --no-run",
             "python scripts/summarize_grid.py --plan configs/fashion_plan.json",
             "python scripts/document_fashion.py",
             "python scripts/make_macros.py",
             "python scripts/make_tables.py",
             "python scripts/make_figures.py", "```", "",
             "The runner uses the same publication gate as the main evaluation. Its output and persistent elapsed-time clock live in `results/fashion/`. "
             "Every raw matrix is retained and hash-checked during validation. The separate download command records checksums; training verifies the complete prepared input set without network access.", "",
             "The summary requires complete unique coverage, identical resolved protocols and library versions, the published frozen method, "
             "and source digests verified against reachable published training commits. It reconstructs every integer ledger and reads the same accuracy in the round and participation logs. "
             "Recorded-only analysis needs no raw matrices or dataset download. Successful training time is reported separately from elapsed time including restarts.", "",
             "## Uncertainty and interpretation", "",
             f"Each interval uses {plan['statistics']['bootstrap_resamples']} fixed-seed bootstrap resamples of the {len(plan['seeds'])} complete seed observations, "
             f"with sample standard deviation and {100*plan['statistics']['confidence']:g}% percentile limits. "
             "Pooled values average the two fixed conditions within each seed before resampling. Undefined reward outcomes retain their counts and invalidate the corresponding complete-cohort estimate.", "",
             f"The descriptive signed-rank comparison has {design['runs']} condition-seed pairs, with Holm correction over both primary costs and all baseline contrasts. "
             f"A separate descriptive test uses seed-level paired averages; its smallest two-sided p-value with all nonzero differences is {design['minimum_seed_p']}. "+
             plan["statistics"]["interpretation"]+". A negative or inconclusive transfer is retained as a result; no additional seed or attack is selected after observing it.", "",
             "The comparison and accuracy figures use single-column width; the tables are sized for double-column placement. "
             "Tables state the common device, seed uncertainty and undefined estimates. "
             "The `fashion` record in `results/summary.json` supplies every empirical macro, table and figure.", "",
             "## Input checksums", "", "| File | SHA-256 |", "|---|---|"]
    lines.extend(f"| {name} | `{digest}` |" for name, digest in dataset["sha256"].items())
    result = summary.get("fashion")
    lines.extend(["", (f"Completed cohort: {result['nominal_cells']} runs on {result['device'].upper()}." if result else
                       "Empirical transfer results are pending the published method freeze and complete training cohort."), ""])
    Path("docs/FASHION.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"Documented {design['runs']} declared transfer runs")


if __name__ == "__main__":
    main()
