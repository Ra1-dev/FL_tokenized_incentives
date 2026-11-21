# Fashion-MNIST transfer

Generated from `results/summary.json` by `python scripts/document_fashion.py`.

## Fixed comparison

The cohort contains 6 CUDA runs, seeds 42, 43, 44, with the fixed mixture and label flipping. The complete original MNIST reward parameters and historical Mstar source are unchanged. No Fashion-MNIST outcome selects a threshold, reputation parameter or penalty setting.

All six rules score the same updates within each run. The model uses fixed sample-count-weighted FedAvg, so accuracy is shared and cannot be attributed to a shadow reward rule. The MNIST reference uses exactly matching seeds, horizons and attack conditions from the unclipped grid. The clipped privacy controls are not used for this dataset comparison.

| Condition | Rule | Attacker share, % [95% CI] | Honest false penalties, % [95% CI] |
|---|---|---|---|
| Fixed mixture | B0 | 17.667 [16.000, 19.500] | 0.000 [0.000, 0.000] |
| Fixed mixture | B1 | 9.936 [7.229, 11.860] | 0.202 [0.000, 0.606] |
| Fixed mixture | B2 | 17.917 [16.250, 18.750] | 20.200 [16.770, 22.619] |
| Fixed mixture | Mstar | 3.639 [1.501, 6.450] | 11.723 [10.559, 13.095] |
| Fixed mixture | CGSV | 8.540 [6.397, 10.924] | 7.098 [5.455, 8.696] |
| Fixed mixture | RFFL | 8.094 [5.835, 10.437] | 6.899 [5.455, 8.696] |
| Label flip 20% | B0 | 21.167 [19.000, 23.500] | 0.000 [0.000, 0.000] |
| Label flip 20% | B1 | 25.367 [23.924, 26.634] | 8.677 [7.407, 10.127] |
| Label flip 20% | B2 | 7.083 [5.625, 8.125] | 5.673 [3.268, 6.962] |
| Label flip 20% | Mstar | 12.891 [10.519, 15.104] | 10.966 [8.497, 13.291] |
| Label flip 20% | CGSV | 7.807 [6.677, 9.691] | 3.597 [3.165, 3.922] |
| Label flip 20% | RFFL | 7.059 [6.025, 8.747] | 3.597 [3.165, 3.922] |

## Does the MNIST ordering hold?

The complete mean ordering is not preserved. The table counts every pair of rules: a reversal means a strict ordering flips direction; other changes include a tie appearing or disappearing. These are descriptive comparisons of means, not tests of equivalence.

| Condition | Cost | Same ordering | Strict reversals | Other changes | Undefined |
|---|---|---|---|---|---|
| Fixed mixture | Attacker share | 13/15 | 2 | 0 | 0 |
| Fixed mixture | Honest false penalties | 13/15 | 1 | 1 | 0 |
| Label flip 20% | Attacker share | 13/15 | 2 | 0 | 0 |
| Label flip 20% | Honest false penalties | 14/15 | 1 | 0 | 0 |

Fixed mixture, attacker share: strict reversals involve B0 / B2, B1 / B2.

Fixed mixture, honest false penalties: strict reversals involve B2 / Mstar.

Label flip 20%, attacker share: strict reversals involve B2 / Mstar, B2 / CGSV.

Label flip 20%, honest false penalties: strict reversals involve B1 / Mstar.

| Condition | Shared final accuracy, % [95% CI] |
|---|---|
| Fixed mixture | 87.607 [87.130, 87.970] |
| Label flip 20% | 81.187 [69.250, 87.350] |

## Uncertainty and scope

Each interval resamples the 3 complete seeds with 20000 fixed-seed draws. Pooled outcomes first average the two conditions within a seed. Paired Mstar contrasts use condition-seed pairs and a separate whole-seed sensitivity test, with Holm correction over the 10 declared baseline/cost contrasts. The smallest possible two-sided nonzero seed-test p-value is 0.25. Intervals and rankings therefore remain descriptive; no claim of reliable rule ordering follows merely from a point estimate.

Different class semantics, partitions and sample counts affect the learning trajectory. Label flipping maps y to 9-y on both datasets, but equal attack fractions do not imply equal poisoning difficulty. MNIST and Fashion-MNIST are small image benchmarks, and this transfer does not establish generality to cross-silo data, larger models or adaptive attackers.

## Reproduction and assets

The six cells are declared in `configs/transfer_study.json`; the historical `configs/fashion_plan.json` is a separate protocol. The committed empirical cohort and frozen extension geometries reconstruct on CPU without downloading data:

```bash
python scripts/run_transfer_study.py --no-run
python scripts/rescore_extensions.py --recorded-only
python scripts/summarize_extensions.py
python scripts/analyze_studies.py --recorded-only
python scripts/document_fashion.py
python scripts/make_macros.py
python scripts/make_tables.py
python scripts/make_figures.py
```

The comparison and accuracy figures are single-column. Wide booktabs tables retain every rule, the matching MNIST means and paired dataset differences, plus the complete ordering counts. The `fashion` summary section supplies all empirical macros and assets. Existing FashionMixture and FashionLabelFlip macro names remain stable; CGSV/RFFL and ordering counts extend them.
