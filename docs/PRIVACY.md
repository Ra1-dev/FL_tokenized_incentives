# Noisy honest clients in the fixed mixture

Generated from `results/summary.json` by `python scripts/document_privacy.py`.

## Scope and fixed design

The comparison contains 12 complete-horizon attempts: 9 new coordinate-noise runs and 3 reused clipped zero-noise controls. All use CUDA, 20 rounds and seeds 42, 43, 44. The fixed mixture includes adversaries; only honest clients clip and perturb their local updates. Adversaries retain their declared behavior and receive no privacy guarantee.

The clipping bound remains C=1.1639285800003483, selected from 1000 attack-free pilot participations. No evaluation result changes the clipping bound, reward parameters, learning rate or failure policy. The original frozen Mstar source is loaded after its digest is checked. B0/B1/B2/Mstar score in shadow mode; frozen CGSV/RFFL use the identical released matrices, inline for controls and offline for the new runs. Every rule in one run shares the same sample-count-weighted FedAvg trajectory. These reward comparisons do not measure closed-loop training gains.

The declared coordinate standard deviations are 0, 0.001, 0.003, 0.01. Each maps to multiplier z=tau/C. The actual floating-point standard deviation is recorded alongside requested tau; there is no division by the square root of model dimension. The zero-noise control still clips honest updates and is distinct from the unclipped main grid.

## Original multiplier experiment

Individually released client noise at the original positive multipliers caused numerical termination in 9 of 12 attempts; 3 zero-noise controls completed. Large coordinate noise accumulates across the full model vector. This local-release design is not central noise added once to an aggregate. The failed runs were retained without retries or retuning; the subsequent coordinate-scale comparison was declared separately.

| Multiplier | Seed | Completed rounds | Outcome |
|---|---|---|---|
| 0 | 42 | 20 | complete |
| 0.5 | 42 | 7 | failed_numerical |
| 1 | 42 | 2 | failed_numerical |
| 1.5 | 42 | 1 | failed_numerical |
| 0 | 43 | 20 | complete |
| 0.5 | 43 | 5 | failed_numerical |
| 1 | 43 | 1 | failed_numerical |
| 1.5 | 43 | 1 | failed_numerical |
| 0 | 44 | 20 | complete |
| 0.5 | 44 | 7 | failed_numerical |
| 1 | 44 | 1 | failed_numerical |
| 1.5 | 44 | 1 | failed_numerical |

A failed round contributes no reward ledger entry. Completed prefixes are diagnostics, never full-horizon utility observations. A failed seed invalidates its complete-cohort interval; no average over surviving seeds replaces it. Runtime and transport errors remain execution errors, rather than being relabeled numerical outcomes.

## Measured rewards and shared accuracy

All 12 coordinate-comparison cells completed; numerical failures: 0. Intervals use 20000 fixed-seed bootstrap resamples of the complete seeds. The small seed count makes these intervals descriptive. Undefined reputation states remain undefined in inference; they are not ordinary zero payouts.

| Coordinate SD | Rule | Attacker share, % [95% CI] | Honest false penalties, % [95% CI] |
|---|---|---|---|
| 0 | B0 | 17.667 [16.000, 19.500] | 0.000 [0.000, 0.000] |
| 0 | B1 | 13.717 [12.268, 14.895] | 1.216 [0.000, 1.863] |
| 0 | B2 | 7.917 [6.250, 10.000] | 10.525 [10.303, 10.714] |
| 0 | Mstar | 3.816 [2.719, 4.781] | 11.564 [10.119, 13.665] |
| 0 | CGSV | 6.948 [5.297, 9.612] | 3.438 [3.106, 3.636] |
| 0 | RFFL | 5.406 [3.669, 8.705] | 3.034 [2.424, 3.571] |
| 0.001 | B0 | 17.667 [16.000, 19.500] | 0.000 [0.000, 0.000] |
| 0.001 | B1 | 13.703 [12.194, 14.737] | 1.216 [0.000, 1.863] |
| 0.001 | B2 | 8.333 [6.250, 11.250] | 10.940 [10.303, 11.801] |
| 0.001 | Mstar | 3.956 [2.736, 4.836] | 10.945 [9.697, 12.422] |
| 0.001 | CGSV | 7.087 [5.424, 9.765] | 3.638 [2.424, 4.762] |
| 0.001 | RFFL | 5.488 [3.722, 8.818] | 3.229 [1.818, 4.762] |
| 0.003 | B0 | 17.667 [16.000, 19.500] | 0.000 [0.000, 0.000] |
| 0.003 | B1 | 13.028 [11.799, 14.124] | 0.198 [0.000, 0.595] |
| 0.003 | B2 | 9.167 [7.500, 11.875] | 11.745 [10.909, 12.422] |
| 0.003 | Mstar | 4.307 [3.133, 5.190] | 10.358 [8.333, 13.043] |
| 0.003 | CGSV | 7.866 [6.061, 10.623] | 4.042 [3.636, 4.762] |
| 0.003 | RFFL | 6.271 [4.524, 9.755] | 4.037 [3.106, 4.762] |
| 0.01 | B0 | 17.667 [16.000, 19.500] | 0.000 [0.000, 0.000] |
| 0.01 | B1 | 10.258 [9.231, 11.104] | 0.000 [0.000, 0.000] |
| 0.01 | B2 | 20.000 [17.500, 22.500] | 22.278 [21.429, 22.981] |
| 0.01 | Mstar | 7.253 [5.288, 8.699] | 12.344 [11.515, 13.095] |
| 0.01 | CGSV | 12.326 [9.817, 15.476] | 7.104 [3.636, 9.938] |
| 0.01 | RFFL | 11.958 [9.362, 15.174] | 6.699 [3.636, 9.317] |

| Coordinate SD | Shared final accuracy, % [95% CI] | Maximum honest-client epsilon across seeds |
|---|---|---|
| 0 | 98.727 [98.660, 98.780] | Infinite |
| 0.001 | 98.733 [98.680, 98.810] | 3.25522e+07 |
| 0.003 | 98.723 [98.710, 98.740] | 3.62551e+06 |
| 0.01 | 98.710 [98.650, 98.760] | 329005 |

At the largest scale, Mstar's attacker share is 7.253 [5.288, 8.699]%, and its honest false-penalty rate is 12.344 [11.515, 13.095]%, compared with 11.564 [10.119, 13.665]% in the clipped control. The table retains intermediate scales: the honest cost is not assumed monotone. A smaller false-penalty rate alone does not establish robust contribution evaluation; it must be read alongside attacker rewards. The attacker and honest-cost tables keep the large-noise Krum and extension costs visible.

## Accountant and interpretation

Delta is 1e-05. Replacement of one honest client's complete data changes an L2-clipped vector by at most 2C. For coordinate SD z*C and t public participations, rho=2*t/z^2. Gaussian RDP composition and conversion give epsilon=rho+2*sqrt(rho*log(1/delta)), minimized over real Renyi orders greater than one and rounded upward in serialization. Identified, individually visible releases receive no sampling amplification. The derivation follows [Mironov (2017)](https://arxiv.org/abs/1702.07476), Propositions 1, 3 and 7.

The implied epsilon values are large. This comparison measures rewards for noisy honest clients, without claiming a strong privacy guarantee. The complete per-client budgets are in each run's `privacy.json` and in `privacy.cells[].per_seed_accounting[].accounting.clients` in the shared summary. The accounting table gives each seed's participation range, maximum honest-client bound and conservative all-round bound; it does not imply identical participation counts.

Public-seeded floating-point draws do not provide deployment privacy; bounds describe ideal undisclosed Gaussian randomness conditional on a clipping bound and auxiliary protocol metadata fixed independently of the protected data. Public-benchmark pilot calibration is not included in this accountant. A positive participation count with zero noise has infinite epsilon, represented by `finite=false` and `epsilon=null`. An unobserved client incurs zero release cost. Public-seeded simulation noise can be subtracted; the floating-point sampler is not a certified deployment mechanism. Public-benchmark clipping calibration is outside the accountant. Hypothetical full-schedule bounds for failed original runs exclude failure signals and data-dependent termination; no privacy guarantee is asserted for those transcripts.

## Reproduction and assets

The published training plans are `configs/transfer_study.json` and `configs/coordinate_noise_study.json`. Use their recorded producer commits for fresh training, preserve caches and run one GPU driver at a time as described in README. The earlier attack-free `configs/privacy_plan.json` remains the historical calibration protocol, not this mixture experiment.

The completed evidence reconstructs on CPU without datasets or raw update arrays:

```bash
python scripts/run_transfer_study.py --no-run
python scripts/run_coordinate_study.py --no-run
python scripts/rescore_extensions.py --recorded-only
python scripts/summarize_extensions.py
python scripts/analyze_studies.py --recorded-only
python scripts/document_privacy.py
python scripts/make_macros.py
python scripts/make_tables.py
python scripts/make_figures.py
```

For a newly trained coordinate cohort, first run `python scripts/analyze_studies.py --coordinate-cache PATH` with its retained cache root. This records both extension geometries before portable replay. Source, plan, identity, clock, complete coverage, accounting and integer ledgers are checked. Original producer files and frozen calibrations must remain unchanged.

The honest-cost/accuracy figure is double-column width; the attacker-share figure is single-column. Coordinate labels are categorical positions, with offsets separating rules; they are not noise multipliers. Tables use booktabs and include attacker share, honest cost, shared accuracy, implied scales, accounting, all outcomes and original divergence. PrivacyA through PrivacyD refer to the declared coordinate groups in order; CoordinateSD and NoiseMultiplier macros distinguish tau from z. All numbers and plot inputs come from the shared summary.
