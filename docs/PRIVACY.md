# Honest clients with clipped Gaussian updates

This is a public-data, reproducible simulation of the cost of adding noise to client updates. Its analytic accountant is a calibration of an ideal mechanism; it is not a deployment privacy guarantee.

## Fixed protocol

The comparison uses MNIST, 20 rounds, seeds 42, 43, 44, 45, 46, and CUDA for every compared run. Noise multipliers are 0.0, 0.5, 1.0, 1.5. The accounting delta is 1e-05. All clients are honest.

Median L2 update norm over every participation in the complete attack-free pilot runs selects one clipping bound from pilot seeds 100, 101, 102, 103, 104. The declaration and mechanism sources are committed before calibration; the selected scale and complete pilot evidence are published before evaluation.

Document generation and completed-outcome analysis verify the recorded calibration and its historical producer sources without recalibrating the clipping bound. Execution still requires the unchanged sources in the [pinned calibration checkout](../README.md#pinned-calibration-checkout).

One L2 projection of the complete local model update before release; the zero-noise control is also clipped. Independent Gaussian coordinates with standard deviation sigma times the clipping bound, applied on the client before the reply. The unchanged sample-count-weighted FedAvg consumes exactly the same perturbed matrices as every shadow scorer.

Each multiplier attempts its own full Flower training trajectory. All reward rules within that run see the identical perturbed matrix. The already-frozen reward parameters remain fixed; Aref isolates the excluding reference without reputation or penalties. A zero multiplier is a clipped control and is not interchangeable with an unclipped main-grid run.

## Accounting derivation

For an individually released vector clipped to L2 norm C, replacement of one client's complete data has sensitivity at most 2C. A Gaussian coordinate standard deviation sigma*C therefore incurs Renyi divergence 2*a/sigma^2 at order a>1 for one participation. The coordinator sees the sampled identity and individual update, so no hidden-sampling or secure-aggregation amplification is applied.

For t scheduled participations, write rho=2*t/sigma^2. Adaptive composition and the standard RDP conversion give epsilon(a)=rho*a+log(1/delta)/(a-1). Minimizing over real orders a>1 yields a=1+sqrt(log(1/delta)/rho) and epsilon=rho+2*sqrt(rho*log(1/delta)). The serialized finite bound is rounded upward. Zero participations cost zero; positive participation with zero noise has infinite epsilon. The JSON represents the latter with finite=false and epsilon=null, never an invented finite bound.

Per-client counts come from the public schedule. A second bound assumes participation in every round. Both are upper bounds for the ideal mechanism, not an exact privacy-loss distribution calculation. The noise is per coordinate: dividing its standard deviation by the square root of the model dimension would change the claimed multiplier.

The derivation follows [Mironov, Renyi Differential Privacy](https://arxiv.org/abs/1702.07476), Propositions 1, 3 and 7. The clipping design is related to [McMahan et al., Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963). Their aggregate-release and sampling analysis does not transfer automatically to this individually visible-update setting.

## Scope and limitations

Replace the full data of one registered client, conditional on fixed public client identities, partition assignment and record counts. The public-benchmark pilot calibration is outside the accountant. The ideal Gaussian bound assumes the chosen clipping bound and auxiliary metadata are public constants independent of a subsequently protected dataset; choosing a scale on private data requires separate accounting.

Public coordinate-seeded Gaussian draws for reproducible simulation; analytic bounds describe ideal undisclosed Gaussian randomness, not a deployment guarantee for these public-seeded runs. Known noise can be subtracted. The PRNG and floating-point Gaussian implementation are not a certified private sampler. The published dataset, seeds, norms and experimental artifacts are research evidence, not a privately released training transcript. Raw pre-noise norms are not returned by the client or used in the run's participation log. High-dimensional noise may destroy model utility; the comparison retains and reports that outcome without reducing noise after evaluation.

## Numerical termination

A non-finite local update, aggregated model or global evaluation terminates that run explicitly as failed_numerical. The failed round produces no reward ledger entry. Completed rounds, their hashes, integer ledgers and diagnostic prefix metrics remain available. The runner records the failed round and stage, then attempts every remaining declared seed and multiplier. Runtime and transport errors remain errors; they are not relabeled as numerical outcomes. No non-finite value is replaced with zero and no noise, learning rate or frozen reward parameter is adjusted.

A direct run_experiment.py invocation returns exit code 20 for a recorded numerical outcome. The privacy runner accepts that code only with matching validated failure evidence; all other nonzero exits remain execution errors. Resumption retains a validated failed seed rather than silently retrying it.

Full-horizon metrics are undefined for a failed run. Every seed remains in its multiplier group; an interval requiring a failed seed is undefined, rather than an average over survivors. Prefix metrics are diagnostics at unequal horizons and are never pooled into the headline comparison. The outcomes table reports completed rounds and failure stages alongside the full-horizon tables.

Accounting uses the entire predeclared public schedule, including rounds that a failed run never reaches. These are hypothetical ideal-mechanism bounds. A numerical failure signal and its data-dependent stopping time are not covered by the Gaussian accountant; no epsilon guarantee is asserted for that transcript.

## Reproduction

```bash
python scripts/privacy_protocol.py --freeze
# Commit and publish the clipping calibration before evaluation.
python scripts/run_privacy.py
python scripts/run_privacy.py --no-run
python scripts/summarize_privacy.py
python scripts/document_privacy.py
```

The runner verifies every retained update matrix. The summary reconstructs ledgers and accounting from committed per-round records without requiring raw matrices. The complete declared corpus is required for a headline summary. Seed intervals and paired false-penalty differences are generated from that summary; undefined reward metrics and the nonprivate control stay explicit. The comparison figure uses double-column width; small horizontal offsets separate rule markers at each declared multiplier. Its intervals resample complete seeds, and its accuracy panel displays the shared trajectory once. A cross above the plotting area denotes an undefined estimate.

## Pilot calibration

Fixed clipping bound: 1.1639285800003483, from 1000 pilot participations. All candidate multipliers use this same scale. The pooled participation median is used, not an average of seed medians.

The [calibration table](../paper/generated/privacy_calibration.tex) retains each seed's count and norm range. It fits a single column. PrivacyCalibrationClipBound, PrivacyCalibrationParticipations and PrivacyCalibrationSeeds are stable macros from the same shared summary; calibration alone is not a privacy-utility result.
