# Portable reward metrics

Generated from `results/summary.json` by `python scripts/document_metric_portability.py`.

## Exact definitions

For sorted nonnegative integer client payouts x_i, Gini is sum((2*i-n-1)*x_i)/(n*sum(x_i)), with rank i starting at one. Both sums and the quotient use Python integers and Fraction; only the final value is converted to a float. Empty and all-zero populations have undefined Gini. No payout is first converted to floating point.

Attacker share, registered headcount share, honest false-penalty rate, precision and recall are exact ratios of integer totals or counts, converted once. Detection F1 is 2*TP/(2*TP+FP+FN), with zero when its denominator is zero. It is not reconstructed through already-rounded precision and recall. Flagging compares the exact binary-float value of a recorded share against the rational threshold 1/(2*K).

Reproduction quality sums are integers. Per-round and pooled means use rational division. Linear quantiles interpolate the two adjacent sorted observations with Fraction arithmetic; undefined observations remain explicit. Seed means and standard deviations use Python's statistics module, whose accumulation is exact rational arithmetic, rather than vectorized floating reductions. Per-observation subtraction and division follow a fixed scalar order.

## Declared migration

The previous Gini implementation converted large integer payouts to float and used a vector dot product. Platform-dependent reduction order changed its final bits. Stored metrics were regenerated from the same participation logs and integer ledgers, without retraining or changing payouts. Each changed run retains `metrics_provenance.json`: previous and new metric hashes, changed fields, training-log hash, original training commit and derivation-source hashes.

Migrated runs: 14. Changed rule-level fields: f1=12, honest_gini=39.

Both environments reconstructed 62 artifacts across 14 runs without any byte changes. Artifact and derivation-source hashes agree between environments.

| Environment | Python | NumPy | PyYAML |
|---|---|---|---|
| Windows-11-10.0.26200-SP0 | 3.12.10 | 2.5.5 | 6.0.3 |
| Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.39 | 3.12.3 | 2.5.5 | 6.0.3 |

## Reproduce

In each prepared environment, run the same command with a platform-specific output name:

```bash
python scripts/check_derived_metrics.py --output results/validation/derived-linux.json
```

Use `derived-windows.json` for the Windows execution. Then run `python scripts/document_metric_portability.py` to verify that both reports match the current artifact bytes. The check invokes ledger verification, reproduction analysis and stale-sensitivity analysis without training or network access.

A deliberate metric-definition migration uses `python scripts/rebuild_ledgers.py --rewrite`, followed by both analysis commands and the platform checks. Normal verification omits `--rewrite`; mismatches fail without an approximate comparison. This evidence concerns reconstruction of fixed logs. It does not claim cross-platform bit identity of newly trained neural networks.
