"""Measured noise and transfer notes generated from the shared summary."""
from pathlib import Path

from evaluation_assets import condition_label


def estimate(observation, scale=100):
    value = observation['estimate']
    return 'Undefined' if value is None else f"{scale*value['mean']:.3f} [{scale*value['lower']:.3f}, {scale*value['upper']:.3f}]"


def privacy_document(summary):
    result = summary['privacy']
    calibration = summary['privacy_calibration']['evidence']
    original = result['original_multiplier_study']
    lines = ['# Noisy honest clients in the fixed mixture', '',
        'Generated from `results/summary.json` by `python scripts/document_privacy.py`.', '',
        '## Scope and fixed design', '',
        f"The comparison contains {result['runs']} complete-horizon attempts: {result['new_runs']} new coordinate-noise runs and "
        f"{result['reused_control_runs']} reused clipped zero-noise controls. All use {result['training_device'].upper()}, "
        f"{result['rounds']} rounds and seeds {', '.join(map(str, result['seeds']))}. "
        'The fixed mixture includes adversaries; only honest clients clip and perturb their local updates. Adversaries retain their declared behavior and receive no privacy guarantee.', '',
        f"The clipping bound remains C={result['clipping_bound']:.17g}, selected from {calibration['participations']} attack-free pilot participations. "
        'No evaluation result changes the clipping bound, reward parameters, learning rate or failure policy. The original frozen Mstar source is loaded after its digest is checked. '
        'B0/B1/B2/Mstar score in shadow mode; frozen CGSV/RFFL use the identical released matrices, inline for controls and offline for the new runs. '
        'Every rule in one run shares the same sample-count-weighted FedAvg trajectory. These reward comparisons do not measure closed-loop training gains.', '',
        'The declared coordinate standard deviations are '+', '.join(f"{row['coordinate_standard_deviation']:g}" for row in result['cells'])+'. '
        'Each maps to multiplier z=tau/C. The actual floating-point standard deviation is recorded alongside requested tau; there is no division by the square root of model dimension. '
        'The zero-noise control still clips honest updates and is distinct from the unclipped main grid.', '',
        '## Original multiplier experiment', '',
        f"Individually released client noise at the original positive multipliers caused numerical termination in {original['numerical_failures']} of "
        f"{original['runs']} attempts; {original['completed_runs']} zero-noise controls completed. "
        'Large coordinate noise accumulates across the full model vector. This local-release design is not central noise added once to an aggregate. '
        'The failed runs were retained without retries or retuning; the subsequent coordinate-scale comparison was declared separately.', '',
        '| Multiplier | Seed | Completed rounds | Outcome |', '|---|---|---|---|']
    for row in original['outcomes']:
        lines.append(f"| {row['noise_multiplier']:g} | {row['seed']} | {row['completed_rounds']} | {row['status']} |")
    lines += ['', 'A failed round contributes no reward ledger entry. Completed prefixes are diagnostics, never full-horizon utility observations. '
        'A failed seed invalidates its complete-cohort interval; no average over surviving seeds replaces it. '
        'Runtime and transport errors remain execution errors, rather than being relabeled numerical outcomes.', '',
        '## Measured rewards and shared accuracy', '',
        f"All {result['completed_runs']} coordinate-comparison cells completed; numerical failures: {result['numerical_failures']}. "
        f"Intervals use {result['statistical_plan']['bootstrap_resamples']} fixed-seed bootstrap resamples of the complete seeds. "
        'The small seed count makes these intervals descriptive. Undefined reputation states remain undefined in inference; they are not ordinary zero payouts.', '',
        '| Coordinate SD | Rule | Attacker share, % [95% CI] | Honest false penalties, % [95% CI] |', '|---|---|---|---|']
    for row in result['cells']:
        for rule in result['rules']:
            lines.append(f"| {row['coordinate_standard_deviation']:g} | {rule} | {estimate(row['metrics'][rule]['attacker_share'])} | "
                         f"{estimate(row['metrics'][rule]['honest_false_penalty_rate'])} |")
    lines += ['', '| Coordinate SD | Shared final accuracy, % [95% CI] | Maximum honest-client epsilon across seeds |', '|---|---|---|']
    for row in result['cells']:
        budgets = [seed['maximum_client_epsilon'] for seed in row['per_seed_accounting']]
        epsilon = 'Infinite' if any(value is None for value in budgets) else f'{max(budgets):.6g}'
        lines.append(f"| {row['coordinate_standard_deviation']:g} | {estimate(row['metrics']['B1']['final_accuracy'])} | {epsilon} |")
    low, high = result['cells'][0], result['cells'][-1]
    lines += ['', f"At the largest scale, Mstar's attacker share is {estimate(high['metrics']['Mstar']['attacker_share'])}%, "
        f"and its honest false-penalty rate is {estimate(high['metrics']['Mstar']['honest_false_penalty_rate'])}%, "
        f"compared with {estimate(low['metrics']['Mstar']['honest_false_penalty_rate'])}% in the clipped control. "
        'The table retains intermediate scales: the honest cost is not assumed monotone. '
        'A smaller false-penalty rate alone does not establish robust contribution evaluation; it must be read alongside attacker rewards. '
        'The attacker and honest-cost tables keep the large-noise Krum and extension costs visible.', '',
        '## Accountant and interpretation', '',
        f"Delta is {result['delta']:g}. Replacement of one honest client's complete data changes an L2-clipped vector by at most 2C. "
        'For coordinate SD z*C and t public participations, rho=2*t/z^2. Gaussian RDP composition and conversion give '
        'epsilon=rho+2*sqrt(rho*log(1/delta)), minimized over real Renyi orders greater than one and rounded upward in serialization. '
        'Identified, individually visible releases receive no sampling amplification. The derivation follows '
        '[Mironov (2017)](https://arxiv.org/abs/1702.07476), Propositions 1, 3 and 7.', '',
        result['interpretation']+' The complete per-client budgets are in each run\'s `privacy.json` and in '
        '`privacy.cells[].per_seed_accounting[].accounting.clients` in the shared summary. The accounting table gives each seed\'s '
        'participation range, maximum honest-client bound and conservative all-round bound; it does not imply identical participation counts.', '',
        result['accounting_limitations']+'. A positive participation count with zero noise has infinite epsilon, represented by '
        '`finite=false` and `epsilon=null`. An unobserved client incurs zero release cost. Public-seeded simulation noise can be subtracted; '
        'the floating-point sampler is not a certified deployment mechanism. Public-benchmark clipping calibration is outside the accountant. '
        'Hypothetical full-schedule bounds for failed original runs exclude failure signals and data-dependent termination; no privacy guarantee is asserted for those transcripts.', '',
        '## Reproduction and assets', '', 'The published training plans are `configs/transfer_study.json` and `configs/coordinate_noise_study.json`. '
        'Use their recorded producer commits for fresh training, preserve caches and run one GPU driver at a time as described in README. '
        'The earlier attack-free `configs/privacy_plan.json` remains the historical calibration protocol, not this mixture experiment.', '',
        'The completed evidence reconstructs on CPU without datasets or raw update arrays:', '', '```bash',
        'python scripts/run_transfer_study.py --no-run', 'python scripts/run_coordinate_study.py --no-run',
        'python scripts/rescore_extensions.py --recorded-only', 'python scripts/summarize_extensions.py',
        'python scripts/analyze_studies.py --recorded-only', 'python scripts/document_privacy.py',
        'python scripts/make_macros.py', 'python scripts/make_tables.py', 'python scripts/make_figures.py', '```', '',
        'For a newly trained coordinate cohort, first run `python scripts/analyze_studies.py --coordinate-cache PATH` with its retained cache root. '
        'This records both extension geometries before portable replay. Source, plan, identity, clock, complete coverage, accounting and integer ledgers are checked. '
        'Original producer files and frozen calibrations must remain unchanged.', '',
        'The honest-cost/accuracy figure is double-column width; the attacker-share figure is single-column. Coordinate labels are categorical positions, '
        'with offsets separating rules; they are not noise multipliers. Tables use booktabs and include attacker share, honest cost, shared accuracy, implied scales, '
        'accounting, all outcomes and original divergence. PrivacyA through PrivacyD refer to the declared coordinate groups in order; '
        'CoordinateSD and NoiseMultiplier macros distinguish tau from z. All numbers and plot inputs come from the shared summary.', '']
    Path('docs/PRIVACY.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def fashion_document(summary):
    result = summary['fashion']
    ordering_text = ('Some pairwise comparisons remain undefined.' if any(row['defined_pairs'] != len(row['pairs']) for row in result['ordering']) else
                     'The complete mean ordering is not preserved.' if any(row['same_order_pairs'] != row['defined_pairs'] for row in result['ordering']) else
                     'Every defined pair retains its mean ordering in this cohort.')
    lines = ['# Fashion-MNIST transfer', '',
        'Generated from `results/summary.json` by `python scripts/document_fashion.py`.', '',
        '## Fixed comparison', '',
        f"The cohort contains {result['nominal_cells']} {result['device'].upper()} runs, seeds {', '.join(map(str, result['seeds']))}, "
        'with the fixed mixture and label flipping. The complete original MNIST reward parameters and historical Mstar source are unchanged. '
        'No Fashion-MNIST outcome selects a threshold, reputation parameter or penalty setting.', '',
        'All six rules score the same updates within each run. The model uses fixed sample-count-weighted FedAvg, so accuracy is shared and cannot '
        'be attributed to a shadow reward rule. The MNIST reference uses exactly matching seeds, horizons and attack conditions from the unclipped grid. '
        'The clipped privacy controls are not used for this dataset comparison.', '',
        '| Condition | Rule | Attacker share, % [95% CI] | Honest false penalties, % [95% CI] |', '|---|---|---|---|']
    for row in result['intervals']:
        for rule in result['rules']:
            lines.append(f"| {condition_label(row)} | {rule} | {estimate(row['metrics'][rule]['attacker_share'])} | "
                         f"{estimate(row['metrics'][rule]['honest_false_penalty_rate'])} |")
    lines += ['', '## Does the MNIST ordering hold?', '',
        ordering_text+' The table counts every pair of rules: a reversal means a strict ordering flips direction; '
        'other changes include a tie appearing or disappearing. These are descriptive comparisons of means, not tests of equivalence.', '',
        '| Condition | Cost | Same ordering | Strict reversals | Other changes | Undefined |', '|---|---|---|---|---|---|']
    explanations = []
    for row in result['ordering']:
        label = 'Attacker share' if row['metric'] == 'attacker_share' else 'Honest false penalties'
        lines.append(f"| {condition_label(row)} | {label} | {row['same_order_pairs']}/{len(row['pairs'])} | {row['reversed_pairs']} | "
                     f"{row['defined_pairs']-row['same_order_pairs']-row['reversed_pairs']} | {len(row['pairs'])-row['defined_pairs']} |")
        reversed_pairs = [item['left']+' / '+item['right'] for item in row['pairs'] if item['reversed']]
        if reversed_pairs:
            explanations.append(f"{condition_label(row)}, {label.lower()}: strict reversals involve {', '.join(reversed_pairs)}.")
    for explanation in explanations:
        lines += ['', explanation]
    lines += ['', '| Condition | Shared final accuracy, % [95% CI] |', '|---|---|']
    for row in result['intervals']:
        lines.append(f"| {condition_label(row)} | {estimate(row['metrics']['B0']['final_accuracy'])} |")
    lines += ['', '## Uncertainty and scope', '',
        f"Each interval resamples the {result['seed_count']} complete seeds with {result['statistical_plan']['bootstrap_resamples']} fixed-seed draws. "
        'Pooled outcomes first average the two conditions within a seed. Paired Mstar contrasts use condition-seed pairs and a separate whole-seed sensitivity '
        f"test, with Holm correction over the {len(result['contrasts'])} declared baseline/cost contrasts. "
        f"The smallest possible two-sided nonzero seed-test p-value is {result['statistical_plan']['minimum_two_sided_seed_p']}. "
        'Intervals and rankings therefore remain descriptive; no claim of reliable rule ordering follows merely from a point estimate.', '',
        'Different class semantics, partitions and sample counts affect the learning trajectory. Label flipping maps y to 9-y on both datasets, '
        'but equal attack fractions do not imply equal poisoning difficulty. MNIST and Fashion-MNIST are small image benchmarks, and this '
        'transfer does not establish generality to cross-silo data, larger models or adaptive attackers.', '',
        '## Reproduction and assets', '',
        'The six cells are declared in `configs/transfer_study.json`; the historical `configs/fashion_plan.json` is a separate protocol. '
        'The committed empirical cohort and frozen extension geometries reconstruct on CPU without downloading data:', '', '```bash',
        'python scripts/run_transfer_study.py --no-run', 'python scripts/rescore_extensions.py --recorded-only',
        'python scripts/summarize_extensions.py', 'python scripts/analyze_studies.py --recorded-only',
        'python scripts/document_fashion.py', 'python scripts/make_macros.py', 'python scripts/make_tables.py',
        'python scripts/make_figures.py', '```', '',
        'The comparison and accuracy figures are single-column. Wide booktabs tables retain every rule, the matching MNIST means and paired dataset '
        'differences, plus the complete ordering counts. The `fashion` summary section supplies all empirical macros and assets. '
        'Existing FashionMixture and FashionLabelFlip macro names remain stable; CGSV/RFFL and ordering counts extend them.', '']
    Path('docs/FASHION.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
