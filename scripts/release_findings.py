"""Paper-support findings derived from the completed shared summary."""

from decimal import Decimal


RULE_NAMES = {'B0': 'Equal', 'B1': 'Cosine', 'B2': 'Krum', 'Mstar': 'Ours', 'CGSV': 'CGSV', 'RFFL': 'RFFL'}


def release_ready(summary):
    """Return whether the summary carries every completed study used by the release findings."""
    try:
        fashion_rules = set(summary['fashion']['pooled']['attacker_share'])
        return ('bias_intervals' in summary['extensions']
                and 'original_multiplier_study' in summary['privacy']
                and bool(fashion_rules) and fashion_rules <= set(RULE_NAMES)
                and 'penalty_ablation' in summary['grid_settlement'])
    except (KeyError, TypeError):
        return False


def release_values(summary):
    extension, privacy, fashion = (summary[key] for key in ('extensions', 'privacy', 'fashion'))
    main_bias = [row for row in extension['bias_intervals'] if row['horizon'] == 'main']
    values = {}
    for metric, label in [('fraction_positive_cosine', 'InclusivePositive'),
                          ('fraction_positive_excluding_cosine', 'ExcludingPositive')]:
        observations = [100*row['metrics'][metric]['estimate']['mean'] for row in main_bias]
        values['Release'+label+'Low'] = f'{min(observations):.1f}'
        values['Release'+label+'High'] = f'{max(observations):.1f}'
    eps = [row['conservative_all_rounds']['epsilon'] for row in privacy['cells']
           if row['conservative_all_rounds']['finite']]
    values['ReleaseAllRoundEpsilonLow'] = f'{min(eps):.6g}'
    values['ReleaseAllRoundEpsilonHigh'] = f'{max(eps):.6g}'
    observed = [row['maximum_client_epsilon'] for cell in privacy['cells'] if cell['noise_multiplier'] > 0
                for row in cell['per_seed_accounting']]
    values['ReleaseObservedMaxEpsilonLow'] = f'{min(observed):.6g}'
    values['ReleaseObservedMaxEpsilonHigh'] = f'{max(observed):.6g}'
    failures = [row for row in privacy['original_multiplier_study']['outcomes'] if row['status'] == 'failed_numerical']
    values['ReleaseNoiseFailures'] = len(failures)
    values['ReleaseLatestFailedRound'] = max(row['completed_rounds']+1 for row in failures)
    for metric, label in [('attacker_share', 'AttackerShare'), ('honest_false_penalty_rate', 'HonestPenalty')]:
        for rule, observation in fashion['pooled'][metric].items():
            name = RULE_NAMES[rule]
            for field, suffix in [('mean', 'Mean'), ('lower', 'Low'), ('upper', 'High')]:
                values['ReleaseFashion'+name+label+suffix] = f"{100*observation['estimate'][field]:.3f}"
    groups = summary['grid_settlement']['penalty_ablation']['aggregates']['main']['groups']['Afull']
    values['ReleaseSlashedCountRatio'] = f"{groups['adversary']['ever_slashed_clients']/groups['honest']['ever_slashed_clients']:.2f}"
    return values


def findings(summary):
    v = release_values(summary)
    extension, privacy = summary['extensions'], summary['privacy']
    def pooled(rule, metric):
        return 100*extension['pooled'][metric]['rules'][rule]['estimate']['mean']
    def cost(cell, rule, metric):
        return 100*cell['metrics'][rule][metric]['estimate']['mean']
    first, last = privacy['cells'][0], privacy['cells'][-1]
    groups = summary['grid_settlement']['penalty_ablation']['aggregates']['main']['groups']['Afull']
    attacker, honest = groups['adversary'], groups['honest']
    fashion = summary['fashion']
    ranking = sorted(fashion['pooled']['attacker_share'], key=lambda rule: fashion['pooled']['attacker_share'][rule]['estimate']['mean'])
    fashion_text = ', '.join(f"{rule} {100*fashion['pooled']['attacker_share'][rule]['estimate']['mean']:.1f}%" for rule in ranking)
    return [
        f"CGSV and RFFL inherit self-inclusion bias. Across main-cohort conditions and participation subsets, positive noise-reference cosines occur in {v['ReleaseInclusivePositiveLow']}–{v['ReleaseInclusivePositiveHigh']}% of participations, versus {v['ReleaseExcludingPositiveLow']}–{v['ReleaseExcludingPositiveHigh']}% after removing the client's own contribution (near chance level). These are ranges of seed-mean proportions; the full condition intervals and eligibility counts remain in the extension tables.",
        f"Reputation and thresholding still reduce pooled attacker share to {pooled('CGSV','attacker_share'):.2f}% for CGSV and {pooled('RFFL','attacker_share'):.2f}% for RFFL, compared with B1 {pooled('B1','attacker_share'):.2f}% and Mstar {pooled('Mstar','attacker_share'):.2f}%. Honest false-penalty rates are {pooled('CGSV','honest_false_penalty_rate'):.1f}% and {pooled('RFFL','honest_false_penalty_rate'):.1f}%, versus Mstar {pooled('Mstar','honest_false_penalty_rate'):.1f}%. Mstar is not Pareto-dominant: its lowest attacker share comes with the highest honest cost.",
        f"The positive multiplier-scale local-noise cells all diverged: {v['ReleaseNoiseFailures']}/{v['ReleaseNoiseFailures']} failures within the first {v['ReleaseLatestFailedRound']} rounds. The clipping bound was C={privacy['clipping_bound']:.3f}; failure prefixes are retained as diagnostics, never complete-horizon utility results.",
        f"At coordinate noise scales {', '.join(str(row['coordinate_standard_deviation']) for row in privacy['cells'])}, Mstar honest false penalties remain approximately {min(cost(row,'Mstar','honest_false_penalty_rate') for row in privacy['cells']):.1f}–{max(cost(row,'Mstar','honest_false_penalty_rate') for row in privacy['cells']):.1f}% while attacker share rises {cost(first,'Mstar','attacker_share'):.1f} → {cost(last,'Mstar','attacker_share'):.1f}%. B2 degrades most: honest false penalties {cost(first,'B2','honest_false_penalty_rate'):.1f} → {cost(last,'B2','honest_false_penalty_rate'):.1f}%, attacker share {cost(first,'B2','attacker_share'):.1f} → {cost(last,'B2','attacker_share'):.1f}%. Insensitivity describes this measured range, not a statistical equivalence claim.",
        f"For positive coordinate noise, conservative every-round epsilon bounds are {v['ReleaseAllRoundEpsilonLow']}–{v['ReleaseAllRoundEpsilonHigh']} at delta={privacy['delta']:g}, under replacement adjacency with sensitivity 2C. Maximal observed per-client bounds across individual runs span {v['ReleaseObservedMaxEpsilonLow']}–{v['ReleaseObservedMaxEpsilonHigh']}. Neither supplies meaningful privacy; this experiment measures fairness to noisy honest clients. Public-seeded simulation draws supply no deployment guarantee.",
        f"Fashion-MNIST pooled attacker shares are {fashion_text}; Mstar honest false penalties are {100*fashion['pooled']['honest_false_penalty_rate']['Mstar']['estimate']['mean']:.1f}%. Its edge over CGSV/RFFL disappears. The broad comparison transfers, but complete ordering does not: condition-specific reversals and ties remain in the transfer table. These {fashion['seed_count']}-seed estimates are descriptive.",
        f"Penalties act on stake only in the measured main horizon: adversaries {attacker['ever_slashed_clients']}/{attacker['clients']} client-runs slashed ({attacker['slash_events']} events, {Decimal(attacker['burned_stake'])/10**18} tokens burned); honest clients {honest['ever_slashed_clients']}/{honest['clients']} ({honest['slash_events']} events, {Decimal(honest['burned_stake'])/10**18} tokens). Unvested reward forfeiture is {Decimal(attacker['burned_rewards'])/10**18} tokens for adversaries and {Decimal(honest['burned_rewards'])/10**18} for honest clients. The slashed client-run count is about {v['ReleaseSlashedCountRatio']} times larger for adversaries; denominators differ, so this is not a normalized incidence ratio. Allocations are identical with and without the penalty component; observed stake losses do not prove behavioral deterrence.",
    ]


LIMITATIONS = [
    'Stale updates retain plausible directions; Mstar underperforms Krum on stale-attacker reward share at every measured fraction.',
    'The lowest pooled attacker share carries the highest honest false-penalty cost; a low reward is not proof of dishonesty.',
    'CGSV/RFFL retain geometric self-inclusion yet offer a different attacker-versus-honest-cost trade-off. Token adaptations are not the complete personalized-gradient algorithms.',
    'Multiplier-scale local noise diverged. Coordinate-noise epsilon values are too large for meaningful privacy; public-seeded floating-point noise is not a private deployment mechanism.',
    'Measured penalties burn stake, not unvested rewards, and do not change allocation shares. Economic loss is measured; rational deterrence is not.',
    'MNIST and Fashion-MNIST are one small image-dataset family; limited seeds, fixed attacks and small models restrict generalization.',
    'Simulation and local-chain replay are not deployment. The coordinator sees updates and posts trusted scores; no adaptive adversary, Sybil resistance or closed-loop reward response is established.',
    'The frozen historical scorer has an underflow edge case. Its numerical repair is a separately identified variant, with unchanged measured grid allocations; historical evidence retains its original producer.',
    'OpenFL and Chen et al. have abstract-level annotations only. Their full methods/results were not read; no detailed omission or quantitative superiority claim follows.',
    'Conditions sharing seeds are dependent. Whole-seed tests and bootstrap intervals accompany descriptive means; rankings are not population-wide significance claims.',
]
