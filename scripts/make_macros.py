"""One result summary feeds every numerical paper macro."""

import json
from pathlib import Path

from fltoken.config import load_config
from fltoken.io import write_json
from fltoken.models import LeNet5


def paired_macros(evaluation):
    """Export the precomputed contrasts without changing their inference units."""
    values = {}
    metrics = {"attacker_share": "AttackerShare", "honest_false_penalty_rate": "HonestPenalty"}
    baselines = {"B0": "Equal", "B1": "Cosine", "B2": "Krum"}
    for row in evaluation["contrasts"]:
        prefix = "Evaluation"+metrics[row["metric"]]+baselines[row["baseline"]]
        paired = row["paired_difference"]
        values[prefix+"PairedDefined"] = paired["defined"]
        values[prefix+"PairedMissing"] = paired["undefined"]
        for key, suffix in [("mean", "Mean"), ("sample_sd", "SD"), ("lower", "Low"), ("upper", "High")]:
            estimate = paired["estimate"]
            values[prefix+"Difference"+suffix] = ("undefined" if estimate is None or estimate[key] is None
                                                  else f"{100*estimate[key]:.3f}")
        for key, label in [("condition_seed_test", "ConditionSeed"), ("seed_test", "Seed")]:
            test = row[key]
            values[prefix+label+"Pairs"] = "undefined" if test is None else test["pairs"]
            for field, suffix in [("p_value", "P"), ("rank_biserial", "RankEffect")]:
                values[prefix+label+suffix] = "undefined" if test is None else f"{test[field]:.6g}"
            adjusted = row[key+"_holm_p"]
            values[prefix+label+"HolmP"] = "undefined" if adjusted is None else f"{adjusted:.6g}"
    return values


def main():
    path = Path("results/summary.json")
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    source = json.loads(Path("results/literature/sendros.json").read_text(encoding="utf-8"))
    config = load_config("configs/sendros_mix.yaml")
    summary["protocol"] = {**config, "parameters": sum(p.numel() for p in LeNet5().parameters())}
    summary["published"] = source
    write_json(path, summary)
    p = summary["protocol"]
    values = {"NumClients": p["num_clients"], "SampledClients": p["clients_per_round"], "NumRounds": p["rounds"],
              "LocalEpochs": p["local_epochs"], "BatchSize": p["batch_size"], "LearningRate": p["lr"],
              "Momentum": p["momentum"], "DirichletAlpha": p["alpha"], "NumParameters": f"{p['parameters']:,}",
              "NoiseSigma": p["noise_sigma"], "StaleLag": p["stale_lag"],
              "PublishedShare": f"{100*source['standard']['attacker_share_mean']:.2f}",
              "PublishedShareSD": f"{100*source['standard']['attacker_share_sd']:.2f}"}
    r = summary.get("reproduction")
    if r:
        values.update(MeasuredShare=f"{100*r['attacker_share_mean']:.2f}",
                      MeasuredShareSD=f"{100*r['attacker_share_sample_sd']:.2f}",
                      BiasMedianError="undefined" if r['relative_error_quantiles']['0.5'] is None else f"{100*r['relative_error_quantiles']['0.5']:.2f}",
                      BiasMaxError="undefined" if r['relative_error_quantiles']['1'] is None else f"{100*r['relative_error_quantiles']['1']:.2f}",
                      NoiseParticipations=r["noise_participations"])
    if "contracts" in summary:
        c, g = summary["contracts"], summary["gas"]
        values.update(ContractTests=c["checks"]["tests_passed"], ContractFuzzRuns=c["checks"]["fuzz_runs"],
                      ExactChainRounds=c["replay_rounds"], StressChainRounds=c["stress_rounds"],
                      StressSlashParticipations=c["stress_slashes"], TransactionGasCap=f"{g['cap']:,}",
                      ExecutedCapacity=g["executed_success_n"], ExecutedCapacityFailure=g["executed_failure_n"])
    if "grid_settlement" in summary:
        from document_grid_settlement import checked_coverage
        grid = checked_coverage(summary["grid_settlement"])
        values.update(SettlementGridRuns=grid["grid_runs"], SettlementJobs=grid["jobs"],
                      SettlementRounds=grid["round_rule_cases"])
        for rule, label in [("B0", "Equal"), ("B1", "Cosine"), ("B2", "Krum"), ("Mstar", "Ours")]:
            values["Settlement"+label+"Runs"] = grid["by_rule"][rule]["runs"]
            values["Settlement"+label+"Rounds"] = grid["by_rule"][rule]["round_rule_cases"]
        for horizon,record in grid["penalty_ablation"]["aggregates"].items():
            for group,label in [("honest","Honest"),("adversary","Attacker")]:
                measured = record["groups"]["Afull"][group]
                prefix = "Settlement"+horizon.title()+label
                values[prefix+"ClientRuns"] = measured["clients"]
                values[prefix+"EverSlashed"] = measured["ever_slashed_clients"]
                values[prefix+"SlashEvents"] = measured["slash_events"]
                from decimal import Decimal
                values[prefix+"StakeBurnTokens"] = format(Decimal(measured["burned_stake"])/10**18, "f")
                values[prefix+"RewardBurnTokens"] = format(Decimal(measured["burned_rewards"])/10**18, "f")
        for row in grid["penalty_ablation"]["intervals"]:
            prefix = "Settlement"+row["horizon"].title()+{"Aema":"EMA","Afull":"Full"}[row["stage"]]+row["group"].title()
            for field,label in [("burned_stake","StakeLoss"),("burned_rewards","RewardLoss"),
                                ("net_token_outcome","NetOutcome"),("ever_slashed_clients","SlashedClients")]:
                for key,suffix in [("mean","Mean"),("sample_sd","SD"),("lower","Low"),("upper","High")]:
                    estimate = row["metrics"][field][key]
                    values[prefix+label+suffix] = "undefined" if estimate is None else f"{estimate:.3f}"
    if "reproduction_overall" in summary:
        overall = summary["reproduction_overall"]
        values.update(OverallMixtureShare=f"{100*overall['attacker_share_mean']:.3f}",
                      OverallMixtureShareSD=f"{100*overall['attacker_share_sample_sd']:.3f}",
                      OverallMixtureShareLow=f"{100*overall['attacker_share_bootstrap']['lower']:.3f}",
                      OverallMixtureShareHigh=f"{100*overall['attacker_share_bootstrap']['upper']:.3f}",
                      OverallMixtureSeeds=len(overall["seeds"]),
                      SourceMixtureShareLow=f"{100*r['attacker_share_bootstrap']['lower']:.3f}",
                      SourceMixtureShareHigh=f"{100*r['attacker_share_bootstrap']['upper']:.3f}")
    if "method_selection" in summary:
        selected = summary["method_selection"]
        parameters = selected["configuration"]["params"]
        settlement = selected["configuration"]["settlement"]
        chosen = selected["selection"]["selected"]
        values.update(MethodEMADecay=parameters["beta"], MethodQualityThreshold=parameters["threshold"],
                      MethodPatience=parameters["patience"], MethodInitialReputation=parameters["initial_reputation"],
                      MethodVestingRounds=settlement["vesting_rounds"], MethodStakeTokens=settlement["stake_tokens"],
                      MethodSlashNumerator=settlement["slash_numerator"], MethodSlashDenominator=settlement["slash_denominator"],
                      PilotCells=selected["selection"]["cells"], PilotCandidates=len(selected["selection"]["candidates"]))
        values.update(PilotSelectedCandidate=chosen["candidate"], PilotObjectiveSum=f"{chosen['objective']:.8f}",
                      PilotAttackerCost=f"{chosen['mean_attacker_share_cost']:.8f}",
                      PilotHonestCost=f"{chosen['mean_honest_false_penalty_rate']:.8f}",
                      PilotUndefinedCells=chosen["undefined_attacker_cells"], PilotEmptyRounds=chosen["empty_rounds"],
                      PilotHonestSlashedSum=chosen["honest_clients_slashed_sum"])
    extension_selection = summary.get("extension_design", {}).get("selection")
    if extension_selection is not None:
        values.update(ExtensionPilotCells=extension_selection["cells"],
                      ExtensionPilotCandidates=len(extension_selection["candidates"]),
                      ExtensionCGSVBeta=extension_selection["configuration"]["params"]["CGSV"]["beta"])
        for family, chosen in extension_selection["selected"].items():
            values["Extension"+family+"PilotObjective"] = f"{chosen['objective']:.8f}"
            values["Extension"+family+"PilotAttackerCost"] = f"{chosen['mean_attacker_share_cost']:.8f}"
            values["Extension"+family+"PilotHonestCost"] = f"{chosen['mean_honest_false_penalty_rate']:.8f}"
    if "privacy_calibration" in summary:
        calibrated = summary["privacy_calibration"]["evidence"]
        values.update(PrivacyCalibrationClipBound=f"{calibrated['clipping_bound']:.17g}",
                      PrivacyCalibrationParticipations=calibrated["participations"],
                      PrivacyCalibrationSeeds=len(calibrated["cells"]))
    if "evaluation" in summary:
        from evaluation_findings import headline_rows
        for row in headline_rows(summary["evaluation"])[2:]:
            fraction = {.1:"Ten", .2:"Twenty", .3:"Thirty"}[row["fraction"]]
            for rule,label in [("B0","Equal"),("B1","Cosine"),("B2","Krum"),("Mstar","Ours")]:
                for key,suffix in [("mean","Mean"),("lower","Low"),("upper","High")]:
                    estimate = row['estimates'][rule]
                    values["EvaluationStale"+fraction+label+suffix] = "undefined" if estimate is None else f"{100*estimate[key]:.3f}"
        evaluation = summary["evaluation"]
        values.update(paired_macros(evaluation))
        values.update(EvaluationRuns=evaluation["nominal_cells"],
                      EvaluationConditions=evaluation["distinct_main_scenarios"],
                      EvaluationSeeds=evaluation["seed_count"])
        if "runtime" in evaluation:
            values.update(EvaluationElapsedSeconds=f"{evaluation['runtime']['elapsed_wall_seconds']:.1f}",
                          EvaluationTrainingSeconds=f"{evaluation['runtime']['training_wall_seconds']:.1f}")
        names = {"B0": "Equal", "B1": "Cosine", "B2": "Krum", "Mstar": "Ours",
                 "Aref": "Reference", "Aema": "EMA", "Afull": "Full"}
        for metric, label in [("attacker_share", "AttackerShare"), ("honest_false_penalty_rate", "HonestPenalty")]:
            for rule, name in names.items():
                estimate = evaluation["pooled"][metric]["rules"][rule]["estimate"]
                for key, suffix in [("mean", "Mean"), ("sample_sd", "SD"), ("lower", "Low"), ("upper", "High")]:
                    values["Evaluation"+name+label+suffix] = "undefined" if estimate is None else f"{100*estimate[key]:.3f}"
    if "extensions" in summary:
        extension = summary["extensions"]
        values.update(ExtensionRuns=extension["runs"], ExtensionSeeds=len(extension["seeds"]))
        metric_names = {"attacker_share": "AttackerShare", "honest_false_penalty_rate": "HonestPenalty"}
        rule_names = {"B1": "Cosine", "Mstar": "Ours", "CGSV": "CGSV", "RFFL": "RFFL"}
        for metric, pooled in extension["pooled"].items():
            for rule, observation in pooled["rules"].items():
                estimate = observation["estimate"]
                prefix = "ExtensionPooled"+rule_names[rule]+metric_names[metric]
                for field, suffix in [("mean", "Mean"), ("lower", "Low"), ("upper", "High")]:
                    values[prefix+suffix] = "undefined" if estimate is None else f"{100*estimate[field]:.4f}"
        for row in extension["contrasts"]:
            prefix = "Extension"+row["family"]+"Vs"+rule_names[row["baseline"]]+metric_names[row["metric"]]
            estimate = row["paired_difference"]["estimate"]
            for field, suffix in [("mean", "Mean"), ("lower", "Low"), ("upper", "High")]:
                values[prefix+"Difference"+suffix] = "undefined" if estimate is None else f"{100*estimate[field]:.4f}"
            for field, suffix in [("condition_seed_test_holm_p", "CellHolmP"), ("seed_test_holm_p", "SeedHolmP")]:
                values[prefix+suffix] = "undefined" if row[field] is None else f"{row[field]:.6g}"
        for row in extension["bias_intervals"]:
            if row["horizon"] != "main" or row["attack"] != "sendros_mix":
                continue
            prefix = "Extension"+row["family"]+('Eligible' if row['subset'] == 'eligible_noise' else '')
            values[prefix+"NoiseParticipations"] = row["participations"]
            values[prefix+"UndefinedGeometry"] = row["undefined_geometry"]
            for metric, label in [("mean_cosine", "Inclusive"), ("mean_excluding_cosine", "Excluding"),
                                  ("mean_self_inclusion", "SelfInclusion"),
                                  ("mean_inclusive_minus_excluding", "Difference")]:
                estimate = row["metrics"][metric]["estimate"]
                for key, suffix in [("mean", "Mean"), ("sample_sd", "SD"), ("lower", "Low"), ("upper", "High")]:
                    values[prefix+label+suffix] = "undefined" if estimate is None else f"{estimate[key]:.4f}"
    if "privacy" in summary:
        from decimal import Decimal
        privacy = summary["privacy"]
        values.update(PrivacyRuns=privacy["runs"], PrivacyCompletedRuns=privacy["completed_runs"],
                      PrivacyNumericalFailures=privacy["numerical_failures"], PrivacySeeds=len(privacy["seeds"]),
                      PrivacyClipBound=f"{privacy['clipping_bound']:.6g}", PrivacyDelta=format(Decimal(str(privacy['delta'])), "f"))
        if 'original_multiplier_study' in privacy:
            original = privacy['original_multiplier_study']
            values.update(PrivacyNewRuns=privacy['new_runs'], PrivacyReusedControls=privacy['reused_control_runs'],
                          PrivacyUniqueTrainingRuns=privacy['unique_privacy_training_runs'],
                          PrivacyOriginalRuns=original['runs'], PrivacyOriginalCompleted=original['completed_runs'],
                          PrivacyOriginalFailures=original['numerical_failures'])
        for index, row in enumerate(privacy["cells"]):
            prefix = "Privacy"+chr(ord("A")+index)
            values[prefix+"NoiseMultiplier"] = row["noise_multiplier"]
            if 'coordinate_standard_deviation' in row:
                values[prefix+'CoordinateSD'] = row['coordinate_standard_deviation']
                budgets = [seed['maximum_client_epsilon'] for seed in row['per_seed_accounting']]
                values[prefix+'MaximumClientEpsilon'] = r'\ensuremath{\infty}' if any(value is None for value in budgets) else f'{max(budgets):.6g}'
            epsilon = row["conservative_all_rounds"]["epsilon"]
            values[prefix+"ConservativeEpsilon"] = r"\ensuremath{\infty}" if epsilon is None else f"{epsilon:.3f}"
            names = dict(B0='Equal', B1='Cosine', B2='Krum', Mstar='Ours', CGSV='CGSV', RFFL='RFFL', Aref='Reference')
            for rule in privacy.get('rules', ['B1', 'Mstar', 'Aref']):
                name = names[rule]
                estimate = row["metrics"][rule]["honest_false_penalty_rate"]["estimate"]
                for key, suffix in [("mean", "Mean"), ("sample_sd", "SD"), ("lower", "Low"), ("upper", "High")]:
                    values[prefix+name+"Penalty"+suffix] = "undefined" if estimate is None or estimate[key] is None else f"{100*estimate[key]:.3f}"
                estimate = row['metrics'][rule]['attacker_share']['estimate']
                for key, suffix in [('mean', 'Mean'), ('sample_sd', 'SD'), ('lower', 'Low'), ('upper', 'High')]:
                    values[prefix+name+'AttackerShare'+suffix] = 'undefined' if estimate is None else f'{100*estimate[key]:.3f}'
            estimate = row['metrics']['B1']['final_accuracy']['estimate']
            for key, suffix in [('mean', 'Mean'), ('sample_sd', 'SD'), ('lower', 'Low'), ('upper', 'High')]:
                values[prefix+'SharedAccuracy'+suffix] = 'undefined' if estimate is None else f'{100*estimate[key]:.3f}'
    if "fashion" in summary:
        fashion = summary["fashion"]
        values.update(FashionRuns=fashion["nominal_cells"], FashionSeeds=fashion["seed_count"])
        for row in fashion["intervals"]:
            condition = "Mixture" if row["attack"] == "sendros_mix" else "LabelFlip"
            for metric, label in [("attacker_share", "AttackerShare"), ("honest_false_penalty_rate", "HonestPenalty")]:
                for rule, name in [("B0", "Equal"), ("B1", "Cosine"), ("B2", "Krum"), ("Mstar", "Ours"), ('CGSV', 'CGSV'), ('RFFL', 'RFFL')]:
                    if rule not in row['metrics']:
                        continue
                    estimate = row["metrics"][rule][metric]["estimate"]
                    for key, suffix in [("mean", "Mean"), ("sample_sd", "SD"), ("lower", "Low"), ("upper", "High")]:
                        values["Fashion"+condition+name+label+suffix] = "undefined" if estimate is None else f"{100*estimate[key]:.3f}"
            estimate = row["metrics"]["B0"]["final_accuracy"]["estimate"]
            for key, suffix in [("mean", "Mean"), ("sample_sd", "SD"), ("lower", "Low"), ("upper", "High")]:
                values["Fashion"+condition+"SharedAccuracy"+suffix] = "undefined" if estimate is None else f"{100*estimate[key]:.3f}"
        for row in fashion.get('ordering', []):
            condition = 'Mixture' if row['attack'] == 'sendros_mix' else 'LabelFlip'
            cost = 'Attacker' if row['metric'] == 'attacker_share' else 'Honest'
            prefix = 'Fashion'+condition+cost+'Ordering'
            for field, label in [('defined_pairs', 'Defined'), ('same_order_pairs', 'Same'), ('reversed_pairs', 'Reversed')]:
                values[prefix+label] = row[field]
    lines = ["% Generated from results/summary.json. Do not edit numerical values here."]
    lines += [f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in values.items()]
    folder = Path("paper/generated")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Generated {len(values)} numerical macros")


if __name__ == "__main__":
    main()
