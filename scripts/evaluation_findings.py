"""Select headline benefits and costs from the same completed comparison."""

PRIMARY = ["B0", "B1", "B2", "Mstar"]


def headline_rows(evaluation):
    rows = []
    for metric, label in [("attacker_share", "Attackers, pooled"),
                          ("honest_false_penalty_rate", "Honest penalties, pooled")]:
        pooled = evaluation["pooled"][metric]
        rows.append(dict(label=label, metric=metric, scenarios=pooled["scenarios"], fraction=None,
                         estimates={rule:pooled["rules"][rule]["estimate"] for rule in PRIMARY}))
    stale = sorted((row for row in evaluation["intervals"] if row["horizon"] == "main" and row["attack"] == "stale"),
                   key=lambda row:row["fraction"])
    if len(stale) != 3 or [row["fraction"] for row in stale] != [.1, .2, .3]:
        raise ValueError("headline comparison requires all three declared stale fractions")
    for row in stale:
        rows.append(dict(label=f"Stale {100*row['fraction']:g}%", metric="attacker_share", scenarios=1,
                         fraction=row["fraction"], estimates={rule:row["metrics"][rule]["attacker_share"]["estimate"]
                                                              for rule in PRIMARY}))
    return rows


def headline_text(evaluation):
    rows = headline_rows(evaluation)
    lines = ["## Measured benefit and costs", "",
             "All values below are percentages of the stated outcome; brackets give 95% seed bootstrap intervals.", "",
             "| Outcome | B0 | B1 | B2 | Mstar |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        values = [("Undefined" if row['estimates'][rule] is None else
                   f"{100*row['estimates'][rule]['mean']:.2f} [{100*row['estimates'][rule]['lower']:.2f}, "
                   f"{100*row['estimates'][rule]['upper']:.2f}]") for rule in PRIMARY]
        lines.append("| "+" | ".join([row["label"], *values])+" |")
    penalties = rows[1]["estimates"]
    if all(value is not None for value in penalties.values()):
        maximum = max(value["mean"] for value in penalties.values())
        worst = [rule for rule,value in penalties.items() if value["mean"] == maximum]
        ranking = f"The highest pooled honest false-penalty rate among the four primary rules belongs to {', '.join(worst)}."
    else:
        ranking = "An undefined honest-cost estimate prevents a complete ranking."
    defined = [row for row in rows[2:] if all(row["estimates"][rule] is not None for rule in ["Mstar", "B2"])]
    stale_losses = sum(row["estimates"]["Mstar"]["mean"] > row["estimates"]["B2"]["mean"] for row in defined)
    attacker = rows[0]["estimates"]
    if any(value is None for value in attacker.values()):
        benefit = "An undefined pooled attacker estimate prevents a complete reward comparison."
    elif all(attacker["Mstar"]["mean"] < attacker[rule]["mean"] for rule in PRIMARY[:-1]):
        benefit = "Mstar reduces pooled attacker rewards but is not uniformly better."
    else:
        benefit = "Mstar does not have lower pooled attacker rewards than every baseline."
    lines += ["", f"Attacker-share pooling uses {rows[0]['scenarios']} attacked conditions; honest-cost pooling uses "
              f"{rows[1]['scenarios']} conditions including attack-free runs. "
              f"Mstar's stale-attacker share exceeds B2 at {stale_losses} of {len(defined)} defined fractions (three planned). "
              +ranking, "",
              "The honest false-penalty rate counts honest participations receiving less than half the equal-share reward. "
              "It is distinct from an executed stake slash. The settlement analysis counts actual slash flags and balance losses.", "",
              "Stale updates can retain honest-looking directions, while the leave-one-out median can assign low scores "
              "to heterogeneous honest updates. These mechanisms explain why reward suppression can miss stale attackers "
              "and harm honest clients; the fixed experiment does not isolate either mechanism causally.", "",
              benefit+" The paired condition-seed tests "
              "and their dependence caveat below must accompany the comparison; the five-seed sensitivity tests "
              "do not establish a population-wide significance claim.", ""]
    return "\n".join(lines)
