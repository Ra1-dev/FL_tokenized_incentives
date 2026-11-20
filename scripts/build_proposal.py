"""Generate the research proposal from recorded protocol and design inputs."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

from fltoken.io import write_json


def main():
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    method = json.loads(Path("results/method/proposed.json").read_text(encoding="utf-8"))
    literature = {row["key"]: row for row in json.loads(Path("results/literature/annotations.json").read_text(encoding="utf-8"))}
    proposal = dict(
        title="Tokenized incentives for federated learning",
        author="Rauan Meiramuly", affiliation="Nazarbayev University", course="CSCI 435",
        stage="Research proposal: mitigation evaluation planned",
        protocol=summary["protocol"], design=method,
        evaluation_seeds=list(range(42, 47)),
        references=[literature[key] for key in ["sendros", "wu2025", "dong2024flock"]],
        gap="Existing blockchain reward and penalty systems do not establish whether removing a client's own update from its reward reference reduces the self-inclusion payment to noise free-riders under a controlled comparison that also measures honest-client costs.",
        question="Can client-excluding references, reputation and delayed stake penalties reduce attacker rewards while preserving useful participation by honest clients?",
        subquestions=[
            "How does self-inclusion affect noise rewards as training converges, and how closely does the published free-rider effect reproduce?",
            "Which reference, reputation and penalty components change the tradeoff between attacker rewards and honest false penalties under free-riding and poisoning?",
            "How does that tradeoff change for privacy-noised honest updates and published similarity-based rules, and what does exact on-chain settlement cost?",
        ],
        planned_caption="Attacker reward share and honest false-penalty rate under a fixed training trajectory. Each rule scores the same sampled updates. Points summarize evaluation seeds and bars show bootstrap confidence intervals over seeds. Panels separate attack types and registered attacker fractions; the long-run mix is shown separately. Component ablations identify the incremental effect of the client-excluding reference, reputation, and slash/vesting. Lower attacker share alone is insufficient if honest penalties rise. This is a planned caption; no mitigation result is asserted.",
    )
    write_json("results/proposal.json", proposal)
    summary["proposal"] = proposal
    write_json("results/summary.json", summary)
    c = proposal["protocol"]
    ink, blue, pale = colors.HexColor("#16243A"), colors.HexColor("#255C91"), colors.HexColor("#EEF3F8")
    styles = {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=ink, spaceAfter=10),
        "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=9, leading=13, textColor=blue, spaceAfter=16),
        "h": ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=blue, spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=9.7, leading=14, textColor=ink, spaceAfter=7, alignment=TA_LEFT),
        "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=8, leading=11, textColor=ink, spaceAfter=6),
    }
    def p(text, style="body"):
        return Paragraph(text, styles[style])
    def section(title, text):
        return [p(title, "h"), p(text)]
    story = [p(proposal["title"], "title"), p(f"{proposal['author']} | {proposal['affiliation']} | {proposal['course']}<br/>{proposal['stage']}", "subtitle")]
    story += section("Problem", "A reward contract can settle payments correctly while rewarding updates that contribute no useful training. In cosine-to-aggregate scoring, an update appears inside its own reference. Sendros et al. identify a positive self-inclusion term that can reward random noise as honest updates shrink [1]. Reward fairness therefore requires evaluating the scoring rule as well as the contract.")
    story += section("Specific gap", proposal["gap"])
    story += section("Position against related work", "Wu and Seneviratne combine alignment-based rewards with blockchain gas analysis, but leave incentive effectiveness unevaluated [2]. FLock already combines peer voting, rewards and stake penalties against poisoning [3]. The proposed contribution is a measured correction to a particular reward-reference failure, with a common-update comparison and exact settlement. It is not the introduction of blockchain rewards, reputation or slashing.")
    story += section("Research question", proposal["question"])
    for i, question in enumerate(proposal["subquestions"], 1):
        story.append(p(f"<b>{i}.</b> {question}"))
    story += section("What would count as useful evidence?", "A reproducible reduction in attacker allocation, accompanied by explicit honest-client cost and uncertainty. A negative outcome is informative: excluding self-inclusion may suppress noise while also rejecting useful non-IID or privacy-noised updates. Accuracy alone cannot distinguish shadow reward rules because their training trajectory is deliberately shared.")
    story.append(PageBreak())
    story += [p("Method and evaluation plan", "title")]
    story += section("Implementation", "Use Flower to train and evaluate the global model. Compare equal split, the reproduced cosine rule, Multi-Krum acceptance, and a proposed rule using a coordinate-wise median of the other sampled updates. Add per-client EMA reputation and persistent-low-score penalties with delayed vesting. Each component remains independently switchable. A Solidity token and reward pool will replay a Python integer ledger, including rewards, slashes and vesting.")
    rows = [
        ["Protocol", f"{c['dataset'].upper()}; {c['parameters']:,}-parameter LeNet; {c['num_clients']} clients, {c['clients_per_round']} per round"],
        ["Local training", f"{c['local_epochs']} epochs; batch {c['batch_size']}; SGD lr {c['lr']}, momentum {c['momentum']}"],
        ["Partition / duration", f"Dirichlet alpha {c['alpha']}; {c['rounds']} main rounds; longer mix for convergence"],
        ["Pilot / evaluation seeds", f"{method['pilot_seeds']} / {proposal['evaluation_seeds']}"],
        ["Attacks", "Zero, noise, stale, label flip, sign flip, mixed and attack-free controls"],
        ["Attacker fractions", ", ".join(f"{100*x:g}%" for x in method["fractions"]) + "; original reproduction mix retained separately"],
    ]
    table = Table([[p(a, "small"), p(b, "small")] for a, b in rows], colWidths=[125, A4[0]-96-125])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), pale), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                               ("LINEBELOW", (0, 0), (-1, -1), .3, colors.HexColor("#D3DFE9"))]))
    story += [table]
    story += section("Measurement and controls", "Freeze all mitigation parameters after the disjoint pilot corpus, before evaluation scoring. Report attacker share, honest false penalties, honest payout inequality, detection precision/recall, accuracy, and settlement gas. Use seed-level bootstrap intervals and paired tests with multiplicity correction. Replay exact base-unit accounting on a local chain. Extend the reference-bias analysis to CGSV and RFFL, privacy-noised honest clients with a stated accountant, and Fashion-MNIST.")
    story += [p("Planned main-figure caption", "h"), p(proposal["planned_caption"], "small")]
    story += section("Scope and limitations", "The coordinator sees updates and posts scores. Fixed attacks do not establish adaptive-attack security or rational-client incentives. Peer agreement can penalize useful heterogeneous data. Shadow payouts do not cause accuracy changes; a separate feedback experiment would be needed to study incentive-driven training. Local gas measurements require explicit compiler and execution settings.")
    story += [p("References", "h")]
    for i, entry in enumerate(proposal["references"], 1):
        story.append(p(f"[{i}] {entry['authors']} ({entry['year']}). {entry['title']}. <link href=\"{entry['url']}\" color=\"#255C91\">Source</link>.", "small"))
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#D3DFE9"));canvas.line(48, 37, A4[0]-48, 37)
        canvas.setFont("Helvetica", 8);canvas.setFillColor(ink)
        canvas.drawString(48, 25, "Research proposal | Tokenized incentives for federated learning")
        canvas.drawRightString(A4[0]-48, 25, str(doc.page))
    SimpleDocTemplate("docs/proposal.pdf", pagesize=A4, leftMargin=48, rightMargin=48,
                      topMargin=43, bottomMargin=48, title=proposal["title"], author=proposal["author"],
                      invariant=1).build(story, onFirstPage=footer, onLaterPages=footer)
    print("Generated docs/proposal.pdf from results/proposal.json and recorded source inputs")


if __name__ == "__main__":
    main()
