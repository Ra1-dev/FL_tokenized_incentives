"""Generate the asset map and final research findings from recorded results."""

import json
from pathlib import Path
import re

from PIL import Image
from pypdf import PdfReader

from release_findings import findings, LIMITATIONS


FIGURES = {
    'architecture': ('System', 'Fixed training, common updates, shadow rewards and independent integer settlement.', 'protocol', 'double'),
    'reward_flow': ('Contract', 'Commit/reveal, trusted score posting, settlement and matured claims after registration.', 'settlement', 'double'),
    'reproduction_trajectory': ('Results RQ1', 'Honest and noise quality trajectories under the original reproduction seeds.', 'reproduction', 'double'),
    'reproduction_bias': ('Results RQ1', 'Predicted self-inclusion cosine, measured cosine and relative-error distribution.', 'reproduction', 'double'),
    'reproduction_attacker_share': ('Results RQ1', 'Reproduced attacker reward shares; bars show seed sample SD and points show seeds.', 'reproduction', 'single'),
    'evaluation_attacker_share': ('Results RQ2', 'Attacker share by condition and rule with seed bootstrap intervals.', 'evaluation', 'single'),
    'evaluation_honest_false_penalty_rate': ('Results RQ2', 'Honest low-reward participations by condition and rule with seed intervals.', 'evaluation', 'single'),
    'evaluation_ablation': ('Results RQ2', 'Reference, reputation and penalty components expose the attacker/honest-cost trade-off.', 'evaluation', 'double'),
    'evaluation_accuracy': ('Results RQ2', 'Shared main-horizon accuracy trajectories; shadow rules do not cause separate accuracy curves.', 'evaluation', 'double'),
    'evaluation_accuracy_long': ('Results RQ2', 'Shared long-horizon mixture accuracy with seed intervals.', 'evaluation', 'single'),
    'gas_scaling': ('Results RQ3', 'Settlement limits and total receipt gas scale with participation under the recorded state history.', 'contract', 'double'),
    'extensions_bias': ('Results RQ4', 'Noise-reference cosines with and without self-inclusion on the same recorded updates.', 'extensions', 'double'),
    'extensions_bias_contrasts': ('Results RQ4', 'CGSV/RFFL self-inclusion components with seed intervals and eligibility distinctions.', 'extensions', 'double'),
    'privacy_false_penalty': ('Results RQ4', 'Honest low-reward rates and shared accuracy across coordinate-noise scales.', 'privacy', 'double'),
    'privacy_attacker_share': ('Results RQ4', 'Attacker rewards as honest coordinate noise increases; categorical scale positions.', 'privacy', 'single'),
    'fashion_comparison': ('Results RQ4', 'Fashion-MNIST attacker rewards and honest costs across frozen rules.', 'fashion', 'single'),
    'fashion_accuracy': ('Results RQ4', 'Fashion-MNIST shared accuracy by attack condition with seed intervals.', 'fashion', 'single'),
}
GROUP_PREFIXES = {
    'protocol': ('Num', 'SampledClients', 'LocalEpochs', 'BatchSize', 'LearningRate', 'Momentum', 'DirichletAlpha', 'NoiseSigma', 'StaleLag'),
    'reproduction': ('PublishedShare', 'MeasuredShare', 'Bias', 'NoiseParticipations', 'OverallMixture', 'SourceMixture'),
    'method': ('Method', 'Pilot'),
    'evaluation': ('Evaluation',),
    'extensions': ('Extension', 'ReleaseInclusive', 'ReleaseExcluding'),
    'privacy': ('Privacy', 'ReleaseAllRound', 'ReleaseObserved', 'ReleaseNoise', 'ReleaseLatest'),
    'fashion': ('Fashion', 'ReleaseFashion'),
    'contract': ('Contract', 'ExactChain', 'Stress', 'TransactionGas', 'ExecutedCapacity'),
    'settlement': ('Settlement', 'ReleaseSlashed'),
}


def table_metadata(stem):
    groups = [('evaluation_', 'Results RQ2', 'evaluation'), ('extensions_', 'Results RQ4', 'extensions'),
              ('privacy_', 'Results RQ4', 'privacy'), ('fashion_', 'Results RQ4', 'fashion'),
              ('settlement_', 'Results RQ3', 'settlement'), ('reproduction_', 'Results RQ1', 'reproduction'),
              ('extension_pilot_', 'Setup', 'extensions'), ('pilot_', 'Method', 'method'),
              ('method_', 'Discussion/Limitations', 'method'), ('gas_', 'Results RQ3', 'contract'),
              ('published_gas', 'Related Work / Results RQ3', 'contract'), ('contract_', 'Contract', 'contract')]
    for prefix, section, group in groups:
        if stem.startswith(prefix):
            caption = stem.replace('_', ' ').capitalize().replace('False penalty', 'Low-reward penalty')
            if 'paired' in stem:
                caption += '; paired effects and multiplicity-adjusted tests with dependence caveats.'
            elif 'accounting' in stem:
                caption += '; per-client and every-round bounds are distinct.'
            elif 'divergence' in stem or 'outcomes' in stem:
                caption += '; failures remain in the declared cohort.'
            elif 'incidence' in stem:
                caption += '; client-run counts, events and token losses use separate denominators.'
            elif 'ordering' in stem:
                caption += '; strict reversals and changed ties are descriptive.'
            else:
                caption += '; retain the displayed units, cohort labels and uncertainty.'
            return section, caption, group
    raise ValueError('unmapped generated table: '+stem)


def asset_inventory(root=Path('.')):
    rows = []
    actual = {p.stem for p in (root/'figures').glob('*.pdf')}
    if actual != set(FIGURES) or actual != {p.stem for p in (root/'figures').glob('*.png')}:
        raise ValueError('every figure pair needs an explicit placement and caption')
    for name, (section, caption, group, width) in FIGURES.items():
        with Image.open(root/'figures'/f'{name}.png') as img:
            dpi = img.info.get('dpi', (0, 0))
            if min(dpi) < 300:
                raise ValueError('figure resolution below print target: '+name)
        reader = PdfReader(root/'figures'/f'{name}.pdf')
        if len(reader.pages) != 1:
            raise ValueError('figure must be a single PDF page: '+name)
        rows.append(dict(path='figures/'+name, kind='figure', section=section, caption=caption, group=group,
                         width=width, dpi=min(dpi), command='python scripts/make_figures.py'))
    for p in sorted((root/'paper/generated').glob('*.tex')):
        if p.stem == 'numbers':
            continue
        section, caption, group = table_metadata(p.stem)
        rows.append(dict(path=p.relative_to(root).as_posix(), kind='table', section=section,
                         caption=caption, group=group, width='double', command='python scripts/make_tables.py'))
    return rows


def macro_groups(text):
    names = re.findall(r'\\newcommand\{\\([^}]+)\}', text)
    if not names or len(names) != len(set(names)) or any(not name.isalpha() or not name.isascii() for name in names):
        raise ValueError('numerical macros must be present and uniquely named')
    groups = {group:[name for name in names if name.startswith(prefixes)] for group,prefixes in GROUP_PREFIXES.items()}
    if set(names) != {name for group in groups.values() for name in group}:
        raise ValueError('all numerical macros require an asset-guide group')
    return groups


def results_table(summary):
    lines = ['| Rule | MNIST attacker share, % [95% CI] | MNIST honest cost, % [95% CI] | Fashion attacker share, % [95% CI] |',
             '|---|---:|---:|---:|']
    def fmt(observation):
        e = observation['estimate']
        return f"{100*e['mean']:.2f} [{100*e['lower']:.2f}, {100*e['upper']:.2f}]"
    for rule in summary['fashion']['rules']:
        main = summary['extensions'] if rule in ['CGSV', 'RFFL'] else summary['evaluation']
        lines.append('| '+' | '.join([rule, fmt(main['pooled']['attacker_share']['rules'][rule]),
            fmt(main['pooled']['honest_false_penalty_rate']['rules'][rule]),
            fmt(summary['fashion']['pooled']['attacker_share'][rule])])+' |')
    return lines


def main():
    summary = json.loads(Path('results/summary.json').read_text(encoding='utf-8'))
    rows = asset_inventory()
    groups = macro_groups(Path('paper/generated/numbers.tex').read_text(encoding='utf-8'))
    lines = ['# Overleaf asset guide', '', 'Generated by `python scripts/document_paper_assets.py` from `results/summary.json` and the generated asset inventory.', '',
        'Upload `overleaf_assets.zip` into the existing authored project, preserving the `figures/` and `paper/generated/` paths. '
        'The archive contains assets and this guide; it contains no manuscript entry point. Use `paper/references.bib` as the bibliography. '
        'Load `paper/generated/numbers.tex` once. Tables require `booktabs`; figure PDFs contain embedded fonts. '
        'The manifest binds each asset by SHA-256 and identifies the shared summary digest. It is a content snapshot, not an external publication receipt.', '',
        'Regenerate the ZIP with `python scripts/package_overleaf.py`. Regenerate the guide with `python scripts/document_paper_assets.py`. '
        'Both commands run from the repository root. Figure and table commands below consume the already reconstructed summary.', '',
        '## Headline findings', '', *['- '+line for line in findings(summary)], '', *results_table(summary), '',
        'MNIST attacker shares pool attacked conditions; honest cost also includes attack-free runs. Fashion first averages its two conditions within each seed. '
        'These cohorts have different sizes and are not an interchangeable cross-dataset treatment effect. Every interval resamples whole seeds.', '',
        '## Limitations to carry into the manuscript', '', *['- '+line for line in LIMITATIONS], '',
        '## Placement and caption suggestions', '',
        'Intro: use the headline trade-off and link to the measured results. Related Work: use the verified bibliography with explicit abstract-level labels for the two limited-access sources. '
        'Setup: cite the protocol macros and pilot-selection tables. Discussion/Limitations: retain the limitations above. '
        'These are asset placement notes and caption suggestions, not manuscript prose or a LaTeX outline.', '',
        'Use IEEE single-column or double-column placement as listed; do not shrink wide tables into a narrow column. '
        'All tables have conservative double-column placement. Figure canvas targets are single-column 3.5 inches or double-column 7.1 inches before tight cropping. '
        'PNG resolution is at least 300 DPI. The corresponding PDF is preferred for vector text.', '',
        'Figures and tables contain values rendered directly from the summary; they do not invoke text macros. '
        'The companion-macro links enumerate exact command names for prose that refers to each asset. Tables already contain their complete cell values. '
        'Where no aggregate macro represents an individual plotted point, cite the plotted/table value without inventing a new number.', '',
        '| Asset | Section | Caption suggestion | Width | Companion macros | Reproduce |', '|---|---|---|---|---|---|']
    for row in rows:
        if row['kind'] == 'figure':
            links = f"[{Path(row['path']).name} PDF](../{row['path']}.pdf), [PNG](../{row['path']}.png)"
        else:
            links = f"[{Path(row['path']).name}]({Path(row['path']).relative_to('paper').as_posix()})"
        lines.append(f"| {links} | {row['section']} | {row['caption']} | {row['width']} | [{row['group']}](#macros-{row['group']}) | `{row['command']}` |")
    lines += ['', '| Supporting asset | Section | Purpose | Reproduce |', '|---|---|---|---|',
        '| [numbers.tex](generated/numbers.tex) | All numerical sections | Stable numerical commands; percentage outputs are already scaled. | `python scripts/make_macros.py` |',
        '| [references.bib](references.bib) | Related Work | Verified citation records; keep abstract-level scope visible. | Publisher records retained in `results/literature/`; validate with `python scripts/document_bibliography.py`. |',
        '| [Annotated bibliography](../docs/annotated_bibliography.md) | Related Work | Problem, method, data, findings and limits of each opened source. | `python scripts/document_bibliography.py` |',
        '| [Defense notes](../docs/defense_notes.md) | All sections | Explain implementation, parameters, evidence and limitations. | `python scripts/document_defense.py` |',
        '| [Proposal](../docs/proposal.pdf) | Intro / Method | Preserved prospective gap, questions and planned figure caption. | `python scripts/build_proposal.py` |', '',
        '## Exact companion macro names', '',
        'Commands below include the leading backslash. Names are stable; do not redefine a command to change its measured value. '
        'Mean/Low/High are point estimate and interval endpoints; SD is sample standard deviation. Undefined values and infinity retain their explicit meanings.', '']
    for group, names in groups.items():
        lines += [f'<a id="macros-{group}"></a>', f'### {group.capitalize()}', '', ', '.join('`\\'+name+'`' for name in names), '']
    Path('paper/README.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    block = ['## What the experiments show', '', *results_table(summary), '',
        'Mstar is not Pareto-dominant: its lower pooled attacker share comes with higher honest cost. '
        'CGSV/RFFL inherit geometric self-inclusion while offering a different reward trade-off. '
        'Fashion-MNIST removes Mstar\'s edge over those reputation rules; coordinate-noise experiments measure fairness, not meaningful privacy.', '',
        'The [paper asset guide](paper/README.md) gives every figure/table, its section, caption, exact companion macros and one regeneration command. '
        'It includes all headline findings, cohort definitions and limitations. [Evaluation](docs/EVALUATION.md), '
        '[extensions](docs/EXTENSIONS.md), [privacy](docs/PRIVACY.md) and [transfer](docs/FASHION.md) retain the full estimates.', '',
        'The Overleaf upload is [paper/overleaf_assets.zip](paper/overleaf_assets.zip); rebuild it with `python scripts/package_overleaf.py`. '
        'Release 1.0.0 is archived on Zenodo: [10.5281/zenodo.24424418](https://doi.org/10.5281/zenodo.24424418); '
        '[10.5281/zenodo.24424417](https://doi.org/10.5281/zenodo.24424417) always resolves to the latest version. '
        'The manuscript and a public testnet demonstration are separate deliverables and are not claimed complete.', '']
    runtime = summary['evaluation']['runtime']
    block += [f"The main CUDA grid contains {summary['evaluation']['nominal_cells']} runs. Successful training took "
              f"{runtime['training_wall_seconds']/3600:.2f} hours; elapsed execution including interruptions, retries and validation was "
              f"{runtime['elapsed_wall_seconds']/3600:.2f} hours. Per-run CPU/GPU timing evidence follows below. "
              'Asset regeneration uses recorded summaries and does not train models.', '']
    start, end = '<!-- generated final findings start -->', '<!-- generated final findings end -->'
    readme = Path('README.md').read_text(encoding='utf-8')
    rendered = start+'\n'+'\n'.join(block)+'\n'+end
    if start in readme:
        readme = readme[:readme.index(start)]+rendered+readme[readme.index(end)+len(end):]
    else:
        readme = readme.replace('<!-- generated validation start -->', rendered+'\n\n<!-- generated validation start -->', 1)
    Path('README.md').write_text(readme, encoding='utf-8', newline='\n')
    print(json.dumps(dict(figures=sum(row['kind']=='figure' for row in rows), tables=sum(row['kind']=='table' for row in rows),
                          macros=sum(map(len,groups.values())), findings=len(findings(summary)), limitations=len(LIMITATIONS)),sort_keys=True))


if __name__ == '__main__':
    main()
