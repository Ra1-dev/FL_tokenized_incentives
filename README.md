# Tokenized incentives for federated learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24424417.svg)](https://doi.org/10.5281/zenodo.24424417)

This research codebase studies whether cosine-based rewards pay noise free-riders
and whether a leave-one-out reference, reputation and stake penalties improve
the reward distribution. The reproduction target is Sendros et al.,
[*Tokenized Reward Systems for Federated Learning: The Privacy vs. Fairness
Tradeoff*](https://arxiv.org/abs/2611.16065).

The mathematical core, Flower experiment harness and full reproduction
measurements are available. Reward scoring and contract settlement are implemented.
The controlled reward comparisons, noise experiments and dataset transfer are complete.
The authored manuscript and external submission remain in progress.

<!-- generated final findings start -->
## What the experiments show

| Rule | MNIST attacker share, % [95% CI] | MNIST honest cost, % [95% CI] | Fashion attacker share, % [95% CI] |
|---|---:|---:|---:|
| B0 | 20.46 [19.49, 21.59] | 0.00 [0.00, 0.00] | 19.42 [17.50, 21.50] |
| B1 | 18.04 [17.28, 18.93] | 5.76 [4.37, 7.01] | 17.65 [15.58, 19.25] |
| B2 | 12.15 [11.04, 13.15] | 10.91 [10.10, 11.86] | 12.50 [11.88, 13.44] |
| Mstar | 5.16 [4.55, 5.77] | 15.23 [13.98, 16.49] | 8.27 [6.01, 9.75] |
| CGSV | 7.95 [7.08, 8.94] | 4.48 [4.00, 4.95] | 8.17 [6.73, 10.31] |
| RFFL | 7.20 [6.25, 8.26] | 4.55 [4.09, 4.98] | 7.58 [6.12, 9.59] |

Mstar is not Pareto-dominant: its lower pooled attacker share comes with higher honest cost. CGSV/RFFL inherit geometric self-inclusion while offering a different reward trade-off. Fashion-MNIST removes Mstar's edge over those reputation rules; coordinate-noise experiments measure fairness, not meaningful privacy.

The [paper asset guide](paper/README.md) gives every figure/table, its section, caption, exact companion macros and one regeneration command. It includes all headline findings, cohort definitions and limitations. [Evaluation](docs/EVALUATION.md), [extensions](docs/EXTENSIONS.md), [privacy](docs/PRIVACY.md) and [transfer](docs/FASHION.md) retain the full estimates.

The Overleaf upload is [paper/overleaf_assets.zip](paper/overleaf_assets.zip); rebuild it with `python scripts/package_overleaf.py`. Release 1.0.0 is archived on Zenodo: [10.5281/zenodo.24424418](https://doi.org/10.5281/zenodo.24424418); [10.5281/zenodo.24424417](https://doi.org/10.5281/zenodo.24424417) always resolves to the latest version. The manuscript and a public testnet demonstration are separate deliverables and are not claimed complete.

The main CUDA grid contains 90 runs. Successful training took 8.78 hours; elapsed execution including interruptions, retries and validation was 16.06 hours. Per-run CPU/GPU timing evidence follows below. Asset regeneration uses recorded summaries and does not train models.

<!-- generated final findings end -->

<!-- generated validation start -->
## Recorded validation

CPU: 4 completed runs, wall times 618.1–767.6 seconds; 4 within the recorded runtime bound.
GPU: 7 completed runs, wall times 241.3–640.1 seconds; 6 within the recorded runtime bound.
Full CPU repeat: byte-identical participation logs, round logs and integer ledger.

See [runtime and repeatability evidence](docs/VALIDATION.md) for every run, resource setting and hash. Regenerate with `python scripts/summarize_validation.py` and `python scripts/document_validation.py`.
<!-- generated validation end -->

## Parameter sensitivity

An exploratory replay of 120 settings of the proposed rule (reputation weight, low-quality
threshold and patience) on the recorded evaluation grid shows that no setting lowers the
honest low-reward rate below 14.49%, the value of the leave-one-out reference alone.
Reputation and penalties only move along a narrow trade-off; CGSV and RFFL lie outside
the region the rule reaches. The replay needs no training, and the frozen setting must
reproduce the published allocations exactly before any other setting is scored. See the
[sensitivity analysis](docs/SENSITIVITY.md).

```bash
python scripts/sensitivity_sweep.py
python scripts/sensitivity_assets.py
python scripts/make_macros.py
```

## Adaptive attackers

The fixed attacks above do not adapt to the reward rule. `configs/adaptive_study.json`
declares two adaptive attackers on the unchanged protocol: full-knowledge ALIE
(Baruch et al.), which reports the honest mean minus z standard deviations, and a
free-rider that reports the previous round's public aggregate. Both run at 10%, 20%
and 30% attackers over five seeds, with all six reward rules scoring the same updates.
The driver runs cells serially, skips completed cells and runs seed by seed, so an
interruption leaves complete seeds.

All 30 runs completed. Both adaptive attackers defeat the proposed rule: pooled over the
three fractions, attackers receive 52.62% of the pool under ALIE and 47.90% under the
mimic, against 20.67% under an equal split, while 43.81% and 45.19% of honest
participations receive less than half of the equal-split reward. Cosine rewards pay the
mimic close to its headcount (22.90%); CGSV and RFFL lie in between. A leave-one-out
median rewards updates placed at the honest consensus above genuine heterogeneous
training. Neither attack noticeably harms accuracy. See [adaptive attackers](docs/ADAPTIVE.md).

```bash
python scripts/run_adaptive_study.py --smoke   # two CPU rounds per attack, written to out/
python scripts/run_adaptive_study.py           # 30 runs; resumes after interruption
python scripts/adaptive_assets.py              # summary, figure, tables and docs/ADAPTIVE.md
python scripts/make_macros.py
```

## Install on Windows (PowerShell)

Clone with LF line endings so recorded artifact hashes are preserved:

```powershell
git -c core.autocrlf=false clone https://github.com/Ra1-dev/FL_tokenized_incentives.git
cd FL_tokenized_incentives
```

The repository also declares LF text checkouts in `.gitattributes`; PDFs and
PNGs remain binary. If an older checkout already converted text to CRLF, save
any local edits before making a fresh clone with the command above.

```powershell
git config core.autocrlf false
py -3.12 -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt -e .
$env:FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION = "1"
python -m pytest -q
```

For CUDA, install the matching PyTorch and torchvision wheels from the
[official PyTorch selector](https://pytorch.org/get-started/locally/) before
installing the requirements. The local development environment uses the CUDA
build; the CPU build is supported. Do not silently change versions when comparing
reproduction runs.

## Install on Linux (CPU)

```bash
git config core.autocrlf false
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -e .
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
python -m pytest -q
```

Tests use synthetic inputs and never download data. Prepare MNIST once, then run:

```bash
python scripts/download_data.py --dataset mnist
python scripts/run_experiment.py --config configs/smoke.yaml --seed 100 --device cpu
```

For a full reproduction run, use `--config configs/sendros_mix.yaml --seed 42`.
Use `--device cuda` to request GPU execution explicitly. The default GPU resource
fraction is recorded in the resolved run config; it reserves the device for one
client at a time to fit the development GPU. CPU runs use two workers by default;
`--workers` changes this without changing the sampled clients or their RNG streams.
`--gpu-fraction` changes the per-client reservation for a recorded concurrency
benchmark; the reservation is a scheduling limit, not a GPU-memory limit.
The training runtime is Flower's `ServerApp`/`ClientApp` Message API on Ray.

The runner records explicit Ray object-store and task-memory reservations.
`--object-store-mb` and `--task-memory-mb` change those runtime budgets without
changing training parameters. Task-memory reservation is a scheduling value,
not a physical-memory guarantee. Windows host metadata uses the version API
and processor environment so a stalled WMI service cannot prevent a run.

Run folders contain per-client statistics and all rule scores, per-round global
evaluation, an integer ledger, metrics, file hashes and provenance. Use `--output`
for an independent repeat; otherwise a rerun replaces the same named result.
Full update arrays are cached under the ignored `results/cache/` directory.

`python scripts/reproduce_sendros.py` runs missing full seeds and regenerates
the source comparison. `--no-run` analyzes only existing complete runs. See
[the reproduction protocol](docs/REPRODUCTION.md) for exact choices, deviations,
data hashes and known limits. Smoke data reductions never count as full results.

Before changing Flower's runtime dependency setting, stop any existing local
SuperLink in the terminal where it was started (Ctrl+C), then restart it with the
environment variable set. The experiment runner will use Flower's simulation
runtime and declare the client CPU/GPU resources in its resolved configuration.

## Reproduce figures and tables

The committed `results/summary.json` is the numerical input to every asset.
After either installation above, run from the repository root:

```bash
python scripts/make_macros.py
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/document_bibliography.py
python scripts/document_defense.py
python scripts/document_paper_assets.py
python scripts/package_overleaf.py
python scripts/package_overleaf.py --check
```

The [asset guide](paper/README.md) maps every figure and table to its individual
regeneration command, caption, units, column width and numerical macro names.
The archive preserves the paths used in the authored Overleaf project.
This asset-only workflow needs no GPU, dataset download or network access.

To reconstruct the underlying analyses or launch fresh experiments, follow
[the experiment commands](docs/EXPERIMENTS.md). They preserve the separately
published producers for the main grid, reputation rules, honest noise and
Fashion-MNIST. Do not launch multiple training drivers. All reward rules within
a run score the same updates under fixed FedAvg; comparing their rewards does
not compare separate trained models.

## Structure and provenance

- `src/fltoken/models.py`: unpadded LeNet architecture.
- `src/fltoken/data.py`: classwise Dirichlet partitions and seeded sampling.
- `src/fltoken/scoring.py`: equal split, cosine quality, Multi-Krum acceptance.
- `src/fltoken/ours.py`: client-excluding quality, stateful reputation and penalties.
- `src/fltoken/attacks.py`: zero, Gaussian, sign and label transforms, stale timing.
- `tests/`: numerical and boundary contracts.

The learning and scoring implementations are written from the mathematical
definitions. The contract uses the vendored token library documented below.
Third-party packages remain under their respective licenses. The project code
is distributed under [LICENSE](LICENSE).

PDF reading tools used during source verification were installed with
`python -m pip install pypdf==7.1.1 pymupdf==2.1`. They are optional research
tools and are not required to run the core tests.

The standalone research proposal is generated from the recorded protocol,
proposed method and verified literature notes:

```bash
python -m pip install -r requirements-artifacts.txt
python scripts/build_proposal.py
```

For visual PDF checks on Windows, `python scripts/install_poppler.py` downloads
the pinned [Poppler package](https://github.com/oschwartz10612/poppler-windows/releases/tag/v26.09.0-0),
verifies its SHA-256 digest and extracts it under `out/tools/`. Render with
`out/tools/poppler/poppler-26.09.0/Library/bin/pdftoppm.exe -png docs/proposal.pdf out/proposal-page`.
On Debian or Ubuntu, install `poppler-utils`, then use
`pdftoppm -png docs/proposal.pdf out/proposal-page`. Create `out/` first if needed.

Validate the recorded smoke and repeatability evidence without training again:

```bash
python scripts/smoke_attacks.py --no-run --output results/validation/attack-smoke
python scripts/summarize_validation.py
python scripts/document_validation.py
```

Omit `--no-run` from the smoke command to run every attack through Flower again.
`python scripts/stale_sensitivity.py` runs the separately labelled own-report
replay sensitivity; `--no-run` analyzes its existing complete outputs.
Persistent stale identities have no honest warm-up in that variant: their first
missing-history report is zero, and subsequent eligible reports copy zero by
recursion. That removes their direct reward and also changes later training
trajectories; the reduction is reported separately from the main definition.

Derived reward metrics use exact integer totals and rational arithmetic. See
[metric definitions and the declared migration](docs/METRICS.md) for Windows and
Linux reconstruction evidence. Verify the stored logs without training:

```bash
python scripts/check_derived_metrics.py --output results/validation/derived-linux.json
```

Use `derived-windows.json` on Windows. The command fails on any changed recorded
metric or derived artifact; it does not apply a numerical tolerance.

## Prepare the Overleaf support package

```bash
python scripts/make_macros.py
python scripts/make_figures.py
python scripts/make_tables.py
python scripts/document_bibliography.py
```

The paper is written in Overleaf. This repository generates its numerical macros
in `paper/generated/numbers.tex`, figures in `figures/`, tables in
`paper/generated/*.tex` and verified references in `paper/references.bib`.
Supporting research notes live in `docs/`. Measurements feed these artifacts
through `results/summary.json`.

After committing the generated assets, `python scripts/package_assets.py`
builds `out/support-assets.zip` directly from that commit's Git objects. Its
manifest records every file hash, the source commit, and which empirical result
sections are still absent. See [the asset-package notes](docs/ASSETS.md) for
Overleaf paths, deterministic archives, and software citation metadata.
`CITATION.cff` and `.zenodo.json` describe software version 1.0.0; the DOI and
external archival publication remain pending. The frozen training module retains
its historical version constant for source-digest compatibility; installed
distribution metadata identifies this release.

Result tables require complete reproduction measurements. Architecture and
multi-panel reproduction figures use a double-column width; the attacker-share
chart uses a single-column width. Error bars on that chart show sample standard
deviations across seeds, and points show the individual seeds.

To build a local copy, export the Overleaf source into `paper/` with an entry
point named `main.tex`, then run:

```bash
python scripts/install_tectonic.py
python scripts/build_paper.py
```

The optional installer downloads the pinned executable from the official
[Tectonic release](https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0)
and verifies the release asset digest when available. It writes only under
`out/tools/`. The first paper build downloads its TeX dependencies into Tectonic's
cache; subsequent builds can use `python scripts/build_paper.py --offline`.
The build checks for unresolved references and LaTeX errors. A manuscript export
has not yet been supplied, so the submission's page count and build status remain
pending.

## Contract settlement and gas

<!-- generated contract start -->
Solidity: 12 tests and 3,072 fuzz cases pass; static analysis reports 0 high and 0 medium findings.
Recorded chain replay matches 180 round-rule cases exactly; state stress checks cover 96 rounds and 228 slash participations.
The executed full-participation gas boundary is 142 successful clients and 143 clients exhausting the settlement transaction limit under the recorded state history.
Selected-method grid settlement matches 4280 round-rule states in 196 jobs; [coverage and economic balances](docs/GRID_SETTLEMENT.md) retain the complete and sampled rules separately.
<!-- generated contract end -->

`python scripts/install_contract_tools.py` installs checksum-pinned Foundry
v1.10.0 and Solidity 0.8.30 inside `out/tools/foundry/` on Windows or Linux
x86-64. The installer uses the official Foundry release and Ethereum solc-bin
mirror. Add that directory to PATH before contract commands. Downloading tools
is a setup step; contract tests must use the installed compiler offline.

The contract reuses the unmodified ERC-20 implementation from
[OpenZeppelin Contracts v5.7.0](https://github.com/OpenZeppelin/openzeppelin-contracts/tree/v5.7.0).
Its minimal dependency files, MIT license and SHA-256 provenance are retained
under `third_party/openzeppelin-contracts/`. Our reward-pool logic is separate.
For isolated Python chain tooling, create `.venv-chain` and install
`requirements-contracts.txt` there; this keeps its Web3 and Slither dependencies
separate from the training environment.

Windows setup and checks:

```powershell
python scripts/install_contract_tools.py
py -3.12 -m venv .venv-chain
.venv-chain/Scripts/python.exe -m pip install -r requirements-contracts.txt
.venv-chain/Scripts/python.exe scripts/check_contracts.py
```

Linux setup and checks:

```bash
python scripts/install_contract_tools.py
python3.12 -m venv .venv-chain
.venv-chain/bin/python -m pip install -r requirements-contracts.txt
.venv-chain/bin/python scripts/check_contracts.py
```

The checker runs offline Solidity tests and static analysis, retaining classified
findings in `results/contracts/checks.json`. It sets the tool PATH internally.
Use the chain environment's Python for `scripts/replay_onchain.py`,
`scripts/check_chain_states.py` and `scripts/measure_gas.py`. Their exact commands,
state order, trust assumptions and cap interpretation are in
[the settlement and gas notes](docs/CONTRACTS.md). The programs start and close
ephemeral loopback-only chains; no wallet credentials or public RPC are needed.

Once grid runs are available, launch the resumable replay driver from the
training environment, with the chain environment's interpreter explicitly:

```powershell
python scripts/replay_grid.py --chain-python .venv-chain/Scripts/python.exe
```

On Linux use `.venv-chain/bin/python` for `--chain-python`. Add `--available`
to process completed cells while training continues, or `--no-run` to validate
saved replay evidence. Partial coverage remains explicitly incomplete in
`results/grid_onchain/coverage.json`. Every cosine and selected-method ledger
is required; equal-split and Krum are sampled on the first declared seed across
all attacks at the largest fraction, plus the attack-free and mixture runs.

In the original training/artifact environment, regenerate the shared summary
and assets from the recorded chain evidence:

```bash
python scripts/summarize_contracts.py
python scripts/document_contracts.py
python scripts/make_macros.py
python scripts/make_figures.py
```

## License and citation

The project is MIT licensed; see [LICENSE](LICENSE). The unmodified OpenZeppelin
dependencies retain their license and provenance under `third_party/`.
No unlicensed implementation is incorporated. Cite the author and exact source
commit using [CITATION.cff](CITATION.cff). The archived release is
[10.5281/zenodo.24424418](https://doi.org/10.5281/zenodo.24424418) (version 1.0.0);
[10.5281/zenodo.24424417](https://doi.org/10.5281/zenodo.24424417) cites all versions.

## Public testnet demonstration

No Sepolia deployment is claimed. The deployment script
`contracts/script/Deploy.s.sol` accepts only Sepolia or a local development chain,
and uses the measured pool policy. With the pinned tools already installed, its
offline simulation is:

```powershell
$env:PATH = (Resolve-Path out/tools/foundry).Path + [IO.Path]::PathSeparator + $env:PATH
forge script --root contracts contracts/script/Deploy.s.sol:Deploy --offline
```

On Linux, prefix the command with `PATH="$PWD/out/tools/foundry:$PATH"`.
Signing, test ETH, RPC access and explorer verification are separate actions.
A completed demonstration must record the actual chain ID, block, pool/token
addresses, source commit and verified explorer links here; none is fabricated.
