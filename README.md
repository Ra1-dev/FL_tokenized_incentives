# Tokenized incentives for federated learning

This research codebase studies whether cosine-based rewards pay noise free-riders
and whether a leave-one-out reference, reputation and stake penalties improve
the reward distribution. The reproduction target is Sendros et al.,
[*Tokenized Reward Systems for Federated Learning: The Privacy vs. Fairness
Tradeoff*](https://arxiv.org/abs/2611.16065).

The mathematical core, Flower experiment harness and full reproduction
measurements are available. Reward scoring and contract settlement are implemented;
pilot selection, the main comparison and the manuscript remain in progress.

<!-- generated validation start -->
## Recorded validation

CPU: 4 completed runs, wall times 618.1–767.6 seconds; 4 within the recorded runtime bound.
GPU: 2 completed runs, wall times 417.4–640.1 seconds; 1 within the recorded runtime bound.
Full CPU repeat: byte-identical participation logs, round logs and integer ledger.

See [runtime and repeatability evidence](docs/VALIDATION.md) for every run, resource setting and hash. Regenerate with `python scripts/summarize_validation.py` and `python scripts/document_validation.py`.
<!-- generated validation end -->

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

## Pilot selection for the proposed rule

The leave-one-out scorer is implemented with independently switchable reference,
reputation and penalties. Its [method specification](docs/METHOD.md) gives the
formulas, development defaults, predeclared candidate space and selection
objective. Empirical selection and evaluation remain pending.

```bash
python scripts/run_pilots.py
python scripts/select_method.py --freeze
```

The pilot plan uses disjoint seeds and the full training protocol. Every candidate
scores the same cached update matrices. These commands require the plan and
scientific implementation to be committed, and stop on a provenance mismatch.
The selected configuration and pilot evidence must be committed and pushed before
evaluation scoring. Later, `python scripts/select_method.py --recorded-only` reconstructs
candidate comparisons from the compact recorded quality tables; recomputing the
qualities themselves requires the ignored update cache.

## Evaluation and uncertainty

The [evaluation protocol](docs/EVALUATION.md) fixes coverage, component stages,
undefined-value handling, seed bootstrap intervals and paired comparisons before
evaluation scoring. The runner checks that the selected method and complete
pilot evidence are published on `origin/main`.

After the pilot freeze is published:

```bash
python scripts/run_grid.py
python scripts/run_grid.py --no-run
python scripts/summarize_grid.py
python scripts/reproduce_sendros.py --grid --no-run
```

The attack-free condition and fixed mixture each occur once per seed;
single attacks vary over the declared fractions. Missing runs or altered
provenance stop analysis.
Keep `results/grid/execution.json` with the run folders. Its elapsed time covers
the first training attempt through completion, including interruptions and
retries; the successful-run time sum is reported separately. Validation retains
the recorded endpoints.
The reproduction refresh uses the source's seed cohort for the direct comparison
and reports all evaluation seeds separately. Later regeneration preserves this
grid basis. Use `--legacy` only to explicitly restore the standalone reproduction.
Every completed grid run retains the full update matrices in the ignored
`results/cache/` directory. Keep that directory when moving runs: later reward
rules score these same matrices offline, with hashes and client order verified
against the training log. Do not retrain a different trajectory for an additional
shadow rule.
`python scripts/document_evaluation.py` regenerates the declared design notes;
it does not create empirical results.

After summarization, regenerate comparison assets with
`python scripts/make_macros.py`, `python scripts/make_tables.py` and
`python scripts/make_figures.py`. They include seed intervals, paired contrasts,
component ablations and the common accuracy trajectory. Tables state the training
device. Main accuracy and ablation panels use double-column width; condition
charts and the long-horizon accuracy curve use single-column width. The method
notes automatically include the frozen parameters and all pilot candidate costs
once their hash-bound evidence exists.

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
