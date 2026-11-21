# Reconstructing and running experiments

The recorded results regenerate without training. Fresh training must use each published producer and frozen configuration below. The main README gives installation and asset-only commands.

## Pilot selection for the proposed rule

The leave-one-out scorer is implemented with independently switchable reference,
reputation and penalties. Its [method specification](../docs/METHOD.md) gives the
formulas, development defaults, predeclared candidate space and selection
objective.

<!-- generated method start -->
Pilot selection is complete: candidate 13 from 18 candidates over 85 cells. The frozen configuration and every candidate cost are recorded in the [method specification](METHOD.md). The complete evaluation summary covers 90 runs.
<!-- generated method end -->

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
Use the [pinned calibration checkout](#pinned-calibration-checkout) to reconstruct
the existing selection. The current checkout can render its recorded method note:
`python scripts/document_method.py` verifies the original producer digests in Git
history and leaves the selected parameters and recorded costs unchanged.

The current library repairs underflow and cancellation in reference geometry and
preserves relative payout weights when positive quality times reputation underflows.
Frozen experiments retain their original implementation and recorded outcomes.
`python scripts/compare_method.py --synthetic` measures the correction on declared
arithmetic boundary traces. With both complete pilot and evaluation corpora and
their retained cache, `python scripts/compare_method.py` compares states, integer
payouts and vesting wealth at the unchanged selected parameters. Synthetic checks
do not establish empirical equivalence, and neither command selects parameters.
`python scripts/audit_method_weights.py` checks the complete grid's recorded
qualities and reconstructs its original allocations without loading update
matrices. It counts subnormal and erased positive weights separately from
legitimate empty rounds. Its explicit `--available --output PATH` mode produces
only a partial diagnostic while the grid is still running.
`python scripts/document_numerical_comparison.py` verifies the complete audit
and same-matrix comparison before exporting the separately labelled
[numerical variant](../docs/NUMERICAL_COMPARISON.md). Exact reconstruction also
requires the original numerical runtime: retain its NumPy build and BLAS thread
settings. Changing reduction parallelism can change floating-point results even
when the scorer source and input matrices are identical.

## Evaluation and uncertainty

The [evaluation protocol](../docs/EVALUATION.md) fixes coverage, component stages,
undefined-value handling, seed bootstrap intervals and paired comparisons before
evaluation scoring. The runner checks that the selected method and complete
pilot evidence are published on `origin/main`.

For the existing freeze, run training and cache validation in the
[pinned calibration checkout](#pinned-calibration-checkout):

```bash
python scripts/run_grid.py
python scripts/run_grid.py --no-run
```

After complete result artifacts and the original execution clock are available,
the current checkout can validate and summarize their historical implementation:

```bash
python scripts/summarize_grid.py
python scripts/reproduce_sendros.py --grid --no-run
```

The attack-free condition and fixed mixture each occur once per seed;
single attacks vary over the declared fractions. Missing runs or altered
provenance stop analysis.
Keep a long-running comparison in its own checkout at the published freeze,
with its own Git object store and unchanged publication reference. Run from
that checkout's root and put its `src` directory first on `PYTHONPATH` when
sharing an existing Python environment. Checks of frozen artifacts and historical
sources use the repository containing the check code, so another checkout's working directory
or inherited Git variables cannot redirect them. A Git process failure reports
its exit status and diagnostic; it does not establish that an artifact is
missing. Keep failed attempts separate and restart an incomplete cell from its
first round. On memory-constrained machines, serialize training, verification
and chain replay. Keep the training checkout pinned until its comparison finishes.
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

## Offline reputation extensions

The [CGSV and RFFL adaptations](../docs/EXTENSIONS.md) retain their inclusive
reputation references and document the changes needed for token shares and
partial participation.

### Pinned calibration checkout

Reconstruct the original method selection or repeat its original training grid
in this published snapshot, which retains the frozen scientific sources and
the corrected source validator:

```bash
git clone --no-checkout https://github.com/Ra1-dev/FL_tokenized_incentives.git out/calibrated-corpus
git -C out/calibrated-corpus config core.autocrlf false
git -C out/calibrated-corpus checkout --detach b4ded6b8e711ddccea71988549d84958e3464822
git -C out/calibrated-corpus update-ref refs/remotes/origin/main HEAD
cd out/calibrated-corpus
```

Activate the existing environment and set this checkout's own `PYTHONPATH` as
below. Copy complete input run folders and their ignored update caches before
reconstruction. Keep this local publication reference pinned while executing;
fetching the latest branch can intentionally trip the source checks. This is
the original method/grid route. The revised noise and transfer experiments
require the separate execution declarations described below.

### Published reconstruction and execution

Use the current published checkout for offline extension scoring. Keep its own
`src` directory first on `PYTHONPATH`, and retain the complete input run folders
and ignored update matrices under `results/cache/`.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location).Path "src")
```

Or in Bash:

```bash
export PYTHONPATH="$PWD/src"
```

The runtime gate checks the original calibration in published Git history and
requires every scientific geometry, state, allocation, metric and cache source
to remain unchanged. Published validator and orchestration changes are allowed
only with a complete, read-only reconstruction of all original pilot candidate
measurements and selected parameters before evaluation. The original selection
files are never rewritten by this reconstruction. Current validation hashes are
recorded separately in the evaluation manifest.

The original frozen Mstar outputs remain recorded baselines. Each training
commit must bind the original scorer digest. Offline extension execution never
uses the current Mstar implementation to replace those results. Initial
calibration still requires exact current-source publication; do not recreate a
freeze to bypass a mismatch.

The earlier privacy calibration retains its original plan and clipping evidence.
The mixture-noise and Fashion-MNIST comparison uses the separate published
`configs/transfer_study.json` declaration and `scripts/run_transfer_study.py`.
The old attack-free privacy driver and old transfer condition remain separate
historical protocols. Running the offline extension commands below does not
start training.

<!-- generated extension start -->
Extension calibration covers 85 pilot cells; CGSV beta is 2.0. RFFL retains its declared literature setting. The complete extension summary covers 90 runs.
<!-- generated extension end -->

After the complete pilot corpus is available, run
`python scripts/select_extensions.py --freeze`, reconstruct it with
`python scripts/select_extensions.py --recorded-only`, and regenerate
`python scripts/document_extensions.py`. Publish the frozen configuration and
complete selection evidence before running `python scripts/rescore_extensions.py`.
It verifies each common cached matrix and saves geometry, reputation histories,
integer payouts and bias diagnostics. `--recorded-only` reconstructs those
outputs without raw matrices; `--available` explicitly permits partial grid
coverage while training continues. Neither command retrains the model.
Once coverage is complete, `python scripts/summarize_extensions.py` verifies
the saved states and ledgers and adds seed-level reward and bias intervals to
the shared summary. The standard figure, table and macro commands then generate
the extension assets. Undefined histories remain visible in the estimates.

## Honest-update noise comparison

The [privacy experiment and accounting notes](../docs/PRIVACY.md) declare a fixed
pilot-calibrated clipping bound and client-side Gaussian perturbations.

<!-- generated privacy start -->
The fixed clipping bound is 1.1639285800003483, calibrated from 1000 participations across 5 attack-free pilot seeds. The privacy summary covers 12 attempted runs: 12 complete and 0 numerical failures. This includes 3 reused zero-noise controls and 9 new runs. The separate original multiplier study retains 9 numerical failures. The coordinate-noise comparison has large epsilon bounds and does not establish a strong privacy guarantee.
<!-- generated privacy end -->

Every multiplier trains its own full
Flower trajectory; within a run, all reward rules score the same released updates.
The public-seeded simulation reports ideal Gaussian RDP bounds and explicitly
does not provide a deployment privacy guarantee.

After the complete pilot corpus and frozen reward method are published:

```bash
python scripts/privacy_protocol.py --freeze
```

The calibration and its evidence are already published. The current mixture
study contains twelve runs: three seeds and four noise multipliers, with only
honest clients perturbed. It then runs six Fashion-MNIST cells: the fixed mixture
and 20% label flipping over the same three seeds. All eighteen runs use CUDA,
one worker and the same six shadow rules. The zero-noise control is clipped.

Use an independent checkout of the recorded producer commit with its own Git
objects and `src` directory first on `PYTHONPATH`. Prepare both datasets once,
then start exactly one driver:

```bash
python scripts/download_data.py --dataset mnist
python scripts/download_data.py --dataset fashion_mnist
python scripts/run_transfer_study.py
python scripts/run_transfer_study.py --no-run
```

The driver verifies input checksums, published execution sources and all three
original calibrations. It loads the original Mstar source from published Git
history after checking its frozen digest, recording that effective producer
separately from current orchestration sources. CGSV/RFFL use their unchanged
pilot-selected kernels. All rules receive the same released updates.
Keep every update cache and the original execution clock. Numerical failures
retain their completed prefix and count as terminal outcomes; they cannot
supply full-horizon metrics. Other failures stop the driver. Archive incomplete
cell folders and their partial cache before an explicit resume. A retained
driver lock requires checking that its process has exited before removing it.
No test suite, WSL workload or offline matrix rescoring should run beside GPU
training. Analyze all declared outcomes after the driver finishes.

`configs/privacy_plan.json` and `scripts/run_privacy.py` preserve the earlier
attack-free protocol; they are separate from this mixture study.

The subsequent coordinate-noise comparison is declared in
`configs/coordinate_noise_study.json`. After the eighteen-cell driver finishes,
use the same prepared datasets in a separate checkout of the published commit:

```bash
python scripts/run_coordinate_study.py
python scripts/run_coordinate_study.py --no-run
```

This serial driver runs seeds 42--44 with honest-client per-coordinate standard
deviations 0.001, 0.003 and 0.01. It retains the original clipping bound and uses
the implied multiplier `z = tau/C`, without dimension scaling. B0/B1/B2 and the
original Mstar run in shadow mode; CGSV/RFFL use the same caches offline. The
three completed zero-noise cells from the first study supply the controls and
are not retrained. Replacement sensitivity remains `2C`. The resulting epsilon
values are large: the comparison measures rewards for noisy honest clients and
does not establish a strong privacy guarantee. The original multiplier-based
cells remain recorded divergence evidence. All failure, provenance and resource
rules above also apply to this nine-cell comparison.

## Fashion-MNIST transfer

Fashion-MNIST transfer uses the published MNIST reward parameters without
retuning. Its six runs are the second part of `run_transfer_study.py` above.
The older `configs/fashion_plan.json` remains a separate 30% poisoning protocol.

The same activated Python environment and Flower settings apply on Windows
and Linux. Training requires the published method freeze. The runner verifies the downloaded input checksums before
training and preserves every common update matrix. The recorded summary can
be rebuilt without downloading data or retaining those matrices. See
[the transfer protocol](../docs/FASHION.md) for conditions, device, seed uncertainty
and limitations. Regenerate the shared figures, tables and macros after the
complete summary exists.

The retained comparisons can be reconstructed on CPU without datasets or model
updates. Start in a checkout of the current published analysis with its own Git
objects and `src` first on `PYTHONPATH`:

```bash
python scripts/rescore_extensions.py --recorded-only
python scripts/summarize_extensions.py
python scripts/run_transfer_study.py --no-run
python scripts/run_coordinate_study.py --no-run
python scripts/analyze_studies.py --recorded-only
python scripts/document_privacy.py
python scripts/document_fashion.py
python scripts/document_extensions.py
python scripts/make_macros.py
python scripts/make_tables.py
python scripts/make_figures.py
```

Training reproduction uses the producer commits recorded in each manifest and
run metadata, preserving the frozen parameters and scientific source hashes.
The original transfer producer is `b51a601d871c534e928fbcbeeb446d0c242e4778`;
the coordinate-noise producer is `4365caad3526c44c4b75515581d474b3f79d0043`.
New training belongs in separate output directories. The analysis preserves
the recorded divergence outcomes and compares Fashion-MNIST with the same
MNIST seeds and attack conditions; it never replaces an earlier run.

