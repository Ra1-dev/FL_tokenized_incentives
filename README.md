# Tokenized incentives for federated learning

This research codebase studies whether cosine-based rewards pay noise free-riders
and whether a leave-one-out reference, reputation and stake penalties improve
the reward distribution. The reproduction target is Sendros et al.,
[*Tokenized Reward Systems for Federated Learning: The Privacy vs. Fairness
Tradeoff*](https://arxiv.org/abs/2611.16065).

The mathematical core is implemented. The Flower experiment harness, contract,
measurements and paper are being developed. No completed research result is
claimed yet.

## Install on Windows (PowerShell)

```powershell
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
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1
python -m pytest -q
```

Tests use synthetic inputs and never download data. Dataset downloading will be
an explicit preparation step, separate from experiment execution.

Before changing Flower's runtime dependency setting, stop any existing local
SuperLink in the terminal where it was started (Ctrl+C), then restart it with the
environment variable set. The experiment runner will use Flower's simulation
runtime and declare the client CPU/GPU resources in its resolved configuration.

## Structure and provenance

- `src/fltoken/models.py`: unpadded LeNet architecture.
- `src/fltoken/data.py`: classwise Dirichlet partitions and seeded sampling.
- `src/fltoken/scoring.py`: equal split, cosine quality, Multi-Krum acceptance.
- `src/fltoken/attacks.py`: zero, Gaussian, sign and label transforms, stale timing.
- `tests/`: numerical and boundary contracts.

The implementations are written from the mathematical definitions. No external
repository code is copied. Third-party packages remain under their respective
licenses. The project code is distributed under [LICENSE](LICENSE).

PDF reading tools used during source verification were installed with
`python -m pip install pypdf==7.1.1 pymupdf==2.1`. They are optional research
tools and are not required to run the core tests.
