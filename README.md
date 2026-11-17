# Closing the Noise Loophole in On-Chain FL Rewards

Reward contract for federated learning that pays clients for useful updates and
penalises free-riders and poisoned updates. We reproduce the cosine self-inclusion
bias reported by Sendros et al. (2026) and test the mitigations they proposed:
a leave-one-out reference, reputation over rounds and stake slashing.

Status: in progress (Nov 2026).

## Structure
- `src/fl` – Flower simulation (MNIST, LeNet-5)
- `src/attacks` – free-rider and poisoning clients
- `src/scoring` – reward rules: equal split, cosine (Sendros et al.), Multi-Krum, ours
- `src/chain` – web3.py bridge to the contract
- `contracts` – Solidity reward contract (Foundry)
- `scripts` – experiment runs and figure generation

## Setup
Python 3.12. Install the CUDA build of PyTorch from pytorch.org (used CUDA 13.4), then `pip install -r requirements.txt`.
Run instructions will be added as components land.

### Running Flower simulations locally
- Disable Flower's per-run dependency install so runs use this venv (with CUDA PyTorch):
  `export FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION=1`
  (stop any running local SuperLink first so the variable takes effect).
- Give clients a share of the GPU:
  `flwr federation simulation-config --client-resources-num-gpus 0.25`
- Tested on Windows 11, Python 3.12, GTX 1650 (4 GB), Flower 1.43.0.