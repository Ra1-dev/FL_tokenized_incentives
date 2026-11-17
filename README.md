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