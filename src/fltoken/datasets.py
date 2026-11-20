"""Offline dataset loading; downloading is an explicit preparation command."""

from functools import lru_cache
from pathlib import Path

import torch
from torchvision.datasets import FashionMNIST, MNIST

from fltoken.io import sha256, write_json

DATASETS = {"mnist": (MNIST, .1307, .3081), "fashion_mnist": (FashionMNIST, .2860, .3530)}


def prepare_dataset(name: str, root: str | Path) -> dict:
    cls, mean, std = DATASETS[name]
    for train in (True, False):
        cls(root=str(root), train=train, download=True)
    raw = Path(root) / cls.__name__ / "raw"
    checksums = {p.name: sha256(p) for p in sorted(raw.iterdir()) if p.is_file()}
    record = {"dataset": name, "normalization": {"mean": mean, "std": std}, "sha256": checksums}
    write_json(Path(root) / cls.__name__ / "checksums.json", record)
    return record


@lru_cache(maxsize=4)
def load_tensors(name: str, root: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    cls, mean, std = DATASETS[name]
    data = cls(root=root, train=train, download=False)
    images = data.data.unsqueeze(1).to(torch.float32).div_(255).sub_(mean).div_(std)
    return images, data.targets.to(torch.int64)
