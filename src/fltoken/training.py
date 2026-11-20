"""Seeded local SGD and centralized evaluation, independent of reward rules."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fltoken.attacks import label_flip
from fltoken.data import seed_for
from fltoken.models import LeNet5


def configure_torch(seed: int, threads: int = 1) -> None:
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def flatten(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().astype(np.float64).ravel() for p in model.parameters()])


def load_flat(model: nn.Module, vector: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            end = offset + parameter.numel()
            parameter.copy_(torch.as_tensor(vector[offset:end].copy(), dtype=parameter.dtype, device=parameter.device).view_as(parameter))
            offset = end
    if offset != len(vector):
        raise ValueError("parameter vector length does not match model")


def local_update(initial: np.ndarray, images: torch.Tensor, labels: torch.Tensor, config: dict,
                 client_id: int, source_round: int, poisoned: bool = False) -> np.ndarray:
    local_seed = seed_for(config["seed"], source_round, client_id, 2) % (2**63 - 1)
    configure_torch(local_seed, config["torch_threads"])
    device = config["device"]
    model = LeNet5().to(device)
    load_flat(model, initial)
    before = flatten(model)
    x = images.to(device)
    y = (label_flip(labels) if poisoned else labels).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=0)
    generator = torch.Generator().manual_seed(local_seed)
    model.train()
    for _ in range(config["local_epochs"]):
        order = torch.randperm(len(y), generator=generator).to(device)
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    return flatten(model) - before


def evaluate(vector: np.ndarray, images: torch.Tensor, labels: torch.Tensor, device: str,
             batch_size: int = 512) -> tuple[float, float]:
    model = LeNet5().to(device)
    load_flat(model, vector)
    model.eval()
    correct, loss = 0, 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            x = images[start:start + batch_size].to(device)
            y = labels[start:start + batch_size].to(device)
            logits = model(x)
            loss += float(nn.functional.cross_entropy(logits, y, reduction="sum").item())
            correct += int((logits.argmax(1) == y).sum().item())
    return correct / len(labels), loss / len(labels)
