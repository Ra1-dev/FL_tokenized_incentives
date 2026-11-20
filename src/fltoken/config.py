"""Resolved experiment parameters with explicit smoke-only reductions."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULTS = dict(
    dataset="mnist", data_dir="data", num_clients=30, clients_per_round=10,
    alpha=.5, rounds=20, local_epochs=5, batch_size=32, lr=.01, momentum=.9,
    attack="sendros_mix", fraction=1 / 6, noise_sigma=.01, stale_lag=5, stale_mode="train_snapshot",
    seed=42, device="cpu", torch_threads=1, workers=2, gpu_fraction=1.0,
    train_limit=0, test_limit=0, rules=["B0", "B1", "B2"], save_updates=True,
)
ATTACKS = {"none", "zero", "noise", "stale", "label_flip", "sign_flip", "sendros_mix"}


def load_config(path: str | Path) -> dict:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    base = load_config(p.parent / raw.pop("extends")) if "extends" in raw else DEFAULTS.copy()
    unknown = raw.keys() - DEFAULTS.keys() - {"privacy"}
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    base.update(raw)
    return validate_config(base)


def validate_config(config: dict) -> dict:
    c = config.copy()
    if c["attack"] not in ATTACKS or c["dataset"] not in {"mnist", "fashion_mnist"}:
        raise ValueError("unsupported attack or dataset")
    if not 3 <= c["clients_per_round"] <= c["num_clients"] or not 0 <= c["fraction"] <= 1:
        raise ValueError("invalid client count or attacker fraction")
    if c["attack"] == "sendros_mix" and c["num_clients"] < 5:
        raise ValueError("Sendros mix requires at least five clients")
    for key in ["rounds", "local_epochs", "batch_size", "torch_threads", "workers"]:
        if not isinstance(c[key], int) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if c["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not 0 < c["gpu_fraction"] <= 1:
        raise ValueError("gpu_fraction must be in (0, 1]")
    if c["stale_mode"] not in {"train_snapshot", "replay_own"}:
        raise ValueError("unsupported stale mode")
    if not isinstance(c["stale_lag"], int) or c["stale_lag"] < 1:
        raise ValueError("stale_lag must be a positive integer")
    if "privacy" in c:
        from fltoken.privacy import validate_privacy
        c["privacy"] = validate_privacy(c["privacy"])
        if c["attack"] != "none" or c["fraction"] != 0:
            raise ValueError("the privacy comparison requires every client to be honest")
    return c
