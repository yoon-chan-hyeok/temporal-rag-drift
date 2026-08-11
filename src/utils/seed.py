"""Seed management helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int | None) -> None:
    """Set process-level random seeds when a seed is provided."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def derive_seed(base_seed: int | None, *parts: object) -> int | None:
    """Derive a deterministic non-negative seed from a base seed and labels."""
    if base_seed is None:
        return None
    value = int(base_seed) & 0xFFFFFFFF
    for part in parts:
        for char in str(part):
            value = ((value * 33) + ord(char)) & 0xFFFFFFFF
    return value
