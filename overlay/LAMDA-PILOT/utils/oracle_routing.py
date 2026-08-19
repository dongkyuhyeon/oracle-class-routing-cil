"""Deterministic ground-truth class routing for Oracle experiments."""

from __future__ import annotations

import torch


def oracle_class_topk(
    targets: torch.Tensor,
    num_loras: int,
    top_k: int,
) -> torch.Tensor:
    """Map each class label to a fixed cyclic set of LoRA experts.

    For K=5:
      * top_k=1: class c -> [c % 5]
      * top_k=2: class c -> [c % 5, (c + 1) % 5]

    This makes routing class-consistent. If class IDs are evenly represented
    modulo K, it also balances class membership across experts.
    """
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if targets.ndim != 1:
        raise ValueError("targets must be a 1-D tensor")
    if num_loras < 1:
        raise ValueError("num_loras must be >= 1")
    if not 1 <= top_k <= num_loras:
        raise ValueError("top_k must be in [1, num_loras]")

    labels = targets.to(dtype=torch.long)
    offsets = torch.arange(top_k, device=labels.device, dtype=torch.long)
    return (labels.unsqueeze(1) + offsets.unsqueeze(0)) % num_loras

