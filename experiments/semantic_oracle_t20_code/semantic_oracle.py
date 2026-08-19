"""Semantic Oracle routing based on a pre-generated ImageNet-A class mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def load_semantic_mapping(
    mapping_path: str,
    num_loras: int = 5,
    expected_num_classes: int = 200,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load and validate a semantic class-to-expert mapping JSON."""
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"semantic_mapping_path does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Semantic mapping JSON root must be an object.")

    entries = payload.get("classes")
    if not isinstance(entries, list):
        raise ValueError("Semantic mapping JSON must contain a 'classes' list.")
    if len(entries) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} mapping entries, found {len(entries)}."
        )

    class_to_expert = torch.full((expected_num_classes,), -1, dtype=torch.long)
    seen_labels = set()

    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("Each mapping entry must be an object.")
        if "label_id" not in row or "expert_id" not in row:
            raise ValueError("Each mapping entry requires label_id and expert_id.")

        label_id = int(row["label_id"])
        expert_id = int(row["expert_id"])

        if not 0 <= label_id < expected_num_classes:
            raise ValueError(
                f"Invalid label_id={label_id}; expected 0..{expected_num_classes - 1}."
            )
        if label_id in seen_labels:
            raise ValueError(f"Duplicate label_id in semantic mapping: {label_id}")
        if not 0 <= expert_id < num_loras:
            raise ValueError(
                f"Invalid expert_id={expert_id}; expected 0..{num_loras - 1}."
            )

        seen_labels.add(label_id)
        class_to_expert[label_id] = expert_id

    missing = (class_to_expert < 0).nonzero(as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(f"Semantic mapping is missing label IDs: {missing}")

    counts = torch.bincount(class_to_expert, minlength=num_loras)
    if expected_num_classes == 200 and num_loras == 5:
        expected = [40] * 5
        if counts.tolist() != expected:
            raise ValueError(
                f"Semantic mapping must contain exactly 40 classes per expert; got {counts.tolist()}."
            )

    return class_to_expert, payload


class SemanticOracleRouter:
    """GT-label Top-1 router using a fixed semantic mapping."""

    def __init__(
        self,
        mapping_path: str,
        num_loras: int = 5,
        expected_num_classes: int = 200,
    ):
        self.num_loras = num_loras
        self.expected_num_classes = expected_num_classes
        self.class_to_expert, self.mapping_payload = load_semantic_mapping(
            mapping_path=mapping_path,
            num_loras=num_loras,
            expected_num_classes=expected_num_classes,
        )

    def route(self, targets: torch.Tensor) -> torch.Tensor:
        """Return expert IDs with shape [B, 1] for GT incremental labels [B]."""
        if not isinstance(targets, torch.Tensor):
            raise TypeError("targets must be a torch.Tensor")
        if targets.ndim != 1:
            raise ValueError(f"targets must be 1-D, got shape={tuple(targets.shape)}")

        labels = targets.to(dtype=torch.long)
        if labels.numel() == 0:
            return torch.empty((0, 1), dtype=torch.long, device=labels.device)

        lo = int(labels.min().item())
        hi = int(labels.max().item())
        if lo < 0 or hi >= self.expected_num_classes:
            raise ValueError(
                f"Semantic Oracle received invalid label range [{lo}, {hi}], "
                f"expected 0..{self.expected_num_classes - 1}."
            )

        mapping = self.class_to_expert.to(labels.device)
        experts = mapping[labels]
        if (experts < 0).any() or (experts >= self.num_loras).any():
            raise RuntimeError("Semantic Oracle produced an invalid expert index.")

        return experts.unsqueeze(1)
