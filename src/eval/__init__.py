"""Mask evaluation metrics and CLI."""

from src.eval.metrics import (
    class_mask_to_rgb,
    compute_confusion_matrix,
    rgb_to_class_mask,
)

__all__ = ["rgb_to_class_mask", "class_mask_to_rgb", "compute_confusion_matrix"]
