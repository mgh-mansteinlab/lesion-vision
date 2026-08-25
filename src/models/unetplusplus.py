"""Backward-compatible import path. Use src.models.segmentation_model."""

from src.models.losses import CompoundLoss
from src.models.architecture import SMPSegmentationNet as UNetPlusPlus
from src.models.segmentation_model import LesionSegmentationModel

__all__ = ["UNetPlusPlus", "CompoundLoss", "LesionSegmentationModel"]
