"""Backward-compatible import path. Use src.analytics.grid."""

from src.analytics.grid import GridMapper
from src.constants import PIXEL_SCALE_UM

__all__ = ["GridMapper", "PIXEL_SCALE_UM"]
