"""Backward-compatible import path. Use src.analytics.lesions."""

from src.analytics.lesions import LesionAnalyzer, LesionMetrics

__all__ = ["LesionAnalyzer", "LesionMetrics"]
