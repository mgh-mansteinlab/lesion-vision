"""Backward-compatible import path. Use src.infer.predictor."""

from src.analytics.lesions import LesionAnalyzer, LesionMetrics
from src.infer.predictor import LesionPredictor

__all__ = ["LesionPredictor", "LesionMetrics", "LesionAnalyzer"]
