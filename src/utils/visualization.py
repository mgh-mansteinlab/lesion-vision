"""Backward-compatible import path. Use src.train.viz."""

from src.train.viz import create_color_mask, plot_training_metrics, visualize_results

__all__ = ["create_color_mask", "visualize_results", "plot_training_metrics"]
