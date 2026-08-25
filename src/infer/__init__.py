"""Inference: tiling, post-process, NDPI, predictor, batch."""

from src.infer.predictor import LesionPredictor
from src.infer.postprocess import post_process_mask

__all__ = ["LesionPredictor", "post_process_mask"]
