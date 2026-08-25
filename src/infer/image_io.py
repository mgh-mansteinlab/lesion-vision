"""Image read / NDPI resolve helpers for inference."""

import os
import shutil
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from src.infer.ndpi import NDPIExtractor


def read_rgb(image_path: str) -> np.ndarray:
    """Read an image from disk and return RGB uint8."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
    except cv2.error as e:
        if "pixels <= CV_IO_MAX_IMAGE_PIXELS" not in str(e):
            raise
        pil_image = Image.open(image_path)
        width, height = pil_image.size
        max_pixels = 2**30
        scale_factor = np.sqrt(max_pixels / (width * height))
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        print(f"Resized large image from {width}x{height} to {new_width}x{new_height}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resolve_ndpi(image_path: str) -> Tuple[str, Optional[str]]:
    """If NDPI, extract first sample to a temp TIF. Caller must clean temp_dir."""
    if not image_path.lower().endswith(".ndpi"):
        return image_path, None

    print(f"Detected NDPI file: {image_path}")
    print("Converting to TIF format...")
    temp_dir = tempfile.mkdtemp()
    extractor = NDPIExtractor(image_path, temp_dir)
    tif_paths = extractor.extract_to_tif()
    if not tif_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("No samples extracted from NDPI file")
    print(f"Extracted {len(tif_paths)} sample(s) from NDPI")
    if len(tif_paths) > 1:
        print(
            f"Warning: NDPI contains {len(tif_paths)} samples. "
            "Only processing the first one. Use batch processing for all."
        )
    return tif_paths[0], temp_dir
