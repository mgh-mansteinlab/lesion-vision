"""Tissue windowing, tile slicing, and probability-space stitching."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TissueWindowConfig:
    padding: int = 64
    bg_percentile: int = 98
    bg_margin: int = 4
    min_fg_ratio: float = 0.005
    # Pale NBTC punches are ~8 mm (~9k px). A crop shorter than this is a failed window.
    min_span_px: int = 4000
    min_contour_frac_of_largest: float = 0.05


def tissue_threshold(gray: np.ndarray, config: TissueWindowConfig) -> float:
    """Brightness threshold separating tissue from bright slide background.

    Pale NBTC sits only a few gray levels below the slide. A large margin plus
    clipping at 242 treats that tissue as background and the bbox collapses to
    a dark stain blob.
    """
    bg_level = float(np.percentile(gray, config.bg_percentile))
    return float(np.clip(bg_level - config.bg_margin, 180.0, 252.0))


def is_background_tile(tile: np.ndarray, threshold: float, config: TissueWindowConfig) -> bool:
    """True if a tile contains essentially no tissue."""
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    fg_frac = float((gray < threshold).mean())
    return fg_frac < config.min_fg_ratio


def _bbox_from_foreground(
    fg: np.ndarray,
    padding: int,
    config: TissueWindowConfig,
) -> Optional[Tuple[int, int, int, int]]:
    """Union bbox of substantial contours. None if nothing usable is found."""
    h_img, w_img = fg.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    min_keep = max(max_area * config.min_contour_frac_of_largest, 1.0)
    kept = [c for c, area in zip(contours, areas) if area >= min_keep]
    pts = np.concatenate(kept)
    x, y, w, h = cv2.boundingRect(pts)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _span_ok(bbox: Optional[Tuple[int, int, int, int]], config: TissueWindowConfig) -> bool:
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return min(x2 - x1, y2 - y1) >= config.min_span_px


def get_tissue_bbox(
    image: np.ndarray,
    config: TissueWindowConfig,
    padding: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Detect the main sample and return a cropped region plus bbox.

    Uses the union of large contours, not only the single largest, so a dark
    stain blob cannot replace a pale punch. If the window is still too small,
    retry with a looser threshold, then fall back to the full image.
    """
    if padding is None:
        padding = config.padding

    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold is None:
        threshold = tissue_threshold(blurred, config)

    attempts = [threshold]
    loose = float(np.clip(np.percentile(blurred, 99.0) - 1.0, 180.0, 252.0))
    if loose > threshold + 0.5:
        attempts.append(loose)

    bbox = None
    used_threshold = threshold
    for cand in attempts:
        fg = (blurred < cand).astype(np.uint8) * 255
        bbox = _bbox_from_foreground(fg, padding, config)
        used_threshold = cand
        if _span_ok(bbox, config):
            break

    if not _span_ok(bbox, config):
        print(
            "Warning: Tissue window too small or missing "
            f"(threshold={used_threshold:.0f}); using full image"
        )
        return image, (0, 0, w_img, h_img)

    x1, y1, x2, y2 = bbox
    print(f"Detected tissue region: ({x1}, {y1}, {x2}, {y2}) (tissue threshold={used_threshold:.0f})")
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def get_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Split an image into overlapping tiles, white-padded at edges."""
    height, width = image.shape[:2]
    tiles = []
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile = image[y:y_end, x:x_end]
            if tile.shape[:2] != (tile_size, tile_size):
                padded = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            tiles.append((tile, (y, x)))
    return tiles


def stitch_predictions(tiles, original_shape, tile_size=512, overlap=128):
    """Blend overlapping tile predictions. Prefers probability-space averaging."""
    height, width = original_shape
    sample_pred = tiles[0][0]
    is_prob = sample_pred.ndim == 3

    if is_prob:
        n_classes = sample_pred.shape[0]
        stitched = np.zeros((n_classes, height, width), dtype=np.float32)
    else:
        stitched = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    for pred, (y, x) in tiles:
        weight = np.ones(pred.shape[-2:], dtype=np.float32)
        if overlap > 0:
            weight[:, :overlap] = np.linspace(0, 1, overlap)[None, :]
            weight[:, -overlap:] = np.linspace(1, 0, overlap)[None, :]
            weight[:overlap, :] *= np.linspace(0, 1, overlap)[:, None]
            weight[-overlap:, :] *= np.linspace(1, 0, overlap)[:, None]

        y_end = min(y + tile_size, height)
        x_end = min(x + tile_size, width)
        h_slice = slice(y, y_end)
        w_slice = slice(x, x_end)
        ph = y_end - y
        pw = x_end - x

        if is_prob:
            for c in range(n_classes):
                stitched[c, h_slice, w_slice] += pred[c, :ph, :pw] * weight[:ph, :pw]
        else:
            stitched[h_slice, w_slice] += pred[:ph, :pw].astype(np.float32) * weight[:ph, :pw]
        weights[h_slice, w_slice] += weight[:ph, :pw]

    safe_w = np.maximum(weights, 1e-6)
    if is_prob:
        for c in range(n_classes):
            stitched[c] /= safe_w
        return np.argmax(stitched, axis=0).astype(np.uint8)
    return np.round(stitched / safe_w).astype(np.uint8)
