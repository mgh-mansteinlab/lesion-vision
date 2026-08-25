"""Flag lesions whose geometry is incomplete at the punch or image boundary."""

from typing import Optional, Tuple

import cv2
import numpy as np

from src.constants import CLASS_ID_ABLATION, CLASS_ID_BACKGROUND, CLASS_ID_COAGULATION, PIXEL_SCALE_UM


def distance_to_background_px(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (px) from each pixel to the nearest background pixel."""
    tissue = (mask != CLASS_ID_BACKGROUND).astype(np.uint8)
    if not np.any(tissue):
        return np.zeros(mask.shape[:2], dtype=np.float32)
    return cv2.distanceTransform(tissue, cv2.DIST_L2, 5)


def lesion_binary(mask: np.ndarray) -> np.ndarray:
    return ((mask == CLASS_ID_COAGULATION) | (mask == CLASS_ID_ABLATION)).astype(np.uint8)


def component_on_image_border(component: np.ndarray) -> bool:
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return False
    h, w = component.shape[:2]
    return bool(
        int(xs.min()) == 0
        or int(ys.min()) == 0
        or int(xs.max()) == w - 1
        or int(ys.max()) == h - 1
    )


def component_touches_background(
    component: np.ndarray,
    mask: np.ndarray,
    dilate_px: int = 1,
) -> bool:
    """True if a 1-pixel ring around the component includes background."""
    if not np.any(component):
        return False
    k = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1)
    ring = (dilated > 0) & (component == 0)
    return bool(np.any(mask[ring] == CLASS_ID_BACKGROUND))


def lesion_edge_flags(
    component: np.ndarray,
    mask: np.ndarray,
    dist_px: Optional[np.ndarray] = None,
    pixel_scale_um: float = PIXEL_SCALE_UM,
) -> Tuple[bool, float]:
    """
    Return (touches_edge, min_distance_to_background_um).

    touches_edge is True if the component meets background, the image border,
    or has a min tissue-to-background distance of < 0.5 px.
    """
    if dist_px is None:
        dist_px = distance_to_background_px(mask)
    pts = component > 0
    if not np.any(pts):
        return False, float("nan")
    min_px = float(dist_px[pts].min())
    dist_um = min_px * pixel_scale_um
    touches = (
        min_px <= 0.5
        or component_touches_background(component, mask)
        or component_on_image_border(component)
    )
    return bool(touches), float(dist_um)


def labeled_lesion_components(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Binary lesion map and connected-component labels (0 = background)."""
    binary = lesion_binary(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    _n, labels = cv2.connectedComponents(closed)
    return closed, labels


def component_near_point(
    labels: np.ndarray,
    x: float,
    y: float,
    search_px: int = 40,
) -> Optional[np.ndarray]:
    """Component containing (x, y), or the nearest labeled pixel within search_px."""
    h, w = labels.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    if 0 <= yi < h and 0 <= xi < w and labels[yi, xi] > 0:
        return labels == labels[yi, xi]

    y0 = max(0, yi - search_px)
    y1 = min(h, yi + search_px + 1)
    x0 = max(0, xi - search_px)
    x1 = min(w, xi + search_px + 1)
    patch = labels[y0:y1, x0:x1]
    if not np.any(patch > 0):
        return None
    ys, xs = np.where(patch > 0)
    dist2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    j = int(np.argmin(dist2))
    label_id = int(patch[ys[j], xs[j]])
    if label_id <= 0:
        return None
    return labels == label_id
