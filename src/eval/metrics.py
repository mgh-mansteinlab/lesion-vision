"""Pixel and boundary metrics for predicted vs GT masks."""

from typing import Dict, List

import cv2
import numpy as np

from src.constants import CLASS_NAME_LIST as CLASS_NAMES, N_CLASSES, colorize_class_mask


def rgb_to_class_mask(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB mask image to a class-index mask (uint8, values 0-3)."""
    h, w = rgb.shape[:2]
    cls = np.zeros((h, w), dtype=np.uint8)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    cls[(g > 200) & (r < 100) & (b < 100)] = 1   # Tissue
    cls[(b > 200) & (r < 100) & (g < 100)] = 2   # Coagulation
    cls[(r > 200) & (g < 100) & (b < 100)] = 3   # Ablation
    # Everything else stays 0 (Background)
    return cls


def class_mask_to_rgb(cls: np.ndarray) -> np.ndarray:
    """Convert a class-index mask back to an RGB image for visualisation."""
    return colorize_class_mask(cls)


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------

def compute_confusion_matrix(gt: np.ndarray, pred: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    """Compute an n x n confusion matrix (rows=GT, cols=Pred)."""
    cm = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            cm[i, j] = np.sum((gt == i) & (pred == j))
    return cm


def per_class_metrics(cm: np.ndarray) -> List[Dict[str, float]]:
    """Derive per-class IoU, Dice, Precision, Recall from a confusion matrix."""
    metrics = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        metrics.append({
            'class': CLASS_NAMES[c],
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        })
    return metrics


def overall_metrics(cm: np.ndarray, class_metrics: List[Dict]) -> Dict[str, float]:
    """Aggregate metrics: accuracy, mean IoU (excl. background), weighted IoU."""
    total = cm.sum()
    accuracy = np.trace(cm) / total if total > 0 else 0.0

    ious = [m['iou'] for m in class_metrics if not np.isnan(m['iou'])]
    ious_no_bg = [m['iou'] for i, m in enumerate(class_metrics)
                  if i > 0 and not np.isnan(m['iou'])]

    class_totals = cm.sum(axis=1).astype(float)
    weight_sum = class_totals.sum()
    weighted_iou = sum(
        class_totals[i] * class_metrics[i]['iou']
        for i in range(len(class_metrics))
        if not np.isnan(class_metrics[i]['iou'])
    ) / weight_sum if weight_sum > 0 else 0.0

    return {
        'pixel_accuracy': accuracy,
        'mean_iou_all': float(np.nanmean(ious)) if ious else 0.0,
        'mean_iou_fg': float(np.nanmean(ious_no_bg)) if ious_no_bg else 0.0,
        'weighted_iou': weighted_iou,
    }


# ---------------------------------------------------------------------------
# Boundary analysis
# ---------------------------------------------------------------------------

def extract_boundary(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Extract class boundaries via morphological gradient (boolean mask)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return (dilated != eroded)


def boundary_metrics(gt_cls: np.ndarray, pred_cls: np.ndarray,
                     tolerances: List[int] = [1, 3, 5]) -> Dict:
    """
    Boundary F1 at multiple tolerance levels.
    For each class boundary pixel in GT, checks if there is a matching
    boundary pixel in pred within `tol` pixels, and vice versa.
    """
    gt_bnd = extract_boundary(gt_cls)
    pred_bnd = extract_boundary(pred_cls)

    results = {}
    for tol in tolerances:
        gt_dilated = cv2.dilate(gt_bnd.astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1)))
        pred_dilated = cv2.dilate(pred_bnd.astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1)))

        prec = pred_bnd & gt_dilated.astype(bool)
        rec = gt_bnd & pred_dilated.astype(bool)

        p = prec.sum() / pred_bnd.sum() if pred_bnd.sum() > 0 else 0.0
        r = rec.sum() / gt_bnd.sum() if gt_bnd.sum() > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[f'boundary_f1_tol{tol}'] = f1
        results[f'boundary_prec_tol{tol}'] = p
        results[f'boundary_rec_tol{tol}'] = r

    return results

