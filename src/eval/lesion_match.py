"""Lesion instance matching between predicted and GT masks."""

from typing import Dict, List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def find_connected_lesions(cls_mask: np.ndarray,
                           coag_class: int = 2,
                           ablation_class: int = 3) -> List[Dict]:
    """
    Find individual lesions as connected components of (coagulation | ablation).
    Returns list of dicts with keys: label, centroid, area_px, bbox,
    ablation_area_px, coagulation_area_px.
    """
    lesion_binary = ((cls_mask == coag_class) | (cls_mask == ablation_class)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lesion_binary = cv2.morphologyEx(lesion_binary, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        lesion_binary, connectivity=8
    )

    min_area = 50
    lesions = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        component_mask = (labels == i)
        lesions.append({
            'label': i,
            'centroid': tuple(centroids[i]),
            'area_px': int(area),
            'bbox': (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            ),
            'ablation_area_px': int(np.sum(component_mask & (cls_mask == ablation_class))),
            'coagulation_area_px': int(np.sum(component_mask & (cls_mask == coag_class))),
            'mask': component_mask,
        })
    return lesions


def match_lesions(gt_lesions: List[Dict], pred_lesions: List[Dict],
                  max_dist: float = 150.0) -> Dict:
    """
    Hungarian matching of GT and pred lesions by centroid distance.
    Returns matched pairs, unmatched GT (missed), unmatched pred (false positives),
    and per-pair IoU.
    """
    n_gt, n_pred = len(gt_lesions), len(pred_lesions)
    if n_gt == 0 and n_pred == 0:
        return {'matched': [], 'missed_gt': [], 'false_pos_pred': [],
                'detection_precision': 1.0, 'detection_recall': 1.0, 'detection_f1': 1.0}
    if n_gt == 0:
        return {'matched': [], 'missed_gt': [], 'false_pos_pred': list(range(n_pred)),
                'detection_precision': 0.0, 'detection_recall': 1.0, 'detection_f1': 0.0}
    if n_pred == 0:
        return {'matched': [], 'missed_gt': list(range(n_gt)), 'false_pos_pred': [],
                'detection_precision': 1.0, 'detection_recall': 0.0, 'detection_f1': 0.0}

    cost = np.full((n_gt, n_pred), 1e9)
    for i, gl in enumerate(gt_lesions):
        for j, pl in enumerate(pred_lesions):
            d = np.hypot(gl['centroid'][0] - pl['centroid'][0],
                         gl['centroid'][1] - pl['centroid'][1])
            if d <= max_dist:
                cost[i, j] = d

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    matched_gt_set = set()
    matched_pred_set = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > max_dist:
            continue
        inter = np.sum(gt_lesions[r]['mask'] & pred_lesions[c]['mask'])
        union = np.sum(gt_lesions[r]['mask'] | pred_lesions[c]['mask'])
        iou = inter / union if union > 0 else 0.0
        matched.append({
            'gt_idx': r,
            'pred_idx': c,
            'distance': cost[r, c],
            'iou': iou,
            'gt_area': gt_lesions[r]['area_px'],
            'pred_area': pred_lesions[c]['area_px'],
            'gt_ablation': gt_lesions[r]['ablation_area_px'],
            'pred_ablation': pred_lesions[c]['ablation_area_px'],
            'gt_coag': gt_lesions[r]['coagulation_area_px'],
            'pred_coag': pred_lesions[c]['coagulation_area_px'],
        })
        matched_gt_set.add(r)
        matched_pred_set.add(c)

    missed_gt = [i for i in range(n_gt) if i not in matched_gt_set]
    false_pos = [j for j in range(n_pred) if j not in matched_pred_set]

    tp = len(matched)
    prec = tp / (tp + len(false_pos)) if (tp + len(false_pos)) > 0 else 0.0
    rec = tp / (tp + len(missed_gt)) if (tp + len(missed_gt)) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'matched': matched,
        'missed_gt': missed_gt,
        'false_pos_pred': false_pos,
        'detection_precision': prec,
        'detection_recall': rec,
        'detection_f1': f1,
    }
