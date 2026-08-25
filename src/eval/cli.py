"""CLI: compare predicted vs ground-truth RGB masks."""

import argparse
import csv
import os
from typing import Dict, List

import cv2
import numpy as np

from src.constants import CLASS_NAME_LIST as CLASS_NAMES, N_CLASSES
from src.eval.lesion_match import find_connected_lesions, match_lesions
from src.eval.metrics import (
    boundary_metrics,
    class_mask_to_rgb,
    compute_confusion_matrix,
    overall_metrics,
    per_class_metrics,
    rgb_to_class_mask,
)
from src.eval.plots import (
    plot_confusion_matrix,
    plot_difference_map,
    plot_lesion_area_scatter,
    plot_lesion_matching,
    plot_per_class_bars,
    plot_per_class_comparison,
    plot_side_by_side,
)


def write_summary_csv(output_path: str, overall: Dict, class_metrics: List[Dict],
                      boundary: Dict, match_result: Dict):
    rows = []
    rows.append(['=== Overall Pixel Metrics ===', ''])
    for k, v in overall.items():
        rows.append([k, f'{v:.6f}'])

    rows.append(['', ''])
    rows.append(['=== Per-Class Metrics ===', 'IoU', 'Dice', 'Precision', 'Recall', 'TP', 'FP', 'FN'])
    for m in class_metrics:
        rows.append([
            m['class'],
            f"{m['iou']:.6f}",
            f"{m['dice']:.6f}",
            f"{m['precision']:.6f}",
            f"{m['recall']:.6f}",
            str(m['tp']),
            str(m['fp']),
            str(m['fn']),
        ])

    rows.append(['', ''])
    rows.append(['=== Boundary Metrics ===', ''])
    for k, v in boundary.items():
        rows.append([k, f'{v:.6f}'])

    rows.append(['', ''])
    rows.append(['=== Lesion Detection ===', ''])
    rows.append(['detection_precision', f"{match_result['detection_precision']:.6f}"])
    rows.append(['detection_recall', f"{match_result['detection_recall']:.6f}"])
    rows.append(['detection_f1', f"{match_result['detection_f1']:.6f}"])
    rows.append(['matched_lesions', str(len(match_result['matched']))])
    rows.append(['missed_gt_lesions', str(len(match_result['missed_gt']))])
    rows.append(['false_positive_lesions', str(len(match_result['false_pos_pred']))])

    if match_result['matched']:
        ious = [p['iou'] for p in match_result['matched']]
        rows.append(['mean_matched_iou', f'{np.mean(ious):.6f}'])
        rows.append(['median_matched_iou', f'{np.median(ious):.6f}'])
        rows.append(['min_matched_iou', f'{np.min(ious):.6f}'])
        rows.append(['max_matched_iou', f'{np.max(ious):.6f}'])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_lesion_csv(output_path: str, match_result: Dict,
                     gt_lesions: List[Dict], pred_lesions: List[Dict]):
    """Per-lesion pair comparison CSV."""
    rows = []
    header = [
        'pair_id', 'gt_idx', 'pred_idx', 'centroid_dist',
        'iou', 'gt_area', 'pred_area', 'area_diff_pct',
        'gt_ablation', 'pred_ablation', 'ablation_diff_pct',
        'gt_coag', 'pred_coag', 'coag_diff_pct',
    ]
    rows.append(header)

    for i, pair in enumerate(match_result['matched']):
        area_diff = ((pair['pred_area'] - pair['gt_area']) / pair['gt_area'] * 100
                     if pair['gt_area'] > 0 else 0)
        ab_diff = ((pair['pred_ablation'] - pair['gt_ablation']) / pair['gt_ablation'] * 100
                   if pair['gt_ablation'] > 0 else 0)
        cg_diff = ((pair['pred_coag'] - pair['gt_coag']) / pair['gt_coag'] * 100
                   if pair['gt_coag'] > 0 else 0)
        rows.append([
            i, pair['gt_idx'], pair['pred_idx'],
            f"{pair['distance']:.1f}",
            f"{pair['iou']:.4f}",
            pair['gt_area'], pair['pred_area'], f"{area_diff:.1f}",
            pair['gt_ablation'], pair['pred_ablation'], f"{ab_diff:.1f}",
            pair['gt_coag'], pair['pred_coag'], f"{cg_diff:.1f}",
        ])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(overall: Dict, class_metrics: List[Dict],
                 boundary: Dict, match_result: Dict,
                 gt_lesions: List[Dict], pred_lesions: List[Dict]):
    sep = '=' * 70
    print(f'\n{sep}')
    print('   LESION SEGMENTATION – PREDICTION vs GROUND TRUTH REPORT')
    print(sep)

    print('\n--- Overall Pixel-Level Metrics ---')
    print(f"  Pixel Accuracy:       {overall['pixel_accuracy']:.4f}  ({overall['pixel_accuracy']*100:.2f}%)")
    print(f"  Mean IoU (all):       {overall['mean_iou_all']:.4f}")
    print(f"  Mean IoU (FG only):   {overall['mean_iou_fg']:.4f}")
    print(f"  Weighted IoU:         {overall['weighted_iou']:.4f}")

    print('\n--- Per-Class Metrics ---')
    print(f"  {'Class':<15} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}   {'TP':>12} {'FP':>12} {'FN':>12}")
    for m in class_metrics:
        print(f"  {m['class']:<15} {m['iou']:>8.4f} {m['dice']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f}   "
              f"{m['tp']:>12,} {m['fp']:>12,} {m['fn']:>12,}")

    print('\n--- Boundary Metrics ---')
    for k, v in boundary.items():
        print(f"  {k:<30} {v:.4f}")

    print('\n--- Lesion-Level Detection ---')
    print(f"  GT lesions found:         {len(gt_lesions)}")
    print(f"  Pred lesions found:       {len(pred_lesions)}")
    print(f"  Matched:                  {len(match_result['matched'])}")
    print(f"  Missed (FN):              {len(match_result['missed_gt'])}")
    print(f"  False Positive:           {len(match_result['false_pos_pred'])}")
    print(f"  Detection Precision:      {match_result['detection_precision']:.4f}")
    print(f"  Detection Recall:         {match_result['detection_recall']:.4f}")
    print(f"  Detection F1:             {match_result['detection_f1']:.4f}")

    if match_result['matched']:
        ious = [p['iou'] for p in match_result['matched']]
        dists = [p['distance'] for p in match_result['matched']]
        print(f"\n  Matched Lesion IoU:  mean={np.mean(ious):.4f}  "
              f"median={np.median(ious):.4f}  min={np.min(ious):.4f}  max={np.max(ious):.4f}")
        print(f"  Centroid Distance:   mean={np.mean(dists):.1f}px  "
              f"median={np.median(dists):.1f}px  max={np.max(dists):.1f}px")

        area_diffs = [
            (p['pred_area'] - p['gt_area']) / p['gt_area'] * 100
            for p in match_result['matched'] if p['gt_area'] > 0
        ]
        if area_diffs:
            print(f"  Area Difference:     mean={np.mean(area_diffs):+.1f}%  "
                  f"median={np.median(area_diffs):+.1f}%  "
                  f"std={np.std(area_diffs):.1f}%")

    print(f'\n{sep}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Compare predicted mask vs ground truth')
    parser.add_argument('--pred', required=True, help='Path to predicted mask PNG (RGB)')
    parser.add_argument('--gt', required=True, help='Path to ground truth mask PNG (RGB)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: alongside pred)')
    args = parser.parse_args()

    pred_path = os.path.abspath(args.pred)
    gt_path = os.path.abspath(args.gt)

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(pred_path), 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(pred_path))[0].replace('_pred', '')

    print(f'Pred: {pred_path}')
    print(f'GT:   {gt_path}')
    print(f'Out:  {out_dir}')

    # ------------------------------------------------------------------
    # Load images
    # ------------------------------------------------------------------
    pred_bgr = cv2.imread(pred_path, cv2.IMREAD_COLOR)
    gt_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    if pred_bgr is None:
        raise FileNotFoundError(f'Cannot read prediction: {pred_path}')
    if gt_bgr is None:
        raise FileNotFoundError(f'Cannot read ground truth: {gt_path}')

    pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

    print(f'Pred shape: {pred_rgb.shape}')
    print(f'GT shape:   {gt_rgb.shape}')

    if pred_rgb.shape != gt_rgb.shape:
        print(f'WARNING: shape mismatch – resizing pred to GT size {gt_rgb.shape[:2]}')
        pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------------
    # Convert to class indices
    # ------------------------------------------------------------------
    pred_cls = rgb_to_class_mask(pred_rgb)
    gt_cls = rgb_to_class_mask(gt_rgb)

    print(f'GT  class distribution:   {dict(zip(CLASS_NAMES, [int((gt_cls==c).sum()) for c in range(N_CLASSES)]))}')
    print(f'Pred class distribution:  {dict(zip(CLASS_NAMES, [int((pred_cls==c).sum()) for c in range(N_CLASSES)]))}')

    # ------------------------------------------------------------------
    # Pixel-level metrics
    # ------------------------------------------------------------------
    print('\nComputing pixel-level metrics...')
    cm = compute_confusion_matrix(gt_cls, pred_cls)
    cls_metrics = per_class_metrics(cm)
    ovr = overall_metrics(cm, cls_metrics)

    # ------------------------------------------------------------------
    # Boundary metrics
    # ------------------------------------------------------------------
    print('Computing boundary metrics...')
    bnd = boundary_metrics(gt_cls, pred_cls, tolerances=[1, 3, 5])

    # ------------------------------------------------------------------
    # Lesion-level matching
    # ------------------------------------------------------------------
    print('Finding GT lesions...')
    gt_lesions = find_connected_lesions(gt_cls)
    print(f'  Found {len(gt_lesions)} GT lesions')

    print('Finding Pred lesions...')
    pred_lesions = find_connected_lesions(pred_cls)
    print(f'  Found {len(pred_lesions)} Pred lesions')

    print('Matching lesions...')
    match = match_lesions(gt_lesions, pred_lesions)

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print_report(ovr, cls_metrics, bnd, match, gt_lesions, pred_lesions)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    print('Generating visualizations...')

    plot_side_by_side(
        class_mask_to_rgb(gt_cls), class_mask_to_rgb(pred_cls),
        os.path.join(out_dir, f'{basename}_side_by_side.png'))

    plot_confusion_matrix(cm, os.path.join(out_dir, f'{basename}_confusion_matrix.png'))

    plot_per_class_bars(cls_metrics, os.path.join(out_dir, f'{basename}_class_metrics.png'))

    plot_difference_map(gt_cls, pred_cls, os.path.join(out_dir, f'{basename}_difference_map.png'))

    plot_per_class_comparison(gt_cls, pred_cls,
                              os.path.join(out_dir, f'{basename}_per_class_comparison.png'))

    plot_lesion_matching(gt_cls, pred_cls, gt_lesions, pred_lesions, match,
                         os.path.join(out_dir, f'{basename}_lesion_matching.png'))

    plot_lesion_area_scatter(match, os.path.join(out_dir, f'{basename}_lesion_area_scatter.png'))

    # ------------------------------------------------------------------
    # CSVs
    # ------------------------------------------------------------------
    write_summary_csv(os.path.join(out_dir, f'{basename}_summary_metrics.csv'),
                      ovr, cls_metrics, bnd, match)
    write_lesion_csv(os.path.join(out_dir, f'{basename}_lesion_pairs.csv'),
                     match, gt_lesions, pred_lesions)

    print(f'\nAll outputs saved to: {out_dir}')
    print('Files:')
    for f in sorted(os.listdir(out_dir)):
        print(f'  {f}')


if __name__ == '__main__':
    main()
