"""Diagnostic plots for mask comparison."""

from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.constants import CLASS_NAME_LIST as CLASS_NAMES, N_CLASSES
from src.eval.metrics import class_mask_to_rgb


def plot_confusion_matrix(cm: np.ndarray, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm[i, j]
            txt = f'{val:,}' if val < 1e6 else f'{val:.2e}'
            color = 'white' if val > cm.max() * 0.5 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix (pixel counts)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_bars(class_metrics: List[Dict], output_path: str):
    names = [m['class'] for m in class_metrics]
    ious = [m['iou'] for m in class_metrics]
    dices = [m['dice'] for m in class_metrics]
    precs = [m['precision'] for m in class_metrics]
    recs = [m['recall'] for m in class_metrics]

    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, ious, w, label='IoU', color='#2196F3')
    ax.bar(x - 0.5*w, dices, w, label='Dice', color='#4CAF50')
    ax.bar(x + 0.5*w, precs, w, label='Precision', color='#FF9800')
    ax.bar(x + 1.5*w, recs, w, label='Recall', color='#F44336')

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Segmentation Metrics')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_difference_map(gt_cls: np.ndarray, pred_cls: np.ndarray, output_path: str):
    """
    Colour-coded difference map:
      - White: both agree
      - Per-class colours show where each class was missed or falsely predicted
    """
    agree = (gt_cls == pred_cls)
    h, w = gt_cls.shape
    diff_rgb = np.full((h, w, 3), 200, dtype=np.uint8)  # light grey = agree
    diff_rgb[agree] = [220, 220, 220]

    for c in range(1, N_CLASSES):
        fn_mask = (gt_cls == c) & (pred_cls != c)  # missed
        fp_mask = (gt_cls != c) & (pred_cls == c)  # false positive
        color = CLASS_COLORS_RGB[c]
        dark = (color * 0.5).astype(np.uint8)
        diff_rgb[fn_mask] = dark       # darker shade = missed
        diff_rgb[fp_mask] = color      # bright = false positive

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(diff_rgb)
    ax.set_title('Difference Map  (bright=FP, dark=FN, grey=agree)')
    ax.axis('off')

    patches = [mpatches.Patch(color=[0.86, 0.86, 0.86], label='Agreement')]
    for c in range(1, N_CLASSES):
        col = CLASS_COLORS_RGB[c] / 255.0
        patches.append(mpatches.Patch(color=col, label=f'{CLASS_NAMES[c]} FP'))
        patches.append(mpatches.Patch(color=col * 0.5, label=f'{CLASS_NAMES[c]} FN'))
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_side_by_side(gt_rgb: np.ndarray, pred_rgb: np.ndarray, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(gt_rgb)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(pred_rgb)
    axes[1].set_title('Prediction', fontsize=14)
    axes[1].axis('off')

    patches = [mpatches.Patch(color=CLASS_COLORS_RGB[i] / 255.0, label=CLASS_NAMES[i])
               for i in range(N_CLASSES)]
    fig.legend(handles=patches, loc='lower center', ncol=N_CLASSES, fontsize=11)
    fig.suptitle('GT vs Prediction', fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_class_comparison(gt_cls: np.ndarray, pred_cls: np.ndarray, output_path: str):
    """Side-by-side binary masks for each foreground class."""
    fg_classes = list(range(1, N_CLASSES))
    fig, axes = plt.subplots(len(fg_classes), 3, figsize=(15, 5 * len(fg_classes)))

    for row, c in enumerate(fg_classes):
        gt_bin = (gt_cls == c).astype(np.uint8) * 255
        pred_bin = (pred_cls == c).astype(np.uint8) * 255
        overlap = np.zeros((*gt_cls.shape, 3), dtype=np.uint8)
        overlap[..., 1] = gt_bin    # GT in green channel
        overlap[..., 0] = pred_bin  # pred in red channel
        # overlap yellow where both

        axes[row, 0].imshow(gt_bin, cmap='gray', vmin=0, vmax=255)
        axes[row, 0].set_title(f'{CLASS_NAMES[c]} – GT')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(pred_bin, cmap='gray', vmin=0, vmax=255)
        axes[row, 1].set_title(f'{CLASS_NAMES[c]} – Pred')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(overlap)
        axes[row, 2].set_title(f'{CLASS_NAMES[c]} – Overlap (green=GT, red=Pred, yellow=both)')
        axes[row, 2].axis('off')

    fig.suptitle('Per-Class Binary Comparison', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_lesion_matching(gt_cls: np.ndarray, pred_cls: np.ndarray,
                         gt_lesions: List[Dict], pred_lesions: List[Dict],
                         match_result: Dict, output_path: str):
    """Overlay showing matched, missed, and false-positive lesions."""
    canvas = (class_mask_to_rgb(gt_cls).astype(float) * 0.3 +
              class_mask_to_rgb(pred_cls).astype(float) * 0.3 +
              np.full((*gt_cls.shape, 3), 100, dtype=float))
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(canvas)

    for pair in match_result['matched']:
        gc = gt_lesions[pair['gt_idx']]['centroid']
        pc = pred_lesions[pair['pred_idx']]['centroid']
        ax.plot(*gc, 'go', markersize=6)
        ax.plot(*pc, 'b^', markersize=6)
        ax.plot([gc[0], pc[0]], [gc[1], pc[1]], 'g-', linewidth=0.8, alpha=0.6)
        mid_x, mid_y = (gc[0] + pc[0]) / 2, (gc[1] + pc[1]) / 2
        ax.text(mid_x, mid_y, f"IoU={pair['iou']:.2f}", fontsize=6,
                color='white', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.5))

    for idx in match_result['missed_gt']:
        c = gt_lesions[idx]['centroid']
        ax.plot(*c, 'rx', markersize=10, markeredgewidth=2)

    for idx in match_result['false_pos_pred']:
        c = pred_lesions[idx]['centroid']
        ax.plot(*c, 'ms', markersize=8, markeredgewidth=2, fillstyle='none')

    patches = [
        mpatches.Patch(color='green', label='Matched GT'),
        mpatches.Patch(color='blue', label='Matched Pred'),
        mpatches.Patch(color='red', label='Missed (FN)'),
        mpatches.Patch(color='magenta', label='False Positive'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    ax.set_title(f'Lesion Matching  '
                 f'(P={match_result["detection_precision"]:.2f}  '
                 f'R={match_result["detection_recall"]:.2f}  '
                 f'F1={match_result["detection_f1"]:.2f})')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_lesion_area_scatter(match_result: Dict, output_path: str):
    """Scatter: GT lesion area vs pred lesion area for matched pairs."""
    if not match_result['matched']:
        return
    gt_areas = [p['gt_area'] for p in match_result['matched']]
    pred_areas = [p['pred_area'] for p in match_result['matched']]
    ious = [p['iou'] for p in match_result['matched']]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sc = axes[0].scatter(gt_areas, pred_areas, c=ious, cmap='RdYlGn', vmin=0, vmax=1,
                         edgecolors='k', linewidth=0.3, s=40)
    lo = min(min(gt_areas), min(pred_areas)) * 0.9
    hi = max(max(gt_areas), max(pred_areas)) * 1.1
    axes[0].plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y=x')
    axes[0].set_xlabel('GT area (px)')
    axes[0].set_ylabel('Pred area (px)')
    axes[0].set_title('Total Lesion Area')
    axes[0].legend()
    fig.colorbar(sc, ax=axes[0], label='IoU')

    gt_ab = [p['gt_ablation'] for p in match_result['matched']]
    pred_ab = [p['pred_ablation'] for p in match_result['matched']]
    axes[1].scatter(gt_ab, pred_ab, c='#F44336', edgecolors='k', linewidth=0.3, s=40, alpha=0.7)
    lo_a = min(min(gt_ab), min(pred_ab)) * 0.9 if gt_ab else 0
    hi_a = max(max(gt_ab), max(pred_ab)) * 1.1 if gt_ab else 1
    axes[1].plot([lo_a, hi_a], [lo_a, hi_a], 'k--', alpha=0.4)
    axes[1].set_xlabel('GT ablation area (px)')
    axes[1].set_ylabel('Pred ablation area (px)')
    axes[1].set_title('Ablation Area')

    gt_cg = [p['gt_coag'] for p in match_result['matched']]
    pred_cg = [p['pred_coag'] for p in match_result['matched']]
    axes[2].scatter(gt_cg, pred_cg, c='#2196F3', edgecolors='k', linewidth=0.3, s=40, alpha=0.7)
    lo_c = min(min(gt_cg), min(pred_cg)) * 0.9 if gt_cg else 0
    hi_c = max(max(gt_cg), max(pred_cg)) * 1.1 if gt_cg else 1
    axes[2].plot([lo_c, hi_c], [lo_c, hi_c], 'k--', alpha=0.4)
    axes[2].set_xlabel('GT coagulation area (px)')
    axes[2].set_ylabel('Pred coagulation area (px)')
    axes[2].set_title('Coagulation Area')

    fig.suptitle('Matched Lesion Area Comparison (GT vs Pred)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

