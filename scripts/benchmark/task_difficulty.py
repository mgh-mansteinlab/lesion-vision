#!/usr/bin/env python
"""Task-difficulty demonstration: weak baselines vs trained models.

Lesion sub-typing (separating coagulation from ablation inside a thermal lesion
on H&E-like tissue scans) is hard. This script quantifies that by running
non-learned baselines on the *raw* tissue images and comparing them to ground
truth:

  - otsu_tissue   : Otsu threshold -> tissue vs background only (no sub-typing)
  - color_thresh  : HSV heuristics for reddish/bluish regions
  - kmeans_oracle : k=4 color clustering with oracle cluster->class assignment
                    (Hungarian on IoU) -- an upper bound for unsupervised color

Optionally, a trained model (--model best_model.pth) is added for contrast.

Outputs (under --out-dir):
  - task_difficulty.csv          per-method metrics
  - confusion_<method>.png       normalized 4x4 confusion matrix
  - task_difficulty_bars.png     lesion-Dice comparison bar chart
  - task_difficulty_report.md

Classes: 0=Background 1=Tissue 2=Coagulation 3=Ablation
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))

import argparse
import csv
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.eval_all_models import (
    CLASS_NAMES,
    DEFAULT_PAIRS,
    colour_png_to_class,
    compute_metrics,
    read_rgb,
)

N_CLASSES = 4


def _downscale(img, gt, long_side=1500):
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img, gt


def _tissue_mask(image):
    """Otsu foreground (non-white) mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th > 0


def baseline_otsu_tissue(image):
    """Everything in the tissue silhouette -> tissue (class 1); cannot sub-type."""
    pred = np.zeros(image.shape[:2], dtype=np.uint8)
    pred[_tissue_mask(image)] = 1
    return pred


def baseline_color_thresh(image):
    """HSV heuristics: reddish -> ablation, bluish/purple -> coagulation, rest tissue."""
    fg = _tissue_mask(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    pred = np.zeros(image.shape[:2], dtype=np.uint8)
    pred[fg] = 1
    reddish = fg & (s > 60) & ((h < 12) | (h > 168))
    bluish = fg & (s > 50) & (h > 110) & (h < 145)
    pred[bluish] = 2
    pred[reddish] = 3
    return pred


def baseline_kmeans_oracle(image, gt, k=N_CLASSES, seed=42):
    """k-means color clusters mapped to classes by oracle Hungarian IoU match."""
    h, w = image.shape[:2]
    samp = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(seed)
    _, labels, _ = cv2.kmeans(samp, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    clusters = labels.reshape(h, w)

    # Cost matrix: -IoU(cluster c, gt class j); maximize IoU via Hungarian.
    cost = np.zeros((k, N_CLASSES), dtype=np.float64)
    for c in range(k):
        cmask = clusters == c
        for j in range(N_CLASSES):
            gmask = gt == j
            inter = np.logical_and(cmask, gmask).sum(dtype=np.int64)
            union = np.logical_or(cmask, gmask).sum(dtype=np.int64)
            cost[c, j] = -(inter / union) if union > 0 else 0.0
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    pred = np.zeros((h, w), dtype=np.uint8)
    for c in range(k):
        pred[clusters == c] = mapping.get(c, 0)
    return pred


def confusion_matrix(pred, gt, n=N_CLASSES):
    idx = gt.reshape(-1).astype(np.int64) * n + pred.reshape(-1).astype(np.int64)
    cm = np.bincount(idx, minlength=n * n).reshape(n, n).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    return cm, np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)


def save_confusion(cm_norm, method, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right'); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Ground truth')
    ax.set_title(f'Confusion (row-normalized)\n{method}')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = os.path.join(out_dir, f'confusion_{method}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


def _trained_predict(model_path, image, tile_size, overlap):
    from src.prediction import LesionPredictor
    predictor = LesionPredictor(model_path)
    pred_crop, bbox = predictor._predict_image(
        image, tile_size=tile_size, overlap=overlap, use_tta=False, use_post_process=False)
    full = np.zeros(image.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    full[y1:y2, x1:x2] = pred_crop
    return full


def main():
    ap = argparse.ArgumentParser(description="Task-difficulty: weak baselines vs trained models")
    ap.add_argument(
        '--data-dir',
        default=os.environ.get('DATA_DIR'),
        help='Directory with image/mask pairs (or set DATA_DIR)',
    )
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'experiments', 'task_difficulty'))
    ap.add_argument('--model', default=None, help='optional trained best_model.pth for contrast')
    ap.add_argument('--tile-size', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--long-side', type=int, default=1500, help='downscale long side for speed')
    ap.add_argument('--max-images', type=int, default=None)
    args = ap.parse_args()
    if not args.data_dir:
        ap.error('--data-dir is required (or set DATA_DIR)')

    os.makedirs(args.out_dir, exist_ok=True)
    pairs = [(t, p) for t, p in DEFAULT_PAIRS
             if os.path.isfile(os.path.join(args.data_dir, t))
             and os.path.isfile(os.path.join(args.data_dir, p))]
    if args.max_images:
        pairs = pairs[:args.max_images]
    if not pairs:
        print("No image/GT pairs found.")
        return
    print(f"Task-difficulty study on {len(pairs)} image(s)")

    methods = {
        'otsu_tissue': lambda img, gt: baseline_otsu_tissue(img),
        'color_thresh': lambda img, gt: baseline_color_thresh(img),
        'kmeans_oracle': lambda img, gt: baseline_kmeans_oracle(img, gt),
    }
    if args.model:
        methods['trained'] = None  # handled specially below

    agg = {m: [] for m in methods}
    cm_accum = {m: np.zeros((N_CLASSES, N_CLASSES)) for m in methods}

    for tif, png in pairs:
        name = os.path.splitext(tif)[0]
        image = read_rgb(os.path.join(args.data_dir, tif))
        gt = colour_png_to_class(read_rgb(os.path.join(args.data_dir, png)))
        image_s, gt_s = _downscale(image, gt, args.long_side)
        print(f"  {name}: {image_s.shape[1]}x{image_s.shape[0]}")

        for m, fn in methods.items():
            if m == 'trained':
                pred = _trained_predict(args.model, image, args.tile_size, args.overlap)
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt_use = gt
            else:
                pred = fn(image_s, gt_s)
                gt_use = gt_s
            metrics = compute_metrics(pred, gt_use)
            agg[m].append(metrics)
            cm, _ = confusion_matrix(pred, gt_use)
            cm_accum[m] += cm
            print(f"    {m:14s} mIoU={metrics['mean_iou']:.3f} lesionDice={metrics['lesion_dice']:.3f}")

    # Aggregate + write CSV
    def nanmean(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float('nan')

    keys = ['pixel_acc', 'mean_iou', 'mean_dice', 'lesion_iou', 'lesion_dice',
            'iou_coagulation', 'iou_ablation']
    rows = []
    for m in methods:
        row = {'method': m, 'n_images': len(agg[m])}
        for k in keys:
            row[k] = nanmean([r.get(k) for r in agg[m]])
        rows.append(row)
    rows.sort(key=lambda r: -(r['lesion_dice'] if not np.isnan(r['lesion_dice']) else -1))

    csv_path = os.path.join(args.out_dir, 'task_difficulty.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'n_images'] + keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Confusion matrices (mean over images, row-normalized)
    cm_paths = {}
    for m in methods:
        rs = cm_accum[m].sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_accum[m], rs, out=np.zeros_like(cm_accum[m]), where=rs > 0)
        cm_paths[m] = save_confusion(cm_norm, m, args.out_dir)

    # Bar chart: lesion Dice per method
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r['method'] for r in rows]
    vals = [0 if np.isnan(r['lesion_dice']) else r['lesion_dice'] for r in rows]
    colors = ['seagreen' if n == 'trained' else 'lightcoral' for n in names]
    ax.bar(names, vals, color=colors, edgecolor='black')
    ax.set_ylabel('Lesion Dice (coag+ablation)')
    ax.set_title('Task difficulty: weak baselines vs trained model')
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    bars_path = os.path.join(args.out_dir, 'task_difficulty_bars.png')
    plt.savefig(bars_path, dpi=150, bbox_inches='tight'); plt.close(fig)

    # Markdown report
    md = os.path.join(args.out_dir, 'task_difficulty_report.md')
    with open(md, 'w') as f:
        f.write("# Task-difficulty demonstration\n\n")
        f.write("Lesion sub-typing on raw tissue scans, weak baselines vs trained model. "
                "`kmeans_oracle` uses oracle cluster->class assignment (an upper bound for "
                "unsupervised color clustering), yet still scores poorly on lesion sub-typing.\n\n")
        f.write("| Method | Lesion Dice | Lesion IoU | Coag IoU | Ablation IoU | mIoU | Pixel Acc |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['lesion_dice']:.4f} | {r['lesion_iou']:.4f} | "
                    f"{r['iou_coagulation']:.4f} | {r['iou_ablation']:.4f} | "
                    f"{r['mean_iou']:.4f} | {r['pixel_acc']:.4f} |\n")

    print(f"\nWrote: {csv_path}\nWrote: {bars_path}\nWrote: {md}")
    for m, p in cm_paths.items():
        print(f"Wrote: {p}")


if __name__ == '__main__':
    main()
