#!/usr/bin/env python
"""Detailed prediction + ground-truth report over an image/mask dataset.

For every image it runs the full prediction pipeline (sliding-window inference,
post-processing, lesion analysis, radial mushrooming figures) via
``LesionPredictor.save_prediction``, then scores the resulting colored mask
against the paired ground-truth PNG (per-class IoU/Dice, pixel accuracy,
lesion-only metrics, class area fractions). Work is sharded one-image-per-GPU.

Dataset layout (auto-detected):
  <data-dir>/images/*.tif + <data-dir>/masks/*.png   (paired by basename)
  and/or  <data-dir>/*.tif with masks in <data-dir>/masks/

Outputs (under --out-dir):
  <name>/images/...                 detailed per-image artifacts
  per_image_metrics.csv / .json     raw scores
  summary.json                      aggregate stats
  metrics_per_class.png             per-class IoU/Dice bar chart
  per_image_ranking.png             per-image mIoU / lesion-Dice
  overlay_montage.png               grid of prediction overlays
  REPORT.md                         human-readable detailed report

Example:
  python scripts/predict_with_report.py \
      --data-dir /path/to/images \
      --model models/single_scale_448/best_model.pth
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))

import argparse
import csv
import glob
import json
import multiprocessing as mp
import sys
import time
import traceback
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.eval_all_models import CLASS_NAMES, colour_png_to_class, compute_metrics, read_rgb

DEFAULT_MODEL = os.path.join(REPO, 'models', 'single_scale_448', 'best_model.pth')

_DEVICE = None
_MODEL_PATH = None
_PREDICTOR = None
_CFG = {}


def discover_pairs(data_dir):
    """Find (image_path, mask_path) pairs. Masks live in <data-dir>/masks/."""
    mask_dir = os.path.join(data_dir, 'masks')
    masks = {}
    for p in glob.glob(os.path.join(mask_dir, '*.png')):
        masks[os.path.splitext(os.path.basename(p))[0]] = p
    images = sorted(glob.glob(os.path.join(data_dir, 'images', '*.tif')))
    images += sorted(glob.glob(os.path.join(data_dir, '*.tif')))
    pairs = []
    seen = set()
    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        if name in seen or '_sample_' in name:
            continue
        seen.add(name)
        if name in masks:
            pairs.append((name, img, masks[name]))
    return sorted(pairs)


def _init_worker(gpu_queue, model_path, cfg):
    global _DEVICE, _MODEL_PATH, _CFG
    import torch
    gid = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gid)
        _DEVICE = f'cuda:{gid}'
    else:
        _DEVICE = 'cpu'
    _MODEL_PATH = model_path
    _CFG = cfg
    print(f"[worker pid={os.getpid()}] device={_DEVICE}", flush=True)


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        from src.prediction import LesionPredictor
        _PREDICTOR = LesionPredictor(_MODEL_PATH, device=_DEVICE)
    return _PREDICTOR


def _process_one(task):
    name, img_path, mask_path, out_dir = task
    import torch
    try:
        predictor = _get_predictor()
        output_path = os.path.join(out_dir, name + '_pred.png')
        t0 = time.perf_counter()
        predictor.save_prediction(
            img_path, output_path,
            tile_size=_CFG['tile_size'], overlap=_CFG['overlap'],
            summary_metrics=_CFG['summary_metrics'], figures=_CFG['figures'],
            create_subdir=True, use_tta=_CFG['use_tta'],
            use_post_process=_CFG['post_process'],
        )
        latency = time.perf_counter() - t0

        pred_png = os.path.join(out_dir, name, 'images', f'{name}_pred.png')
        pred_rgb = read_rgb(pred_png)
        pred = colour_png_to_class(pred_rgb)
        gt = colour_png_to_class(read_rgb(mask_path))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        m = compute_metrics(pred, gt)
        m['name'] = name
        m['latency_s'] = latency
        for c in range(4):
            m[f'pred_frac_{CLASS_NAMES[c]}'] = float((pred == c).mean())
            m[f'gt_frac_{CLASS_NAMES[c]}'] = float((gt == c).mean())
        # Save a small overlay thumbnail for the montage.
        overlay = os.path.join(out_dir, name, 'images', f'{name}_overlay.png')
        if os.path.isfile(overlay):
            ov = cv2.imread(overlay)
            if ov is not None:
                h, w = ov.shape[:2]
                s = 360 / max(h, w)
                thumb = cv2.resize(ov, (max(1, int(w * s)), max(1, int(h * s))))
                tdir = os.path.join(out_dir, '_thumbs')
                os.makedirs(tdir, exist_ok=True)
                cv2.imwrite(os.path.join(tdir, f'{name}.png'), thumb)
        print(f"[{name}] mIoU={m['mean_iou']:.3f} mDice={m['mean_dice']:.3f} "
              f"lesionDice={m['lesion_dice']:.3f} acc={m['pixel_acc']:.3f} {latency:.1f}s", flush=True)
        return m
    except Exception as e:
        traceback.print_exc()
        return {'name': name, 'error': str(e)}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _nanmean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float('nan')


def _nanstd(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.std(vals)) if vals else float('nan')


def _make_figures(rows, out_dir):
    good = [r for r in rows if 'error' not in r]
    if not good:
        return
    # 1) per-class IoU/Dice bars
    classes = CLASS_NAMES
    iou_means = [_nanmean([r.get(f'iou_{c}') for r in good]) for c in classes]
    dice_means = [_nanmean([r.get(f'dice_{c}') for r in good]) for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, iou_means, 0.4, label='IoU', color='steelblue')
    ax.bar(x + 0.2, dice_means, 0.4, label='Dice', color='seagreen')
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1); ax.set_ylabel('Score'); ax.set_title('Mean per-class IoU / Dice vs ground truth')
    for i, (a, b) in enumerate(zip(iou_means, dice_means)):
        if not np.isnan(a): ax.text(i - 0.2, a + 0.02, f'{a:.2f}', ha='center', fontsize=8)
        if not np.isnan(b): ax.text(i + 0.2, b + 0.02, f'{b:.2f}', ha='center', fontsize=8)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'metrics_per_class.png'), dpi=150); plt.close(fig)

    # 2) per-image ranking
    sd = sorted(good, key=lambda r: (r.get('mean_dice') if r.get('mean_dice') is not None and not np.isnan(r.get('mean_dice')) else -1))
    names = [r['name'] for r in sd]
    mdice = [r.get('mean_dice', np.nan) for r in sd]
    ldice = [r.get('lesion_dice', np.nan) for r in sd]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(names))))
    y = np.arange(len(names))
    ax.barh(y - 0.2, mdice, 0.4, label='mean Dice', color='slateblue')
    ax.barh(y + 0.2, ldice, 0.4, label='lesion Dice', color='darkorange')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1); ax.set_xlabel('Dice'); ax.set_title('Per-image Dice (sorted)')
    ax.legend(loc='lower right'); ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'per_image_ranking.png'), dpi=150); plt.close(fig)

    # 3) overlay montage
    tdir = os.path.join(out_dir, '_thumbs')
    thumbs = sorted(glob.glob(os.path.join(tdir, '*.png')))
    if thumbs:
        n = len(thumbs); cols = min(6, n); rows_n = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.4, rows_n * 2.4))
        axes = np.atleast_1d(axes).ravel()
        for i, tp in enumerate(thumbs):
            im = cv2.cvtColor(cv2.imread(tp), cv2.COLOR_BGR2RGB)
            axes[i].imshow(im); axes[i].set_title(os.path.splitext(os.path.basename(tp))[0], fontsize=7)
            axes[i].axis('off')
        for j in range(n, len(axes)):
            axes[j].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'overlay_montage.png'), dpi=130); plt.close(fig)


def _write_report(rows, summary, cfg, out_dir):
    good = [r for r in rows if 'error' not in r]
    failed = [r for r in rows if 'error' in r]
    md = os.path.join(out_dir, 'REPORT.md')
    with open(md, 'w') as f:
        f.write("# Lesion Segmentation - Detailed Prediction Report\n\n")
        f.write(f"- **Dataset:** `{cfg['data_dir']}`\n")
        f.write(f"- **Model:** `{cfg['model']}` (backbone: {cfg['backbone']}, {cfg['params_m']:.1f}M params)\n")
        f.write(f"- **Images scored:** {len(good)} / {len(rows)}"
                + (f"  ({len(failed)} failed)" if failed else "") + "\n")
        f.write(f"- **Inference:** tile={cfg['tile_size']} overlap={cfg['overlap']} "
                f"TTA={cfg['use_tta']} post_process={cfg['post_process']}\n")
        f.write(f"- **Mean latency:** {summary['latency_s_mean']:.1f}s / image\n\n")

        f.write("## Overall accuracy vs ground truth\n\n")
        f.write("| Metric | Mean | Std |\n|---|---:|---:|\n")
        for label, key in [("Pixel accuracy", "pixel_acc"), ("Mean IoU (4-class)", "mean_iou"),
                           ("Mean Dice (4-class)", "mean_dice"), ("Lesion IoU (coag+abl)", "lesion_iou"),
                           ("Lesion Dice (coag+abl)", "lesion_dice")]:
            f.write(f"| {label} | {summary[key+'_mean']:.4f} | {summary[key+'_std']:.4f} |\n")
        f.write("\n### Per-class\n\n| Class | IoU mean | IoU std | Dice mean | Dice std |\n|---|---:|---:|---:|---:|\n")
        for c in CLASS_NAMES:
            f.write(f"| {c} | {summary[f'iou_{c}_mean']:.4f} | {summary[f'iou_{c}_std']:.4f} | "
                    f"{summary[f'dice_{c}_mean']:.4f} | {summary[f'dice_{c}_std']:.4f} |\n")

        f.write("\n![Per-class metrics](metrics_per_class.png)\n")
        f.write("\n![Per-image ranking](per_image_ranking.png)\n")
        if os.path.isfile(os.path.join(out_dir, 'overlay_montage.png')):
            f.write("\n![Overlay montage](overlay_montage.png)\n")

        f.write("\n## Per-image metrics\n\n")
        f.write("| Image | Pixel Acc | mIoU | mDice | Coag IoU | Abl IoU | Lesion Dice | Latency (s) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(good, key=lambda r: -(r.get('mean_dice') if r.get('mean_dice') is not None and not np.isnan(r.get('mean_dice')) else -1)):
            f.write(f"| {r['name']} | {r['pixel_acc']:.4f} | {r['mean_iou']:.4f} | {r['mean_dice']:.4f} | "
                    f"{r.get('iou_coagulation', float('nan')):.4f} | {r.get('iou_ablation', float('nan')):.4f} | "
                    f"{r.get('lesion_dice', float('nan')):.4f} | {r['latency_s']:.1f} |\n")

        if failed:
            f.write("\n## Failures\n\n")
            for r in failed:
                f.write(f"- `{r['name']}`: {r['error']}\n")

        f.write("\n## Per-image detailed artifacts\n\n")
        f.write("Each image has a folder `<name>/` containing: `images/<name>_pred.png` (colored mask), "
                "`images/<name>_overlay.png`, `images/<name>_analysis.png`, radial mushrooming diagrams, "
                "and `csv/` with lesion measurements.\n")
    return md


def main():
    ap = argparse.ArgumentParser(description="Detailed prediction + GT report over an image/mask dataset")
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--tile-size', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--gpus', type=int, default=None)
    ap.add_argument('--no-figures', action='store_true', help='skip per-image lesion figures (faster)')
    ap.add_argument('--no-post-process', action='store_true')
    ap.add_argument('--use-tta', action='store_true')
    ap.add_argument('--names', nargs='*', default=None, help='optional subset of image basenames')
    args = ap.parse_args()

    import torch
    n_gpus = args.gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    n_gpus = max(1, n_gpus)

    out_dir = args.out_dir or os.path.join(REPO, 'predictions',
                                           os.path.basename(os.path.normpath(args.data_dir)) + '_report')
    os.makedirs(out_dir, exist_ok=True)

    pairs = discover_pairs(args.data_dir)
    if args.names:
        want = set(args.names)
        pairs = [p for p in pairs if p[0] in want]
        missing = want - {p[0] for p in pairs}
        if missing:
            print(f"Warning: names not found: {sorted(missing)}")
    if not pairs:
        print(f"No image/mask pairs found under {args.data_dir}")
        return
    print(f"Found {len(pairs)} image/mask pairs; using {min(n_gpus, len(pairs))} GPU worker(s)")

    backbone, params_m = 'unknown', float('nan')
    cfg_path = os.path.join(os.path.dirname(args.model), 'config.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            backbone = json.load(fh).get('backbone', 'unknown')

    cfg = {
        'tile_size': args.tile_size, 'overlap': args.overlap,
        'summary_metrics': True, 'figures': not args.no_figures,
        'use_tta': args.use_tta, 'post_process': not args.no_post_process,
        'data_dir': args.data_dir, 'model': args.model, 'backbone': backbone,
    }

    n_workers = min(n_gpus, len(pairs))
    ctx = mp.get_context('spawn')
    gpu_queue = ctx.Queue()
    n_phys = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    for i in range(n_workers):
        gpu_queue.put(i % n_phys)

    tasks = [(name, img, mask, out_dir) for name, img, mask in pairs]
    rows = []
    with ctx.Pool(processes=n_workers, initializer=_init_worker,
                  initargs=(gpu_queue, args.model, cfg)) as pool:
        for r in pool.imap_unordered(_process_one, tasks):
            rows.append(r)

    # Param count (load once, cheap, on CPU/GPU0).
    try:
        from src.prediction import LesionPredictor
        p = LesionPredictor(args.model, device='cpu')
        net = getattr(p.model, 'model', p.model)
        params_m = sum(pp.numel() for pp in net.parameters()) / 1e6
        del p
    except Exception:
        pass
    cfg['params_m'] = params_m

    # Persist raw rows.
    with open(os.path.join(out_dir, 'per_image_metrics.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    good = [r for r in rows if 'error' not in r]
    if good:
        keys = [k for k in good[0].keys() if k != 'name']
        with open(os.path.join(out_dir, 'per_image_metrics.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + keys)
            w.writeheader()
            for r in good:
                w.writerow({k: r.get(k) for k in ['name'] + keys})

    # Aggregate summary.
    summary = {'n_images': len(rows), 'n_scored': len(good)}
    agg_keys = ['pixel_acc', 'mean_iou', 'mean_dice', 'lesion_iou', 'lesion_dice', 'latency_s'] + \
        [f'iou_{c}' for c in CLASS_NAMES] + [f'dice_{c}' for c in CLASS_NAMES]
    for k in agg_keys:
        summary[k + '_mean'] = _nanmean([r.get(k) for r in good])
        summary[k + '_std'] = _nanstd([r.get(k) for r in good])
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    _make_figures(rows, out_dir)
    md = _write_report(rows, summary, cfg, out_dir)

    print("\n==== SUMMARY ====")
    print(f"Scored {len(good)}/{len(rows)} images")
    print(f"  Pixel acc : {summary['pixel_acc_mean']:.4f} +/- {summary['pixel_acc_std']:.4f}")
    print(f"  Mean IoU  : {summary['mean_iou_mean']:.4f} +/- {summary['mean_iou_std']:.4f}")
    print(f"  Mean Dice : {summary['mean_dice_mean']:.4f} +/- {summary['mean_dice_std']:.4f}")
    print(f"  LesionDice: {summary['lesion_dice_mean']:.4f} +/- {summary['lesion_dice_std']:.4f}")
    print(f"\nReport: {md}")
    print(f"Outputs: {out_dir}")


if __name__ == '__main__':
    main()
