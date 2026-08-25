#!/usr/bin/env python
"""Multi-architecture benchmark over a fixed ground-truth set.

Standardizes the comparison started in ``scripts/eval_all_models.py`` and adds,
for every trained model (and optionally SAM3):
  - per-class IoU/Dice, mean IoU/Dice, lesion-only IoU/Dice, pixel accuracy
  - backbone architecture (from config.json) and parameter count (millions)
  - mean inference latency per image (seconds)

Outputs (under --out-dir):
  - benchmark.csv            one row per model, ranked by mean Dice
  - benchmark_per_image.json raw per-(model, image) rows
  - benchmark_report.md      human-readable ranked table

Work is sharded one worker per GPU. Raw model output is compared (TTA and
post-processing disabled) so the ranking reflects the architectures themselves.

Examples:
    python scripts/benchmark/run_benchmark.py --data-dir /path/to/data
    python scripts/benchmark/run_benchmark.py --max-models 2 --max-images 1 --gpus 1
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.eval_all_models import (
    DEFAULT_PAIRS,
    colour_png_to_class,
    compute_metrics,
    read_rgb,
)

_DEVICE = None
_IMG_CACHE = {}
_GT_CACHE = {}

METRIC_KEYS = [
    'pixel_acc', 'mean_iou', 'mean_dice', 'lesion_iou', 'lesion_dice',
    'iou_background', 'iou_tissue', 'iou_coagulation', 'iou_ablation',
    'dice_background', 'dice_tissue', 'dice_coagulation', 'dice_ablation',
]


def _model_meta(model_dir):
    """Read backbone + n_classes from the model's config.json (best effort)."""
    cfg_path = os.path.join(model_dir, 'config.json')
    backbone, n_classes = 'unknown', 4
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            backbone = cfg.get('backbone', 'unknown')
            n_classes = cfg.get('n_classes', 4)
        except Exception:
            pass
    return backbone, n_classes


def _init_worker(gpu_queue, data_dir, pairs):
    global _DEVICE
    import torch
    gpu_id = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        _DEVICE = f'cuda:{gpu_id}'
    else:
        _DEVICE = 'cpu'
    for tif, png in pairs:
        name = os.path.splitext(tif)[0]
        _IMG_CACHE[name] = read_rgb(os.path.join(data_dir, tif))
        _GT_CACHE[name] = colour_png_to_class(read_rgb(os.path.join(data_dir, png)))
    print(f"[worker pid={os.getpid()}] device={_DEVICE} cached {len(_IMG_CACHE)} images", flush=True)


def _bench_one_model(args):
    model_dir, pairs, tile_size, overlap = args
    import torch
    from src.prediction import LesionPredictor

    name = os.path.basename(model_dir)
    backbone, _ = _model_meta(model_dir)
    rows = []
    try:
        predictor = LesionPredictor(os.path.join(model_dir, 'best_model.pth'), device=_DEVICE)
        net = getattr(predictor.model, 'model', predictor.model)
        params_m = sum(p.numel() for p in net.parameters()) / 1e6
    except Exception as e:
        traceback.print_exc()
        return {'model': name, 'backbone': backbone, 'error': str(e), 'rows': []}

    for tif, png in pairs:
        img_name = os.path.splitext(tif)[0]
        try:
            image = _IMG_CACHE[img_name]
            gt = _GT_CACHE[img_name]
            t0 = time.perf_counter()
            pred_crop, bbox = predictor._predict_image(
                image, tile_size=tile_size, overlap=overlap,
                use_tta=False, use_post_process=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0

            full = np.zeros(image.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            full[y1:y2, x1:x2] = pred_crop
            if full.shape != gt.shape:
                full = cv2.resize(full, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            m = compute_metrics(full, gt)
            m.update({'model': name, 'backbone': backbone, 'image': img_name,
                      'params_m': params_m, 'latency_s': latency})
            rows.append(m)
            print(f"[{name}|{backbone}] {img_name}: mDice={m['mean_dice']:.3f} "
                  f"lesDice={m['lesion_dice']:.3f} {latency:.1f}s {params_m:.1f}M", flush=True)
        except Exception as e:
            traceback.print_exc()
            rows.append({'model': name, 'backbone': backbone, 'image': img_name, 'error': str(e)})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'model': name, 'backbone': backbone, 'rows': rows}


def _nanmean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float('nan')


def _aggregate(all_rows):
    by_model = defaultdict(list)
    for r in all_rows:
        if 'error' in r:
            continue
        by_model[r['model']].append(r)

    agg = []
    for model, rows in by_model.items():
        row = {
            'model': model,
            'backbone': rows[0].get('backbone', 'unknown'),
            'params_m': rows[0].get('params_m', float('nan')),
            'n_images': len(rows),
            'latency_s': _nanmean([r.get('latency_s') for r in rows]),
        }
        for k in METRIC_KEYS:
            row[k] = _nanmean([r.get(k) for r in rows])
        agg.append(row)
    agg.sort(key=lambda r: -(r['mean_dice'] if not np.isnan(r['mean_dice']) else -1))
    return agg


def _write_report(agg, out_dir):
    md = os.path.join(out_dir, 'benchmark_report.md')
    with open(md, 'w') as f:
        f.write("# Multi-architecture benchmark\n\n")
        f.write("Raw model output (no TTA, no post-processing), ranked by mean Dice "
                "over 4 classes (background/tissue/coagulation/ablation).\n\n")
        f.write("| Rank | Model | Backbone | Params (M) | mDice | mIoU | "
                "Lesion Dice | Lesion IoU | Pixel Acc | Latency (s) |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, r in enumerate(agg, 1):
            f.write(f"| {i} | {r['model']} | {r['backbone']} | {r['params_m']:.1f} | "
                    f"{r['mean_dice']:.4f} | {r['mean_iou']:.4f} | "
                    f"{r['lesion_dice']:.4f} | {r['lesion_iou']:.4f} | "
                    f"{r['pixel_acc']:.4f} | {r['latency_s']:.2f} |\n")
    return md


def main():
    ap = argparse.ArgumentParser(description="Multi-architecture segmentation benchmark")
    ap.add_argument(
        '--data-dir',
        default=os.environ.get('DATA_DIR'),
        help='Directory with image/mask pairs (or set DATA_DIR)',
    )
    ap.add_argument('--models-glob', default=os.path.join(REPO, 'models', '*'))
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'experiments', 'benchmark'))
    ap.add_argument('--tile-size', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--gpus', type=int, default=None)
    ap.add_argument('--max-models', type=int, default=None, help='limit models (debug)')
    ap.add_argument('--max-images', type=int, default=None, help='limit images (debug)')
    args = ap.parse_args()
    if not args.data_dir:
        ap.error('--data-dir is required (or set DATA_DIR)')

    import torch
    n_gpus = args.gpus if args.gpus is not None else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    n_gpus = max(1, n_gpus)
    os.makedirs(args.out_dir, exist_ok=True)

    model_dirs = []
    for d in sorted(glob.glob(args.models_glob)):
        if not os.path.isdir(d) or os.path.basename(d) == 'visualizations':
            continue
        if os.path.isfile(os.path.join(d, 'best_model.pth')):
            model_dirs.append(d)
    if args.max_models:
        model_dirs = model_dirs[:args.max_models]

    pairs = [(t, p) for t, p in DEFAULT_PAIRS
             if os.path.isfile(os.path.join(args.data_dir, t))
             and os.path.isfile(os.path.join(args.data_dir, p))]
    if args.max_images:
        pairs = pairs[:args.max_images]

    print(f"Benchmarking {len(model_dirs)} models on {len(pairs)} images using {n_gpus} GPU worker(s)")
    if not model_dirs or not pairs:
        print("Nothing to benchmark (no models or no image/GT pairs found).")
        return

    ctx = mp.get_context('spawn')
    gpu_queue = ctx.Queue()
    n_phys = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    for i in range(n_gpus):
        gpu_queue.put(i % n_phys)

    tasks = [(d, pairs, args.tile_size, args.overlap) for d in model_dirs]
    all_rows = []
    with ctx.Pool(processes=n_gpus, initializer=_init_worker,
                  initargs=(gpu_queue, args.data_dir, pairs)) as pool:
        for result in pool.imap_unordered(_bench_one_model, tasks):
            if result.get('error'):
                print(f"MODEL FAILED {result['model']}: {result['error']}", flush=True)
            all_rows.extend(result.get('rows', []))

    with open(os.path.join(args.out_dir, 'benchmark_per_image.json'), 'w') as f:
        json.dump(all_rows, f, indent=2)

    agg = _aggregate(all_rows)
    csv_path = os.path.join(args.out_dir, 'benchmark.csv')
    fieldnames = ['model', 'backbone', 'params_m', 'n_images', 'latency_s'] + METRIC_KEYS
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in agg:
            w.writerow({k: row.get(k) for k in fieldnames})

    md = _write_report(agg, args.out_dir)

    print("\n==== BENCHMARK (by mean Dice) ====")
    print(f"{'rank':>4} {'model':28s} {'backbone':22s} {'params':>7} {'mDice':>7} {'lesDice':>8} {'lat(s)':>7}")
    for i, r in enumerate(agg, 1):
        print(f"{i:>4} {r['model']:28s} {r['backbone']:22s} {r['params_m']:7.1f} "
              f"{r['mean_dice']:7.4f} {r['lesion_dice']:8.4f} {r['latency_s']:7.2f}")
    print(f"\nWrote: {csv_path}\nWrote: {md}")


if __name__ == '__main__':
    main()
