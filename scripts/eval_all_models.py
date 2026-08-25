#!/usr/bin/env python
"""
Evaluate every trained model (models/*/best_model.pth) against a small set of
ground-truth tif/png pairs and rank them by segmentation quality.

For each (model x image) we run the standard sliding-window predictor, build a
full-resolution class mask, convert the ground-truth colour PNG to class
indices, and compute per-class IoU / Dice plus pixel accuracy.

Work is sharded across the available GPUs with one worker process per GPU.
Raw model output is compared (post-processing and TTA disabled) so the ranking
reflects the models themselves rather than the shared post-processing rules.

Classes: 0=Background(white) 1=Tissue(green) 2=Coagulation(blue) 3=Ablation(red)
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))

import argparse
import glob
import json
import multiprocessing as mp
import sys
import traceback

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CLASS_NAMES = ['background', 'tissue', 'coagulation', 'ablation']
LESION_CLASSES = [2, 3]

# Default 5 test pairs (top-level data files that contain real lesions).
DEFAULT_PAIRS = [
    ('20H_02.tif', '20H_02.png'),
    ('RH_01.tif', 'RH_01.png'),
    ('CPBSV1_02.tif', 'CPBSV1_02.png'),
    ('cvb1_02.tif', 'cvb1_02.png'),
    ('CUA0_02.tif', 'CUA0_02.png'),
]

# Per-worker caches.
_DEVICE = None
_IMG_CACHE = {}
_GT_CACHE = {}


def colour_png_to_class(mask_rgb):
    """Convert an RGB ground-truth mask to a class-index map (matches dataset.py)."""
    h, w = mask_rgb.shape[:2]
    cm = np.zeros((h, w), dtype=np.uint8)
    white = (mask_rgb[..., 0] > 240) & (mask_rgb[..., 1] > 240) & (mask_rgb[..., 2] > 240)
    cm[white] = 0
    green = (mask_rgb[..., 1] > 200) & (mask_rgb[..., 0] < 100) & (mask_rgb[..., 2] < 100)
    cm[green] = 1
    blue = (mask_rgb[..., 2] > 200) & (mask_rgb[..., 0] < 100) & (mask_rgb[..., 1] < 100)
    cm[blue] = 2
    red = (mask_rgb[..., 0] > 200) & (mask_rgb[..., 1] < 100) & (mask_rgb[..., 2] < 100)
    cm[red] = 3
    return cm


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_metrics(pred, gt, n_classes=4):
    """Per-class IoU/Dice + pixel accuracy for one image."""
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    res = {}
    ious, dices = [], []
    for c in range(n_classes):
        p = pred == c
        g = gt == c
        inter = np.logical_and(p, g).sum(dtype=np.int64)
        psum = p.sum(dtype=np.int64)
        gsum = g.sum(dtype=np.int64)
        union = psum + gsum - inter
        present = gsum > 0
        iou = (inter / union) if union > 0 else np.nan
        dice = (2 * inter / (psum + gsum)) if (psum + gsum) > 0 else np.nan
        res[f'iou_{CLASS_NAMES[c]}'] = float(iou) if union > 0 else np.nan
        res[f'dice_{CLASS_NAMES[c]}'] = float(dice) if (psum + gsum) > 0 else np.nan
        res[f'gt_frac_{CLASS_NAMES[c]}'] = float(gsum / gt.size)
        if present and union > 0:
            ious.append(iou)
            dices.append(dice)
    res['pixel_acc'] = float((pred == gt).sum(dtype=np.int64) / gt.size)
    res['mean_iou'] = float(np.mean(ious)) if ious else np.nan
    res['mean_dice'] = float(np.mean(dices)) if dices else np.nan
    # lesion-only (coag + ablation) where present in GT
    les_iou, les_dice = [], []
    for c in LESION_CLASSES:
        if not np.isnan(res[f'iou_{CLASS_NAMES[c]}']) and (gt == c).sum() > 0:
            les_iou.append(res[f'iou_{CLASS_NAMES[c]}'])
            les_dice.append(res[f'dice_{CLASS_NAMES[c]}'])
    res['lesion_iou'] = float(np.mean(les_iou)) if les_iou else np.nan
    res['lesion_dice'] = float(np.mean(les_dice)) if les_dice else np.nan
    return res


def _init_worker(gpu_queue, data_dir, pairs):
    global _DEVICE
    import torch
    gpu_id = gpu_queue.get()
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        _DEVICE = f'cuda:{gpu_id}'
    else:
        _DEVICE = 'cpu'
    # Pre-load images + ground truth once per worker.
    for tif, png in pairs:
        name = os.path.splitext(tif)[0]
        _IMG_CACHE[name] = read_rgb(os.path.join(data_dir, tif))
        _GT_CACHE[name] = colour_png_to_class(read_rgb(os.path.join(data_dir, png)))
    print(f"[worker pid={os.getpid()}] device={_DEVICE} cached {len(_IMG_CACHE)} images", flush=True)


def _eval_one_model(args):
    model_dir, pairs, out_dir, tile_size, overlap = args
    import torch
    from src.prediction import LesionPredictor
    name = os.path.basename(model_dir)
    rows = []
    try:
        predictor = LesionPredictor(os.path.join(model_dir, 'best_model.pth'), device=_DEVICE)
    except Exception as e:
        traceback.print_exc()
        return {'model': name, 'error': str(e), 'rows': []}

    for tif, png in pairs:
        img_name = os.path.splitext(tif)[0]
        try:
            image = _IMG_CACHE[img_name]
            gt = _GT_CACHE[img_name]
            pred_crop, bbox = predictor._predict_image(
                image, tile_size=tile_size, overlap=overlap,
                use_tta=False, use_post_process=False,
            )
            full = np.zeros(image.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            full[y1:y2, x1:x2] = pred_crop
            if full.shape != gt.shape:
                full = cv2.resize(full, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            m = compute_metrics(full, gt)
            m['model'] = name
            m['image'] = img_name
            rows.append(m)
            # Save a downscaled coloured prediction for later visual inspection.
            cmap = {0: (255, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255)}
            small = cv2.resize(full, (full.shape[1] // 6, full.shape[0] // 6),
                               interpolation=cv2.INTER_NEAREST)
            col = np.zeros((*small.shape, 3), dtype=np.uint8)
            for k, v in cmap.items():
                col[small == k] = v
            od = os.path.join(out_dir, 'pred_masks', img_name)
            os.makedirs(od, exist_ok=True)
            cv2.imwrite(os.path.join(od, f'{name}.png'), col)
            print(f"[{name}] {img_name}: mIoU={m['mean_iou']:.3f} "
                  f"lesionDice={m['lesion_dice']:.3f} acc={m['pixel_acc']:.3f}", flush=True)
        except Exception as e:
            traceback.print_exc()
            rows.append({'model': name, 'image': img_name, 'error': str(e)})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'model': name, 'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--data-dir',
        default=os.environ.get('DATA_DIR'),
        help='Directory with image/mask pairs (or set DATA_DIR)',
    )
    ap.add_argument('--models-glob', default=os.path.join(REPO, 'models', '*'))
    ap.add_argument('--out-dir', default=os.path.join(REPO, 'experiments', 'model_eval'))
    ap.add_argument('--tile-size', type=int, default=512)
    ap.add_argument('--overlap', type=int, default=128)
    ap.add_argument('--gpus', type=int, default=None, help='number of GPUs to use')
    args = ap.parse_args()
    if not args.data_dir:
        ap.error('--data-dir is required (or set DATA_DIR)')

    import torch
    n_gpus = args.gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    n_gpus = max(1, n_gpus)

    os.makedirs(args.out_dir, exist_ok=True)

    model_dirs = []
    for d in sorted(glob.glob(args.models_glob)):
        if not os.path.isdir(d):
            continue
        if os.path.basename(d) == 'visualizations':
            continue
        if os.path.isfile(os.path.join(d, 'best_model.pth')):
            model_dirs.append(d)
    print(f"Found {len(model_dirs)} models, using {n_gpus} GPUs")

    pairs = [(t, p) for t, p in DEFAULT_PAIRS
             if os.path.isfile(os.path.join(args.data_dir, t))
             and os.path.isfile(os.path.join(args.data_dir, p))]
    print(f"Using {len(pairs)} image/GT pairs: {[t for t, _ in pairs]}")

    ctx = mp.get_context('spawn')
    gpu_queue = ctx.Queue()
    # Round-robin GPU ids; one slot per worker.
    for i in range(n_gpus):
        gpu_queue.put(i % max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1))

    tasks = [(d, pairs, args.out_dir, args.tile_size, args.overlap) for d in model_dirs]

    all_rows = []
    with ctx.Pool(processes=n_gpus, initializer=_init_worker,
                  initargs=(gpu_queue, args.data_dir, pairs)) as pool:
        for result in pool.imap_unordered(_eval_one_model, tasks):
            if result.get('error'):
                print(f"MODEL FAILED {result['model']}: {result['error']}", flush=True)
            all_rows.extend(result.get('rows', []))

    # Persist raw per-image rows.
    with open(os.path.join(args.out_dir, 'per_image_metrics.json'), 'w') as f:
        json.dump(all_rows, f, indent=2)

    # Aggregate per model.
    import csv
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_rows:
        if 'error' in r:
            continue
        by_model[r['model']].append(r)

    def nanmean(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float('nan')

    agg = []
    metric_keys = ['pixel_acc', 'mean_iou', 'mean_dice', 'lesion_iou', 'lesion_dice',
                   'iou_background', 'iou_tissue', 'iou_coagulation', 'iou_ablation',
                   'dice_background', 'dice_tissue', 'dice_coagulation', 'dice_ablation']
    for model, rows in by_model.items():
        row = {'model': model, 'n_images': len(rows)}
        for k in metric_keys:
            row[k] = nanmean([r.get(k) for r in rows])
        agg.append(row)
    agg.sort(key=lambda r: (-(r['mean_dice'] if not np.isnan(r['mean_dice']) else -1)))

    csv_path = os.path.join(args.out_dir, 'model_ranking.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'n_images'] + metric_keys)
        w.writeheader()
        for row in agg:
            w.writerow(row)

    print("\n==== RANKING (by mean Dice over 4 classes) ====")
    print(f"{'rank':>4} {'model':32s} {'mDice':>7} {'mIoU':>7} {'lesDice':>8} {'acc':>7}")
    for i, row in enumerate(agg, 1):
        print(f"{i:>4} {row['model']:32s} {row['mean_dice']:7.4f} {row['mean_iou']:7.4f} "
              f"{row['lesion_dice']:8.4f} {row['pixel_acc']:7.4f}")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {os.path.join(args.out_dir, 'per_image_metrics.json')}")


if __name__ == '__main__':
    main()
