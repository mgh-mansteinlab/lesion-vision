#!/usr/bin/env python
"""Save whole-section class probabilities (no TTA, no post-processing).

Same tissue window, tiling (512 px, 128 overlap), model and probability-space
blending as LesionPredictor, but the blended softmax is kept instead of being
reduced to a class mask. argmax of the saved probabilities equals the
'no TTA, no post-processing' prediction. Output per section:
  <out-dir>/<name>.npz with key 'p': uint8 (H, W, 4), probability x 255,
  background = 1 outside the tissue window.
Inputs: sections with masks in <data-dir>/masks/<name>.png and images in
<data-dir>/images/<name>.tif or <data-dir>/<name>.tif.

Usage:
  python scripts/predict_probs.py --data-dir <dir> --model <ckpt.pth> --out-dir <dir> --gpus 8
"""
import argparse
import glob
import os
import sys
from multiprocessing import get_context

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)


def find_sections(data_dir, images_only=False):
    if images_only:  # unannotated series: every <data-dir>/images/*.tif
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in sorted(glob.glob(os.path.join(data_dir, 'images', '*.tif')))]
    out = []
    for m in sorted(glob.glob(os.path.join(data_dir, 'masks', '*.png'))):
        name = os.path.splitext(os.path.basename(m))[0]
        for p in (os.path.join(data_dir, 'images', name + '.tif'), os.path.join(data_dir, name + '.tif')):
            if os.path.exists(p):
                out.append((name, p))
                break
    return out


def section_probs(pred, image, tile_size=512, overlap=128):
    import cv2
    from src.infer.tiles import TissueWindowConfig  # noqa: F401  (config lives on pred)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = pred._tissue_threshold(cv2.GaussianBlur(gray, (5, 5), 0))
    cropped, (x1, y1, x2, y2) = pred.get_tissue_bbox(image, threshold=threshold)
    h, w = cropped.shape[:2]
    acc = np.zeros((4, h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    weight = np.ones((tile_size, tile_size), np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        weight[:, :overlap] = ramp[None, :]; weight[:, -overlap:] = ramp[::-1][None, :]
        weight[:overlap, :] *= ramp[:, None]; weight[-overlap:, :] *= ramp[::-1][:, None]
    for tile, (y, x) in pred.get_tiles(cropped, tile_size, overlap):
        if pred._is_background_tile(tile, threshold):
            p = np.zeros((4, tile_size, tile_size), np.float32); p[0] = 1.0
        else:
            p = pred.predict_tile(tile)
        ph, pw = min(tile_size, h - y), min(tile_size, w - x)
        acc[:, y:y + ph, x:x + pw] += p[:, :ph, :pw] * weight[:ph, :pw]
        wsum[y:y + ph, x:x + pw] += weight[:ph, :pw]
    acc /= np.maximum(wsum, 1e-6)
    full = np.zeros(image.shape[:2] + (4,), np.uint8)
    full[..., 0] = 255
    full[y1:y2, x1:x2] = np.clip(np.round(np.moveaxis(acc, 0, -1) * 255), 0, 255).astype(np.uint8)
    return full


def worker(args):
    gpu, jobs, model, out_dir = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    from src.infer.predictor import LesionPredictor
    pred = LesionPredictor(model_path=model, device='cuda')
    done = []
    for name, path in jobs:
        out = os.path.join(out_dir, name + '.npz')
        if os.path.exists(out):
            done.append(name); continue
        image = pred._read_image(path)
        np.savez_compressed(out + '.tmp.npz', p=section_probs(pred, image))
        os.replace(out + '.tmp.npz', out)
        print(f'{name}: saved', flush=True)
        done.append(name)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--gpus', type=int, default=1)
    ap.add_argument('--images-only', action='store_true', help='predict every image in <data-dir>/images (no masks needed)')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    secs = find_sections(a.data_dir, a.images_only)
    shards = [secs[i::a.gpus] for i in range(a.gpus)]
    with get_context('spawn').Pool(a.gpus) as pool:
        res = pool.map(worker, [(g, s, a.model, a.out_dir) for g, s in enumerate(shards) if s])
    print(f'{sum(len(r) for r in res)} sections -> {a.out_dir}')


if __name__ == '__main__':
    main()
