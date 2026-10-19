#!/usr/bin/env python3
"""Square-root inverse-frequency class weights from the TRAINING split only.

Uses the training code's own split (get_train_val_dataset) so the weights
never see validation or test punches, and counts every pixel of every
training mask tile (all tile sizes and shift grids, as the loader sees them),
classified with the loader's colour thresholds. Same formula and
normalisation as TiledLesionDataset.compute_class_weights:
w_c = sqrt(total / (n_classes * count_c)), divided by min(w).

Usage:
  python scripts/analysis/compute_class_weights.py --data-dir <tiles_dir> \
      --exclude-groups PVcont --out configs/class_weights_model2.json
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from src.data.dataset import get_train_val_dataset  # noqa: E402

CLASS_NAMES = ['background', 'tissue', 'coagulation', 'ablation']


def count(path):
    m = cv2.imread(path, cv2.IMREAD_COLOR)
    b, g, r = (m[..., i].astype(np.int16) for i in range(3))
    tis = (g > 200) & (r < 100) & (b < 100)
    coag = (b > 200) & (r < 100) & (g < 100)
    abl = (r > 200) & (g < 100) & (b < 100)
    n = m.shape[0] * m.shape[1]
    return np.array([n - tis.sum() - coag.sum() - abl.sum(), tis.sum(), coag.sum(), abl.sum()], np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--tile-sizes', type=int, nargs='+', default=[448, 768])
    ap.add_argument('--exclude-groups', nargs='*', default=['PVcont'])
    ap.add_argument('--val-split', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    train, val = get_train_val_dataset(a.data_dir, disk_tile_sizes=a.tile_sizes, split_level='punch',
                                       exclude_groups=a.exclude_groups, val_split=a.val_split, random_state=a.seed)
    with Pool(48) as pool:
        counts = np.sum(pool.map(count, train.mask_paths, chunksize=256), axis=0)
    total = counts.sum()
    w = np.sqrt(total / (len(CLASS_NAMES) * counts + 1e-6))
    w = w / w.min()
    out = {'method': 'sqrt_inv_freq', 'computed_on': 'training split only',
           'split': {'split_level': 'punch', 'exclude_groups': a.exclude_groups, 'val_split': a.val_split, 'seed': a.seed},
           'n_mask_tiles': len(train.mask_paths), 'tile_sizes': a.tile_sizes,
           'pixel_fraction': dict(zip(CLASS_NAMES, (counts / total).round(6).tolist())),
           'weights': [round(float(x), 4) for x in w], 'class_names': CLASS_NAMES}
    json.dump(out, open(a.out, 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
