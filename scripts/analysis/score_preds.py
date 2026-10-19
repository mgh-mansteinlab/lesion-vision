#!/usr/bin/env python3
"""Pooled per-class metrics and lesion-level counts for a folder of saved predictions.

Scores <pred-root>/<name>/images/<name>_pred.png against <gt-dir>/<name>.png
(colour masks, loader thresholds). Pixel metrics are pooled over all images
from the confusion counts. Lesion level: a predicted lesion (connected
coagulation+ablation component, >= 100 px) is "real" if >= 20% of it lies on
GT lesion, otherwise "phantom"; a GT lesion is "found" if >= 20% of it is
covered by predicted lesion.

Usage:
  python scripts/analysis/score_preds.py --pred-root predictions/r3_val_tta_pp \
      --gt-dir $LESION_DATA_ROOT/data_val_r3/masks [--out x.json]
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import argparse, glob, json, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from diff_maps import classes

CL = ['background', 'tissue', 'coagulation', 'ablation']
MIN_PX, OVERLAP = 100, 0.2


def score(job):
    name, pred_p, gt_p = job
    gt = classes(cv2.cvtColor(cv2.imread(str(gt_p)), cv2.COLOR_BGR2RGB))
    pr = classes(cv2.cvtColor(cv2.imread(str(pred_p)), cv2.COLOR_BGR2RGB))
    if pr.shape != gt.shape:
        pr = cv2.resize(pr, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
    conf = {c: [int(((pr == i) & (gt == i)).sum()), int(((pr == i) & (gt != i)).sum()), int(((pr != i) & (gt == i)).sum())]
            for i, c in enumerate(CL)}
    gl, pl = (gt >= 2).astype(np.uint8), (pr >= 2).astype(np.uint8)
    def comps(m, other):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        hit = tot = 0
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < MIN_PX:
                continue
            tot += 1
            hit += other[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i].mean() >= OVERLAP
        return int(hit), int(tot)
    real, n_pred = comps(pl, gl)
    found, n_gt = comps(gl, pl)
    return name, conf, {'pred_lesions': n_pred, 'real': real, 'phantom': n_pred - real, 'gt_lesions': n_gt, 'gt_found': found}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-root', type=Path, required=True)
    ap.add_argument('--gt-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--names', nargs='*', default=None)
    a = ap.parse_args()
    jobs = []
    for p in sorted(glob.glob(str(a.pred_root / '*' / 'images' / '*_pred.png'))):
        name = os.path.basename(p)[:-len('_pred.png')]
        if (a.names is None or name in a.names) and (a.gt_dir / f'{name}.png').exists():
            jobs.append((name, p, a.gt_dir / f'{name}.png'))
    with Pool(min(12, len(jobs))) as pool:
        res = pool.map(score, jobs)
    tot = {c: np.zeros(3) for c in CL}
    les = {k: 0 for k in res[0][2]}
    for _, conf, l in res:
        for c in CL:
            tot[c] += conf[c]
        for k in les:
            les[k] += l[k]
    pc = {}
    for c, (tp, fp, fn) in tot.items():
        pc[c] = {'dice': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float('nan'),
                 'precision': tp / (tp + fp) if tp + fp else float('nan'), 'recall': tp / (tp + fn) if tp + fn else float('nan')}
    out = {'pred_root': str(a.pred_root), 'n_images': len(res), 'macro_dice': float(np.mean([v['dice'] for v in pc.values()])),
           'per_class': pc, 'lesions': les, 'per_image': {n: {'confusion_tp_fp_fn': c, 'lesions': l} for n, c, l in res}}
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)
    print(f"{a.pred_root.name:22s} n={len(res)} macro {out['macro_dice']:.4f} | "
          + ' '.join(f"{c[:4]} {v['dice']:.3f}(P{v['precision']:.2f}/R{v['recall']:.2f})" for c, v in pc.items())
          + f" | lesions found {les['gt_found']}/{les['gt_lesions']}, phantom {les['phantom']}/{les['pred_lesions']}")


if __name__ == '__main__':
    main()
