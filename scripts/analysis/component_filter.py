#!/usr/bin/env python3
"""Object-level filter on predicted lesions, tuned on validation, applied once to test.

Input: probability files from scripts/predict_probs.py (or ensemble_probs.py).
Base prediction = argmax. Each predicted lesion (8-connected coagulation +
ablation component) gets three features:
  eq_diam_um      equivalent diameter
  mean_prob       mean of p(coagulation) + p(ablation) over the component
                  (for an ensemble this is the cross-model agreement)
  edge_um         minimum distance to the predicted tissue boundary
Rule: keep a component if eq_diam_um >= 80 (fixed in advance from training GT)
AND mean_prob >= tau AND NOT (edge_um < d AND eq_diam_um < 150); removed
components become tissue. tau in {0, .5, .6, .7, .8, .9, .95}, d in {0, 50, 100, 150}
um are chosen on VALIDATION by lesion-detection F1 (a predicted lesion is real
if >= 20% of it lies on GT lesion; a GT lesion is found if >= 20% is covered),
ties broken by macro-Dice, then by the simpler rule (smaller tau, then d).

Usage:
  tune:  python scripts/analysis/component_filter.py tune --probs-dir <val probs> --gt-dir <val masks> --out params.json
  apply: python scripts/analysis/component_filter.py apply --probs-dir <test probs> --params params.json --out-dir <pred root>
"""
from __future__ import annotations

import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import argparse
import glob
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from src.eval.metrics import gt_class_mask  # noqa: E402

UM = 0.8832
MIN_DIAM, EDGE_SMALL = 80.0, 150.0
TAUS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DS = (0.0, 50.0, 100.0, 150.0)
OVERLAP = 0.2
COLORS = np.array([[255, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0]], np.uint8)


RIM_UM = 10.0  # minimum coagulation rim: 90% of expert train/val lesions keep >= 10 um all round
DECODE = 'argmax'
INTERIOR_BG = False  # step 0 of nested decoding; chosen on validation by greedy_ensemble.py
FILL_CRATER = False  # step 2b: coagulation enclosed by crater becomes crater; chosen on validation (r3_fill_crater.py)


def nested_decode(p, rim_um=RIM_UM):
    """Biologically nested decoding: tissue > coagulation > ablation.

    1. lesion = p(coag)+p(abl) > p(bg)+p(tissue); enclosed non-lesion holes are lesion
    2. ablation = lesion & p(abl) > p(coag)
    3. containment: ablation closer than rim_um to the lesion edge becomes
       coagulation, so every crater has a complete cuff of at least rim_um
    Non-lesion pixels are tissue or background by the larger probability.
    """
    from scipy import ndimage
    p = p.astype(np.int16)
    if INTERIOR_BG:
        # 0. Inside the section, background may not touch a crater (expert
        # annotation: lesion boundary touches interior background 0.1-0.9%).
        # Background components inside the section outline that touch a
        # provisional crater are re-decoded among the non-background classes,
        # so an empty crater read as background becomes crater (with its rim)
        # and a gap in tissue becomes tissue. Off by default.
        am = p.argmax(-1)
        env = ndimage.binary_fill_holes(cv2.morphologyEx((am >= 1).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)))
        bg_in = (am == 0) & env
        les0 = (p[..., 2] + p[..., 3]) > (p[..., 0] + p[..., 1])
        n, lab = cv2.connectedComponents(bg_in.astype(np.uint8), connectivity=8)
        if n > 1:
            crat0 = les0 & (p[..., 3] > p[..., 2])
            touch = np.unique(lab[cv2.dilate(crat0.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & bg_in])
            sel = np.isin(lab, touch[touch > 0])
            p = p.copy(); p[sel, 0] = -1
    les = (p[..., 2] + p[..., 3]) > (p[..., 0] + p[..., 1])
    les = ndimage.binary_fill_holes(les)
    mask = np.where(p[..., 1] >= p[..., 0], 1, 0).astype(np.uint8)  # p0 = -1 on re-decoded components -> tissue unless lesion
    mask[les] = 2
    abl = les & (p[..., 3] > p[..., 2])
    if FILL_CRATER:
        # 2b. coagulation lines the crater from outside and cannot lie inside it:
        # coagulation islands fully enclosed by crater are crater
        abl = ndimage.binary_fill_holes(abl) & les
    d_out = cv2.distanceTransform(les.astype(np.uint8), cv2.DIST_L2, 5) * UM
    mask[abl & (d_out > rim_um)] = 3
    return mask


def components(p):
    mask = nested_decode(p) if DECODE == 'nested' else p.argmax(-1).astype(np.uint8)
    les = (mask >= 2).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(les, connectivity=8)
    lp = (p[..., 2].astype(np.float32) + p[..., 3]) / 255.0
    edge = cv2.distanceTransform((mask >= 1).astype(np.uint8), cv2.DIST_L2, 5) * UM
    feats = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        sl = (slice(y, y + h), slice(x, x + w)); c = lab[sl] == i
        feats.append({'label': i, 'eq_diam_um': 2 * np.sqrt(a / np.pi) * UM, 'mean_prob': float(lp[sl][c].mean()),
                      'edge_um': float(edge[sl][c].min()), 'bbox': (x, y, w, h)})
    return mask, lab, feats


def keep(f, tau, d):
    return f['eq_diam_um'] >= MIN_DIAM and f['mean_prob'] >= tau and not (f['edge_um'] < d and f['eq_diam_um'] < EDGE_SMALL)


def prep_val(args):
    """Per-section tables so every grid point is scored without re-reading images."""
    name, probs, gt_path = args
    os.nice(19)
    p = np.load(probs)['p']
    mask, lab, feats = components(p)
    gt = gt_class_mask(cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB))
    if gt.shape != mask.shape:
        gt = cv2.resize(gt, mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
    base = np.zeros((4, 4), np.int64)  # rows pred, cols gt
    np.add.at(base, (mask.ravel(), gt.ravel()), 1)
    gl = (gt >= 2)
    ng, glab, gst, _ = cv2.connectedComponentsWithStats(gl.astype(np.uint8), connectivity=8)
    for f in feats:
        x, y, w, h = f['bbox']; sl = (slice(y, y + h), slice(x, x + w)); c = lab[sl] == f['label']
        conf = np.zeros((4, 4), np.int64)
        np.add.at(conf, (mask[sl][c], gt[sl][c]), 1)
        f['conf'] = conf
        f['real'] = bool(gl[sl][c].mean() >= OVERLAP)
    gts = []  # per GT lesion: area and overlap with each predicted component
    for j in range(1, ng):
        x, y, w, h, a = gst[j]
        if a < 100:
            continue
        sl = (slice(y, y + h), slice(x, x + w)); c = glab[sl] == j
        ids, cnt = np.unique(lab[sl][c], return_counts=True)
        gts.append({'area': int(a), 'ov': {int(i): int(k) for i, k in zip(ids, cnt) if i > 0}})
    for f in feats:
        f.pop('bbox')
    return {'name': name, 'base': base, 'feats': feats, 'gts': gts}


def score(tables, tau, d):
    conf = np.zeros((4, 4), np.int64); tp_pred = n_pred = found = n_gt = 0
    for t in tables:
        c = t['base'].copy(); kept = set()
        for f in t['feats']:
            if keep(f, tau, d):
                kept.add(f['label'])
                n_pred += 1; tp_pred += f['real']
            else:  # removed component: its pixels become tissue
                c[1] += f['conf'][2] + f['conf'][3]; c[2] -= f['conf'][2]; c[3] -= f['conf'][3]
        conf += c
        for g in t['gts']:
            n_gt += 1
            found += sum(v for k, v in g['ov'].items() if k in kept) >= OVERLAP * g['area']
    prec = tp_pred / n_pred if n_pred else 0.0; rec = found / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tp = np.diag(conf); fp = conf.sum(1) - tp; fn = conf.sum(0) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        dice = 2 * tp / (2 * tp + fp + fn)
    return {'tau': tau, 'd_um': d, 'f1': f1, 'precision': prec, 'recall': rec, 'macro_dice': float(np.nanmean(dice)),
            'n_pred': n_pred, 'n_gt': n_gt}


def apply_one(args):
    name, probs, out_dir, tau, d = args
    os.nice(19)
    from src.analytics.lesions import LesionAnalyzer
    p = np.load(probs)['p']
    mask, lab, feats = components(p)
    drop = np.zeros(len(feats) + 1, bool)
    for f in feats:
        drop[f['label']] = not keep(f, tau, d)
    mask[drop[lab] & (mask >= 2)] = 1
    dd = Path(out_dir) / name
    (dd / 'images').mkdir(parents=True, exist_ok=True); (dd / 'csv').mkdir(exist_ok=True)
    cv2.imwrite(str(dd / 'images' / f'{name}_pred.png'), cv2.cvtColor(COLORS[mask], cv2.COLOR_RGB2BGR))
    LesionAnalyzer().analyze_mask(mask, None, str(dd / 'csv'), summary_metrics=True, figures=False)
    return name, int(drop.sum()), len(feats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['tune', 'apply'])
    ap.add_argument('--probs-dir', required=True)
    ap.add_argument('--gt-dir')
    ap.add_argument('--params')
    ap.add_argument('--out')
    ap.add_argument('--out-dir')
    ap.add_argument('--raw', action='store_true', help='apply: no filtering (tau=0, d=0, no size rule) to write the argmax baseline')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--interior-bg', action='store_true',
                    help='nested decoding step 0: re-decode interior background that touches a crater')
    ap.add_argument('--fill-crater', action='store_true',
                    help='nested decoding step 2b: coagulation enclosed by crater becomes crater')
    ap.add_argument('--decode', choices=['argmax', 'nested'], default='argmax',
                    help="'nested': lesion vs non-lesion, then crater, with a >= RIM_UM coagulation rim")
    a = ap.parse_args()
    global MIN_DIAM, DECODE, INTERIOR_BG, FILL_CRATER
    DECODE = a.decode
    INTERIOR_BG = a.interior_bg
    FILL_CRATER = a.fill_crater
    os.nice(19)
    probs = sorted(glob.glob(os.path.join(a.probs_dir, '*.npz')))
    if a.mode == 'tune':
        jobs = [(Path(p).stem, p, os.path.join(a.gt_dir, Path(p).stem + '.png')) for p in probs
                if os.path.exists(os.path.join(a.gt_dir, Path(p).stem + '.png'))]
        with Pool(min(a.workers, len(jobs))) as pool:
            tables = pool.map(prep_val, jobs)
        grid = [score(tables, t, d) for t, d in itertools.product(TAUS, DS)]
        best = max(grid, key=lambda r: (round(r['f1'], 6), round(r['macro_dice'], 6), -r['tau'], -r['d_um']))
        out = {'chosen': best, 'grid': grid, 'decode': a.decode, 'interior_bg': a.interior_bg, 'fill_crater': a.fill_crater, 'rim_um': RIM_UM if a.decode == 'nested' else None, 'rule': f'keep if eqD>={MIN_DIAM} and mean_prob>=tau and not (edge<d and eqD<{EDGE_SMALL})',
               'n_val_sections': len(tables), 'probs_dir': a.probs_dir}
        json.dump(out, open(a.out, 'w'), indent=1)
        print(f"chosen tau={best['tau']} d={best['d_um']} | val F1 {best['f1']:.3f} (P {best['precision']:.3f}, R {best['recall']:.3f}), macro-Dice {best['macro_dice']:.4f}")
        base = [g for g in grid if g['tau'] == 0.0 and g['d_um'] == 0.0][0]
        print(f"size rule only (tau=0, d=0): F1 {base['f1']:.3f} (P {base['precision']:.3f}, R {base['recall']:.3f}), macro-Dice {base['macro_dice']:.4f}")
    else:
        if a.raw:
            MIN_DIAM = 0.0
            tau, d = 0.0, 0.0
        else:
            prm = json.load(open(a.params))['chosen']; tau, d = prm['tau'], prm['d_um']
        jobs = [(Path(p).stem, p, a.out_dir, tau, d) for p in probs]
        with Pool(min(a.workers, len(jobs))) as pool:
            res = pool.map(apply_one, jobs)
        json.dump({'tau': tau, 'd_um': d, 'raw': a.raw, 'decode': a.decode, 'removed': sum(r[1] for r in res), 'components': sum(r[2] for r in res)},
                  open(os.path.join(a.out_dir, 'filter_applied.json'), 'w'), indent=1)
        print(f"applied tau={tau} d={d}: removed {sum(r[1] for r in res)} of {sum(r[2] for r in res)} components")


if __name__ == '__main__':
    main()
