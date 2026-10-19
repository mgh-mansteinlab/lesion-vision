#!/usr/bin/env python3
"""All round-3 statistics for the annotated punch (PVcont), from saved predictions.

For each prediction pipeline (and for the expert ground truth itself):
  1. pixel metrics, pooled over sections, per class and per depth band, with
     95% cluster-bootstrap CIs (sections resampled);
  2. object-level detection (one-to-one centroid matching) at several
     tolerances, Bland-Altman of ablation diameter and coagulation width, and
     median absolute coagulation-width error per band;
  3. the complete-field census (detections, lumen-bearing fraction,
     within-section CV, nearest-neighbour pitch, coag/ablation relation),
     computed identically on GT and on predictions so they can be compared.

GT lesion tables are produced with the same LesionAnalyzer.analyze_mask call
that wrote the prediction lesion_summary.csv files.

Depth = serial rank x 20 um over all 35 annotated sections (missing serials are
not accounted for). Bands: 0-220 um (12), 240-460 um (12), 480-680 um (11).

Usage:
  python scripts/analysis/r3_stats.py --out predictions/r3_stats.json \
      --pipeline frozen=predictions/r3_PVcont_report \
      --pipeline tta=predictions/r3_PVcont_tta_raw
"""
from __future__ import annotations

import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import argparse
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analytics.lesions import LesionAnalyzer  # noqa: E402
from src.constants import PIXEL_SCALE_UM  # noqa: E402
from src.eval.metrics import rgb_to_class_mask  # noqa: E402

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data' / 'LesionSegmentation'))

GT_DIR = DATA_ROOT / 'data/PVcont/masks'
CL = ['background', 'tissue', 'coagulation', 'ablation']
SECTION_UM = 20.0
BANDS = [('superficial', 0, 220), ('middle', 240, 460), ('deep', 480, 680)]
TOLS = (100.0, 150.0, 250.0)
N_BOOT = 2000


def gt_class_mask(rgb):
    """GT class map with anti-aliased boundary pixels resolved.

    The annotation PNGs are anti-aliased: pixels between two classes carry
    blended colours that rgb_to_class_mask maps to background, giving every
    lesion a thin false background rim. Pixels that are neither a class colour
    nor near-white are assigned the majority class of their classified 3x3
    neighbours, iterated until none remain. Only near-white stays background.
    """
    cls = rgb_to_class_mask(rgb)
    white = np.all(rgb > 240, axis=2)
    unk = (cls == 0) & ~white
    k = np.ones((3, 3), np.float32)
    for _ in range(20):
        if not unk.any():
            break
        known = ~unk
        counts = np.stack([cv2.filter2D(((cls == c) & known).astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
                           for c in range(4)])
        best = counts.argmax(0).astype(np.uint8)
        fill = unk & (counts.max(0) > 0)
        cls[fill] = best[fill]
        unk &= ~fill
    return cls


def order(s):
    return int(s[6:].split('_')[0]), s


def depth_of(sections):
    return {s: i * SECTION_UM for i, s in enumerate(sorted(sections, key=order))}


def band_of(d):
    for name, lo, hi in BANDS:
        if lo <= d <= hi:
            return name
    return None


# ---------------------------------------------------------------- GT lesions
def _gt_csv(args):
    name, out_root = args
    out = Path(out_root) / name / 'csv'
    if (out / 'lesion_summary.csv').exists():
        return name
    out.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(cv2.imread(str(GT_DIR / f'{name}.png')), cv2.COLOR_BGR2RGB)
    mask = gt_class_mask(rgb)
    LesionAnalyzer().analyze_mask(mask, None, str(out), summary_metrics=True, figures=False)
    if not (out / 'lesion_summary.csv').exists():  # no lesions: write an empty table
        pd.DataFrame(columns=['Lesion_ID', 'Coagulation_Width_um', 'Ablation_Diameter_um', 'Has_Ablation',
                              'Lesion_Diameter_um', 'Center_X_px', 'Center_Y_px', 'Touches_Edge']).to_csv(
            out / 'lesion_summary.csv', index=False)
    return name


def load_lesions(root, sections):
    rows = []
    for s in sections:
        p = Path(root) / s / 'csv' / 'lesion_summary.csv'
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:  # section without lesions
                continue
            if len(df):
                df['section_id'] = s
                rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df['touches_edge'] = df['Touches_Edge'].astype(str).str.lower().isin(['true', '1'])
    df['has_abl'] = df['Ablation_Diameter_um'].astype(float) > 0
    return df


# ------------------------------------------------------------ pixel metrics
def _confusion(args):
    name, pred_root = args
    gt = gt_class_mask(cv2.cvtColor(cv2.imread(str(GT_DIR / f'{name}.png')), cv2.COLOR_BGR2RGB))
    pr = rgb_to_class_mask(cv2.cvtColor(cv2.imread(f'{pred_root}/{name}/images/{name}_pred.png'), cv2.COLOR_BGR2RGB))
    if pr.shape != gt.shape:
        pr = cv2.resize(pr, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return name, np.array([[((pr == i) & (gt == i)).sum(), ((pr == i) & (gt != i)).sum(), ((pr != i) & (gt == i)).sum()]
                           for i in range(4)], dtype=np.int64)


def dice_from(conf):  # conf: (4, 3) tp fp fn
    tp, fp, fn = conf[:, 0], conf[:, 1], conf[:, 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        dice = 2 * tp / (2 * tp + fp + fn)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        iou = tp / (tp + fp + fn)
    return dice, prec, rec, iou


def pixel_block(confs, names, rng):
    C = np.stack([confs[n] for n in names])
    dice, prec, rec, iou = dice_from(C.sum(0))
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(names), len(names))
        d = dice_from(C[idx].sum(0))[0]
        boots.append(np.r_[np.nanmean(d), d])
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    out = {'n_sections': len(names), 'macro_dice': float(np.nanmean(dice)),
           'macro_dice_ci95': [float(lo[0]), float(hi[0])], 'macro_iou': float(np.nanmean(iou)), 'per_class': {}}
    for i, c in enumerate(CL):
        out['per_class'][c] = {'dice': float(dice[i]), 'dice_ci95': [float(lo[i + 1]), float(hi[i + 1])],
                               'precision': float(prec[i]), 'recall': float(rec[i]), 'iou': float(iou[i])}
    return out


# ------------------------------------------------------------- object level
def match(gt, pr, tol_um):
    """Greedy one-to-one nearest-centroid matching within tol_um."""
    if not len(gt) or not len(pr):
        return []
    g = gt[['Center_X_px', 'Center_Y_px']].to_numpy(float)
    p = pr[['Center_X_px', 'Center_Y_px']].to_numpy(float)
    d = np.sqrt(((g[:, None, :] - p[None, :, :]) ** 2).sum(-1)) * PIXEL_SCALE_UM
    pairs, used_g, used_p = [], set(), set()
    for k in np.argsort(d, axis=None):
        i, j = np.unravel_index(k, d.shape)
        if d[i, j] > tol_um:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j); pairs.append((i, j, d[i, j]))
    return pairs


def object_block(gt_les, pr_les, sections, depth):
    out = {}
    for tol in TOLS:
        res = {'all': [0, 0, 0]}
        res.update({b[0]: [0, 0, 0] for b in BANDS})
        for s in sections:
            g = gt_les[gt_les.section_id == s].reset_index(drop=True)
            p = pr_les[pr_les.section_id == s].reset_index(drop=True)
            m = len(match(g, p, tol))
            for key in ('all', band_of(depth[s])):
                res[key][0] += m; res[key][1] += len(g); res[key][2] += len(p)
        out[f'tol_{int(tol)}um'] = {k: {'matched': v[0], 'gt': v[1], 'pred': v[2],
                                        'recall': v[0] / v[1] if v[1] else None,
                                        'precision': v[0] / v[2] if v[2] else None} for k, v in res.items()}
    # paired morphometry at the 250 um tolerance
    rows = []
    for s in sections:
        g = gt_les[gt_les.section_id == s].reset_index(drop=True)
        p = pr_les[pr_les.section_id == s].reset_index(drop=True)
        for i, j, dist in match(g, p, 250.0):
            rows.append({'section_id': s, 'depth_um': depth[s], 'band': band_of(depth[s]), 'match_um': dist,
                         'gt_abl': g.Ablation_Diameter_um[i], 'pr_abl': p.Ablation_Diameter_um[j],
                         'gt_coag': g.Coagulation_Width_um[i], 'pr_coag': p.Coagulation_Width_um[j]})
    pairs = pd.DataFrame(rows)

    def ba(x, y):
        dlt = y - x
        md, sd = float(np.mean(dlt)), float(np.std(dlt, ddof=1))
        return {'n': int(len(dlt)), 'bias': md, 'loa': [md - 1.96 * sd, md + 1.96 * sd],
                'median_abs_err': float(np.median(np.abs(dlt))), 'pearson_r': float(stats.pearsonr(x, y)[0]) if len(x) > 2 else None}
    both_abl = pairs[(pairs.gt_abl > 0) & (pairs.pr_abl > 0)]
    out['bland_altman'] = {'ablation_diameter_um': ba(both_abl.gt_abl.to_numpy(float), both_abl.pr_abl.to_numpy(float)),
                           'coagulation_width_um': ba(pairs.gt_coag.to_numpy(float), pairs.pr_coag.to_numpy(float))}
    out['coag_median_abs_err_by_band_um'] = {b: float(np.median(np.abs(pairs[pairs.band == b].pr_coag - pairs[pairs.band == b].gt_coag)))
                                             for b, _, _ in BANDS if (pairs.band == b).any()}
    out['lumen_agreement'] = {'gt_lumen_pred_none': int(((pairs.gt_abl > 0) & (pairs.pr_abl == 0)).sum()),
                              'gt_none_pred_lumen': int(((pairs.gt_abl == 0) & (pairs.pr_abl > 0)).sum())}
    return out, pairs


# ------------------------------------------------------------------- census
def nn_um(df):
    c = df[['Center_X_px', 'Center_Y_px']].to_numpy(float)
    if len(c) < 2:
        return np.array([])
    d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
    return d.min(1) * PIXEL_SCALE_UM


def census(les, sections, depth):
    les = les.copy(); les['depth_um'] = les.section_id.map(depth)
    inter = les[~les.touches_edge]
    per_sec = les.groupby('section_id').size().reindex(sections, fill_value=0)
    dep = np.array([depth[s] for s in sections])
    lumen = les.groupby('section_id').has_abl.mean().reindex(sections)
    cv = []
    for s, part in inter.groupby('section_id'):
        a = part.Ablation_Diameter_um.to_numpy(float); a = a[a > 0]
        if len(a) >= 10:
            cv.append((depth[s], np.std(a, ddof=1) / np.mean(a)))
    cv = np.array(cv)
    sup = inter[inter.depth_um < 3 * SECTION_UM]; sup_a = sup[sup.has_abl]
    nn_sup = [np.median(nn_um(p)) for _, p in sup.groupby('section_id') if len(p) > 1]
    band_stats = {}
    for b, lo, hi in BANDS:
        ss = [s for s in sections if lo <= depth[s] <= hi]
        sub = les[les.section_id.isin(ss)]
        band_stats[b] = {'detections_per_section': float(per_sec[ss].mean()), 'lumen_fraction': float(sub.has_abl.mean()) if len(sub) else None,
                         'within_section_abl_cv_mean': float(cv[(cv[:, 0] >= lo) & (cv[:, 0] <= hi), 1].mean()) if len(cv) and ((cv[:, 0] >= lo) & (cv[:, 0] <= hi)).any() else None}
    valid = ~np.isnan(lumen.to_numpy(float))
    return {
        'detections': int(len(les)), 'interior': int(len(inter)),
        'detections_per_section_superficial3': float(per_sec.iloc[:3].mean()),
        'counts_vs_depth_spearman': dict(zip(('rho', 'p'), map(float, stats.spearmanr(dep, per_sec.to_numpy())))),
        'lumen_fraction_vs_depth_spearman': dict(zip(('rho', 'p'), map(float, stats.spearmanr(dep[valid], lumen.to_numpy(float)[valid])))),
        'within_section_abl_cv_vs_depth_spearman': (dict(zip(('rho', 'p'), map(float, stats.spearmanr(cv[:, 0], cv[:, 1])))) | {'n_sections': int(len(cv))}) if len(cv) > 2 else None,
        'superficial3_within_section_abl_cv': float(np.std(sup_a.Ablation_Diameter_um, ddof=1) / np.mean(sup_a.Ablation_Diameter_um)) if len(sup_a) > 2 else None,
        'superficial3_per_section_cv_range': [float(cv[cv[:, 0] < 3 * SECTION_UM, 1].min()), float(cv[cv[:, 0] < 3 * SECTION_UM, 1].max())] if len(cv) and (cv[:, 0] < 3 * SECTION_UM).any() else None,
        'superficial3_nn_pitch_um': float(np.median(nn_sup)) if nn_sup else None,
        'superficial3_coag_vs_abl_pearson': dict(zip(('r', 'p'), map(float, stats.pearsonr(sup_a.Ablation_Diameter_um, sup_a.Coagulation_Width_um)))) | {'n': int(len(sup_a))} if len(sup_a) > 2 else None,
        'superficial3_coag_over_abl_ratio': {'median': float(np.median(sup_a.Coagulation_Width_um / sup_a.Ablation_Diameter_um)),
                                             'iqr': list(map(float, np.percentile(sup_a.Coagulation_Width_um / sup_a.Ablation_Diameter_um, [25, 75])))} if len(sup_a) else None,
        'by_band': band_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', action='append', required=True, help='name=pred_root')
    ap.add_argument('--gt-out', type=Path, default=ROOT / 'predictions' / 'r3_PVcont_GT')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    pipes = dict(p.split('=', 1) for p in a.pipeline)
    sections = sorted({Path(p).stem for p in glob.glob(str(GT_DIR / '*.png'))}, key=order)
    depth = depth_of(sections)
    with Pool(12) as pool:
        pool.map(_gt_csv, [(s, str(a.gt_out)) for s in sections])
    gt_les = load_lesions(a.gt_out, sections)
    rng = np.random.default_rng(0)
    result = {'sections': sections, 'depth_um': depth, 'bands': BANDS, 'pixel_scale_um': PIXEL_SCALE_UM,
              'census': {'gt': census(gt_les, sections, depth)}, 'pipelines': {}}
    for name, root in pipes.items():
        with Pool(12) as pool:
            confs = dict(pool.map(_confusion, [(s, root) for s in sections]))
        pix = {'all': pixel_block(confs, sections, rng)}
        for b, lo, hi in BANDS:
            pix[b] = pixel_block(confs, [s for s in sections if lo <= depth[s] <= hi], rng)
        pr_les = load_lesions(root, sections)
        obj, pairs = object_block(gt_les, pr_les, sections, depth)
        pairs.to_csv(a.out.with_name(a.out.stem + f'_{name}_pairs.csv'), index=False)
        result['pipelines'][name] = {'pred_root': root, 'pixel': pix, 'object': obj}
        result['census'][name] = census(pr_les, sections, depth)
    a.out.write_text(json.dumps(result, indent=1, default=float))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
