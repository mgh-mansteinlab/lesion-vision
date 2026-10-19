#!/usr/bin/env python3
"""CPU-only extended report on round-3 model-1 predictions vs expert ground truth.

Runs at low priority (nice 19, ionice idle) on a bounded worker pool so it can
share the machine with GPU training. Inputs are saved predictions only.

Sections (PVcont, 35 serials, frozen and TTA pipelines):
  1. per-section pixel metrics vs depth
  2. boundary metrics (HD95, ASSD, boundary F1 at 5 um) per class
  3. lesion level: per-lesion IoU, size-stratified recall, error decomposition
  4. phantom-lesion profile (size, lumen, edge distance, stain intensity)
  5. per-section census agreement GT vs model (Lin's CCC)
  6. Major 2: section size (shrinkage) and lesion pitch/density from expert GT
     on all annotated known-scale slides
  7. contact sheet of all sections
Outputs: <out>/REPORT.md, figures/*.png, tables/*.csv, report.json

Usage:
  nice -n 19 ionice -c3 python scripts/analysis/r3_cpu_report.py --workers 32 \
      --out predictions/r3_cpu_report
"""
from __future__ import annotations

import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import argparse
import csv
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from r3_stats import gt_class_mask, order, BANDS, band_of, SECTION_UM  # noqa: E402
from src.eval.metrics import rgb_to_class_mask  # noqa: E402

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data' / 'LesionSegmentation'))

UM = 0.8832
PV = DATA_ROOT / 'data/PVcont'
SRC = DATA_ROOT / 'data_tiling_sources'
PIPES = {'frozen': ROOT / 'predictions' / 'r3_PVcont_report', 'tta': ROOT / 'predictions' / 'r3_PVcont_tta_raw'}
CL = ['background', 'tissue', 'coagulation', 'ablation']
BF_TOL_UM = 5.0
COLORS = np.array([[255, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0]], np.uint8)


def low_priority():
    os.nice(19)


def img_path(name):
    for p in (PV / 'images' / f'{name}.tif', PV / f'{name}.tif'):
        if p.exists():
            return p


def read_pred(pipe, name, shape):
    pr = rgb_to_class_mask(cv2.cvtColor(cv2.imread(str(PIPES[pipe] / name / 'images' / f'{name}_pred.png')), cv2.COLOR_BGR2RGB))
    if pr.shape != shape:
        pr = cv2.resize(pr, shape[::-1], interpolation=cv2.INTER_NEAREST)
    return pr


def boundary(m):
    m = m.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def surface_metrics(gt_m, pr_m):
    """HD95, ASSD (um) and boundary F1 at BF_TOL_UM between two binary masks."""
    if not gt_m.any() and not pr_m.any():
        return None
    if not gt_m.any() or not pr_m.any():
        return {'hd95_um': float('nan'), 'assd_um': float('nan'), 'bf1': 0.0}
    bg, bp = boundary(gt_m), boundary(pr_m)
    dg = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 5)
    dp = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 5)
    d_pg, d_gp = dg[bp] * UM, dp[bg] * UM
    allp = np.r_[d_pg, d_gp]
    prec = (d_pg <= BF_TOL_UM).mean(); rec = (d_gp <= BF_TOL_UM).mean()
    return {'hd95_um': float(np.percentile(allp, 95)), 'assd_um': float(allp.mean()),
            'bf1': float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0}


def section_job(name):
    low_priority()
    gt = gt_class_mask(cv2.cvtColor(cv2.imread(str(PV / 'masks' / f'{name}.png')), cv2.COLOR_BGR2RGB))
    img = cv2.cvtColor(cv2.imread(str(img_path(name)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != gt.shape:
        img = cv2.resize(img, gt.shape[::-1], interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # NBTC blue-purple formazan: blueness = B - mean(R, G)
    blue = img[..., 2].astype(np.float32) - 0.5 * (img[..., 0].astype(np.float32) + img[..., 1].astype(np.float32))
    tissue_gt = gt >= 1
    edge_dist = cv2.distanceTransform(tissue_gt.astype(np.uint8), cv2.DIST_L2, 5) * UM
    out = {'section': name, 'pixel': {}, 'surface': {}, 'lesions': {}, 'phantoms': [], 'gt_lesion_rows': []}
    gl = (gt >= 2)
    ng, glab, gst, _ = cv2.connectedComponentsWithStats(gl.astype(np.uint8), connectivity=8)
    for pipe in PIPES:
        pr = read_pred(pipe, name, gt.shape)
        out['pixel'][pipe] = {c: [int(((pr == i) & (gt == i)).sum()), int(((pr == i) & (gt != i)).sum()), int(((pr != i) & (gt == i)).sum())]
                              for i, c in enumerate(CL)}
        conf = np.zeros((4, 4), np.int64)
        for a in range(4):
            for b in range(4):
                conf[a, b] = int(((gt == a) & (pr == b)).sum())
        out['pixel'][pipe]['confusion_gt_rows_pred_cols'] = conf.tolist()
        out['surface'][pipe] = {k: surface_metrics(g, p) for k, g, p in
                                (('coagulation', gt == 2, pr == 2), ('ablation', gt == 3, pr == 3), ('lesion', gl, pr >= 2))}
        # lesion level
        pl = pr >= 2
        npr, plab, pst, _ = cv2.connectedComponentsWithStats(pl.astype(np.uint8), connectivity=8)
        gt_rows = []
        for i in range(1, ng):
            x, y, w, h, a = gst[i]
            if a < 100:
                continue
            comp = glab[y:y + h, x:x + w] == i
            ids = plab[y:y + h, x:x + w][comp]; ids = ids[ids > 0]
            if len(ids):
                j = np.bincount(ids).argmax()
                inter = int((comp & (plab[y:y + h, x:x + w] == j)).sum())
                union = int(a + pst[j, cv2.CC_STAT_AREA] - inter)
                iou = inter / union
            else:
                iou = 0.0
            gt_rows.append({'pipe': pipe, 'section': name, 'eq_diam_um': 2 * np.sqrt(a / np.pi) * UM, 'iou': iou,
                            'found': bool((pl[y:y + h, x:x + w][comp]).mean() >= 0.2),
                            'gt_has_lumen': bool((gt[y:y + h, x:x + w][comp] == 3).any())})
        out['gt_lesion_rows'] += gt_rows
        if pipe == 'frozen':
            ring_k = np.ones((41, 41), np.uint8)
            for j in range(1, npr):
                x, y, w, h, a = pst[j]
                if a < 100:
                    continue
                pad = 25
                sl = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
                comp = plab[sl] == j
                ov = gl[sl][comp].mean()
                ring = cv2.dilate(comp.astype(np.uint8), ring_k).astype(bool) & ~comp & tissue_gt[sl] & ~gl[sl]
                out['phantoms'].append({
                    'section': name, 'kind': 'real' if ov >= 0.2 else 'phantom', 'eq_diam_um': 2 * np.sqrt(a / np.pi) * UM,
                    'has_lumen': bool((pr[sl][comp] == 3).any()), 'edge_dist_um': float(edge_dist[sl][comp].min()),
                    'gray_in': float(gray[sl][comp].mean()), 'gray_ring': float(gray[sl][ring].mean()) if ring.any() else float('nan'),
                    'blue_in': float(blue[sl][comp].mean()), 'blue_ring': float(blue[sl][ring].mean()) if ring.any() else float('nan')})
    out['tissue_gray_mean'] = float(gray[tissue_gt & ~gl].mean()) if (tissue_gt & ~gl).any() else float('nan')
    out['tissue_blue_mean'] = float(blue[tissue_gt & ~gl].mean()) if (tissue_gt & ~gl).any() else float('nan')
    return out


def gt_geometry_job(args):
    """Section size and lesion pitch/density from expert GT on a known-scale source."""
    low_priority()
    sid, mask_path, um = args
    gt = gt_class_mask(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))
    tis = (gt >= 1).astype(np.uint8)
    tis = cv2.morphologyEx(tis, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(tis)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    sec = (lab == k).astype(np.uint8)
    area_mm2 = sec.sum() * um ** 2 / 1e6
    cnts, _ = cv2.findContours(sec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (_, _), (ma, mi), _ = cv2.fitEllipse(c) if len(c) >= 5 else ((0, 0), (0, 0), 0)
    les = ((gt >= 2) & (sec > 0)).astype(np.uint8)
    nl, _, lst, cen = cv2.connectedComponentsWithStats(les, connectivity=8)
    keep = [i for i in range(1, nl) if lst[i, cv2.CC_STAT_AREA] * um ** 2 >= 1000]  # >= ~36 um diameter
    pts = cen[keep]
    nn = None
    if len(pts) >= 3:
        d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
        nn = d.min(1) * um
    return {'slide': sid, 'um_per_px': um, 'section_area_mm2': area_mm2, 'eq_diam_mm': 2 * np.sqrt(area_mm2 / np.pi),
            'ellipse_major_mm': max(ma, mi) * um / 1000, 'ellipse_minor_mm': min(ma, mi) * um / 1000,
            'n_lesions': len(pts), 'lesions_per_mm2': len(pts) / area_mm2 if area_mm2 else float('nan'),
            'nn_pitch_median_um': float(np.median(nn)) if nn is not None else float('nan'),
            'nn_pitch_cv': float(nn.std() / nn.mean()) if nn is not None else float('nan')}


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    if len(x) < 3:
        return float('nan'), len(x)
    sxy = np.cov(x, y, bias=True)[0, 1]
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)), len(x)


def thumb_job(name):
    low_priority()
    gt = gt_class_mask(cv2.cvtColor(cv2.imread(str(PV / 'masks' / f'{name}.png')), cv2.COLOR_BGR2RGB))
    img = cv2.cvtColor(cv2.imread(str(img_path(name))), cv2.COLOR_BGR2RGB)
    pr = read_pred('frozen', name, gt.shape)
    f = 260 / max(gt.shape); sz = (int(gt.shape[1] * f), int(gt.shape[0] * f))
    diff = np.full(gt.shape + (3,), 235, np.uint8); diff[gt == 0] = 255
    for m, col in (((gt >= 2) & (pr >= 2), (120, 120, 120)), ((pr >= 2) & (gt < 2), (230, 0, 0)), ((gt >= 2) & (pr < 2), (0, 160, 255))):
        diff[cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)] = col
    return name, [cv2.resize(x, sz, interpolation=cv2.INTER_AREA if x is img else cv2.INTER_NEAREST) for x in (img, COLORS[gt], COLORS[pr], diff)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--out', type=Path, default=ROOT / 'predictions' / 'r3_cpu_report')
    a = ap.parse_args()
    low_priority()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs, tabs = a.out / 'figures', a.out / 'tables'
    figs.mkdir(parents=True, exist_ok=True); tabs.mkdir(parents=True, exist_ok=True)
    sections = sorted({Path(p).stem for p in glob.glob(str(PV / 'masks' / '*.png'))}, key=order)
    depth = {s: i * SECTION_UM for i, s in enumerate(sections)}

    with Pool(a.workers) as pool:
        res = pool.map(section_job, sections, chunksize=1)
    res = {r['section']: r for r in res}

    # 1. per-section pixel metrics
    rows = []
    for s in sections:
        for pipe in PIPES:
            for c in CL:
                tp, fp, fn = res[s]['pixel'][pipe][c]
                rows.append({'section': s, 'depth_um': depth[s], 'band': band_of(depth[s]), 'pipeline': pipe, 'class': c,
                             'dice': 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
                             'precision': tp / (tp + fp) if tp + fp else np.nan, 'recall': tp / (tp + fn) if tp + fn else np.nan,
                             'gt_px': tp + fn, 'pred_px': tp + fp})
    px = pd.DataFrame(rows); px.to_csv(tabs / 'per_section_pixel_metrics.csv', index=False)
    conf = {p: np.sum([res[s]['pixel'][p]['confusion_gt_rows_pred_cols'] for s in sections], axis=0) for p in PIPES}

    # 2. surface metrics
    srows = []
    for s in sections:
        for pipe in PIPES:
            for k, v in res[s]['surface'][pipe].items():
                if v is not None:
                    srows.append({'section': s, 'depth_um': depth[s], 'band': band_of(depth[s]), 'pipeline': pipe, 'structure': k, **v})
    sf = pd.DataFrame(srows); sf.to_csv(tabs / 'per_section_surface_metrics.csv', index=False)
    surf_band = sf.groupby(['pipeline', 'structure', 'band'])[['hd95_um', 'assd_um', 'bf1']].median().round(2)
    surf_band.to_csv(tabs / 'surface_metrics_by_band_median.csv')

    # 3. lesion level
    gl = pd.DataFrame([r for s in sections for r in res[s]['gt_lesion_rows']])
    gl['depth_um'] = gl.section.map(depth); gl['band'] = gl.depth_um.map(band_of)
    gl.to_csv(tabs / 'gt_lesion_level.csv', index=False)
    bins = [0, 150, 200, 250, 300, 1e9]; labels = ['<150', '150-200', '200-250', '250-300', '>=300']
    gl['size_bin'] = pd.cut(gl.eq_diam_um, bins, labels=labels, right=False)
    size_tab = gl.groupby(['pipe', 'size_bin'], observed=False).agg(n=('found', 'size'), recall=('found', 'mean'), median_iou=('iou', 'median')).round(3)
    size_tab.to_csv(tabs / 'recall_by_gt_lesion_size.csv')
    lumen_tab = gl.groupby(['pipe', 'gt_has_lumen']).agg(n=('found', 'size'), recall=('found', 'mean'), median_iou=('iou', 'median')).round(3)

    # 4. phantoms
    ph = pd.DataFrame([r for s in sections for r in res[s]['phantoms']])
    ph['depth_um'] = ph.section.map(depth); ph['band'] = ph.depth_um.map(band_of)
    tg = {s: res[s]['tissue_gray_mean'] for s in sections}; tb = {s: res[s]['tissue_blue_mean'] for s in sections}
    ph['gray_minus_section_tissue'] = ph.gray_in - ph.section.map(tg)
    ph['blue_minus_ring'] = ph.blue_in - ph.blue_ring
    ph.to_csv(tabs / 'pred_lesion_profile.csv', index=False)
    ph_sum = ph.groupby('kind').agg(n=('eq_diam_um', 'size'), eqD_med=('eq_diam_um', 'median'), lumen_frac=('has_lumen', 'mean'),
                                    edge_dist_med=('edge_dist_um', 'median'), within_100um_of_edge=('edge_dist_um', lambda x: (x < 100).mean()),
                                    gray_in_med=('gray_in', 'median'), gray_vs_tissue_med=('gray_minus_section_tissue', 'median'),
                                    blue_in_med=('blue_in', 'median'), blue_in_minus_ring_med=('blue_minus_ring', 'median')).round(3)
    ph_band = ph.groupby(['band', 'kind']).size().unstack(fill_value=0)

    # 5. census agreement per section (from r3_stats lesion tables)
    def les_table(root):
        out = []
        for s in sections:
            p = Path(root) / s / 'csv' / 'lesion_summary.csv'
            if p.exists() and p.stat().st_size > 0:
                try:
                    d = pd.read_csv(p)
                except pd.errors.EmptyDataError:
                    continue
                if len(d):
                    d['section'] = s; out.append(d)
        d = pd.concat(out, ignore_index=True)
        d['interior'] = ~d.Touches_Edge.astype(str).str.lower().isin(['true', '1'])
        d['lumen'] = d.Ablation_Diameter_um > 0
        return d
    cen = {}
    for k, root in (('gt', ROOT / 'predictions' / 'r3_PVcont_GT'), ('frozen', PIPES['frozen']), ('tta', PIPES['tta'])):
        d = les_table(root)
        per = pd.DataFrame(index=sections)
        per['count'] = d.groupby('section').size().reindex(sections).fillna(0)
        per['lumen_frac'] = d.groupby('section').lumen.mean().reindex(sections)
        def cv_(g):
            v = g[g.interior & g.lumen].Ablation_Diameter_um.to_numpy(float)
            return v.std(ddof=1) / v.mean() if len(v) >= 10 else np.nan
        per['abl_cv'] = d.groupby('section').apply(cv_, include_groups=False).reindex(sections)
        per['abl_diam_mean'] = d[d.lumen].groupby('section').Ablation_Diameter_um.mean().reindex(sections)
        cen[k] = per
    agree = []
    for m in ('frozen', 'tta'):
        for col in ('count', 'lumen_frac', 'abl_cv', 'abl_diam_mean'):
            for scope, ss in (('all', sections), ('0-460um', [s for s in sections if depth[s] <= 460])):
                r, n = ccc(cen['gt'].loc[ss, col], cen[m].loc[ss, col])
                agree.append({'model': m, 'measure': col, 'scope': scope, 'lin_ccc': round(r, 3), 'n_sections': n})
    agree = pd.DataFrame(agree); agree.to_csv(tabs / 'census_agreement_ccc.csv', index=False)
    pd.concat({k: v for k, v in cen.items()}, axis=1).to_csv(tabs / 'census_per_section_gt_vs_model.csv')

    # 6. Major 2: geometry from expert GT on known-scale annotated slides
    man = [r for r in csv.DictReader(open(SRC / 'scale_manifest.csv')) if r['status'] == 'written' and r['quantify_um'] == 'yes'
           and not r['slide_id'].startswith('NDX')]
    jobs = [(r['slide_id'], str(SRC / 'masks' / f"{r['slide_id']}.png"), float(r['tiled_um_per_px'])) for r in man]
    with Pool(min(a.workers, len(jobs))) as pool:
        geo = pd.DataFrame(pool.map(gt_geometry_job, jobs))
    geo['series'] = geo.slide.str.replace(r'_\d+$', '', regex=True).str.replace(r'\d+$', '', regex=True)
    geo = geo.sort_values(['series', 'slide']); geo.round(3).to_csv(tabs / 'gt_section_geometry_pitch.csv', index=False)
    geo_series = geo.groupby('series').agg(n_sections=('slide', 'size'), eq_diam_mm_med=('eq_diam_mm', 'median'),
                                           eq_diam_mm_max=('eq_diam_mm', 'max'), lesions_per_mm2_med=('lesions_per_mm2', 'median'),
                                           nn_pitch_um_med=('nn_pitch_median_um', 'median')).round(3)

    # 7. figures
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, c in enumerate(['ablation', 'coagulation', 'tissue']):
        for pipe, ls in (('frozen', '-'), ('tta', '--')):
            d = px[(px['class'] == c) & (px.pipeline == pipe)]
            ax[i].plot(d.depth_um, d.dice, ls, marker='o', ms=3, label=pipe)
        ax[i].set_title(f'{c} Dice per section'); ax[i].set_xlabel('depth (serial rank x 20 um)'); ax[i].set_ylim(0, 1.02); ax[i].legend()
    fig.tight_layout(); fig.savefig(figs / 'dice_vs_depth.png', dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, k in enumerate(['hd95_um', 'assd_um', 'bf1']):
        for st, col in (('coagulation', 'tab:blue'), ('ablation', 'tab:red'), ('lesion', 'k')):
            d = sf[(sf.structure == st) & (sf.pipeline == 'frozen')]
            ax[i].plot(d.depth_um, d[k], marker='o', ms=3, color=col, label=st)
        ax[i].set_title(f'{k} per section (frozen)'); ax[i].set_xlabel('depth (um)'); ax[i].legend()
    fig.tight_layout(); fig.savefig(figs / 'surface_vs_depth.png', dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for kind, col in (('real', 'tab:green'), ('phantom', 'tab:red')):
        d = ph[ph.kind == kind]
        ax[0].hist(d.eq_diam_um, bins=np.arange(0, 450, 15), alpha=0.6, color=col, label=f'{kind} (n={len(d)})')
        ax[1].hist(d.edge_dist_um.clip(upper=1500), bins=40, alpha=0.6, color=col, label=kind)
        ax[2].hist(d.gray_minus_section_tissue, bins=40, alpha=0.6, color=col, label=kind)
        ax[3].hist(d.blue_minus_ring, bins=40, alpha=0.6, color=col, label=kind)
    for a_, t in zip(ax, ['equivalent diameter (um)', 'distance to tissue edge (um, clipped 1500)',
                          'grey inside minus section tissue grey', 'blueness inside minus surrounding ring']):
        a_.set_xlabel(t); a_.legend()
    fig.suptitle('Predicted lesions (frozen): real vs phantom'); fig.tight_layout(); fig.savefig(figs / 'phantom_profile.png', dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for i, col in enumerate(['count', 'lumen_frac', 'abl_cv', 'abl_diam_mean']):
        x = cen['gt'][col]; y = cen['frozen'][col]; dd = np.array([depth[s] for s in sections])
        sc = ax[i].scatter(x, y, c=dd, cmap='viridis', s=22)
        lim = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
        ax[i].plot(lim, lim, 'k--', lw=1); ax[i].set_xlabel(f'GT {col}'); ax[i].set_ylabel(f'model {col}')
        r = agree[(agree.model == 'frozen') & (agree.measure == col) & (agree.scope == 'all')].lin_ccc.iloc[0]
        ax[i].set_title(f'{col}: CCC {r}')
    fig.colorbar(sc, ax=ax[-1], label='depth (um)'); fig.tight_layout(); fig.savefig(figs / 'census_agreement.png', dpi=110); plt.close(fig)

    with Pool(a.workers) as pool:
        th = dict(pool.map(thumb_job, sections, chunksize=1))
    fig, axs = plt.subplots(len(sections), 4, figsize=(10, 2.6 * len(sections)))
    for row, s in zip(axs, sections):
        for a_, im, t in zip(row, th[s], ['image', 'GT', 'pred (frozen)', 'diff: grey match, red FP, blue FN']):
            a_.imshow(im); a_.axis('off'); a_.set_title(f'{s} ({depth[s]:.0f} um) {t}' if t == 'image' else t, fontsize=6)
    fig.tight_layout(); fig.savefig(figs / 'contact_sheet_35_sections.png', dpi=80); plt.close(fig)

    # report
    def tbl(df):
        return df.to_markdown() if hasattr(df, 'to_markdown') else df.to_string()
    lines = ['# Round-3 extended CPU report (model 1, PVcont, 35 sections)', '',
             'Inputs: saved predictions (frozen pipeline, TTA pipeline) vs expert GT with anti-aliasing resolved. '
             'Depth = serial rank x 20 um (missing serials not accounted for).', '',
             '## 1. Pixel confusion (all sections, frozen; rows GT, columns prediction)', '',
             tbl(pd.DataFrame(conf['frozen'], index=[f'GT {c}' for c in CL], columns=[f'pred {c}' for c in CL])), '',
             'Per-section metrics: tables/per_section_pixel_metrics.csv; figure: figures/dice_vs_depth.png', '',
             '## 2. Boundary metrics (median per band)', '', tbl(surf_band), '',
             f'Boundary F1 tolerance {BF_TOL_UM} um. Figure: figures/surface_vs_depth.png', '',
             '## 3. Lesion level', '', '### Recall by GT lesion size', '', tbl(size_tab), '',
             '### Recall by GT lumen (ablation-bearing vs coagulation-only)', '', tbl(lumen_tab), '',
             '## 4. Real vs phantom predicted lesions (frozen)', '', tbl(ph_sum), '', '### Count by depth band', '', tbl(ph_band), '',
             'Figure: figures/phantom_profile.png', '',
             '## 5. Census agreement per section (Lin CCC, GT vs model)', '', tbl(agree.set_index(['model', 'measure', 'scope'])), '',
             'Figure: figures/census_agreement.png', '',
             '## 6. Major 2: section size and lesion pitch from expert GT (known-scale slides)', '', tbl(geo_series), '',
             'Per slide: tables/gt_section_geometry_pitch.csv. Nominal 8 mm punch; nominal DeepFX pitch 714 um.', '',
             '## 7. Contact sheet', '', 'figures/contact_sheet_35_sections.png']
    (a.out / 'REPORT.md').write_text('\n'.join(lines))
    json.dump({'phantom_summary': json.loads(ph_sum.to_json()), 'census_ccc': agree.to_dict('records'),
               'geometry_by_series': json.loads(geo_series.to_json()), 'size_recall': json.loads(size_tab.reset_index().astype(str).to_json())},
              open(a.out / 'report.json', 'w'), indent=1)
    print('wrote', a.out / 'REPORT.md')


if __name__ == '__main__':
    main()
