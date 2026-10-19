#!/usr/bin/env python3
"""Ground truth vs prediction difference maps and predicted-ablation blob audit.

For each section: image | GT | prediction | difference map, plus zoomed crops of
the largest false-ablation clusters, and a CSV of every predicted ablation
connected component (area, overlap with GT lesion, coagulation ring coverage).

Usage:
  python scripts/analysis/diff_maps.py --pred-root predictions/r3_PVcont_report \
      --out predictions/r3_diffmaps PVcont13_02 PVcont16_01 ...
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np, tifffile
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data' / 'LesionSegmentation'))

UM = 0.8832
DATA = DATA_ROOT / 'data/PVcont'
FADED = np.array([[255, 255, 255], [205, 235, 205], [200, 210, 245], [245, 200, 200]], np.uint8)
CLASS_RGB = np.array([[255, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0]], np.uint8)
ERR = {'FP ablation': (230, 0, 0), 'FN ablation': (255, 200, 0), 'FP coag': (0, 60, 255), 'FN coag': (0, 220, 230)}


def classes(rgb):
    r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
    c = np.zeros(rgb.shape[:2], np.uint8)
    c[(g > 200) & (r < 100) & (b < 100)] = 1
    c[(b > 200) & (r < 100) & (g < 100)] = 2
    c[(r > 200) & (g < 100) & (b < 100)] = 3
    return c


def find(name):
    for p in (DATA / 'images' / f'{name}.tif', DATA / f'{name}.tif'):
        if p.exists():
            return p
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+')
    ap.add_argument('--pred-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--width', type=int, default=1300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    blobs_all, summary = [], {}
    for name in a.names:
        img = tifffile.imread(find(name))[..., :3]
        gt = classes(cv2.cvtColor(cv2.imread(str(DATA / 'masks' / f'{name}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        pr_rgb = cv2.cvtColor(cv2.imread(str(a.pred_root / name / 'images' / f'{name}_pred.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if pr_rgb.shape[:2] != gt.shape:
            pr_rgb = cv2.resize(pr_rgb, gt.shape[::-1], interpolation=cv2.INTER_NEAREST)
        pr = classes(pr_rgb)

        err = {'FP ablation': (pr == 3) & (gt != 3), 'FN ablation': (gt == 3) & (pr != 3),
               'FP coag': (pr == 2) & (gt != 2) & (gt != 3), 'FN coag': (gt == 2) & (pr != 2) & (pr != 3)}
        diff = FADED[gt].copy()
        k = np.ones((7, 7), np.uint8)
        for key in ('FN coag', 'FP coag', 'FN ablation', 'FP ablation'):  # ablation drawn on top
            m = cv2.dilate(err[key].astype(np.uint8), k).astype(bool)
            diff[m] = ERR[key]

        # predicted ablation blobs
        gt_les = (gt >= 2).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats((pr == 3).astype(np.uint8), connectivity=8)
        ring_k = np.ones((9, 9), np.uint8)
        rows = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            pad = 6
            sl = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
            comp = (lab[sl] == i).astype(np.uint8)
            ring = cv2.dilate(comp, ring_k).astype(bool) & ~comp.astype(bool)
            ring_coag = float((pr[sl][ring] == 2).mean()) if ring.any() else 0.0
            ov_les = float(gt_les[sl][comp.astype(bool)].mean())
            ov_abl = float((gt[sl][comp.astype(bool)] == 3).mean())
            rows.append({'section': name, 'blob': i, 'area_px': int(area), 'eq_diam_um': round(2 * np.sqrt(area / np.pi) * UM, 1),
                         'cx': int(cen[i][0]), 'cy': int(cen[i][1]), 'frac_on_gt_ablation': round(ov_abl, 3),
                         'frac_on_gt_lesion': round(ov_les, 3), 'coag_ring_frac': round(ring_coag, 3),
                         'kind': 'TP' if ov_abl > 0.1 else ('in GT coag' if ov_les > 0.5 else 'isolated FP')})
        blobs_all += rows
        tot = {c: int((gt == c).sum()) for c in (2, 3)}
        s = {'gt_ablation_px': tot[3], 'gt_coag_px': tot[2], 'pred_ablation_px': int((pr == 3).sum()), 'pred_coag_px': int((pr == 2).sum()),
             **{k2 + '_px': int(v.sum()) for k2, v in err.items()},
             'pred_abl_blobs': len(rows), **{f'blobs_{kk}': sum(r['kind'] == kk for r in rows) for kk in ('TP', 'in GT coag', 'isolated FP')}}
        summary[name] = s

        # figure: 4 panels
        f = a.width / max(gt.shape)
        sz = (int(gt.shape[1] * f), int(gt.shape[0] * f))
        th = lambda x, it=cv2.INTER_AREA: cv2.resize(x, sz, interpolation=it)
        fig, ax = plt.subplots(1, 4, figsize=(22, 6.2))
        for axx, im, t in zip(ax, [th(img), th(CLASS_RGB[gt], cv2.INTER_NEAREST), th(CLASS_RGB[pr], cv2.INTER_NEAREST), th(diff, cv2.INTER_NEAREST)],
                              ['image', f'GT  (abl {tot[3]} px, coag {tot[2]} px)', f'prediction  (abl {s["pred_ablation_px"]} px, coag {s["pred_coag_px"]} px)',
                               'difference']):
            axx.imshow(im); axx.set_title(t, fontsize=10); axx.set_xticks([]); axx.set_yticks([])
        bar = 1000 / UM * f
        ax[3].plot([10, 10 + bar], [sz[1] - 15] * 2, 'k', lw=4); ax[3].text(10, sz[1] - 25, '1 mm', fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255) for c in ERR.values()]
        ax[3].legend(handles, [f'{k2} ({s[k2 + "_px"]} px)' for k2 in ERR], loc='upper right', fontsize=7, framealpha=0.85)
        fig.suptitle(f'{name}: predicted ablation blobs {len(rows)} = TP {s["blobs_TP"]}, inside GT coag {s["blobs_in GT coag"]}, '
                     f'isolated FP {s["blobs_isolated FP"]}', fontsize=12)
        fig.tight_layout(); fig.savefig(a.out / f'{name}_diff.png', dpi=85); plt.close(fig)

        # zoomed crops around the 6 largest non-TP ablation blobs
        fps = sorted([r for r in rows if r['kind'] != 'TP'], key=lambda r: -r['area_px'])[:6]
        if fps:
            fig, axs = plt.subplots(len(fps), 4, figsize=(14, 3.6 * len(fps)))
            axs = np.atleast_2d(axs)
            for row_ax, r in zip(axs, fps):
                half = 220
                y0, x0 = max(0, r['cy'] - half), max(0, r['cx'] - half)
                sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
                for axx, im, t in zip(row_ax, [img[sl], CLASS_RGB[gt[sl]], CLASS_RGB[pr[sl]], diff[sl]], ['image', 'GT', 'pred', 'diff']):
                    axx.imshow(im); axx.set_xticks([]); axx.set_yticks([]); axx.set_title(t, fontsize=8)
                row_ax[0].set_ylabel(f"{r['kind']}\n{r['eq_diam_um']} um, ring coag {r['coag_ring_frac']:.0%}", fontsize=8)
            fig.suptitle(f'{name}: largest non-TP predicted ablation blobs (crop 390 um)', fontsize=11)
            fig.tight_layout(); fig.savefig(a.out / f'{name}_fp_crops.png', dpi=80); plt.close(fig)
        print(name, json.dumps(s), flush=True)

    with open(a.out / 'pred_ablation_blobs.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(blobs_all[0])); w.writeheader(); w.writerows(blobs_all)
    json.dump(summary, open(a.out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
