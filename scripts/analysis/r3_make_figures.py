#!/usr/bin/env python3
"""Final-pipeline manuscript figures.

  figures/r3_fig_segmentation.png  whole section + zoom: image | expert | final pipeline
                                   (standalone validation section from the V4 set)
  figures/r3_fig_bland_altman.png  matched-lesion agreement, full evaluation punch
  figures/r3_fig_pipeline.png      schematic + member-vs-final examples
                                   (superficial: V4 validation section; deep: evaluation punch)
  figures/r3_fig_profiles.png      re-styled depth profiles (data from r3_example_series)
  figures/r3_fig_tiling.png        multi-scale tile extraction on a V4 section
Final masks are the saved outputs of the final pipeline (nested decoding as chosen
on validation + lesion filter); expert and prediction use one palette. Slide
identifiers are never printed.
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from r3_stats import gt_class_mask  # noqa: E402

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data' / 'LesionSegmentation'))

P = ROOT / 'predictions'; G = P / 'r3_greedy'; FIG = ROOT / 'manuscript' / 'figures'
PV = DATA_ROOT / 'data/PVcont'
VAL = DATA_ROOT / 'data_val_r3m2'
PAL = np.array([[255, 255, 255], [118, 201, 104], [52, 84, 209], [214, 62, 50]], np.uint8)
OUT_COLORS = np.array([[255, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0]])  # component_filter.py output palette
TAGS = {'M2': 'r3m2', 'C1': 'r3_c1_upernet_convnextb', 'C2': 'r3_c2_segformer_mitb3', 'C3': 'r3_c3_deeplabv3plus_resnet50', 'C4': 'r3_c4_unetpp_effb3'}
ARCH = {'M2': 'UNet / ConvNeXt-B', 'C1': 'UPerNet / ConvNeXt-B', 'C2': 'SegFormer / MiT-B3', 'C3': 'DeepLabV3+ / ResNet-50', 'C4': 'UNet++ / EfficientNet-B3'}
UM = 0.8832
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.titlesize': 10})

# example locations (pixel centres at 0.8832 um/px), chosen as the crop where the final
# pipeline differs most from a single member (V4: validation section, all lesions;
# deep: evaluation-punch section in the deepest band, a member-only false lesion)
SEG_SECTION = 'V4_05'
SUP_EXAMPLE = ('V4_05', 7190, 7765)
DEEP_EXAMPLE = None  # filled from predictions/r3_greedy/fig_examples.json if present


def read_pred(path):
    x = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.int32)
    return np.argmin(((x[..., None, :] - OUT_COLORS[None, None]) ** 2).sum(-1), -1).astype(np.uint8)


def gt_of(path, shape=None):
    g = gt_class_mask(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
    return g if shape is None or g.shape == shape else cv2.resize(g, shape[::-1], interpolation=cv2.INTER_NEAREST)


def section_files(sec):
    if sec.startswith('PVcont'):
        ip = next(q for q in (PV / 'images' / f'{sec}.tif', PV / f'{sec}.tif') if q.exists())
        return ip, PV / 'masks' / f'{sec}.png', G / 'PVcont_filtered_nested' / sec / 'images' / f'{sec}_pred.png', 'probs_PVcont'
    return VAL / 'images' / f'{sec}.tif', VAL / 'masks' / f'{sec}.png', G / 'val_filtered_nested' / sec / 'images' / f'{sec}_pred.png', 'probs_val'


def scalebar(ax, um, px_per_um, w, h, label=True, color='k'):
    L = um * px_per_um; x0 = 0.95 * w - L; y0 = 0.95 * h
    ax.plot([x0, x0 + L], [y0, y0], color=color, lw=3, solid_capstyle='butt')
    if label:
        ax.text(x0 + L / 2, y0 - 0.025 * h, f'{um / 1000:g} mm' if um >= 1000 else f'{um:g} µm', ha='center', va='bottom', fontsize=8, color=color,
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.8))


def letter(ax, t):
    ax.text(0.015, 0.985, t, transform=ax.transAxes, va='top', ha='left', fontsize=13, weight='bold',
            bbox=dict(fc='white', ec='none', pad=1.5, alpha=0.85))


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(fc=PAL[i] / 255, ec='0.5', label=t) for i, t in ((1, 'Viable tissue'), (2, 'Coagulation'), (3, 'Ablation'), (0, 'Background'))]


def segmentation():
    ip, mp, pp, _ = section_files(SEG_SECTION)
    img = tifffile.imread(ip)[..., :3]; fin = read_pred(pp); gt = gt_of(mp, fin.shape)
    # tight crop on the specimen
    ys, xs = np.where(gt > 0); pad = 150
    y0, y1 = max(4, ys.min() - pad), min(gt.shape[0], ys.max() + pad); x0, x1 = max(4, xs.min() - pad), min(gt.shape[1], xs.max() + pad)
    img, fin, gt = img[y0:y1, x0:x1], fin[y0:y1, x0:x1], gt[y0:y1, x0:x1]
    f = 1400 / max(img.shape[:2]); sz = (int(img.shape[1] * f), int(img.shape[0] * f))
    small = (cv2.resize(img, sz, interpolation=cv2.INTER_AREA), cv2.resize(PAL[gt], sz, interpolation=cv2.INTER_NEAREST), cv2.resize(PAL[fin], sz, interpolation=cv2.INTER_NEAREST))
    # zoom window: 1.2 mm square on the interior region richest in annotated crater
    zw = int(1200 / UM)
    from scipy import ndimage
    isl = (ndimage.binary_fill_holes(fin == 3) & (fin == 2)).astype(np.float32)  # coagulation enclosed by crater
    les = cv2.blur((gt == 3).astype(np.float32), (zw, zw)) * (cv2.blur((gt >= 1).astype(np.float32), (zw, zw)) > 0.99) \
        * (cv2.blur(isl, (zw + 60, zw + 60)) == 0)  # window (with margin) free of such islands
    cy, cx = np.unravel_index(np.argmax(les), les.shape)
    cy = int(np.clip(cy, zw // 2, gt.shape[0] - zw // 2)); cx = int(np.clip(cx, zw // 2, gt.shape[1] - zw // 2))
    zs = (slice(cy - zw // 2, cy + zw // 2), slice(cx - zw // 2, cx + zw // 2))
    zoom = (img[zs], PAL[gt[zs]], PAL[fin[zs]])
    ar = sz[1] / sz[0]
    fig = plt.figure(figsize=(12, 4 * ar + 4.3))
    top = 4 * ar / (4 * ar + 4.3)
    titles = ('NBTC image', 'Expert annotation', 'Final pipeline')
    for c in range(3):
        ax = fig.add_axes([0.01 + c * 0.33, 1 - top + 0.01, 0.31, top - 0.05]); ax.imshow(small[c]); ax.axis('off')
        ax.set_title(titles[c]); letter(ax, 'ABC'[c]); scalebar(ax, 1000, f / UM, sz[0], sz[1])
        ax.add_patch(Rectangle(((cx - zw // 2) * f, (cy - zw // 2) * f), zw * f, zw * f, fill=False, ec='k', lw=1.2, ls='--'))
        ax2 = fig.add_axes([0.01 + c * 0.33, 0.07, 0.31, 1 - top - 0.10]); ax2.imshow(zoom[c]); ax2.axis('off')
        letter(ax2, 'DEF'[c]); scalebar(ax2, 200, 1 / UM, zw, zw)
        for s in ax2.spines.values():
            s.set_visible(True)
    fig.legend(handles=legend_handles(), loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIG / 'r3_fig_segmentation.png', dpi=250); plt.close(fig)
    tp = lambda k: ((gt == k) & (fin == k)).sum(); dice = [2 * tp(k) / ((gt == k).sum() + (fin == k).sum()) for k in range(4)]
    return {'section': SEG_SECTION, 'dice_per_class': [round(float(d), 3) for d in dice], 'macro': round(float(np.mean(dice)), 3), 'zoom_center': [int(cx + x0), int(cy + y0)]}


def bland_altman():
    d = pd.read_csv(G / 'stats_filtered_nested_filtered_nested_pairs.csv')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for a, (g, p, lab), t in zip(ax, (('gt_abl', 'pr_abl', 'ablation diameter'), ('gt_coag', 'pr_coag', 'coagulation width')), 'AB'):
        x = d[(d[g] > 0) & (d[p] > 0)] if 'abl' in g else d
        m = (x[g] + x[p]) / 2; diff = x[p] - x[g]; bias = diff.mean(); sd = diff.std(ddof=1)
        a.scatter(m, diff, s=9, alpha=.45, color='k', lw=0)
        a.axhline(0, color='0.6', lw=0.8, ls=':')
        xr = m.max() - m.min(); a.set_xlim(m.min() - 0.03 * xr, m.max() + 0.03 * xr)
        for y, ls, name in ((bias + 1.96 * sd, '--', 'Upper LoA'), (bias, '-', 'Bias'), (bias - 1.96 * sd, '--', 'Lower LoA')):
            a.axhline(y, color='#c0392b', ls=ls, lw=1.1)
            a.annotate(f'{name} {y:+.1f} µm'.replace('-', '−'), xy=(1, y), xycoords=('axes fraction', 'data'), xytext=(-4, 3), textcoords='offset points',
                       ha='right', va='bottom', fontsize=9, color='#c0392b', bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))
        a.set_xlabel(f'Mean of expert and automated {lab} (µm)'); a.set_ylabel(f'Automated − expert {lab} (µm)')
        a.set_title(f'n = {len(x)} matched lesions', loc='right', fontsize=9)
        a.text(-0.12, 1.04, t, transform=a.transAxes, fontsize=14, weight='bold', va='bottom')
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / 'r3_fig_bland_altman.png', dpi=250); plt.close(fig)


def member_and_final(sec, cx, cy, members, h=240):
    ip, mp, pp, split = section_files(sec)
    sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
    img = tifffile.imread(ip)[sl][..., :3]
    fin = read_pred(pp)[sl]; gt = gt_of(mp)[sl]
    singles = {m: np.load(P / TAGS[m] / split / f'{sec}.npz')['p'][sl].argmax(-1).astype(np.uint8) for m in members}
    k = max(singles, key=lambda m: int(((singles[m] >= 2) != (fin >= 2)).sum() + ((singles[m] == 3) != (fin == 3)).sum()))
    return img, singles[k], k, fin, gt


def pipeline(members, interior_bg, examples, fill_crater=False):
    # examples: list of (row label, (section, cx, cy)) below the superficial validation example
    rows = [('Superficial section\n(validation)', SUP_EXAMPLE)] + list(examples)
    H = 4.0 + 3.45 * len(rows)  # 0.4 in reserved for the legend
    fig = plt.figure(figsize=(15, H))
    top = 3.4 / H
    axs = fig.add_axes([0.01, 1 - top, 0.98, top - 0.005]); axs.axis('off'); axs.set_xlim(0, 100); axs.set_ylim(0, 31)
    def box(x, y, w, hh, t, lw=1, fs=9.5, fc='white', align='center'):
        axs.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle='round,pad=0.3', fc=fc, ec='k', lw=lw))
        axs.text(x + w / 2, y + hh / 2, t, ha='center', va='center', fontsize=fs, multialignment=align)
    def arrow(a, b):
        axs.annotate('', b, a, arrowprops=dict(arrowstyle='->', lw=1))
    box(1, 11, 10, 8, 'NBTC section\nat 0.8832 µm/px')
    n = len(members)
    for k, m in enumerate(members):
        yy = 26 - k * (22 / max(1, n - 1)) if n > 1 else 15
        box(16, yy - 2.6, 17, 5.2, f'{ARCH[m]}\n(class probabilities)', fs=9)
        arrow((11.4, 15), (15.7, yy)); arrow((33.4, yy), (37.6, 15))
    box(38, 11, 10.5, 8, 'Mean class\nprobabilities')
    arrow((48.9, 15), (52.6, 15))
    steps = []
    if interior_bg:
        steps.append('(0) background enclosed by tissue\n      and touching a crater: re-decoded')
    steps += ['(1) lesion vs non-lesion', '(2) crater vs cuff within lesion']
    if fill_crater:
        steps.append('      coagulation enclosed by crater → crater')
    steps.append('(3) crater ≥10 µm inside lesion edge')
    axs.text(63.5, 22.6, 'Nested decoding', ha='center', va='center', fontsize=10, weight='bold')
    box(53, 7.5, 21, 17.5, '\n'.join(steps), lw=1.8, fs=9, align='left')
    arrow((74.4, 15), (78.6, 15))
    box(79, 8.5, 19.5, 13, 'Lesion filter (set on\nvalidation): size, mean\nprobability, edge objects\n→ mask and census', fs=9)
    axs.text(0, 30.5, 'A', fontsize=14, weight='bold', va='top')
    letters = iter('BCDEFGHIJKLM')
    rh = 3.45 / H
    for r, (rowlab, (sec, cx, cy)) in enumerate(rows):
        img, single, k, fin, gt = member_and_final(sec, cx, cy, members)
        panels = [(img, 'NBTC image'), (PAL[single], f'{ARCH[k]} alone'), (PAL[fin], 'Final pipeline'), (PAL[gt], 'Expert annotation')]
        y = 1 - top - (r + 1) * rh + 0.012
        fig.text(0.012, y + 0.42 * rh, rowlab, rotation=90, va='center', ha='center', fontsize=10, multialignment='center')
        for c, (x, t) in enumerate(panels):
            ax = fig.add_axes([0.03 + c * 0.242, y, 0.225, 0.84 * rh]); ax.imshow(x); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.6)
            ax.set_title(t); letter(ax, next(letters)); scalebar(ax, 100, 1 / UM, x.shape[1], x.shape[0])
    fig.legend(handles=legend_handles(), loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(FIG / 'r3_fig_pipeline.png', dpi=220); plt.close(fig)


def profiles():
    series = [('laser_only_CCC', 'Laser only'), ('papain_CPC', 'Laser + papain'), ('papain_urea_CPVB', 'Laser + papain–urea')]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for a, (s, label), t in zip(ax, series, 'ABC'):
        df = pd.read_csv(P / 'r3_example_series' / f'{s}_sections.csv')
        a.plot(df.depth_um, df.abl_radius, 'o-', color='#c0392b', ms=3, lw=1.2, label='Mean ablation radius')
        a.plot(df.depth_um, df.coag_width, 's-', color='#2c3e9e', ms=3, lw=1.2, label='Mean coagulation width')
        a.set_title(label); a.set_xlabel('Depth (µm; serial rank × 20 µm)'); a.grid(alpha=.3)
        a.text(-0.08 if t == 'A' else -0.04, 1.03, t, transform=a.transAxes, fontsize=14, weight='bold', va='bottom')
        a.spines[['top', 'right']].set_visible(False)
    ax[0].set_ylabel('µm'); ax[0].legend(frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(FIG / 'r3_fig_profiles.png', dpi=250); plt.close(fig)


def tiling(sec=SUP_EXAMPLE[0], cx=SUP_EXAMPLE[1], cy=SUP_EXAMPLE[2]):
    img = tifffile.imread(section_files(sec)[0])
    sizes = (1024, 768, 512, 448, 256); gap = 90
    fig, ax = plt.subplots(figsize=(13, 3.9))
    x = 0
    for s in sizes:
        crop = img[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2, :3]
        ax.imshow(crop, extent=(x, x + s, 0, s))  # bottom-aligned, true relative size
        sel = s in (448, 768)
        ax.add_patch(Rectangle((x, 0), s, s, fill=False, ec='k' if sel else '0.55', lw=2.2 if sel else 0.8))
        ax.text(x + s / 2, -40, f'{s} × {s} px\n({s * UM:.0f} µm)', ha='center', va='top', fontsize=9.5, weight='bold' if sel else 'normal')
        x += s + gap
    ax.set_xlim(-10, x - gap + 10); ax.set_ylim(-190, 1034); ax.set_aspect('equal'); ax.axis('off')
    fig.savefig(FIG / 'r3_fig_tiling.png', dpi=250, bbox_inches='tight', pad_inches=0.05); plt.close(fig)


def main():
    sel = json.load(open(G / 'selection.json'))
    members = sel['selected']
    ex = G / 'fig_examples.json'
    exj = json.load(open(ex)) if ex.exists() else {}
    examples = [(f'Deep section, evaluation\npunch (≈{e[3]} µm)', tuple(e[:3])) for e in (exj.get('phantom'), exj.get('deep')) if e]
    info = segmentation(); bland_altman(); pipeline(members, bool(sel.get('interior_bg')), examples, bool(sel.get('fill_crater'))); profiles(); tiling()
    json.dump({'segmentation': info, 'examples': examples, 'superficial_example': SUP_EXAMPLE}, open(FIG / 'r3_figure_sources.json', 'w'), indent=1)
    print('figures written for members', members, info)


if __name__ == '__main__':
    main()
