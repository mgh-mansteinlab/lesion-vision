#!/usr/bin/env python3
"""Final-pipeline census of the unannotated example series at verified scale.

Series (extracted by the metadata-aware NDPI reader at 0.8832 um/px):
  data_example_series/{laser_only_CCC, papain_CPC, papain_urea_CPVB}/images
Steps: probabilities from every member of the validation-selected final
pipeline (predictions/r3_greedy/selection.json) -> equal-weight mean ->
nested decoding + the validation-tuned lesion filter -> lesion tables ->
depth profiles (figures/r3_fig_profiles.png) and within-section CV table
(manuscript/generated/table_cv.tex). Section order = (slide number, sample
index), depth = serial rank x 20 um, as in the manuscript.
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data' / 'LesionSegmentation'))

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'predictions'
DATA = DATA_ROOT / 'data_example_series'
OUT = P / 'r3_example_series'
SERIES = [('laser_only_CCC', 'Laser only'), ('papain_CPC', 'Laser + papain'), ('papain_urea_CPVB', 'Laser + papain--urea')]
TAGS = {'M2': 'r3m2', 'C1': 'r3_c1_upernet_convnextb', 'C2': 'r3_c2_segformer_mitb3',
        'C3': 'r3_c3_deeplabv3plus_resnet50', 'C4': 'r3_c4_unetpp_effb3'}
PY = sys.executable
MIN_TISSUE_MM2 = 2.0


def run(cmd):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def order_key(name):
    m = re.search(r'(\d+)\s*-\s*20\d\d', name) or re.search(r'_(\d+)\s', name)
    slide = int(m.group(1)) if m else 0
    s = re.search(r'sample_(\d+)', name)
    return slide, int(s.group(1)) if s else 0


def main():
    sel = json.load(open(P / 'r3_greedy' / 'selection.json'))
    members = sel['selected']
    params = P / 'r3_greedy' / 'filter_params_nested.json'
    for s, _ in SERIES:
        for m in members:
            ck = json.load(open(P / TAGS[m] / 'summary.json'))['selected_checkpoint']
            d = OUT / s / f'probs_{m}'
            if len(glob.glob(f'{d}/*.npz')) < len(glob.glob(f'{DATA}/{s}/images/*.tif')):
                run([PY, 'scripts/predict_probs.py', '--images-only', '--data-dir', DATA / s, '--model', ck, '--out-dir', d, '--gpus', '8'])
        run([PY, 'scripts/analysis/ensemble_probs.py', '--out-dir', OUT / s / 'probs_final'] + [OUT / s / f'probs_{m}' for m in members])
        run([PY, 'scripts/analysis/component_filter.py', 'apply', '--decode', 'nested', '--probs-dir', OUT / s / 'probs_final',
             '--params', params, '--out-dir', OUT / s / 'pred'] + (['--interior-bg'] if sel.get('interior_bg') else [])
            + (['--fill-crater'] if sel.get('fill_crater') else []))

    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cv_rows, prof = [], {}
    for s, label in SERIES:
        recs = []
        for c in sorted(glob.glob(f'{OUT}/{s}/pred/*/csv/lesion_summary.csv'), key=lambda p: order_key(Path(p).parents[1].name)):
            name = Path(c).parents[1].name
            mask = cv2.imread(f'{OUT}/{s}/pred/{name}/images/{name}_pred.png')
            tissue_mm2 = float((~np.all(mask > 240, axis=2)).sum()) * 0.8832 ** 2 / 1e6
            if tissue_mm2 < MIN_TISSUE_MM2 or os.path.getsize(c) == 0:
                continue
            try:
                d = pd.read_csv(c)
            except pd.errors.EmptyDataError:
                continue
            d = d[~d.Touches_Edge.astype(str).str.lower().isin(['true', '1'])]
            if d.empty:
                continue
            recs.append((name, d))
        rows = []
        for rank, (name, d) in enumerate(recs):
            lum = d[d.Ablation_Diameter_um > 0]
            rows.append({'section': name, 'depth_um': rank * 20, 'n': len(d), 'n_lumen': len(lum),
                         'abl_radius': lum.Ablation_Diameter_um.mean() / 2 if len(lum) else np.nan,
                         'coag_width': d.Coagulation_Width_um.mean(),
                         'cv_abl': lum.Ablation_Diameter_um.std(ddof=1) / lum.Ablation_Diameter_um.mean() if len(lum) >= 10 else np.nan,
                         'cv_coag': d.Coagulation_Width_um.std(ddof=1) / d.Coagulation_Width_um.mean() if len(d) >= 10 else np.nan,
                         'cv_les': d.Lesion_Diameter_um.std(ddof=1) / d.Lesion_Diameter_um.mean() if len(d) >= 10 else np.nan})
        df = pd.DataFrame(rows); df.to_csv(OUT / f'{s}_sections.csv', index=False); prof[label] = df
        ok = df[df.n_lumen >= 10]
        sup = ok[ok.depth_um < 60]
        f = lambda x: f'${x.mean():.2f} \\pm {x.std(ddof=1):.2f}$' if x.notna().sum() >= 2 else ('--' if x.notna().sum() == 0 else f'${x.dropna().iloc[0]:.2f}$')
        cv_rows.append((label, f(sup.cv_abl), f(sup.cv_coag), f(sup.cv_les), f(ok.cv_abl), f(ok.cv_coag), f(ok.cv_les), len(df)))
    L = [r'\begin{tabular}{@{}lccc@{}}', r'\toprule', r'Series & Ablation CV & Coagulation-width CV & Lesion-diameter CV \\', r'\midrule',
         r'\multicolumn{4}{@{}l}{\emph{Superficial (serials 1--3)}} \\']
    L += [f'{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\' for r in cv_rows]
    L += [r'\addlinespace', r'\multicolumn{4}{@{}l}{\emph{All depths}} \\']
    L += [f'{r[0]} & {r[4]} & {r[5]} & {r[6]} \\\\' for r in cv_rows]
    L += [r'\bottomrule', r'\end{tabular}']
    (ROOT / 'manuscript' / 'generated' / 'table_cv.tex').write_text('\n'.join(L) + '\n')
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for a, (label, df) in zip(ax, prof.items()):
        a.plot(df.depth_um, df.abl_radius, 'o-', color='#c0392b', ms=3, label='ablation radius')
        a.plot(df.depth_um, df.coag_width, 's-', color='#2c3e9e', ms=3, label='coagulation width')
        a.set_title(label.replace('--', '–')); a.set_xlabel('depth (µm, serial rank × 20)'); a.grid(alpha=.3)
    ax[0].set_ylabel('µm'); ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(ROOT / 'manuscript' / 'figures' / 'r3_fig_profiles.png', dpi=200)
    print('done', OUT)


if __name__ == '__main__':
    main()
