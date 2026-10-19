#!/usr/bin/env python3
"""How well does a 10-lesion manual sample estimate the full-field geometry?

Uses expert ground truth only (no model). For every section with at least
--min-lesions interior, lumen-bearing lesions it compares the full-field
within-section ablation-diameter CV and mean with:
  random:      10 lesions drawn at random (n_draws repeats, without replacement)
  well-formed: the 10 largest ablation craters (imitates selecting clearly
               visible, well-formed lesions, as manual practice does)
Sections: superficial PVcont serials (first 3) and all annotated known-scale
slides (scale_manifest quantify_um == yes, excluding tissue-only NDX strips).

Usage:
  nice -n 19 python scripts/analysis/manual_sampling_sim.py --out predictions/r3_manual_sampling
"""
from __future__ import annotations

import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from r3_stats import gt_class_mask  # noqa: E402
from src.analytics.lesions import LesionAnalyzer  # noqa: E402

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data' / 'LesionSegmentation'))

SRC = DATA_ROOT / 'data_tiling_sources'
PV_GT = ROOT / 'predictions' / 'r3_PVcont_GT'
K = 10


def lesions_for(args):
    os.nice(19)
    sid, mask_path, out_root = args
    out = Path(out_root) / sid / 'csv'
    if not (out / 'lesion_summary.csv').exists():
        out.mkdir(parents=True, exist_ok=True)
        m = gt_class_mask(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))
        LesionAnalyzer().analyze_mask(m, None, str(out), summary_metrics=True, figures=False)
    p = out / 'lesion_summary.csv'
    if not p.exists() or p.stat().st_size == 0:
        return sid, None
    d = pd.read_csv(p)
    d = d[(~d.Touches_Edge.astype(str).str.lower().isin(['true', '1'])) & (d.Ablation_Diameter_um > 0)]
    return sid, d.Ablation_Diameter_um.to_numpy(float)


def cv(x):
    return x.std(ddof=1) / x.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--min-lesions', type=int, default=15)
    ap.add_argument('--n-draws', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    os.nice(19)
    a.out.mkdir(parents=True, exist_ok=True)
    man = [r for r in csv.DictReader(open(SRC / 'scale_manifest.csv')) if r['status'] == 'written' and r['quantify_um'] == 'yes'
           and not r['slide_id'].startswith(('NDX', 'PVcont'))]
    jobs = [(r['slide_id'], str(SRC / 'masks' / f"{r['slide_id']}.png"), str(a.out / 'gt_lesions')) for r in man]
    with Pool(a.workers) as pool:
        data = dict(pool.map(lesions_for, jobs))
    for s in ('PVcont1_01', 'PVcont1_02', 'PVcont1_03'):  # superficial serials, GT tables from r3_stats
        d = pd.read_csv(PV_GT / s / 'csv' / 'lesion_summary.csv')
        d = d[(~d.Touches_Edge.astype(str).str.lower().isin(['true', '1'])) & (d.Ablation_Diameter_um > 0)]
        data[s] = d.Ablation_Diameter_um.to_numpy(float)
    rng = np.random.default_rng(0)
    rows = []
    for sid, x in sorted(data.items()):
        if x is None or len(x) < a.min_lesions:
            continue
        full_cv, full_mean = cv(x), x.mean()
        draws = np.array([rng.choice(x, K, replace=False) for _ in range(a.n_draws)])
        cvs = draws.std(1, ddof=1) / draws.mean(1)
        means = draws.mean(1)
        top = np.sort(x)[-K:]
        rows.append({'section': sid, 'n_lesions': len(x), 'full_cv': full_cv, 'full_mean_um': full_mean,
                     'random10_cv_p2.5': np.percentile(cvs, 2.5), 'random10_cv_p97.5': np.percentile(cvs, 97.5),
                     'random10_cv_rel_sd': cvs.std() / full_cv,
                     'random10_cv_within_20pct': float((np.abs(cvs - full_cv) <= 0.2 * full_cv).mean()),
                     'random10_mean_rel_sd': means.std() / full_mean,
                     'wellformed10_cv': cv(top), 'wellformed10_cv_bias_pct': 100 * (cv(top) - full_cv) / full_cv,
                     'wellformed10_mean_bias_pct': 100 * (top.mean() - full_mean) / full_mean})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(a.out / 'manual_sampling_simulation.csv', index=False)
    summ = {'n_sections': len(df), 'k': K, 'n_draws': a.n_draws,
            'median_random10_cv_rel_sd': float(df.random10_cv_rel_sd.median()),
            'median_prob_random10_cv_within_20pct': float(df.random10_cv_within_20pct.median()),
            'median_wellformed10_cv_bias_pct': float(df.wellformed10_cv_bias_pct.median()),
            'median_wellformed10_mean_bias_pct': float(df.wellformed10_mean_bias_pct.median())}
    json.dump(summ, open(a.out / 'summary.json', 'w'), indent=1)
    print(df.to_string(index=False)); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
