#!/usr/bin/env python3
"""Secondary analysis: remove predicted lesions below a minimum size.

A predicted lesion (connected coagulation+ablation component) whose
equivalent diameter is below --min-diam-um is relabelled as tissue. The
threshold is fixed in advance from the training/validation ground truth
(80 um: 0.4% of 499 annotated training/validation lesions are smaller); it is
not tuned on the test punch. Writes filtered prediction PNGs and lesion tables
in the same layout as predict_with_report, so score_preds.py and r3_stats.py
can score them.

Usage:
  nice -n 19 python scripts/analysis/size_rule_postproc.py --in predictions/r3_PVcont_report \
      --out predictions/r3_PVcont_report_size80 --min-diam-um 80 --um-per-px 0.8832
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analytics.lesions import LesionAnalyzer  # noqa: E402
from src.eval.metrics import rgb_to_class_mask  # noqa: E402

COLORS = np.array([[255, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0]], np.uint8)


def run(args):
    os.nice(19)
    src, out_root, min_d, um = args
    name = Path(src).name[:-len('_pred.png')]
    m = rgb_to_class_mask(cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB))
    n, lab, st, _ = cv2.connectedComponentsWithStats((m >= 2).astype(np.uint8), connectivity=8)
    min_area = np.pi * (min_d / 2 / um) ** 2
    small = np.zeros(n, bool); small[1:] = st[1:, cv2.CC_STAT_AREA] < min_area
    removed = int(small.sum())
    m[small[lab] & (m >= 2)] = 1
    d = Path(out_root) / name
    (d / 'images').mkdir(parents=True, exist_ok=True); (d / 'csv').mkdir(exist_ok=True)
    cv2.imwrite(str(d / 'images' / f'{name}_pred.png'), cv2.cvtColor(COLORS[m], cv2.COLOR_RGB2BGR))
    LesionAnalyzer().analyze_mask(m, None, str(d / 'csv'), summary_metrics=True, figures=False)
    return name, removed, int(n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--min-diam-um', type=float, default=80.0)
    ap.add_argument('--um-per-px', type=float, default=0.8832)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    os.nice(19)
    srcs = sorted(glob.glob(str(a.src / '*' / 'images' / '*_pred.png')))
    with Pool(a.workers) as pool:
        res = pool.map(run, [(s, str(a.out), a.min_diam_um, a.um_per_px) for s in srcs])
    summ = {'min_diam_um': a.min_diam_um, 'source': str(a.src),
            'removed_components': sum(r[1] for r in res), 'components_before': sum(r[2] for r in res),
            'per_section': {r[0]: {'removed': r[1], 'before': r[2]} for r in res}}
    json.dump(summ, open(a.out / 'size_rule.json', 'w'), indent=1)
    print(f"removed {summ['removed_components']} of {summ['components_before']} components (< {a.min_diam_um} um)")


if __name__ == '__main__':
    main()
