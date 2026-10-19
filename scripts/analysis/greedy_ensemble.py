#!/usr/bin/env python3
"""Greedy forward ensemble selection on validation, then the final pipeline.

Pre-specified in experiments/protocol_r3_comparison_*/AMENDMENT_final_model_selection.md.

Candidates: M2, C1, C2, C3, C4 (validation probabilities from predict_probs.py).
Score of a set = pooled 4-class macro-Dice on the validation sections of the
nested-decoded equal-weight mean of the members' probabilities. Start with the
best single candidate; repeatedly add the candidate that most increases the
score; stop when no addition increases it (ties -> fewer members).
The selected set then: mean probabilities on validation and PVcont ->
object filter tuned on validation (nested decoding) -> applied once to PVcont
-> r3_stats. Outputs in predictions/r3_greedy/.
"""
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))
import glob
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import component_filter  # noqa: E402
from component_filter import nested_decode  # noqa: E402
from src.eval.metrics import gt_class_mask  # noqa: E402

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data' / 'LesionSegmentation'))

P = ROOT / 'predictions'
CANDIDATES = {'M2': 'r3m2', 'C1': 'r3_c1_upernet_convnextb', 'C2': 'r3_c2_segformer_mitb3',
              'C3': 'r3_c3_deeplabv3plus_resnet50', 'C4': 'r3_c4_unetpp_effb3'}
VAL = str(DATA_ROOT) + '/data_val_r3m2'
OUT = P / 'r3_greedy'
PY = sys.executable


def run(cmd):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=ROOT, check=True)


def macro_dice(conf):
    tp = np.diag(conf).astype(float); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        return float(np.nanmean(2 * tp / (2 * tp + fp + fn)))


def main():
    os.nice(19)
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(Path(p).stem for p in glob.glob(f'{VAL}/masks/*.png'))
    avail = {k: t for k, t in CANDIDATES.items() if all((P / t / 'probs_val' / f'{n}.npz').exists() for n in names)}
    missing = sorted(set(CANDIDATES) - set(avail))
    print('candidates with validation probabilities:', sorted(avail), '| missing:', missing, flush=True)
    gts = {n: gt_class_mask(cv2.cvtColor(cv2.imread(f'{VAL}/masks/{n}.png'), cv2.COLOR_BGR2RGB)) for n in names}
    probs = {k: {n: np.load(P / t / 'probs_val' / f'{n}.npz')['p'] for n in names} for k, t in avail.items()}

    cache = {}
    def score(members):
        key = tuple(sorted(members))
        if key not in cache:
            conf = np.zeros((4, 4), np.int64)  # rows GT, cols prediction
            for n in names:
                mean = np.round(np.mean([probs[m][n].astype(np.float32) for m in key], axis=0)).astype(np.uint8)
                pr = nested_decode(mean)
                g = gts[n] if gts[n].shape == pr.shape else cv2.resize(gts[n], pr.shape[::-1], interpolation=cv2.INTER_NEAREST)
                np.add.at(conf, (g.ravel(), pr.ravel()), 1)
            cache[key] = macro_dice(conf)
            print(f'  {"+".join(key):20s} val macro-Dice {cache[key]:.4f}', flush=True)
        return cache[key]

    singles = {k: score([k]) for k in avail}
    selected = [max(singles, key=lambda k: (singles[k], k))]
    best = singles[selected[0]]
    history = [{'members': list(selected), 'val_macro_dice': best}]
    while True:
        trials = {k: score(selected + [k]) for k in avail if k not in selected}
        if not trials:
            break
        k = max(trials, key=lambda x: (trials[x], x))
        if trials[k] <= best:
            break
        selected.append(k); best = trials[k]
        history.append({'members': list(selected), 'val_macro_dice': best})
    print('SELECTED:', selected, f'val macro-Dice {best:.4f}', flush=True)
    # decoding variant chosen on validation for the selected set: nested, or
    # nested + re-decoding interior background that touches a crater
    component_filter.INTERIOR_BG = True; cache.clear()
    with_bg = score(selected)
    component_filter.INTERIOR_BG = False; cache.clear()
    use_bg = with_bg > best
    print(f'decoding: nested {best:.4f} vs nested+interior-bg {with_bg:.4f} -> {"interior-bg" if use_bg else "nested"}', flush=True)
    bgflag = ['--interior-bg'] if use_bg else []
    sel = {'selected': selected, 'val_macro_dice': best, 'interior_bg': bool(use_bg), 'val_macro_dice_interior_bg': with_bg, 'history': history, 'single_scores': singles,
           'missing_candidates': missing, 'all_scored_sets': {'+'.join(k): v for k, v in cache.items()}}
    json.dump(sel, open(OUT / 'selection.json', 'w'), indent=1)

    for split in ('probs_val', 'probs_PVcont'):
        run([PY, 'scripts/analysis/ensemble_probs.py', '--out-dir', OUT / split] + [P / CANDIDATES[m] / split for m in selected])
    params = OUT / 'filter_params_nested.json'
    run([PY, 'scripts/analysis/component_filter.py', 'tune', '--decode', 'nested', '--probs-dir', OUT / 'probs_val',
         '--gt-dir', f'{VAL}/masks', '--out', params] + bgflag)
    run([PY, 'scripts/analysis/component_filter.py', 'apply', '--decode', 'nested', '--probs-dir', OUT / 'probs_PVcont',
         '--params', params, '--out-dir', OUT / 'PVcont_filtered_nested'] + bgflag)
    run([PY, 'scripts/analysis/component_filter.py', 'apply', '--decode', 'nested', '--raw', '--probs-dir', OUT / 'probs_PVcont',
         '--out-dir', OUT / 'PVcont_nested'] + bgflag)
    if (ROOT / 'scripts' / 'analysis' / 'r3_stats.py').exists():  # evaluation statistics (analysis code, not part of the pipeline)
        run([PY, 'scripts/analysis/r3_stats.py', '--out', OUT / 'stats_filtered_nested.json',
             '--pipeline', f'nested={OUT / "PVcont_nested"}', '--pipeline', f'filtered_nested={OUT / "PVcont_filtered_nested"}'])
    sel['filter'] = json.load(open(params))['chosen']
    json.dump(sel, open(OUT / 'selection.json', 'w'), indent=1)
    print('done', OUT, flush=True)


if __name__ == '__main__':
    main()
