#!/usr/bin/env python3
"""Probability-based secondary analyses for one model, and the ensemble.

Pre-specified in experiments/protocol_r3_comparison_*/AMENDMENT_object_filter_ensemble.md.

Per model (--tag, --ckpt, --val-data):
  1. class probabilities, no TTA, no post-processing, on its validation
     sections and on PVcont (scripts/predict_probs.py)
  2. object filter tuned on validation (scripts/analysis/component_filter.py),
     applied once to PVcont; the unfiltered argmax is written alongside
  3. statistics (r3_stats.py) -> predictions/<tag>/stats_filtered.json
Then, when every ensemble member has probabilities, the ensemble (mean of M2
and C1-C4 probabilities) goes through the same steps under tag r3_ensemble.

Usage:
  python scripts/r3_postprob.py --tag r3m2 --ckpt <ckpt> --val-data $LESION_DATA_ROOT/data_val_r3m2
  python scripts/r3_postprob.py --ensemble-only
"""
import os
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data' / 'LesionSegmentation'))

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PV_DATA = str(DATA_ROOT) + '/data/PVcont'
ENS_VAL = str(DATA_ROOT) + '/data_val_r3m2'
MEMBERS = ['r3m2', 'r3_c1_upernet_convnextb', 'r3_c2_segformer_mitb3', 'r3_c3_deeplabv3plus_resnet50', 'r3_c4_unetpp_effb3']


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def run(cmd):
    log('$ ' + ' '.join(map(str, cmd)))
    r = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        print(r.stdout[-4000:]); raise RuntimeError(f'failed: {cmd[:3]}')
    return r.stdout


def n_sections(data):
    return len(glob.glob(f'{data}/masks/*.png'))


def n_npz(d):
    return len(glob.glob(f'{d}/*.npz'))


def filter_and_stats(out, val_probs, val_data, pv_probs, decode='argmax'):
    sfx = '' if decode == 'argmax' else '_nested'
    params = out / f'filter_params{sfx}.json'
    raw_name = 'argmax' if decode == 'argmax' else 'nested'
    dec = ['--decode', decode]
    run([PY, 'scripts/analysis/component_filter.py', 'tune', '--probs-dir', val_probs, '--gt-dir', f'{val_data}/masks', '--out', params] + dec)
    run([PY, 'scripts/analysis/component_filter.py', 'apply', '--probs-dir', pv_probs, '--params', params, '--out-dir', out / f'PVcont_filtered{sfx}'] + dec)
    run([PY, 'scripts/analysis/component_filter.py', 'apply', '--probs-dir', pv_probs, '--raw', '--out-dir', out / f'PVcont_{raw_name}'] + dec)
    run([PY, 'scripts/analysis/r3_stats.py', '--out', out / f'stats_filtered{sfx}.json',
         '--pipeline', f'{raw_name}={out / f"PVcont_{raw_name}"}', '--pipeline', f'filtered{sfx}={out / f"PVcont_filtered{sfx}"}'])
    pr = json.load(open(params))['chosen']
    st = json.load(open(out / f'stats_filtered{sfx}.json'))
    res = {'filter_params': {k: pr[k] for k in ('tau', 'd_um')},
           'val_filtered': {k: pr[k] for k in ('f1', 'precision', 'recall', 'macro_dice')}}
    for name, p in st['pipelines'].items():
        a, o = p['pixel']['all'], p['object']['tol_250um']
        res[name] = {'macro_dice': a['macro_dice'], 'macro_dice_ci95': a['macro_dice_ci95'],
                     'bands': {b: p['pixel'][b]['macro_dice'] for b in ('superficial', 'middle', 'deep')},
                     'detection_recall': o['all']['recall'], 'detection_precision': o['all']['precision'],
                     'deep_precision': o['deep']['precision']}
    json.dump(res, open(out / f'filter_summary{sfx}.json', 'w'), indent=1)
    f = res[f'filtered{sfx}']
    log(f'{out.name} [{decode}]: filter {res["filter_params"]} | PVcont {raw_name} {res[raw_name]["macro_dice"]:.3f} P {res[raw_name]["detection_precision"]:.3f} '
        f'-> filtered {f["macro_dice"]:.3f} P {f["detection_precision"]:.3f} R {f["detection_recall"]:.3f}')


def per_model(tag, ckpt, val_data, gpus):
    out = ROOT / 'predictions' / tag
    out.mkdir(parents=True, exist_ok=True)
    for data, d in ((val_data, out / 'probs_val'), (PV_DATA, out / 'probs_PVcont')):
        if n_npz(d) < n_sections(data):
            run([PY, 'scripts/predict_probs.py', '--data-dir', data, '--model', ckpt, '--out-dir', d, '--gpus', gpus])
    for decode, sfx in (('argmax', ''), ('nested', '_nested')):
        if not (out / f'filter_summary{sfx}.json').exists():
            filter_and_stats(out, out / 'probs_val', val_data, out / 'probs_PVcont', decode)


def ensemble():
    ready = all(n_npz(ROOT / 'predictions' / m / 'probs_val') >= n_sections(ENS_VAL) and
                n_npz(ROOT / 'predictions' / m / 'probs_PVcont') >= n_sections(PV_DATA) for m in MEMBERS)
    out = ROOT / 'predictions' / 'r3_ensemble'
    if not ready:
        log('ensemble: not all members have probabilities yet'); return
    if (out / 'filter_summary_nested.json').exists():
        log('ensemble: already done'); return
    for split in ('probs_val', 'probs_PVcont'):
        run([PY, 'scripts/analysis/ensemble_probs.py', '--out-dir', out / split] + [ROOT / 'predictions' / m / split for m in MEMBERS])
    json.dump({'members': MEMBERS, 'weights': 'equal', 'member_inference': 'no TTA, no post-processing'}, open(out / 'members.json', 'w'))
    for decode in ('argmax', 'nested'):  # nested = pre-specified primary ensemble pipeline
        filter_and_stats(out, out / 'probs_val', ENS_VAL, out / 'probs_PVcont', decode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag'); ap.add_argument('--ckpt'); ap.add_argument('--val-data')
    ap.add_argument('--gpus', default='8')
    ap.add_argument('--ensemble-only', action='store_true')
    a = ap.parse_args()
    if not a.ensemble_only:
        per_model(a.tag, a.ckpt, a.val_data, a.gpus)
    ensemble()


if __name__ == '__main__':
    main()
