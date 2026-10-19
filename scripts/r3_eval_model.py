#!/usr/bin/env python3
"""Evaluate one trained round-3 model by the frozen protocol.

  1. score every checkpoint_epoch_*.pth on the validation sections (whole
     sections, no TTA, post-processing on); select the highest pooled 4-class
     macro-Dice (ties: earlier epoch)
  2. choose TTA x post-processing on validation with that checkpoint
  3. score the test punch (PVcont) once with the chosen configuration and,
     if different, also with the default one (no TTA, post-processing on)
  4. secondary: 80 um size rule on each PVcont prediction set
  5. statistics (r3_stats.py) for every PVcont prediction set
Outputs go to predictions/<tag>/; summary.json holds every decision.

Usage:
  python scripts/r3_eval_model.py --run-dir models/r3_model2_hardneg/model_20261017_095449 --tag r3m2
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# data location: set LESION_DATA_ROOT, default <repo>/data/LesionSegmentation
DATA_ROOT = Path(os.environ.get('LESION_DATA_ROOT', Path(__file__).resolve().parents[1] / 'data' / 'LesionSegmentation'))

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PV_DATA = str(DATA_ROOT) + '/data/PVcont'
PV_GT = str(DATA_ROOT) + '/data/PVcont/masks'
VAL_DATA = str(DATA_ROOT) + '/data_val_r3m2'
CONFIGS = {'notta_pp': [], 'notta_raw': ['--no-post-process'], 'tta_pp': ['--use-tta'], 'tta_raw': ['--use-tta', '--no-post-process']}
DEFAULT_CFG = 'notta_pp'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def run(cmd):
    log('$ ' + ' '.join(map(str, cmd)))
    r = subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode:
        print(r.stdout[-4000:])
        raise RuntimeError(f'command failed ({r.returncode}): {cmd[:4]}')
    return r.stdout


def predict(model, data, out, cfg, gpus):
    if len(glob.glob(f'{out}/*/images/*_pred.png')) and Path(out, 'REPORT.md').exists():
        log(f'reuse {out}')
        return
    run([PY, 'scripts/predict_with_report.py', '--data-dir', data, '--model', model, '--out-dir', out,
         '--gpus', gpus, '--no-figures', *CONFIGS[cfg]])


def score(pred_root, gt_dir, out_json):
    run([PY, 'scripts/analysis/score_preds.py', '--pred-root', pred_root, '--gt-dir', gt_dir, '--out', out_json])
    return json.load(open(out_json))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gpus', default='8')
    a = ap.parse_args()
    out = ROOT / 'predictions' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    summ_p = out / 'summary.json'
    summ = json.load(open(summ_p)) if summ_p.exists() else {'run_dir': a.run_dir, 'tag': a.tag}
    save = lambda: json.dump(summ, open(summ_p, 'w'), indent=1)
    val_gt = f'{VAL_DATA}/masks'

    # 1. checkpoint selection on validation
    ckpts = sorted(glob.glob(f'{a.run_dir}/checkpoint_epoch_*.pth'), key=lambda p: int(re.findall(r'(\d+)\.pth', p)[0]))
    vals = summ.get('val_checkpoints', {})
    for c in ckpts:
        ep = int(re.findall(r'(\d+)\.pth', c)[0])
        if str(ep) in vals:
            continue
        d = out / 'val_ckpt' / f'epoch_{ep:02d}'
        predict(c, VAL_DATA, d, DEFAULT_CFG, a.gpus)
        s = score(d, val_gt, d / 'score.json')
        vals[str(ep)] = {'macro_dice': s['macro_dice'], 'per_class_dice': {k: v['dice'] for k, v in s['per_class'].items()},
                         'lesions': s['lesions']}
        summ['val_checkpoints'] = vals; save()
        log(f'epoch {ep}: val macro-Dice {s["macro_dice"]:.4f}')
    best_ep = max(sorted(vals, key=int), key=lambda e: (vals[e]['macro_dice'], -int(e)))
    best_ckpt = f'{a.run_dir}/checkpoint_epoch_{best_ep}.pth'
    summ['selected_epoch'] = int(best_ep); summ['selected_checkpoint'] = best_ckpt; save()
    log(f'selected epoch {best_ep} (val macro-Dice {vals[best_ep]["macro_dice"]:.4f})')

    # 2. inference configuration on validation
    cfgs = summ.get('val_configs', {})
    for cfg in CONFIGS:
        if cfg in cfgs:
            continue
        d = out / 'val_ckpt' / f'epoch_{int(best_ep):02d}' if cfg == DEFAULT_CFG else out / 'val_cfg' / cfg
        if cfg != DEFAULT_CFG:
            predict(best_ckpt, VAL_DATA, d, cfg, a.gpus)
        s = score(d, val_gt, d / 'score.json') if cfg != DEFAULT_CFG else json.load(open(d / 'score.json'))
        cfgs[cfg] = s['macro_dice']; summ['val_configs'] = cfgs; save()
        log(f'config {cfg}: val macro-Dice {s["macro_dice"]:.4f}')
    chosen = max(CONFIGS, key=lambda k: (cfgs[k], k == DEFAULT_CFG))
    summ['chosen_config'] = chosen; save()
    log(f'chosen inference config: {chosen}')

    # 3-5. PVcont, size rule, statistics
    pv_sets = [chosen] + ([DEFAULT_CFG] if chosen != DEFAULT_CFG else [])
    pipelines = []
    for cfg in pv_sets:
        d = out / f'PVcont_{cfg}'
        predict(best_ckpt, PV_DATA, d, cfg, a.gpus)
        pipelines.append(f'{cfg}={d}')
        d80 = out / f'PVcont_{cfg}_size80'
        if not (d80 / 'size_rule.json').exists():
            run([PY, 'scripts/analysis/size_rule_postproc.py', '--in', d, '--out', d80, '--workers', '16'])
        pipelines.append(f'{cfg}_size80={d80}')
    stats_p = out / 'stats.json'
    run([PY, 'scripts/analysis/r3_stats.py', '--out', stats_p] + sum([['--pipeline', p] for p in pipelines], []))
    st = json.load(open(stats_p))
    summ['pvcont'] = {}
    for name, p in st['pipelines'].items():
        a_, o = p['pixel']['all'], p['object']['tol_250um']
        summ['pvcont'][name] = {'macro_dice': a_['macro_dice'], 'macro_dice_ci95': a_['macro_dice_ci95'],
                                'dice': {k: v['dice'] for k, v in a_['per_class'].items()},
                                'bands_macro_dice': {b: p['pixel'][b]['macro_dice'] for b in ('superficial', 'middle', 'deep')},
                                'detection_recall': o['all']['recall'], 'detection_precision': o['all']['precision'],
                                'precision_by_band': {b: o[b]['precision'] for b in ('superficial', 'middle', 'deep')},
                                'bland_altman': p['object']['bland_altman']}
    summ['status'] = 'done'; save()
    log(f'done: {summ_p}')

    # 6. secondary analyses on class probabilities: object filter (tuned on
    # validation) and, once all members exist, the cross-architecture ensemble
    try:
        run([PY, 'scripts/r3_postprob.py', '--tag', a.tag, '--ckpt', best_ckpt, '--val-data', VAL_DATA, '--gpus', a.gpus])
        summ['postprob'] = {k: json.load(open(out / f)) for k, f in (('argmax', 'filter_summary.json'), ('nested', 'filter_summary_nested.json')) if (out / f).exists()}; save()
    except Exception as e:  # secondary: never fail the primary evaluation
        log(f'postprob failed: {e}')


if __name__ == '__main__':
    main()
