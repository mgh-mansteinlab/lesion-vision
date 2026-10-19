#!/usr/bin/env python3
"""Average class probabilities of several models (deep ensemble across architectures).

Members are probability folders from scripts/predict_probs.py (same section
names). Output: <out-dir>/<name>.npz with key 'p' = mean probability (uint8);
argmax gives the ensemble prediction, and mean p(coag)+p(abl) over a predicted
lesion is the cross-model agreement used by component_filter.py.

Usage:
  python scripts/analysis/ensemble_probs.py --out-dir <dir> <member_dir> <member_dir> ...
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np


def avg(args):
    name, members, out = args
    os.nice(19)
    acc = None
    for m in members:
        p = np.load(os.path.join(m, name + '.npz'))['p'].astype(np.uint16)
        acc = p if acc is None else acc + p
    np.savez_compressed(out + '.tmp.npz', p=np.round(acc / len(members)).astype(np.uint8))
    os.replace(out + '.tmp.npz', out)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('members', nargs='+')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    names = sorted(set.intersection(*[{os.path.basename(p)[:-4] for p in glob.glob(os.path.join(m, '*.npz'))} for m in a.members]))
    with Pool(a.workers) as pool:
        done = pool.map(avg, [(n, a.members, os.path.join(a.out_dir, n + '.npz')) for n in names])
    print(f'{len(done)} sections averaged over {len(a.members)} members -> {a.out_dir}')


if __name__ == '__main__':
    main()
