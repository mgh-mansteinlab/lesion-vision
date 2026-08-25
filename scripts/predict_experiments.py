#!/usr/bin/env python3
"""Predict remaining experiment folders (CC, CP, TPU, TC) into predictions/experiments/."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = "/home/ubuntu/SAQ/LesionSegmentation/data"
OUT = os.path.join(ROOT, "predictions", "experiments")
PYTHON = sys.executable
PREDICT = os.path.join(ROOT, "scripts", "predict_directory.py")
MODEL = os.path.join(ROOT, "models", "single_scale_448", "best_model.pth")

GROUPS = [
    ("CC", os.path.join(DATA, "CC")),
    ("CP", os.path.join(DATA, "CP")),
    ("TPU", os.path.join(DATA, "TPU")),
    ("TC", os.path.join(DATA, "TC")),
]


def main() -> None:
    only = sys.argv[1:]
    for group_id, data_dir in GROUPS:
        if only and group_id not in only:
            continue
        out_dir = os.path.join(OUT, group_id)
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            PYTHON, PREDICT,
            "--data-dir", data_dir,
            "--out-dir", out_dir,
            "--recursive",
            "--model", MODEL,
        ]
        print(f"\n======== {group_id} ========", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
