#!/usr/bin/env python3
"""Flatten per-sample prediction masks into a single directory.

``LesionPredictor.save_prediction`` writes each image's outputs under
``<out>/<name>/images/<name>_pred.png``. For 3D reconstruction (and for
sharing) it is convenient to gather every ``*_pred.png`` into one flat folder.

Importable:
    from scripts.collect_predictions import collect_predictions
    n = collect_predictions(src_dir, dest_dir)

CLI:
    python scripts/collect_predictions.py --src <pred_root> --dest <flat_dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path
from typing import List


def find_pred_masks(src_dir: os.PathLike | str) -> List[Path]:
    """Recursively find ``*_pred.png`` masks under ``src_dir``."""
    src_dir = os.fspath(src_dir)
    matches = glob.glob(os.path.join(src_dir, "**", "*_pred.png"), recursive=True)
    return sorted({Path(m).resolve() for m in matches})


def collect_predictions(
    src_dir: os.PathLike | str,
    dest_dir: os.PathLike | str,
    move: bool = False,
    overwrite: bool = True,
) -> int:
    """Copy (or move) every ``*_pred.png`` under ``src_dir`` into ``dest_dir``.

    Files keep their original basename. Returns the number of masks collected.
    """
    dest = Path(os.fspath(dest_dir)).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    masks = find_pred_masks(src_dir)
    n = 0
    for mask in masks:
        # Skip masks that already live in the destination directory.
        if mask.parent == dest:
            continue
        target = dest / mask.name
        if target.exists() and not overwrite:
            continue
        if move:
            shutil.move(str(mask), str(target))
        else:
            shutil.copy2(str(mask), str(target))
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Flatten per-sample *_pred.png masks into one directory")
    ap.add_argument("--src", required=True, help="Root directory to search recursively")
    ap.add_argument("--dest", required=True, help="Flat output directory for collected masks")
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    args = ap.parse_args()

    n = collect_predictions(args.src, args.dest, move=args.move)
    verb = "Moved" if args.move else "Copied"
    print(f"{verb} {n} prediction mask(s) into {Path(args.dest).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
