#!/usr/bin/env python3
"""
Repository cleanup / disk reclamation.

The big offender is per-epoch training checkpoints: each models/<run>/ keeps
checkpoint_epoch_1..N.pth (~1.1 GB each) that are only needed to resume
training. Inference (src/prediction.py) only ever loads best_model.pth, so the
epoch checkpoints can be removed safely.

SAFETY:
  - Dry-run by default; nothing is deleted unless --apply is passed.
  - best_model.pth and config.json are NEVER deleted.
  - Each cleanup category is opt-in via its own flag (or --all).

Examples:
    # See what checkpoint cleanup would reclaim (no deletion)
    python scripts/cleanup_repo.py --checkpoints

    # Actually delete per-epoch checkpoints
    python scripts/cleanup_repo.py --checkpoints --apply

    # Everything (checkpoints + pycache + old predictions + stale logs), dry-run
    python scripts/cleanup_repo.py --all

    # Keep epoch checkpoints for a model still being trained
    python scripts/cleanup_repo.py --checkpoints --apply --keep-epochs single_scale_448
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KEEP_NAMES = {"best_model.pth", "config.json"}


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"


def size_of(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for root, _dirs, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total
    except OSError:
        return 0


class Cleaner:
    def __init__(self, apply: bool):
        self.apply = apply
        self.reclaimed = 0
        self.n_files = 0

    def remove_file(self, fp: Path) -> None:
        sz = size_of(fp)
        self.reclaimed += sz
        self.n_files += 1
        tag = "DELETE" if self.apply else "would delete"
        print(f"  [{tag}] {fp.relative_to(PROJECT_ROOT)}  ({human(sz)})")
        if self.apply:
            try:
                fp.unlink()
            except OSError as e:
                print(f"    ! failed: {e}")

    def remove_tree(self, d: Path) -> None:
        sz = size_of(d)
        self.reclaimed += sz
        self.n_files += 1
        tag = "DELETE DIR" if self.apply else "would delete dir"
        print(f"  [{tag}] {d.relative_to(PROJECT_ROOT)}  ({human(sz)})")
        if self.apply:
            shutil.rmtree(d, ignore_errors=True)


def clean_checkpoints(c: Cleaner, keep_epochs: List[str]) -> None:
    print("\n== Per-epoch training checkpoints ==")
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.is_dir():
        print("  (no models/ dir)")
        return
    keep_set = set(keep_epochs)
    ckpts = sorted(models_dir.glob("*/checkpoint_epoch_*.pth"))
    ckpts = sorted(set(ckpts))
    for ckpt in ckpts:
        model_name = ckpt.parent.name
        if model_name in keep_set:
            print(f"  [keep] {ckpt.relative_to(PROJECT_ROOT)} (--keep-epochs)")
            continue
        if ckpt.name in KEEP_NAMES:
            continue
        c.remove_file(ckpt)


def clean_pycache(c: Cleaner) -> None:
    print("\n== __pycache__ directories ==")
    for d in sorted(PROJECT_ROOT.rglob("__pycache__")):
        if "venv" in d.parts:
            continue
        c.remove_tree(d)


def clean_predictions(c: Cleaner) -> None:
    print("\n== Old prediction outputs (predictions/) ==")
    pred = PROJECT_ROOT / "predictions"
    if not pred.is_dir():
        print("  (no predictions/ dir)")
        return
    for child in sorted(pred.iterdir()):
        if child.is_dir():
            c.remove_tree(child)
        else:
            c.remove_file(child)


def clean_logs(c: Cleaner, min_mb: float) -> None:
    print(f"\n== Stale large logs in scripts/ (> {min_mb} MB) ==")
    for log in sorted((PROJECT_ROOT / "scripts").glob("*.log")):
        if size_of(log) >= min_mb * 1024 * 1024:
            c.remove_file(log)


def main() -> int:
    ap = argparse.ArgumentParser(description="Repository cleanup / disk reclamation")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    ap.add_argument("--all", action="store_true", help="run all cleanup categories")
    ap.add_argument("--checkpoints", action="store_true", help="delete per-epoch checkpoint_epoch_*.pth")
    ap.add_argument("--pycache", action="store_true", help="delete __pycache__ dirs")
    ap.add_argument("--predictions", action="store_true", help="delete predictions/ outputs")
    ap.add_argument("--logs", action="store_true", help="delete large scripts/*.log")
    ap.add_argument("--keep-epochs", nargs="*", default=[],
                    help="model dir names whose epoch checkpoints should be kept")
    ap.add_argument("--logs-min-mb", type=float, default=1.0, help="size threshold for --logs")
    args = ap.parse_args()

    do_ckpt = args.checkpoints or args.all
    do_pycache = args.pycache or args.all
    do_pred = args.predictions or args.all
    do_logs = args.logs or args.all

    if not any([do_ckpt, do_pycache, do_pred, do_logs]):
        ap.error("nothing to do: pass --checkpoints/--pycache/--predictions/--logs or --all")

    mode = "APPLY (deleting)" if args.apply else "DRY-RUN (no deletion)"
    print("=" * 70)
    print(f"Repository cleanup - {mode}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 70)

    c = Cleaner(apply=args.apply)
    if do_ckpt:
        clean_checkpoints(c, args.keep_epochs)
    if do_pred:
        clean_predictions(c)
    if do_logs:
        clean_logs(c, args.logs_min_mb)
    if do_pycache:
        clean_pycache(c)

    print("\n" + "=" * 70)
    verb = "Reclaimed" if args.apply else "Would reclaim"
    print(f"{verb}: {human(c.reclaimed)} across {c.n_files} items")
    if not args.apply:
        print("Re-run with --apply to delete.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
