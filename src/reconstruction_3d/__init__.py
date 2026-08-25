"""3D volumetric reconstruction from serial-section lesion masks.

This subpackage vendors the LSA-3D registration + marching-cubes surface
pipeline (``run_pipeline.py`` + ``pipeline/s1..s6``). It consumes the color
``_pred.png`` masks produced by :mod:`src.prediction` (green=tissue,
blue=coagulation, red=ablation) and only makes sense for a *stack* of serial
sections -- a single image cannot be reconstructed in 3D.

The public entry point is :func:`run_reconstruction`, a thin wrapper that sets
the ``LSA_SRC`` / ``LSA_WORK`` / ``LSA_TISSUE_SRC`` environment variables the
stage scripts expect and shells out to ``run_pipeline.py``.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
RUN_PIPELINE = PACKAGE_ROOT / "run_pipeline.py"

MIN_STACK_SIZE = 2

__all__ = ["run_reconstruction", "count_masks", "MIN_STACK_SIZE", "ReconstructionError"]


class ReconstructionError(RuntimeError):
    """Raised when the 3D reconstruction cannot be run."""


def count_masks(masks_dir: os.PathLike | str) -> int:
    """Number of ``*.png`` section masks in ``masks_dir`` (non-recursive)."""
    masks_dir = os.fspath(masks_dir)
    return len([f for f in glob.glob(os.path.join(masks_dir, "*.png"))])


def run_reconstruction(
    masks_dir: os.PathLike | str,
    work_dir: Optional[os.PathLike | str] = None,
    samples_dir: Optional[os.PathLike | str] = None,
    stages: Optional[Sequence[int]] = None,
    python_executable: Optional[str] = None,
) -> Path:
    """Run the 3D reconstruction pipeline over a stack of section masks.

    Args:
        masks_dir: Directory of color-coded section mask PNGs (the stack).
        work_dir: Output directory. Defaults to ``<masks_dir>/../recon_3d``.
        samples_dir: Optional directory of original tissue-scan PNGs (improves
            registration; enables stage 2).
        stages: Optional explicit stage numbers (1..6). ``None`` runs all
            applicable stages.
        python_executable: Interpreter to use (defaults to the current one).

    Returns:
        Path to the working directory containing ``model/lesion_3d_surfaces.html``.

    Raises:
        ReconstructionError: if the stack has fewer than ``MIN_STACK_SIZE``
            masks, paths are invalid, or the pipeline exits non-zero.
    """
    masks_dir = Path(os.fspath(masks_dir)).expanduser().resolve()
    if not masks_dir.is_dir():
        raise ReconstructionError(f"masks_dir is not a directory: {masks_dir}")

    n_masks = count_masks(masks_dir)
    if n_masks < MIN_STACK_SIZE:
        raise ReconstructionError(
            f"3D reconstruction requires a stack of >= {MIN_STACK_SIZE} serial "
            f"sections, but found {n_masks} mask PNG(s) in {masks_dir}. "
            "Use a directory of collected per-section masks."
        )

    if work_dir is None:
        work_dir = masks_dir.parent / "recon_3d"
    work_dir = Path(os.fspath(work_dir)).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        python_executable or sys.executable,
        str(RUN_PIPELINE),
        "--masks",
        str(masks_dir),
        "--work",
        str(work_dir),
    ]

    if samples_dir is not None:
        samples_dir = Path(os.fspath(samples_dir)).expanduser().resolve()
        if not samples_dir.is_dir():
            raise ReconstructionError(f"samples_dir is not a directory: {samples_dir}")
        cmd += ["--samples", str(samples_dir)]

    if stages:
        cmd += [str(int(s)) for s in stages]

    print(f"[reconstruction_3d] {n_masks} sections -> {work_dir}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise ReconstructionError(
            f"3D reconstruction pipeline failed (exit {proc.returncode})."
        )
    return work_dir
