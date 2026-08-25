"""SAM3 segmentation backend adapter.

Wraps the trained SAM3 lesion-segmentation predictor that lives outside this
repo (``SAM2LS/predict_sam3.py``). That script mirrors
:class:`src.prediction.LesionPredictor` (tissue bbox -> tiling -> per-tile SAM3
text-prompt inference -> probability-space stitching -> concentric-lesion
post-processing -> colored mask + IoU/Dice), and writes ``<name>_pred.png`` so
its outputs feed the same collector / 3D reconstruction chain.

Because the SAM3 package and its ~10 GB checkpoint are heavyweight and ship
their own (sometimes partial) environment, this adapter shells out to
``predict_sam3.py`` with a configurable interpreter and SAM2LS root rather than
importing ``sam3`` into this process.

Configuration (override via args or environment):
    SAM2LS_ROOT   root of the SAM2LS checkout (required unless sam_root is passed)
    SAM_PYTHON    interpreter that can import ``sam3`` (default: current python)
    SAM_CHECKPOINT  path to checkpoint.pt
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_SAM_ROOT_ENV = os.environ.get("SAM2LS_ROOT")
DEFAULT_SAM_ROOT = Path(_SAM_ROOT_ENV) if _SAM_ROOT_ENV else None

__all__ = ["run_sam_prediction", "resolve_sam_paths", "SamBackendError"]


class SamBackendError(RuntimeError):
    """Raised when the SAM backend cannot be located or invoked."""


def resolve_sam_paths(
    sam_root: Optional[os.PathLike | str] = None,
    checkpoint: Optional[os.PathLike | str] = None,
) -> tuple[Path, Path, Path]:
    """Resolve and validate (sam_root, predict_script, checkpoint).

    Raises SamBackendError with an actionable message if anything is missing.
    """
    root_src = sam_root or DEFAULT_SAM_ROOT
    if root_src is None:
        raise SamBackendError(
            "SAM2LS root not set. Set SAM2LS_ROOT or pass sam_root."
        )
    root = Path(os.fspath(root_src)).expanduser().resolve()
    if not root.is_dir():
        raise SamBackendError(
            f"SAM2LS root not found: {root}. Set SAM2LS_ROOT or pass sam_root."
        )

    script = root / "predict_sam3.py"
    if not script.is_file():
        raise SamBackendError(f"predict_sam3.py not found under {root}")

    if checkpoint is not None:
        ckpt = Path(os.fspath(checkpoint)).expanduser().resolve()
    else:
        env_ckpt = os.environ.get("SAM_CHECKPOINT")
        ckpt = (
            Path(env_ckpt).expanduser().resolve()
            if env_ckpt
            else root / "exp_log" / "sam3_lesion_seg" / "checkpoints" / "checkpoint.pt"
        )
    if not ckpt.is_file():
        raise SamBackendError(
            f"SAM3 checkpoint not found: {ckpt}. Set SAM_CHECKPOINT or pass checkpoint."
        )
    return root, script, ckpt


def run_sam_prediction(
    output_dir: os.PathLike | str,
    image: Optional[os.PathLike | str] = None,
    input_dir: Optional[os.PathLike | str] = None,
    checkpoint: Optional[os.PathLike | str] = None,
    sam_root: Optional[os.PathLike | str] = None,
    python_executable: Optional[str] = None,
    tile_size: int = 448,
    overlap: int = 112,
    device: str = "cuda:0",
    score_threshold: float = 0.3,
    use_tta: bool = False,
    post_process: bool = True,
    gt_dir: Optional[os.PathLike | str] = None,
) -> int:
    """Run SAM3 prediction on a single image or a directory.

    Exactly one of ``image`` / ``input_dir`` must be provided. Returns the
    subprocess exit code (0 on success).
    """
    if (image is None) == (input_dir is None):
        raise SamBackendError("Provide exactly one of image= or input_dir=.")

    root, script, ckpt = resolve_sam_paths(sam_root, checkpoint)
    py = python_executable or os.environ.get("SAM_PYTHON") or sys.executable

    output_dir = Path(os.fspath(output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        py,
        str(script),
        "--checkpoint",
        str(ckpt),
        "--output-dir",
        str(output_dir),
        "--tile-size",
        str(tile_size),
        "--overlap",
        str(overlap),
        "--device",
        device,
        "--score-threshold",
        str(score_threshold),
    ]
    if image is not None:
        cmd += ["--image", str(Path(os.fspath(image)).expanduser().resolve())]
    else:
        cmd += ["--input-dir", str(Path(os.fspath(input_dir)).expanduser().resolve())]
    if gt_dir is not None:
        cmd += ["--gt-dir", str(Path(os.fspath(gt_dir)).expanduser().resolve())]
    if use_tta:
        cmd.append("--use-tta")
    if not post_process:
        cmd.append("--no-post-process")

    print(f"[sam_backend] running SAM3 via {py}\n  cwd={root}")
    # Run with cwd at the SAM root so predict_sam3.py's relative sam3 imports
    # and default bpe asset path resolve correctly.
    proc = subprocess.run(cmd, cwd=str(root), check=False)
    return proc.returncode
