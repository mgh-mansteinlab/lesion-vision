#!/usr/bin/env python3
"""
Interactive pipeline CLI for Lesion Vision.

Author: Manstein Lab, CBRC, Massachusetts General Hospital
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Light-weight, torch-free helpers can be imported eagerly. The torch-backed
# predictors are imported lazily inside their workflows to keep CLI startup fast.
from scripts.collect_predictions import collect_predictions, find_pred_masks
from src.reconstruction_3d import (
    MIN_STACK_SIZE,
    ReconstructionError,
    run_reconstruction,
)


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


def color(text: str, c: str) -> str:
    return f"{c}{text}{C.RESET}"


def header() -> None:
    line = "=" * 86
    print(color(line, C.CYAN))
    print(color("Lesion Vision - Unified Pipeline CLI", C.BOLD + C.CYAN))
    print(color("Author: Manstein Lab, CBRC, Massachusetts General Hospital", C.BOLD + C.WHITE))
    print(color(line, C.CYAN))
    print(
        color(
            "Guided workflows for: tiling, training, prediction, and analysis/evaluation.",
            C.DIM + C.WHITE,
        )
    )
    print()


def info(msg: str) -> None:
    print(color(f"[INFO] {msg}", C.BLUE))


def ok(msg: str) -> None:
    print(color(f"[OK] {msg}", C.GREEN))


def warn(msg: str) -> None:
    print(color(f"[WARN] {msg}", C.YELLOW))


def err(msg: str) -> None:
    print(color(f"[ERROR] {msg}", C.RED))


def ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(color(f"{prompt}{suffix}: ", C.MAGENTA)).strip()
    if not raw and default is not None:
        return default
    return raw


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        raw = input(color(f"{prompt} ({default_str}): ", C.MAGENTA)).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        warn("Please answer with y/yes or n/no.")


def ask_choice(prompt: str, choices: Iterable[str], default: Optional[str] = None) -> str:
    choices_list = list(choices)
    printable = "/".join(choices_list)
    while True:
        raw = ask(f"{prompt} ({printable})", default=default).strip()
        if raw in choices_list:
            return raw
        warn(f"Invalid choice: {raw}")


def ask_int(prompt: str, default: int, minimum: Optional[int] = None) -> int:
    while True:
        raw = ask(prompt, default=str(default))
        try:
            v = int(raw)
        except ValueError:
            warn("Enter an integer.")
            continue
        if minimum is not None and v < minimum:
            warn(f"Value must be >= {minimum}.")
            continue
        return v


def ask_float(prompt: str, default: float, minimum: Optional[float] = None) -> float:
    while True:
        raw = ask(prompt, default=str(default))
        try:
            v = float(raw)
        except ValueError:
            warn("Enter a numeric value.")
            continue
        if minimum is not None and v < minimum:
            warn(f"Value must be >= {minimum}.")
            continue
        return v


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for token in raw.replace(",", " ").split():
        vals.append(int(token))
    if not vals:
        raise ValueError("No values provided")
    return vals


def ask_path(
    prompt: str,
    default: Optional[str] = None,
    must_exist: bool = False,
    must_be_dir: Optional[bool] = None,
    create_if_missing: bool = False,
) -> Path:
    while True:
        raw = ask(prompt, default=default).strip()
        if not raw:
            warn("A path is required.")
            continue
        p = Path(raw).expanduser().resolve()
        if must_exist and not p.exists():
            warn(f"Path does not exist: {p}")
            continue
        if must_be_dir is True and p.exists() and not p.is_dir():
            warn(f"Expected directory, got file: {p}")
            continue
        if must_be_dir is False and p.exists() and not p.is_file():
            warn(f"Expected file, got directory: {p}")
            continue
        if create_if_missing and not p.exists():
            if must_be_dir is False:
                if p.parent and not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
            else:
                p.mkdir(parents=True, exist_ok=True)
        return p


def run_subprocess(cmd: List[str], env: Optional[dict] = None) -> int:
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(color("\nRunning command:", C.CYAN))
    print(color(pretty, C.WHITE))
    print()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
        if proc.returncode == 0:
            ok("Command completed successfully.")
        else:
            err(f"Command failed with exit code {proc.returncode}.")
        return proc.returncode
    except KeyboardInterrupt:
        warn("Interrupted by user.")
        return 130


def run_tiling_workflow() -> int:
    print(color("\n--- Tiling Workflow ---", C.BOLD + C.CYAN))
    mode = ask_choice("Tiling mode", ["directory", "single_pair"], default="directory")

    output_dir = ask_path(
        "Output directory for generated tiles/masks",
        default=str(PROJECT_ROOT / "data"),
        must_be_dir=True,
        create_if_missing=True,
    )
    tile_sizes = parse_int_list(ask("Tile sizes (space/comma separated)", default="448 768"))
    workers = ask_int("Number of workers", default=32, minimum=1)
    visualize = ask_yes_no("Create sample tile visualization?", default=False)
    dry_run = ask_yes_no("Dry run (discover only, no tile writing)?", default=False)

    cmd = [sys.executable, str(PROJECT_ROOT / "utils" / "TileGen.py"), "--output", str(output_dir)]
    cmd += ["--tile-size"] + [str(s) for s in tile_sizes]
    cmd += ["--workers", str(workers)]
    if visualize:
        cmd.append("--visualize")
    if dry_run:
        cmd.append("--dry-run")

    if mode == "directory":
        input_dir = ask_path("Input directory containing images/masks", must_exist=True, must_be_dir=True)
        cmd += ["--dir", str(input_dir)]
    else:
        image = ask_path("Image path (.tif)", must_exist=True, must_be_dir=False)
        mask = ask_path("Mask path (.png)", must_exist=True, must_be_dir=False)
        cmd += ["--image", str(image), "--mask", str(mask)]

    return run_subprocess(cmd)


def run_train_workflow() -> int:
    print(color("\n--- Training Workflow ---", C.BOLD + C.CYAN))

    tiles_ready = ask_yes_no("Are training tiles already prepared?", default=True)
    if not tiles_ready:
        info("Starting tiling first because training depends on tile/mask directories.")
        code = run_tiling_workflow()
        if code != 0:
            warn("Tiling did not finish successfully. Aborting training.")
            return code
        if not ask_yes_no("Proceed to training now?", default=True):
            warn("Training cancelled by user.")
            return 0

    train_mode = ask_choice(
        "Training launch mode",
        ["manual_cli", "experiment_yaml", "train_sh"],
        default="manual_cli",
    )

    if train_mode == "train_sh":
        return run_subprocess(["bash", str(PROJECT_ROOT / "scripts" / "train.sh")])

    if train_mode == "experiment_yaml":
        yml = ask_path(
            "Experiment YAML path",
            default=str(PROJECT_ROOT / "configs" / "experiments" / "baseline_multi_scale.yaml"),
            must_exist=True,
            must_be_dir=False,
        )
        env = os.environ.copy()
        data_dir = ask_path("DATA_DIR (tiled data root)", must_exist=True, must_be_dir=True)
        output_dir = ask_path(
            "OUTPUT_DIR (model outputs)",
            default=str(PROJECT_ROOT / "models"),
            must_be_dir=True,
            create_if_missing=True,
        )
        env["DATA_DIR"] = str(data_dir)
        env["OUTPUT_DIR"] = str(output_dir)
        if ask_yes_no("Set WORLD_SIZE for distributed launch?", default=False):
            env["WORLD_SIZE"] = str(ask_int("WORLD_SIZE", default=1, minimum=1))
        return run_subprocess([sys.executable, str(PROJECT_ROOT / "scripts" / "launch_experiment.py"), str(yml)], env=env)

    data_dir = ask_path("Data directory (contains tile_*/mask_* folders)", must_exist=True, must_be_dir=True)
    output_dir = ask_path(
        "Output directory for checkpoints/logs",
        default=str(PROJECT_ROOT / "models"),
        must_be_dir=True,
        create_if_missing=True,
    )
    backbone = ask("Backbone", default="tu-convnext_base")
    img_size = ask_int("Model image size", default=512, minimum=64)
    tile_sizes = parse_int_list(ask("Disk tile sizes (space/comma)", default="448 768"))
    shift_type = ask_choice(
        "Shift type",
        ["all", "no_shift", "ovlp_20", "ovlp_40", "ovlp_60", "ovlp_80"],
        default="all",
    )

    epochs = ask_int("Epochs", default=100, minimum=1)
    batch_size = ask_int("Batch size (per GPU)", default=2, minimum=1)
    grad_acc = ask_int("Gradient accumulation steps", default=4, minimum=1)
    lr = ask_float("Learning rate", default=1e-4, minimum=1e-8)
    val_split = ask_float("Validation split", default=0.2, minimum=0.0)
    patience = ask_int("Early stopping patience", default=15, minimum=1)
    seed = ask_int("Random seed", default=42, minimum=0)
    num_workers = ask_int("DataLoader workers", default=4, minimum=0)

    loss_type = ask_choice("Loss type", ["ce", "compound", "focal"], default="compound")
    scheduler = ask_choice("Scheduler", ["plateau", "cosine"], default="cosine")
    warmup_epochs = ask_int("Warmup epochs", default=5, minimum=0)

    class_weights = ask(
        'Class weights ("auto", "none", or comma floats)',
        default="auto",
    )
    weighted_sampling = ask_yes_no("Enable weighted sampling?", default=True)
    sampling_alpha = ask_float("Sampling alpha", default=0.75, minimum=0.0)

    augment = ask_yes_no("Enable augmentation?", default=True)
    activation_checkpointing = ask_yes_no("Enable activation checkpointing?", default=True)
    deterministic = ask_yes_no("Enable deterministic algorithms?", default=False)
    amp = ask_yes_no("Enable mixed precision (AMP)?", default=True)
    use_tta = ask_yes_no("Use TTA during validation visualization?", default=False)

    resume_path: Optional[Path] = None
    if ask_yes_no("Resume from checkpoint?", default=False):
        resume_path = ask_path("Checkpoint path (.pth)", must_exist=True, must_be_dir=False)

    use_wandb = ask_yes_no("Enable Weights & Biases logging?", default=False)
    wandb_project = ask("W&B project", default="lesion-vision") if use_wandb else None
    wandb_entity = ask("W&B entity/team (optional)", default="") if use_wandb else None
    wandb_name = ask("W&B run name (optional)", default="") if use_wandb else None
    wandb_tags = ask("W&B tags (space-separated, optional)", default="") if use_wandb else None
    wandb_log_model = ask_yes_no("Upload model artifact to W&B?", default=False) if use_wandb else False

    distributed = ask_yes_no("Run distributed training?", default=False)
    world_size = ask_int("World size (num GPUs)", default=1, minimum=1) if distributed else None
    dist_backend = ask_choice("Distributed backend", ["nccl", "gloo"], default="nccl") if distributed else None

    cmd: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train.py"),
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(output_dir),
        "--backbone",
        backbone,
        "--img_size",
        str(img_size),
        "--shift_type",
        shift_type,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_acc),
        "--learning_rate",
        str(lr),
        "--val_split",
        str(val_split),
        "--patience",
        str(patience),
        "--seed",
        str(seed),
        "--num_workers",
        str(num_workers),
        "--loss_type",
        loss_type,
        "--class_weights",
        class_weights,
        "--scheduler_type",
        scheduler,
        "--warmup_epochs",
        str(warmup_epochs),
    ]
    cmd += ["--tile_size"] + [str(s) for s in tile_sizes]

    if augment:
        cmd.append("--augment")
    if weighted_sampling:
        cmd += ["--weighted_sampling", "--sampling_alpha", str(sampling_alpha)]
    if activation_checkpointing:
        cmd.append("--activation_checkpointing")
    if deterministic:
        cmd.append("--deterministic")
    if use_tta:
        cmd.append("--use_tta")
    if not amp:
        cmd.append("--no_amp")
    if resume_path is not None:
        cmd += ["--resume", str(resume_path)]
    if distributed and world_size is not None and dist_backend is not None:
        cmd += ["--distributed", "--world_size", str(world_size), "--dist_backend", dist_backend]

    if use_wandb and wandb_project is not None:
        cmd += ["--use_wandb", "--wandb_project", wandb_project]
        if wandb_entity:
            cmd += ["--wandb_entity", wandb_entity]
        if wandb_name:
            cmd += ["--wandb_name", wandb_name]
        if wandb_tags:
            cmd += ["--wandb_tags"] + wandb_tags.split()
        if wandb_log_model:
            cmd.append("--wandb_log_model")

    return run_subprocess(cmd)


def run_predict_single_workflow() -> int:
    print(color("\n--- Predict: Single Image ---", C.BOLD + C.CYAN))
    model_path = ask_path("Model checkpoint path (.pth)", must_exist=True, must_be_dir=False)
    image_path = ask_path("Input image path (.tif/.ndpi)", must_exist=True, must_be_dir=False)
    output_root = ask_path(
        "Output directory",
        default=str(PROJECT_ROOT / "predictions"),
        must_be_dir=True,
        create_if_missing=True,
    )
    tile_size = ask_int("Tile size", default=512, minimum=64)
    overlap = ask_int("Tile overlap", default=128, minimum=0)
    summary_metrics = ask_yes_no("Generate summary CSV metrics?", default=True)
    figures = ask_yes_no("Generate lesion figures?", default=True)
    use_tta = ask_yes_no("Use test-time augmentation?", default=False)
    use_post_process = ask_yes_no("Enable topology post-processing?", default=True)

    output_path = output_root / f"{image_path.stem}_pred.png"
    info(f"Prediction output target: {output_path}")

    try:
        from src.prediction import LesionPredictor

        predictor = LesionPredictor(model_path=str(model_path))
        predictor.save_prediction(
            image_path=str(image_path),
            output_path=str(output_path),
            tile_size=tile_size,
            overlap=overlap,
            summary_metrics=summary_metrics,
            figures=figures,
            use_tta=use_tta,
            use_post_process=use_post_process,
        )
        ok("Single-image prediction completed.")
        return 0
    except Exception as exc:
        err(f"Prediction failed: {exc}")
        return 1


def run_predict_directory_workflow(multi_gpu: bool = False) -> int:
    title = "Predict: Directory (Multi-GPU)" if multi_gpu else "Predict: Directory"
    print(color(f"\n--- {title} ---", C.BOLD + C.CYAN))
    model_path = ask_path("Model checkpoint path (.pth)", must_exist=True, must_be_dir=False)
    input_dir = ask_path("Input directory (.tif/.ndpi files)", must_exist=True, must_be_dir=True)
    output_dir = ask_path(
        "Output directory",
        default=str(PROJECT_ROOT / "predictions" / ("multi_gpu_batch" if multi_gpu else "batch")),
        must_be_dir=True,
        create_if_missing=True,
    )
    tile_size = ask_int("Tile size", default=512, minimum=64)
    overlap = ask_int("Tile overlap", default=128, minimum=0)
    summary_metrics = ask_yes_no("Generate summary CSV metrics?", default=True)
    figures = ask_yes_no("Generate lesion figures?", default=True)
    use_tta = ask_yes_no("Use test-time augmentation?", default=False)

    try:
        if multi_gpu:
            from src.multi_gpu_batch_predictor import MultiGPUBatchPredictor

            num_gpus_raw = ask("Number of GPUs (blank=auto)", default="")
            num_gpus = int(num_gpus_raw) if num_gpus_raw.strip() else None
            workers_per_gpu = ask_int("Max workers per GPU", default=1, minimum=1)
            predictor = MultiGPUBatchPredictor(
                model_path=str(model_path),
                num_gpus=num_gpus,
                max_workers_per_gpu=workers_per_gpu,
            )
            predictor.process_directory(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                tile_size=tile_size,
                overlap=overlap,
                summary_metrics=summary_metrics,
                figures=figures,
                use_tta=use_tta,
            )
        else:
            from src.prediction import LesionPredictor

            use_post_process = ask_yes_no("Enable topology post-processing?", default=True)
            predictor = LesionPredictor(model_path=str(model_path))
            predictor.process_directory(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                tile_size=tile_size,
                overlap=overlap,
                summary_metrics=summary_metrics,
                figures=figures,
                use_tta=use_tta,
                use_post_process=use_post_process,
            )
        ok("Directory prediction completed.")
        offer_3d_chain(output_dir)
        return 0
    except Exception as exc:
        err(f"Directory prediction failed: {exc}")
        return 1


def offer_3d_chain(masks_root: Path) -> None:
    """After a multi-section prediction, optionally collect masks + build 3D.

    3D reconstruction only makes sense for a *stack* of serial sections, so this
    is skipped (with a note) when fewer than two prediction masks are present.
    """
    masks = find_pred_masks(masks_root)
    n = len(masks)
    if n < MIN_STACK_SIZE:
        if n == 1:
            info("Only one section predicted; 3D reconstruction needs a stack (>= 2). Skipping.")
        return

    print()
    if not ask_yes_no(
        f"Found {n} prediction masks. Collect them and build a 3D model?",
        default=False,
    ):
        return

    dest = masks_root / "all_predictions"
    copied = collect_predictions(masks_root, dest)
    ok(f"Collected {copied} mask(s) into {dest}")
    _run_reconstruction_prompts(default_masks=dest)


def _run_reconstruction_prompts(default_masks: Optional[Path] = None) -> int:
    """Shared prompts + execution for the 3D reconstruction pipeline."""
    masks_dir = ask_path(
        "Directory of section mask PNGs (the stack)",
        default=str(default_masks) if default_masks else None,
        must_exist=True,
        must_be_dir=True,
    )
    n_masks = len(find_pred_masks(masks_dir)) or len(list(Path(masks_dir).glob("*.png")))
    if n_masks < MIN_STACK_SIZE:
        err(
            f"3D reconstruction requires >= {MIN_STACK_SIZE} section masks; "
            f"found {n_masks} in {masks_dir}."
        )
        return 1

    work_dir = ask_path(
        "Output (work) directory for 3D results",
        default=str(Path(masks_dir).parent / "recon_3d"),
        must_be_dir=True,
        create_if_missing=True,
    )

    samples_dir: Optional[Path] = None
    if ask_yes_no("Provide original tissue-scan PNGs to improve registration?", default=False):
        samples_dir = ask_path("Tissue-scan directory", must_exist=True, must_be_dir=True)

    stages: Optional[List[int]] = None
    if ask_yes_no("Run a subset of stages only (1=classify..6=snapshots)?", default=False):
        stages = parse_int_list(ask("Stage numbers (e.g. '5 6')", default="1 3 4 5 6"))

    try:
        out = run_reconstruction(
            masks_dir=str(masks_dir),
            work_dir=str(work_dir),
            samples_dir=str(samples_dir) if samples_dir else None,
            stages=stages,
        )
    except ReconstructionError as exc:
        err(str(exc))
        return 1

    html = Path(out) / "model" / "lesion_3d_surfaces.html"
    ok(f"3D reconstruction completed. Interactive surfaces: {html}")
    return 0


def run_3d_reconstruction_workflow() -> int:
    print(color("\n--- 3D Reconstruction from a Stack of Section Masks ---", C.BOLD + C.CYAN))
    info("Requires a directory of >= 2 serial-section color masks (green/blue/red).")
    return _run_reconstruction_prompts()


def run_collect_predictions_workflow() -> int:
    print(color("\n--- Collect Prediction Masks into One Directory ---", C.BOLD + C.CYAN))
    src = ask_path(
        "Prediction root to search recursively (for *_pred.png)",
        must_exist=True,
        must_be_dir=True,
    )
    dest = ask_path(
        "Destination directory for collected masks",
        default=str(Path(src) / "all_predictions"),
        must_be_dir=True,
        create_if_missing=True,
    )
    move = ask_yes_no("Move files instead of copying?", default=False)
    n = collect_predictions(src, dest, move=move)
    ok(f"Collected {n} prediction mask(s) into {dest}")
    if n >= MIN_STACK_SIZE and ask_yes_no("Build a 3D model from these masks now?", default=False):
        return _run_reconstruction_prompts(default_masks=Path(dest))
    return 0


def run_predict_sam_workflow() -> int:
    print(color("\n--- Predict with SAM3 (Segment Anything) ---", C.BOLD + C.CYAN))
    from src.sam_backend import SamBackendError, resolve_sam_paths, run_sam_prediction

    sam_root = ask_path(
        "SAM2LS root directory",
        default=os.environ.get("SAM2LS_ROOT"),
        must_exist=True,
        must_be_dir=True,
    )
    checkpoint_default = str(Path(sam_root) / "exp_log" / "sam3_lesion_seg" / "checkpoints" / "checkpoint.pt")
    checkpoint = ask_path(
        "SAM3 checkpoint (.pt)",
        default=checkpoint_default,
        must_exist=True,
        must_be_dir=False,
    )

    try:
        resolve_sam_paths(sam_root=str(sam_root), checkpoint=str(checkpoint))
    except SamBackendError as exc:
        err(str(exc))
        return 1

    sam_python = ask(
        "Python interpreter that can import sam3 (blank = current)",
        default=os.environ.get("SAM_PYTHON", ""),
    ).strip()

    target = ask_choice("Predict a single image or a directory?", ["single", "directory"], default="single")
    image = input_dir = None
    if target == "single":
        image = str(ask_path("Input image (.tif/.ndpi)", must_exist=True, must_be_dir=False))
    else:
        input_dir = str(ask_path("Input directory", must_exist=True, must_be_dir=True))

    output_dir = ask_path(
        "Output directory",
        default=str(PROJECT_ROOT / "predictions" / "sam3"),
        must_be_dir=True,
        create_if_missing=True,
    )
    tile_size = ask_int("Tile size", default=448, minimum=64)
    overlap = ask_int("Tile overlap", default=112, minimum=0)
    device = ask("Device", default="cuda:0")
    use_tta = ask_yes_no("Use test-time augmentation?", default=False)
    post_process = ask_yes_no("Enable topology post-processing?", default=True)

    try:
        code = run_sam_prediction(
            output_dir=str(output_dir),
            image=image,
            input_dir=input_dir,
            checkpoint=str(checkpoint),
            sam_root=str(sam_root),
            python_executable=sam_python or None,
            tile_size=tile_size,
            overlap=overlap,
            device=device,
            use_tta=use_tta,
            post_process=post_process,
        )
    except SamBackendError as exc:
        err(str(exc))
        return 1

    if code != 0:
        err(f"SAM3 prediction failed (exit {code}).")
        return code

    ok("SAM3 prediction completed.")
    if input_dir is not None:
        offer_3d_chain(output_dir)
    return 0


def run_benchmark_workflow() -> int:
    print(color("\n--- Multi-Architecture Benchmark + Task Difficulty ---", C.BOLD + C.CYAN))
    mode = ask_choice("Benchmark mode", ["models", "task_difficulty"], default="models")
    data_dir = ask_path(
        "Ground-truth data directory (tif + png pairs)",
        default=os.environ.get("DATA_DIR"),
        must_exist=True,
        must_be_dir=True,
    )

    if mode == "models":
        out_dir = ask_path(
            "Output directory",
            default=str(PROJECT_ROOT / "experiments" / "benchmark"),
            must_be_dir=True,
            create_if_missing=True,
        )
        models_glob = ask("Models glob", default=str(PROJECT_ROOT / "models" / "*"))
        gpus_raw = ask("Number of GPU workers (blank=auto)", default="")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark" / "run_benchmark.py"),
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
            "--models-glob", models_glob,
        ]
        if gpus_raw.strip():
            cmd += ["--gpus", gpus_raw.strip()]
        return run_subprocess(cmd)

    out_dir = ask_path(
        "Output directory",
        default=str(PROJECT_ROOT / "experiments" / "task_difficulty"),
        must_be_dir=True,
        create_if_missing=True,
    )
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark" / "task_difficulty.py"),
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
    ]
    if ask_yes_no("Include a trained model for contrast?", default=False):
        model_path = ask_path("Trained model (best_model.pth)", must_exist=True, must_be_dir=False)
        cmd += ["--model", str(model_path)]
    return run_subprocess(cmd)


def run_fem_workflow() -> int:
    print(color("\n--- FEM Biomechanical Mushrooming Simulation ---", C.BOLD + C.CYAN))
    from src.fem_mushrooming import (
        MaterialModel,
        build_mesh,
        load_class_mask,
        simulate_mushrooming,
        validate_against_observed,
    )

    mask_path = ask_path("Segmented mask PNG (green/blue/red)", must_exist=True, must_be_dir=False)
    out_dir = ask_path(
        "Output directory",
        default=str(PROJECT_ROOT / "experiments" / "fem"),
        must_be_dir=True,
        create_if_missing=True,
    )
    max_cells = ask_int("Mesh resolution (cells along long side)", default=120, minimum=20)
    stretch = ask_float("Radial stretch fraction (e.g. 0.1 = 10%)", default=0.1, minimum=0.0)
    e_tissue = ask_float("E tissue (relative)", default=1.0, minimum=1e-6)
    e_coag = ask_float("E coagulation (relative)", default=3.0, minimum=1e-6)
    e_abl = ask_float("E ablation (relative)", default=0.4, minimum=1e-6)
    nu = ask_float("Poisson ratio", default=0.45, minimum=0.0)

    pixel_scale_default = 0.221 * (2 ** 2)
    pixel_scale = ask_float("Pixel scale (um per full-res pixel)", default=pixel_scale_default, minimum=1e-6)

    try:
        class_mask = load_class_mask(str(mask_path))
        mesh, elem_class, _small, center, orig_per_mesh = build_mesh(class_mask, max_cells=max_cells)
        material = MaterialModel(E_tissue=e_tissue, E_coagulation=e_coag, E_ablation=e_abl, nu=nu)
        sim = simulate_mushrooming(
            mesh, elem_class, center,
            material=material,
            stretch=stretch,
            out_dir=str(out_dir),
            pixel_scale_um=pixel_scale * orig_per_mesh,
        )
        ok(f"FEM simulation completed. Figures in {out_dir}")
    except Exception as exc:
        err(f"FEM simulation failed: {exc}")
        return 1

    if ask_yes_no("Validate against an observed radial-CV CSV?", default=False):
        csv_path = ask_path("Observed CV CSV (RadialMushroomingAnalyzer output)", must_exist=True, must_be_dir=False)
        try:
            import pandas as pd

            observed = pd.read_csv(csv_path)
            res = validate_against_observed(sim, observed, str(out_dir))
            ok(f"Validation written: {res['png']} (Spearman r={res['spearman']:.2f})")
        except Exception as exc:
            err(f"Validation failed: {exc}")
            return 1
    return 0


def run_compare_masks_workflow() -> int:
    print(color("\n--- Compare Predicted vs Ground Truth Masks ---", C.BOLD + C.CYAN))
    pred = ask_path("Predicted mask path (.png)", must_exist=True, must_be_dir=False)
    gt = ask_path("Ground-truth mask path (.png)", must_exist=True, must_be_dir=False)
    out = ask_path(
        "Output directory for comparison results",
        default=str(PROJECT_ROOT / "predictions" / "comparison"),
        must_be_dir=True,
        create_if_missing=True,
    )
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "eval_masks.py"),
        "--pred",
        str(pred),
        "--gt",
        str(gt),
        "--output_dir",
        str(out),
    ]
    return run_subprocess(cmd)


def run_stats_workflow() -> int:
    print(color("\n--- Statistical Analysis Tools ---", C.BOLD + C.CYAN))
    choice = ask_choice("Stats mode", ["demo", "bland_altman", "mixedlm"], default="demo")
    script = str(PROJECT_ROOT / "scripts" / "analysis" / "stats_pipeline.py")

    if choice == "demo":
        return run_subprocess([sys.executable, script, "demo"])

    if choice == "bland_altman":
        csv_path = ask_path("CSV path", must_exist=True, must_be_dir=False)
        col1 = ask("Reference column (--col1)", default="manual_um")
        col2 = ask("Test column (--col2)", default="auto_um")
        return run_subprocess(
            [sys.executable, script, "bland-altman", "--csv", str(csv_path), "--col1", col1, "--col2", col2]
        )

    csv_path = ask_path("CSV path", must_exist=True, must_be_dir=False)
    value = ask("Outcome column (--value)", default="coag_width_um")
    group = ask("Group/fixed effect column (--group, blank for none)", default="")
    sample_col = ask("Random intercept group column (--sample-col)", default="sample_id")
    extras = ask("Additional formula terms (--extras, optional)", default="")
    cmd = [sys.executable, script, "mixedlm", "--csv", str(csv_path), "--value", value, "--sample-col", sample_col]
    if group:
        cmd += ["--group", group]
    if extras:
        cmd += ["--extras", extras]
    return run_subprocess(cmd)


def run_end_to_end_workflow() -> int:
    print(color("\n--- Guided End-to-End Workflow ---", C.BOLD + C.CYAN))
    info("This flow can run: tiling -> training -> prediction in sequence.")

    if ask_yes_no("Step 1: Run tiling now?", default=True):
        code = run_tiling_workflow()
        if code != 0:
            return code

    if ask_yes_no("Step 2: Run training now?", default=True):
        code = run_train_workflow()
        if code != 0:
            return code

    if ask_yes_no("Step 3: Run prediction now?", default=True):
        pred_mode = ask_choice("Prediction mode", ["single", "directory", "directory_multi_gpu"], default="single")
        if pred_mode == "single":
            return run_predict_single_workflow()
        if pred_mode == "directory":
            return run_predict_directory_workflow(multi_gpu=False)
        return run_predict_directory_workflow(multi_gpu=True)

    ok("End-to-end flow completed.")
    return 0


MENU_ACTIONS: List[tuple[str, Callable[[], int]]] = [
    ("Guided end-to-end (tiling -> train -> predict)", run_end_to_end_workflow),
    ("Create tiles", run_tiling_workflow),
    ("Train model", run_train_workflow),
    ("Predict single image", run_predict_single_workflow),
    ("Predict directory (single GPU/CPU)", lambda: run_predict_directory_workflow(multi_gpu=False)),
    ("Predict directory (multi-GPU)", lambda: run_predict_directory_workflow(multi_gpu=True)),
    ("Predict with SAM3 (Segment Anything)", run_predict_sam_workflow),
    ("Collect prediction masks into one directory", run_collect_predictions_workflow),
    ("3D reconstruction from a stack of section masks", run_3d_reconstruction_workflow),
    ("Multi-architecture benchmark + task difficulty", run_benchmark_workflow),
    ("FEM biomechanical mushrooming simulation", run_fem_workflow),
    ("Compare predicted vs GT masks", run_compare_masks_workflow),
    ("Run statistics tools (demo/bland-altman/mixedlm)", run_stats_workflow),
]


def interactive_main() -> int:
    header()
    while True:
        print(color("Main Menu", C.BOLD + C.YELLOW))
        for idx, (label, _) in enumerate(MENU_ACTIONS, start=1):
            print(color(f"  {idx}) {label}", C.WHITE))
        print(color(f"  {len(MENU_ACTIONS) + 1}) Exit", C.WHITE))

        choice_raw = ask("Select an option", default="1")
        try:
            choice = int(choice_raw)
        except ValueError:
            warn("Please enter a numeric menu option.")
            continue

        if choice == len(MENU_ACTIONS) + 1:
            ok("Goodbye.")
            return 0
        if not (1 <= choice <= len(MENU_ACTIONS)):
            warn("Invalid option number.")
            continue

        _, action = MENU_ACTIONS[choice - 1]
        print()
        code = action()
        print()
        if code != 0:
            warn("Workflow finished with non-zero status. Review errors above.")
        if not ask_yes_no("Return to main menu?", default=True):
            return code
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified colored CLI for tiling, training, prediction, and analysis workflows."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "menu",
            "end_to_end",
            "tiling",
            "train",
            "predict_single",
            "predict_dir",
            "predict_dir_multi_gpu",
            "predict_sam",
            "collect",
            "reconstruct_3d",
            "benchmark",
            "fem_simulate",
            "compare",
            "stats",
        ],
        default="menu",
        help="Launch directly into a specific workflow mode",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "menu":
        return interactive_main()
    if args.mode == "end_to_end":
        return run_end_to_end_workflow()
    if args.mode == "tiling":
        return run_tiling_workflow()
    if args.mode == "train":
        return run_train_workflow()
    if args.mode == "predict_single":
        return run_predict_single_workflow()
    if args.mode == "predict_dir":
        return run_predict_directory_workflow(multi_gpu=False)
    if args.mode == "predict_dir_multi_gpu":
        return run_predict_directory_workflow(multi_gpu=True)
    if args.mode == "predict_sam":
        return run_predict_sam_workflow()
    if args.mode == "collect":
        return run_collect_predictions_workflow()
    if args.mode == "reconstruct_3d":
        return run_3d_reconstruction_workflow()
    if args.mode == "benchmark":
        return run_benchmark_workflow()
    if args.mode == "fem_simulate":
        return run_fem_workflow()
    if args.mode == "compare":
        return run_compare_masks_workflow()
    if args.mode == "stats":
        return run_stats_workflow()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
