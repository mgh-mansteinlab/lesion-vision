#!/usr/bin/env python3
"""Build train.py argv from configs/experiments/*.yaml and exec training."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_class_weights_csv(project_dir: Path) -> str:
    p = project_dir / "configs" / "class_weights.json"
    if not p.is_file():
        return "auto"
    data = json.loads(p.read_text())
    return ",".join(str(v) for v in data["weights"])


def build_argv(yaml_path: Path, project_dir: Path) -> list[str]:
    import yaml

    cfg = yaml.safe_load(yaml_path.read_text())
    t = cfg.get("train", {})
    tiles = t.get("tile_size", [512])
    if not isinstance(tiles, list):
        tiles = [tiles]

    data_dir = os.environ.get("DATA_DIR", str(project_dir / "data"))
    output_dir = os.environ.get("OUTPUT_DIR", str(project_dir / "models"))
    if "class_weights" in t:
        class_weights = str(t["class_weights"])
    else:
        class_weights = os.environ.get("CLASS_WEIGHTS", load_class_weights_csv(project_dir))

    argv = [
        sys.executable,
        str(project_dir / "scripts" / "train.py"),
        "--data_dir",
        data_dir,
        "--output_dir",
        output_dir,
        "--backbone",
        str(t.get("backbone", "tu-convnext_base")),
        "--img_size",
        str(int(t.get("img_size", 512))),
        "--tile_size",
        *sum(([str(s)] for s in tiles), []),
        "--shift_type",
        str(t.get("shift_type", "all")),
        "--epochs",
        str(int(t.get("epochs", 100))),
        "--batch_size",
        str(int(t.get("batch_size", 2))),
        "--gradient_accumulation_steps",
        str(int(t.get("gradient_accumulation_steps", 1))),
        "--learning_rate",
        str(float(t.get("learning_rate", 1e-4))),
        "--patience",
        str(int(t.get("patience", 15))),
        "--seed",
        str(int(t.get("seed", 42))),
        "--val_split",
        str(float(t.get("val_split", 0.2))),
        "--split_level",
        str(t.get("split_level", "tile")),
        *(["--holdout_groups", *map(str, t["holdout_groups"])] if t.get("holdout_groups") else []),
        *(["--exclude_groups", *map(str, t["exclude_groups"])] if t.get("exclude_groups") else []),
        *(["--train_slides", *map(str, t["train_slides"])] if t.get("train_slides") else []),
        "--loss_type",
        str(t.get("loss_type", "compound")),
        "--scheduler_type",
        str(t.get("scheduler_type", "cosine")),
        "--warmup_epochs",
        str(int(t.get("warmup_epochs", 5))),
        "--sampling_alpha",
        str(float(t.get("sampling_alpha", 0.85))),
        "--class_weights",
        class_weights,
    ]
    if "num_workers" in t:
        argv.extend(["--num_workers", str(int(t["num_workers"]))])
    if t.get("augment"):
        argv.append("--augment")
    if t.get("weighted_sampling"):
        argv.append("--weighted_sampling")
    if t.get("activation_checkpointing"):
        argv.append("--activation_checkpointing")
    if t.get("deterministic"):
        argv.append("--deterministic")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        argv.extend(["--distributed", "--world_size", str(world_size), "--dist_backend", "nccl"])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch training from experiment YAML")
    parser.add_argument("yaml", type=Path, help="Path to configs/experiments/*.yaml")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    yaml_path = args.yaml if args.yaml.is_absolute() else project_dir / args.yaml
    if not yaml_path.is_file():
        sys.exit(f"Not found: {yaml_path}")

    argv = build_argv(yaml_path, project_dir)
    print("Running:", " ".join(argv[:6]), "...")
    os.chdir(project_dir)
    raise SystemExit(subprocess.call(argv))


if __name__ == "__main__":
    main()
