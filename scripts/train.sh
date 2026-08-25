#!/bin/bash
# Generic training wrapper. Set DATA_DIR to the tiled image/mask root.
# Override GPU count with WORLD_SIZE and CUDA_VISIBLE_DEVICES.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -z "${DATA_DIR:-}" ]; then
    echo "Set DATA_DIR to the tiled data root (tile_448/, mask_448/, ...)." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"

WEIGHTS_JSON="$PROJECT_DIR/configs/class_weights.json"
if [ -f "$WEIGHTS_JSON" ]; then
    CLASS_WEIGHTS=$(python3 -c "import json; w=json.load(open('$WEIGHTS_JSON'))['weights']; print(','.join(str(v) for v in w))")
    echo "Loaded class weights from $WEIGHTS_JSON: $CLASS_WEIGHTS"
else
    CLASS_WEIGHTS="auto"
    echo "No class_weights.json found – will compute automatically"
fi

MODEL_DIR="${OUTPUT_DIR:-$PROJECT_DIR/models}"
BACKBONE="${BACKBONE:-tu-convnext_base}"
DISK_TILE_SIZES="${DISK_TILE_SIZES:-448 768}"
IMG_SIZE="${IMG_SIZE:-512}"
RESUME="${RESUME:-}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
PATIENCE="${PATIENCE:-20}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WORLD_SIZE="${WORLD_SIZE:-1}"
LOSS_TYPE="${LOSS_TYPE:-compound}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SAMPLING_ALPHA="${SAMPLING_ALPHA:-0.85}"

USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-lesion-vision}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

mkdir -p "$MODEL_DIR"

DIST_ARGS=()
if [ "$WORLD_SIZE" -gt 1 ]; then
    DIST_ARGS+=(--distributed --world_size "$WORLD_SIZE" --dist_backend nccl)
fi

RESUME_ARGS=()
if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
    RESUME_ARGS+=(--resume "$RESUME")
fi

WANDB_ARGS=()
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS+=(--use_wandb --wandb_project "$WANDB_PROJECT")
    if [ -n "$WANDB_NAME" ]; then
        WANDB_ARGS+=(--wandb_name "$WANDB_NAME")
    fi
    if [ -n "$WANDB_TAGS" ]; then
        WANDB_ARGS+=(--wandb_tags $WANDB_TAGS)
    fi
fi

python "$PROJECT_DIR/scripts/train.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$MODEL_DIR" \
    --backbone "$BACKBONE" \
    --tile_size $DISK_TILE_SIZES \
    --img_size "$IMG_SIZE" \
    "${RESUME_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --patience "$PATIENCE" \
    "${DIST_ARGS[@]}" \
    --augment \
    --loss_type "$LOSS_TYPE" \
    --class_weights "$CLASS_WEIGHTS" \
    --scheduler_type "$SCHEDULER_TYPE" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --weighted_sampling \
    --sampling_alpha "$SAMPLING_ALPHA" \
    --num_workers "$NUM_WORKERS" \
    --activation_checkpointing \
    "${WANDB_ARGS[@]}"
