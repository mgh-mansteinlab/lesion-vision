# Experiment matrix (ablations and external validation)

YAML files under `configs/experiments/` describe training presets. The launcher expands them into `scripts/train.py` CLI flags.

**Default schedule:** all presets currently use **`epochs: 10`** and **`warmup_epochs: 2`** (short smoke / ablation runs). For full paper-quality training, copy a YAML and raise `epochs` (e.g. 100) and `warmup_epochs` (e.g. 5).

## Core / scale

| File | Purpose |
|------|---------|
| `baseline_multi_scale.yaml` | 448+768 tiles, all shifts, compound loss (tile-path split). |
| `baseline_multi_scale_slide_split.yaml` | Same stack with a punch-disjoint train/val split (paper-aligned). |
| `baseline_deterministic.yaml` | Same + `--deterministic` for reproducibility. |
| `single_scale_768.yaml` | 768-only tiles. |
| `single_scale_448.yaml` | 448-only tiles. |

## Shifts (needs matching TileGen folders)

| File | Purpose |
|------|---------|
| `shift_no_shift.yaml` | `no_shift` grid only. |
| `shift_ovlp_20.yaml` | `ovlp_20` offset grid only. |

## Loss / schedule / sampling

| File | Purpose |
|------|---------|
| `loss_ce_only.yaml` | CE only (no Dice in compound). |
| `loss_focal.yaml` | Focal loss. |
| `ce_no_class_weights.yaml` | CE + `class_weights: none`. |
| `scheduler_plateau.yaml` | ReduceLROnPlateau. |
| `no_augment.yaml` | No Albumentations augment. |
| `no_weighted_sampling.yaml` | Uniform dataloader sampling. |

## Backbones

| File | Purpose |
|------|---------|
| `backbone_efficientnet_b3.yaml` | EfficientNet-B3. |
| `backbone_efficientnet_b0.yaml` | EfficientNet-B0 (lighter). |
| `backbone_convnext_small.yaml` | ConvNeXt-Small. |
| `backbone_resnet34.yaml` | ResNet34. |

## Other

| File | Purpose |
|------|---------|
| `no_activation_checkpointing.yaml` | More VRAM per step, no checkpointing. |
| `external_val_holdout_template.yaml` | Copy + define hold-out cohort. |

## Post-processing ablation (inference)

Compare metrics with and without topology post-processing using `use_post_process=False` on `LesionPredictor.predict`, `save_prediction`, or `process_directory` (see `src/prediction.py`).

Log Dice/IoU **before and after** post-processing in `EXPERIMENT_LOG.md`.

## Launcher

Optional YAML keys (see `scripts/launch_experiment.py`): `class_weights`, `num_workers`, `deterministic`.

```bash
export DATA_DIR=/path/to/tiled/data
export OUTPUT_DIR=./models
bash scripts/run_experiment_train.sh configs/experiments/baseline_multi_scale.yaml
```

For multi-GPU, set `WORLD_SIZE` (e.g. `8`); the launcher adds `--distributed`.

## Run all experiments in the background (`nohup`)

Runs every `configs/experiments/*.yaml` **one after another** (same GPU is not shared across concurrent trainings). Each run writes its own log; a TSV summary and a Markdown journal are updated automatically.

```bash
export DATA_DIR=/path/to/tiled/data
export OUTPUT_DIR=./models
export WORLD_SIZE=1          # or 8 for DDP

mkdir -p experiments/logs
nohup bash scripts/run_all_experiments_nohup.sh > experiments/logs/nohup_master_latest.log 2>&1 &
tail -f experiments/logs/nohup_master_latest.log
```

**Outputs**

| File | Contents |
|------|----------|
| `experiments/logs/<experiment>_<timestamp>.log` | Full `stdout`/`stderr` for that training run |
| `experiments/logs/run_summary.tsv` | One row per run: time, config path, exit code, duration, log path |
| `experiments/logs/RUN_JOURNAL.md` | Human-readable run blocks |
| `experiments/EXPERIMENT_LOG_APPEND.tsv` | Rows you can merge into spreadsheets or `EXPERIMENT_LOG.md` |

**Env**

- `SKIP_TEMPLATES=1` (default): skips `*template*.yaml` (e.g. external hold-out placeholder).
- `SKIP_TEMPLATES=0`: include templates.
- `YAML_GLOB=baseline*.yaml`: run only a subset.
