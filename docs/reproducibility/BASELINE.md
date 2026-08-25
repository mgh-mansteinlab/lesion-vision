# Baseline reproducibility snapshot

Default training configuration matching the manuscript methods.

## Manuscript-to-code

| Manuscript element | Source |
|--------------------|--------|
| Four-class UNet + ConvNeXt-Base | `src/models/architecture.py` (`tu-convnext_base` → `smp.Unet`) |
| 448 + 768 tiles, resized to 512 | `scripts/train.py --tile_size 448 768 --img_size 512` |
| Five overlap grids | `shift_type=all` (no_shift + ovlp_20/40/60/80) |
| Compound CE+Dice and class weights | `--loss_type compound`; `configs/class_weights.json` |
| Tile-level 80/20 split, seed 42 | `src/data/dataset.py` (not slide-disjoint) |
| Topology post-process | `src/infer/postprocess.py` (on by default) |
| Per-lesion morphometry | `src/analytics/lesions.py` |
| NDPI extraction | `src/infer/ndpi.py` (default OpenSlide level 2) |

## Default train command

See the README “Train” section, or `bash scripts/train.sh` with `DATA_DIR` set.

Record for every run: git commit, `models/<run>/config.json`, and W&B id if used.

Checkpoints and full imaging data are on request (`models/DATA_ON_REQUEST.txt`, `data/DATA_ON_REQUEST.txt`). One extracted punch is in `data/sample/`.

## Experiment tracking

YAML presets: `configs/experiments/`. Helper: `scripts/run_experiment_train.sh`. Log template: `experiments/EXPERIMENT_LOG.md`.
