# Lesion Vision

Four-class deep learning pipeline for **segmentation and complete-field morphometry** of ablative fractional laser (AFL) lesions on horizontally sectioned, nitroblue tetrazolium chloride (NBTC)-stained whole-slide images.

The model is a **UNet** decoder with a **ConvNeXt-Base** encoder (`segmentation-models-pytorch`). It labels background, viable tissue, coagulation, and ablation, then measures every detection: equivalent-circle ablation diameter, median radial coagulation width, lesion diameter, and centroid.

## Publication

Quadri SA, Marks HL, Wang-Evers M, Manstein D. *Automated histomorphometry of ablative fractional laser lesions enables a complete-field census of geometric heterogeneity.* Scientific Reports (revision).

Code: [github.com/mgh-mansteinlab/lesion-vision](https://github.com/mgh-mansteinlab/lesion-vision)

If you use this software, please cite the paper when it is published, and this repository (`CITATION.cff`).

Human tissue in the associated study was discarded abdominoplasty skin collected under Massachusetts General Hospital IRB \#2017P000027.

## Features

- **Four-class segmentation** of AFL histology (background / tissue / coagulation / ablation)
- **NDPI and TIFF** whole-slide input, including multi-punch NDPI files split into per-sample TIFFs
- **Joint multi-scale training**: 448×448 and 768×768 tiles, both rescaled to 512×512, with five overlap grids
- **Complete-field morphometry** on every detection that passes the area filter (not a 10-lesion subsample)
- **Topology post-processing** to enforce ablation-inside-coagulation
- **Object-level evaluation**: detection recall/precision and paired Bland–Altman versus expert outlines
- Single-GPU or multi-GPU (DDP) training; optional Weights & Biases logging

## Install

Python 3.10+. OpenSlide is required for NDPI.

```bash
git clone https://github.com/mgh-mansteinlab/lesion-vision.git
cd lesion-vision
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Editable install (optional): `pip install -e ".[dev]"`

## Weights

Trained checkpoints are **not** in git (GitHub size limits, and institutional distribution). Request them from the corresponding author and place:

```text
models/<run_name>/best_model.pth
```

The manuscript model is ConvNeXt-Base UNet trained jointly on 448 and 768 px tiles. Directory names for the ablation sweep are listed in `models/CHECKPOINT_INDEX.txt`. If `models/single_scale_448/best_model.pth` is present, inference uses it by default; otherwise pass `--model` / `model_path`.

## Sample image

`data/sample/CCA-1_punch.tif` is **one punch extracted from an NDPI** at the paper's analysis scale (OpenSlide level 2, 0.884 µm/pixel, JPEG-TIFF, ~39 MB). Full NDPI files are 200–500 MB and are not stored here; they are available with the imaging dataset on request (`data/DATA_ON_REQUEST.txt`).

## Predict

Requires a checkpoint under `models/` (see Weights).

Python:

```python
from src.infer.predictor import LesionPredictor

predictor = LesionPredictor(model_path="models/<run_name>/best_model.pth")
predictor.save_prediction(
    "data/sample/CCA-1_punch.tif",
    "predictions/sample_pred.png",
    summary_metrics=True,
    figures=True,
)
```

CLI (directory of `.tif` / `.ndpi`):

```bash
python scripts/predict.py \
  --data-dir data/sample \
  --out-dir predictions/sample \
  --model models/<run_name>/best_model.pth
```

`scripts/predict.py` is an alias for `scripts/predict_directory.py`. Interactive menus: `python scripts/pipeline_cli.py`.

Outputs per image:

- `images/` — class mask, overlay, visual summary
- `csv/` — per-lesion table and aggregate metrics
- `individual_lesions/` — per-lesion crops

## Train

Tiled image/mask pairs under `DATA_DIR`:

```text
DATA_DIR/
  tile_448/{no_shift,ovlp_20,ovlp_40,ovlp_60,ovlp_80}/
  mask_448/{...}/
  tile_768/{...}/
  mask_768/{...}/
```

Tile filenames must match the corresponding masks. Generate tiles with `python utils/TileGen.py` (implementation: `src/data/tiling.py`).

Paper-aligned training (tile-path 80/20 split, seed 42; **not** slide-disjoint):

```bash
export DATA_DIR=/path/to/tiles
export OUTPUT_DIR=./models

python scripts/train.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --backbone tu-convnext_base \
  --tile_size 448 768 \
  --img_size 512 \
  --epochs 100 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --loss_type compound \
  --scheduler_type cosine \
  --warmup_epochs 5 \
  --augment \
  --weighted_sampling \
  --sampling_alpha 0.85 \
  --activation_checkpointing \
  --seed 42
```

Or: `export DATA_DIR=/path/to/tiles && bash scripts/train.sh`

Class ids: `0` background (white), `1` tissue (green), `2` coagulation (blue), `3` ablation (red). Reported class weights are in `configs/class_weights.json`.

## Repository layout

| Path | Role |
|------|------|
| `src/infer/` | Prediction, NDPI split, tiling/stitch, post-process |
| `src/train/` | Training loop, DDP, CLI |
| `src/models/` | SMP UNet / UNet++ wrapper and losses |
| `src/data/` | Dataset, augmentations, tile generation |
| `src/analytics/` | Lesion morphometry |
| `src/eval/` | Pixel metrics and lesion matching |
| `scripts/` | `train.py`, `predict.py`, `pipeline_cli.py`, eval helpers |
| `configs/experiments/` | Ablation YAML files |
| `data/sample/` | Public test punch |
| `tests/` | Unit tests (`pytest tests/ -v`) |

## License

MIT. See `LICENSE`.
