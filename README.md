# Lesion Vision

Four-class deep-learning pipeline for **segmentation and complete-field morphometry** of ablative fractional laser (AFL) lesions on horizontally sectioned, nitroblue tetrazolium chloride (NBTC)-stained whole-slide images.

Each section is segmented into background, viable tissue, coagulation and ablation, and every detected lesion is measured: equivalent-circle ablation diameter, median radial coagulation width (72 rays), total lesion diameter, ablation- and coagulation-zone area, and centroid.

## Publication

Quadri SA, Marks HL, Wang-Evers M, Manstein D. *Automated histomorphometry of ablative fractional laser lesions enables a complete-field census of geometric heterogeneity.* Scientific Reports (in revision).

If you use this software, please cite the paper when published and this repository (`CITATION.cff`). Human tissue in the associated study was discarded abdominoplasty skin collected under Massachusetts General Hospital IRB #2017P000027.

## What is in the pipeline

| Stage | What it does | Code |
|---|---|---|
| Scale | Every image is analysed at 0.8832 µm/pixel; NDPI files are read at the pyramid level matching that scale | `src/infer/ndpi.py` |
| Segmentation models | UNet, UNet++, DeepLabV3+, UPerNet or SegFormer with any `timm`/SMP encoder (`--arch`) | `src/models/` |
| Training | Joint 448 + 768 px tiles rescaled to 512, compound CE + Dice loss, punch-disjoint split, DDP | `src/train/`, `scripts/train.py` |
| Probabilities | Whole-section class probabilities with probability-space tile blending | `scripts/predict_probs.py` |
| Ensemble | Equal-weight mean of member probabilities; members chosen by greedy forward selection on validation | `scripts/analysis/ensemble_probs.py`, `scripts/analysis/greedy_ensemble.py` |
| Nested decoding | Lesion vs non-lesion, then crater vs cuff, then a ≥10 µm coagulation rim, so every crater is enclosed by coagulation | `scripts/analysis/component_filter.py` |
| Lesion filter | Removes objects < 80 µm, low-probability objects and small edge objects; thresholds tuned on validation | `scripts/analysis/component_filter.py` |
| Morphometry | Per-lesion measurements and per-section summaries | `src/analytics/lesions.py` |
| Evaluation | Pixel metrics with bootstrap CIs, lesion matching, Bland–Altman, depth bands, census | `scripts/analysis/r3_stats.py` |

## Install

Python 3.10+. OpenSlide is required for NDPI input.

```bash
git clone https://github.com/mgh-mansteinlab/lesion-vision.git
cd lesion-vision
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Editable install (optional): `pip install -e ".[dev]"`

## Data and weights

Imaging data and trained checkpoints are not stored in git; both are available from the corresponding author on reasonable request (`data/DATA_ON_REQUEST.txt`, `models/DATA_ON_REQUEST.txt`). One extracted punch at the analysis scale is included for testing: `data/sample/CCA-1_punch.tif`.

Class ids: `0` background (white), `1` viable tissue (green), `2` coagulation (blue), `3` ablation (red).

The evaluation and analysis scripts read whole sections from `LESION_DATA_ROOT` (default `data/LesionSegmentation`):

```text
$LESION_DATA_ROOT/
  data/PVcont/{images,masks}/        evaluation punch (35 serial sections)
  data_val_r3m2/{images,masks}/      validation sections
  data_tiling_sources/               annotated sources + scale_manifest.csv
  data_example_series/<series>/images/   unannotated example series
```

## Predict with one model

```python
from src.infer.predictor import LesionPredictor

predictor = LesionPredictor(model_path="models/<run>/best_model.pth")
predictor.save_prediction("data/sample/CCA-1_punch.tif", "predictions/sample_pred.png",
                          summary_metrics=True, figures=True)
```

Directory of `.tif` / `.ndpi` files (NDPI files are split into per-sample TIFFs at the analysis scale automatically):

```bash
python scripts/predict.py --data-dir data/sample --out-dir predictions/sample --model models/<run>/best_model.pth
```

Outputs per image: `images/` (class mask, overlay, summary), `csv/` (per-lesion table `lesion_summary.csv` and aggregates), `individual_lesions/`.

## Final pipeline (ensemble + nested decoding + lesion filter)

```bash
# 1. class probabilities from each member model
python scripts/predict_probs.py --images-only --data-dir <sections> --model models/<member>/checkpoint_epoch_N.pth \
       --out-dir predictions/<member>/probs --gpus 8

# 2. equal-weight mean of the members
python scripts/analysis/ensemble_probs.py --out-dir predictions/final/probs predictions/<m1>/probs predictions/<m2>/probs ...

# 3. nested decoding + lesion filter (thresholds from validation), then morphometry
python scripts/analysis/component_filter.py apply --decode nested --interior-bg \
       --probs-dir predictions/final/probs --params predictions/r3_greedy/filter_params_nested.json \
       --out-dir predictions/final/pred
```

Filter thresholds are tuned with `component_filter.py tune --decode nested --probs-dir <validation probs> --gt-dir <validation masks> --out <params.json>`. `scripts/analysis/greedy_ensemble.py` runs member selection, the decoding choice, filter tuning and the evaluation-punch application end to end.

## Train

Tiles under `DATA_DIR` (`tile_448/`, `mask_448/`, `tile_768/`, `mask_768/`, each with `no_shift` and `ovlp_20…80` grids; tile and mask names must match; tiling in `src/data/tiling.py`).

Models of the paper, one YAML each:

| Model | Architecture / encoder | Config |
|---|---|---|
| M1 (pre-specified) | UNet / ConvNeXt-Base | `configs/experiments/baseline_multi_scale_slide_split.yaml` |
| M2 | UNet / ConvNeXt-Base, + 14 lesion-free sections | `configs/experiments/r3_model2_hardneg.yaml` |
| C1 | UPerNet / ConvNeXt-Base | `configs/experiments/r3_c1_upernet_convnextb.yaml` |
| C2 | SegFormer / MiT-B3 | `configs/experiments/r3_c2_segformer_mitb3.yaml` |
| C3 | DeepLabV3+ / ResNet-50 | `configs/experiments/r3_c3_deeplabv3plus_resnet50.yaml` |
| C4 | UNet++ / EfficientNet-B3 | `configs/experiments/r3_c4_unetpp_effb3.yaml` |

```bash
export DATA_DIR=/path/to/tiles
WORLD_SIZE=8 python scripts/launch_experiment.py configs/experiments/r3_c2_segformer_mitb3.yaml
```

All use a punch-disjoint split (`split_level: punch`: every slide of a punch series in one subset, seed 42), with the evaluation punch removed from training and validation (`exclude_groups: [PVcont]`). Training resumes from a checkpoint with `--resume <ckpt> --resume_history <train_history.json>`, which keeps the early-stopping state. Class weights are square-root inverse frequency on the training split (`scripts/analysis/compute_class_weights.py`).

## Evaluation and paper tables

```bash
python scripts/r3_eval_model.py --run-dir models/<run> --tag <tag>   # checkpoint + inference setting chosen on validation, then the evaluation punch
python scripts/analysis/greedy_ensemble.py                          # final pipeline
python scripts/analysis/r3_example_series.py                        # census of the example series
python scripts/analysis/r3_cpu_report.py                            # section geometry and pitch
python scripts/analysis/manual_sampling_sim.py                      # 10-lesion sampling simulation
python scripts/analysis/r3_glahn_comparison.py                      # comparison with published manual areas
python scripts/analysis/r3_make_tables.py                           # every table and in-text number
python scripts/analysis/r3_make_figures.py                          # result figures
python scripts/analysis/r3_export_source_data.py                    # tables and figure data as CSV
```

## Repository layout

| Path | Role |
|---|---|
| `src/infer/` | Prediction, NDPI reading at the analysis scale, tiling/stitching, post-processing |
| `src/train/` | Training loop, DDP, early stopping, CLI |
| `src/models/` | SMP architectures and losses |
| `src/data/` | Dataset, punch-level split, augmentations, tiling |
| `src/analytics/` | Lesion morphometry |
| `src/eval/` | Pixel metrics and lesion matching |
| `scripts/` | Training, prediction, probability export, per-model evaluation |
| `scripts/analysis/` | Ensemble, nested decoding, lesion filter, statistics, tables and figures |
| `configs/experiments/` | Training presets |
| `experiments/` | Experiment notes |
| `data/sample/` | Public test punch |
| `tests/` | Unit tests (`pytest tests/ -v`) |

## License

MIT. See `LICENSE`.
