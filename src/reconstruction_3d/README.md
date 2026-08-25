# LSA-3D — Registration & 3D Surface Reconstruction

Reconstructs interactive 3D lesion surfaces from serial-section color masks.

**Labels in mask PNGs:** green = live tissue, blue = coagulation, red = ablation.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. Needs ~2 GB RAM for typical datasets.

## Usage

```bash
# Masks only (registration uses mask geometry; stage 2 skipped)
python run_pipeline.py --masks /path/to/mask/pngs

# Masks + original NBTC tissue scans (better registration)
python run_pipeline.py --masks /path/to/masks --samples /path/to/tissue_scans

# Custom output directory
python run_pipeline.py --masks /path/to/masks --work /path/to/output

# Run specific stages (after prior stages have written their outputs)
python run_pipeline.py --masks /path/to/masks --work /path/to/output 5 6

# List stages
python run_pipeline.py --list
```

### Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--masks` | yes | Directory of color-coded section mask PNGs |
| `--samples` | no | Directory of original NBTC tissue-scan PNGs (same FOV as masks) |
| `--work` | no | Output directory (default: `./output`) |

## Pipeline stages

1. **s1_extract** — classify masks, build label maps + `meta.json`
2. **s2_tissue** — downsample tissue scans (only if `--samples` given)
3. **s3_register** — pairwise slice registration → `aligned/`
4. **s4_clean** — lesion cleanup, column tracking → `aligned_clean/`
5. **s5_build_model** — marching-cubes 3D surfaces → `model/lesion_3d_surfaces.html`
6. **s6_snapshot** — static PNG renders → `model/*.png`

## Outputs (`<work>/model/`)

- `lesion_3d_surfaces.html` — interactive 3D surface viewer
- `lesion_3d_pointcloud.html` — point cloud viewer
- `lesion_pointcloud.ply` — PLY export
- `depth_profile.png` — damage area vs depth
- `tps_*_iso/front/side.png` — snapshot renders

## Mask file naming

Supports filenames like `PVC1_m_01.png` or `CPVV1 - ..._sample_01_pred.png`. Files are sorted by extracted case/slice numbers.

## Tissue scan pairing (optional)

When `--samples` is provided, tissue PNGs are matched to masks by numeric case + sub-slice indices in the filename (same convention as the PVC dataset).
