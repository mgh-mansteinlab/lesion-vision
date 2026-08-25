# Sample slide

`CCA-1_punch.tif` is one 8 mm-scale punch extracted from a laser-only NDPI (`CCA-1`) at OpenSlide pyramid level 2 (0.884 µm/pixel), matching the analysis scale in the manuscript.

It is stored as JPEG-compressed TIFF (~39 MB) so it fits GitHub. Native NDPI files from the same scanner are typically 200–500 MB and are distributed with the imaging dataset on request (see `../DATA_ON_REQUEST.txt`).

```bash
python scripts/predict.py \
  --data-dir data/sample \
  --out-dir predictions/sample \
  --model models/<run_name>/best_model.pth
```
