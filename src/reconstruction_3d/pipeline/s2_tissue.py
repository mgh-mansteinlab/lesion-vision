"""Stage 2: downsample the full-resolution NBTC tissue scans (t/) to the SAME working grid as
the masks (they share field-of-view), so tissue texture/edges can drive registration.
Uses OpenCV reduced decode (1/8) to avoid loading 100+ MB per file."""
import os, re, json
import cv2
import numpy as np

SRC = os.environ.get('LSA_TISSUE_SRC')
WORK = os.environ.get('LSA_WORK') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')
if not SRC:
    raise SystemExit('LSA_TISSUE_SRC not set. Pass --samples <path> or skip stage 2.')
SRC = os.path.abspath(SRC)
OUT = os.path.join(WORK, 'tissue'); os.makedirs(OUT, exist_ok=True)
meta = json.load(open(os.path.join(WORK, 'meta.json')))

# index t files by (PVC number, sub-slice 01/02/03)
tfiles = {}
for f in os.listdir(SRC):
    if not f.lower().endswith('.png'): continue
    nums = re.findall(r'\d+', f)
    if len(nums) >= 2:
        tfiles[(int(nums[0]), int(nums[1]))] = f

missing = []
for m in meta:
    nums = re.findall(r'\d+', m['file'])
    N, sub = int(nums[0]), int(nums[-1])
    key = (N, sub)
    if key not in tfiles:
        missing.append((m['idx'], m['file'])); continue
    path = os.path.join(SRC, tfiles[key])
    img = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if img is None:
        missing.append((m['idx'], m['file'])); continue
    g = cv2.resize(img, (m['w'], m['h']), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, 't_%02d.png' % m['idx']), g)
    print(f"{m['idx']:2d} {m['file']:18s} <- {tfiles[key]:22s}  {g.shape}")
print("missing:", missing)
print("done")
