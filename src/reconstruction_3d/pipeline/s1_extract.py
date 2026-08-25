"""Stage 1: classify each full-res mask into a label map at working resolution.

Labels: 0 = background, 1 = green (live tissue), 2 = blue (coagulation), 3 = red (ablation)
Also computes the cleaned tissue silhouette and lesion (red+blue) connected components.
"""
import os, re, json
import cv2
import numpy as np
from scipy import ndimage as ndi

SRC = os.environ.get('LSA_SRC')
WORK = os.environ.get('LSA_WORK') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')
if not SRC:
    raise SystemExit('LSA_SRC not set. Run via: python run_pipeline.py --masks <path>')
SRC = os.path.abspath(SRC)
LBL_DIR = os.path.join(WORK, 'labels')
PREV_DIR = os.path.join(WORK, 'label_prev')
os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(PREV_DIR, exist_ok=True)

LONG_SIDE = 1600  # working resolution (longest side)

files = [f for f in os.listdir(SRC) if f.lower().endswith('.png')]
def key(f):
    # <prefix><case> ..._sample_<nn>_pred.png  (prediction datasets: CPBSV, CPVV, ...)
    m = re.match(r'[A-Za-z]+(\d+).*?_sample_(\d+)', f)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # PVC<case>..._<nn>.png  (original dataset)
    m = re.match(r'PVC(\d+).*?_(\d+)\.png', f)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (999, 999)
files_sorted = sorted(files, key=key)

# Color overlay for label previews (BGR)
PAL = np.array([[0,0,0],[0,180,0],[255,40,0],[0,0,255]], dtype=np.uint8)

meta = []
for idx, f in enumerate(files_sorted):
    img = cv2.imread(os.path.join(SRC, f))  # BGR
    H, W = img.shape[:2]
    scale = LONG_SIDE / max(H, W)
    w, h = int(round(W*scale)), int(round(H*scale))
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    b = small[:,:,0].astype(np.int16); g = small[:,:,1].astype(np.int16); r = small[:,:,2].astype(np.int16)

    bg = (b > 200) & (g > 200) & (r > 200)
    stack = np.stack([g, b, r], axis=-1)        # channel 0->green,1->blue,2->red
    cls = np.argmax(stack, axis=-1)             # 0 green,1 blue,2 red
    label = np.zeros((h, w), np.uint8)
    label[cls == 0] = 1
    label[cls == 1] = 2
    label[cls == 2] = 3
    label[bg] = 0

    # tissue silhouette = any non-background, cleaned (fill holes, keep largest comp)
    tissue = label > 0
    tissue = ndi.binary_fill_holes(tissue)
    lab, n = ndi.label(tissue)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n+1))
        keep = np.argmax(sizes) + 1
        tissue = lab == keep
    # restore label within tissue, drop stray bg islands' labels outside
    label[~tissue] = 0

    np.save(os.path.join(LBL_DIR, f'lbl_{idx:02d}.npy'), label)

    # lesion components (red+blue together)
    lesion = (label == 2) | (label == 3)
    lab2, n2 = ndi.label(lesion)
    cents = ndi.center_of_mass(lesion, lab2, index=np.arange(1, n2+1))
    sizes = ndi.sum(np.ones_like(lab2), lab2, index=np.arange(1, n2+1))
    # keep blobs with meaningful area
    cents_keep = [(float(c[1]), float(c[0])) for c, s in zip(cents, sizes) if s >= 25]

    ty, tx = ndi.center_of_mass(tissue)
    area = int(tissue.sum())

    meta.append(dict(idx=idx, file=f, H=H, W=W, h=h, w=w, scale=scale,
                     tissue_centroid=[float(tx), float(ty)], tissue_area=area,
                     n_lesions=len(cents_keep), lesion_centroids=cents_keep,
                     n_red=int((label==3).sum()), n_blue=int((label==2).sum()),
                     n_green=int((label==1).sum())))

    prev = PAL[label]
    cv2.imwrite(os.path.join(PREV_DIR, f'lbl_{idx:02d}.png'), prev)
    print(f"{idx:2d} {f:18s} size {w}x{h}  lesions={len(cents_keep):2d}  "
          f"red={int((label==3).sum()):6d} blue={int((label==2).sum()):6d} green={int((label==1).sum()):7d}")

with open(os.path.join(WORK, 'meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=1)
print('done -> meta.json')
