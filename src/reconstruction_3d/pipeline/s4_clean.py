"""Stage 4: domain-aware lesion cleanup + column tracking.

Domain model (laser-tissue interaction):
  * A lesion column = a centred ABLATION core (label 3, red) wrapped by a
    COAGULATION zone (label 2, blue), embedded in live tissue (label 1, green).
  * Ablation is shallow (near the laser entry); with depth a column may become
    coagulation-only.  Ablation WITHOUT surrounding coagulation is impossible.
  * Disconnected coagulation blobs that belong to no column are noise.

Pipeline:
  1. Seed columns from ablation cores (clustered across slices)  -> the true lesion count.
  2. Track each column top->end; assign exactly one coag blob per slice (nearest, gated).
  3. Drop every lesion blob not assigned to a column (removes stray blue noise).
  4. Enforce ablation - coag: ablation kept only inside its column's coag (a ring is
     synthesised if a core lacks coag).
  5. Outliers (centre jumps / area spikes / gaps) are replaced by the average of the
     adjacent slices, so a single torn/!missegmented slice can't distort the column.

Outputs: aligned_clean/alc_XX.npy (cleaned label maps) + columns_clean.json + diagnostics.
"""
import os, json, shutil
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

WORK = os.environ.get('LSA_WORK') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')
AL = os.path.join(WORK, 'aligned')
OUTD = os.path.join(WORK, 'aligned_clean')
DIAG = os.path.join(WORK, 'clean_diag')
for d in (OUTD, DIAG):
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
meta = json.load(open(os.path.join(WORK, 'meta.json')))
N = len(meta)
CW = CH = 2200
LIVE, COAG, ABL = 1, 2, 3

COAG_MIN = 200     # min coag-blob area (px) to be a real lesion blob
ABL_MIN  = 120     # min ablation-core area (px) to seed / mark ablation
SEED_GATE = float(os.environ.get('LSA_SEED_GATE', '70'))   # px: cluster ablation cores into columns
SUP_MIN  = int(os.environ.get('LSA_SUP_MIN', '4'))         # min slices a column's ablation must appear in to be real

maps = [np.load(os.path.join(AL, f'al_{i:02d}.npy')) for i in range(N)]

# ---- per-slice lesion blobs (coag union ablation) ----
def clean_tissue(m):
    t = (m > 0).astype(np.uint8)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    t = ndi.binary_fill_holes(t)
    lab, n = ndi.label(t)
    if n > 1:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
        t = lab == (np.argmax(sz) + 1)
    return t

blobs = []          # blobs[i] = list of dict(mask_crop, abl_crop, cen(x,y), bbox(y0,x0), area, abl_area)
tissue = []
for i, m in enumerate(maps):
    tis = clean_tissue(m); tissue.append(tis)
    les = ((m == COAG) | (m == ABL)) & tis
    lab, n = ndi.label(les)
    bl = []
    for cid in range(1, n + 1):
        mask = lab == cid
        area = int(mask.sum())
        if area < COAG_MIN: continue
        ys, xs = np.where(mask)
        y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        abl = mask & (m == ABL)
        bl.append(dict(mask=mask[y0:y1, x0:x1].copy(), abl=abl[y0:y1, x0:x1].copy(),
                       cen=np.array([xs.mean(), ys.mean()]), bbox=(int(y0), int(x0)),
                       area=area, abl_area=int(abl.sum())))
    blobs.append(bl)

# ---- seed columns from ablation cores ----
seeds = []
for i, bl in enumerate(blobs):
    for b in bl:
        if b['abl_area'] >= ABL_MIN:
            ys, xs = np.where(b['abl'])
            seeds.append(np.array([xs.mean() + b['bbox'][1], ys.mean() + b['bbox'][0]]))
clusters = []
for p in seeds:
    best, bd = None, 1e9
    for c in clusters:
        d = np.hypot(*(p - c['m']))
        if d < bd: bd, best = d, c
    if best is not None and bd < SEED_GATE:
        best['pts'].append(p); best['m'] = np.mean(best['pts'], 0)
    else:
        clusters.append({'pts': [p], 'm': p.copy()})
clusters = [c for c in clusters if len(c['pts']) >= SUP_MIN]
centers = np.array([c['m'] for c in clusters])
if centers.size == 0:
    raise SystemExit(
        f"no lesion columns seeded from {len(seeds)} ablation cores at SUP_MIN={SUP_MIN}, "
        f"SEED_GATE={SEED_GATE:.0f}px.\nThis stack has too little cross-slice ablation overlap "
        f"(expected when stacking unrelated specimens). Retry with a lower LSA_SUP_MIN "
        f"(e.g. LSA_SUP_MIN=2) and/or larger LSA_SEED_GATE.")
order = np.argsort(centers[:, 0])           # stable left->right id
centers = centers[order]
K = len(centers)
pitch = np.median(cKDTree(centers).query(centers, k=2)[0][:, 1])
ASSIGN_GATE = max(55.0, 0.6 * pitch)
OUT_DIST = 0.55 * pitch
print(f"columns K={K}  pitch={pitch:.0f}px  assign_gate={ASSIGN_GATE:.0f}  out_dist={OUT_DIST:.0f}")

# ---- assign one blob per column per slice (greedy by distance, gated) ----
assign = [[None] * N for _ in range(K)]      # assign[k][i] = blob dict or None
for i, bl in enumerate(blobs):
    if not bl: continue
    bc = np.array([b['cen'] for b in bl])
    cand = []
    for k in range(K):
        d = np.hypot(bc[:, 0] - centers[k, 0], bc[:, 1] - centers[k, 1])
        for bi in range(len(bl)):
            if d[bi] < ASSIGN_GATE: cand.append((d[bi], k, bi))
    cand.sort()
    uk, ub = set(), set()
    for d, k, bi in cand:
        if k in uk or bi in ub: continue
        assign[k][i] = bl[bi]; uk.add(k); ub.add(bi)

# ---- per-column trajectory, radii, outlier correction (average from adjacent) ----
def smooth_fill(vals, zs, span, sigma=1.6):
    fz = np.arange(span[0], span[1] + 1)
    v = np.interp(fz, zs, vals)
    return fz, ndi.gaussian_filter1d(v, sigma)

columns = {}
for k in range(K):
    zs = [i for i in range(N) if assign[k][i] is not None]
    if len(zs) < 2: continue
    cx = np.array([assign[k][i]['cen'][0] for i in zs])
    cy = np.array([assign[k][i]['cen'][1] for i in zs])
    span = (zs[0], zs[-1])
    fz, sx = smooth_fill(cx, zs, span); _, sy = smooth_fill(cy, zs, span)
    smc = {int(z): np.array([sx[j], sy[j]]) for j, z in enumerate(fz)}
    # mark centre outliers -> replace assignment centre by smoothed (use neighbour shape)
    outlier = {}
    for i in zs:
        if np.hypot(*(assign[k][i]['cen'] - smc[i])) > OUT_DIST:
            outlier[i] = True
    # coag radius + ablation presence, with area-spike outlier detection
    ar = np.array([assign[k][i]['area'] for i in zs], float)
    med = np.median(ar)
    for j, i in enumerate(zs):
        if ar[j] > 2.5 * med: outlier[i] = True            # torn / merged slice
    # ablation must be a CONTIGUOUS shallow run from the top (entry); isolated deep
    # ablation after a coag-only gap is noise -> excluded (allow <=2-slice gaps).
    abl_slices = sorted(i for i in zs if assign[k][i]['abl_area'] >= ABL_MIN and i not in outlier)
    abl_top = abl_bot = None
    if abl_slices:
        abl_top = abl_bot = abl_slices[0]
        for z in abl_slices[1:]:
            if z - abl_bot <= 2: abl_bot = z
            else: break
    columns[k] = dict(span=span, zs=zs, smc=smc, outlier=outlier,
                      abl_top=abl_top, abl_bot=abl_bot)

# a real laser lesion enters at the tissue surface, so every column must start near
# the top; a column first appearing mid-depth (only coagulation) is not a valid lesion.
TOP_MAX = 3
dropped = [k for k in columns if columns[k]['span'][0] > TOP_MAX]
for k in dropped:
    print(f"  drop col {k} (starts mid-depth at slice {columns[k]['span'][0]} -> not surface-connected)")
    del columns[k]
robust = sorted(columns)
print(f"robust columns: {len(robust)}")
for k in robust:
    c = columns[k]
    ab = '' if c['abl_top'] is None else f" abl[{c['abl_top']}-{c['abl_bot']}]"
    print(f"  col {k:2d} @({centers[k,0]:.0f},{centers[k,1]:.0f}) span {c['span'][0]:2d}-{c['span'][1]:2d}"
          f" n={len(c['zs']):2d} outliers={sorted(c['outlier'])}{ab}")

# ---- synthesise cleaned label maps ----
def paste(dst, crop, top, val, tis):
    h, w = crop.shape; y, x = top
    y0, x0 = max(0, y), max(0, x); y1, x1 = min(CH, y + h), min(CW, x + w)
    if y1 <= y0 or x1 <= x0: return
    sub = crop[y0 - y:y1 - y, x0 - x:x1 - x]
    region = dst[y0:y1, x0:x1]
    region[sub & tis[y0:y1, x0:x1]] = val

def nearest_inlier(k, i):
    c = columns[k]
    cands = [z for z in c['zs'] if z not in c['outlier'] and assign[k][z] is not None]
    if not cands: cands = [z for z in c['zs'] if assign[k][z] is not None]
    return min(cands, key=lambda z: abs(z - i)) if cands else None

ring = np.ones((9, 9), np.uint8)
for i in range(N):
    out = np.zeros((CH, CW), np.uint8)
    tis = tissue[i]
    out[tis] = LIVE
    for k in robust:
        c = columns[k]
        if not (c['span'][0] <= i <= c['span'][1]): continue
        target = c['smc'][i]
        use_self = (assign[k][i] is not None) and (i not in c['outlier'])
        src_i = i if use_self else nearest_inlier(k, i)
        if src_i is None: continue
        b = assign[k][src_i]
        srccen = b['cen']; off = np.round(target - srccen).astype(int)
        top = (b['bbox'][0] + off[1], b['bbox'][1] + off[0])
        coag = b['mask']
        # ensure coagulation actually rings the ablation (ablation-only is impossible)
        if b['abl_area'] > 0 and (b['area'] - b['abl_area']) < 0.3 * b['abl_area']:
            coag = cv2.dilate(b['mask'].astype(np.uint8), ring, iterations=2).astype(bool)
        paste(out, coag, top, COAG, tis)
        draw_abl = (c['abl_top'] is not None and c['abl_top'] <= i <= c['abl_bot'])
        if draw_abl:
            abl = b['abl']
            if abl.any():
                inside = cv2.erode(coag.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                abl = abl & inside[:abl.shape[0], :abl.shape[1]] if abl.shape == inside.shape else abl
                paste(out, abl, top, ABL, tis)
    np.save(os.path.join(OUTD, f'alc_{i:02d}.npy'), out)

json.dump({'K': int(K), 'centers': centers.tolist(),
           'columns': {int(k): {'span': list(columns[k]['span']),
                                 'abl_top': columns[k]['abl_top'], 'abl_bot': columns[k]['abl_bot'],
                                 'outliers': sorted(columns[k]['outlier'])} for k in robust}},
          open(os.path.join(WORK, 'columns_clean.json'), 'w'), indent=1)

# ---- diagnostics: montage before/after ----
PAL = np.array([[0, 0, 0], [0, 150, 0], [255, 40, 0], [0, 0, 255]], np.uint8)  # BGR: live,coag,abl
cell = 300; cols_n = 7; rows = int(np.ceil(N / cols_n))
mont = np.zeros((rows * cell, cols_n * cell, 3), np.uint8)
for i in range(N):
    cm = np.load(os.path.join(OUTD, f'alc_{i:02d}.npy'))
    im = cv2.resize(PAL[cm], (cell, cell), interpolation=cv2.INTER_NEAREST)
    cv2.putText(im, str(i), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
    r, cN = divmod(i, cols_n); mont[r * cell:(r + 1) * cell, cN * cell:(cN + 1) * cell] = im
cv2.imwrite(os.path.join(DIAG, 'montage_clean.png'), mont)
print('wrote', os.path.join(DIAG, 'montage_clean.png'))
print('done')
