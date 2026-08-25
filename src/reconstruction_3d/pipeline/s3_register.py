"""Stage 3: richer pairwise registration.

For every consecutive pair (src=i -> dst=i-1) we evaluate several candidate transforms
and keep the one with the best tissue IoU:
  - identity (on the pre-scaled slices),
  - ECC affine on the tissue silhouette  -> loosely matches sample height/width + pose,
  - point ICP on lesions + internal white holes (multi-start) -> uses all correspondences.

A robust isotropic pre-scale (lesion pitch / tissue diameter) is applied first so global
scale cannot drift while the per-pair affine still adapts H/W to improve overlap.
"""
import os, json
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

WORK = os.environ.get('LSA_WORK') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')
LBL_DIR = os.path.join(WORK, 'labels')
OUT_DIR = os.path.join(WORK, 'aligned')
DIAG_DIR = os.path.join(WORK, 'reg_diag')
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(DIAG_DIR, exist_ok=True)
meta = json.load(open(os.path.join(WORK, 'meta.json')))
N = len(meta)
labels = [np.load(os.path.join(LBL_DIR, f'lbl_{i:02d}.npy')) for i in range(N)]
CW = CH = 2200
KS = 600                       # small canvas for ECC / IoU
k = KS / CW
HOLE_MIN = 150                 # min hole area (working-res px)

# ---- NBTC tissue scans (same FOV as masks): structural edge/texture image ----
# The grayscale histology has far more context than the 3-colour mask: tissue
# boundary, internal tears, vessels and the bright lesion dots.  We turn each into a
# blurred gradient-magnitude ("edge energy") image so ECC can align real structure
# regardless of absolute stain intensity.
TIS_DIR = os.path.join(WORK, 'tissue')
def load_struct(i):
    g = cv2.imread(os.path.join(TIS_DIR, f't_{i:02d}.png'), cv2.IMREAD_GRAYSCALE)
    if g is None or g.shape != labels[i].shape:
        g = np.zeros(labels[i].shape, np.uint8) if g is None else cv2.resize(g, (labels[i].shape[1], labels[i].shape[0]))
    gf = cv2.GaussianBlur(g, (0, 0), 1.2).astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0)
    mag *= (labels[i] >= 0)  # keep full frame; background gradient is ~0 anyway
    m = np.percentile(mag, 99.5) or 1.0
    return np.clip(mag / m * 255.0, 0, 255).astype(np.uint8)
tstruct = [load_struct(i) for i in range(N)]

# ---- affine helpers ----
def to33(M):
    o = np.eye(3); o[:2] = M; return o
def compose(A, B):             # apply B first then A
    return (to33(A) @ to33(B))[:2]
def apply_aff(M, p):
    p = np.asarray(p, float); return p @ M[:, :2].T + M[:, 2]
def rot2d(d):
    a = np.deg2rad(d); c, s = np.cos(a), np.sin(a); return np.array([[c, -s], [s, c]])

def kabsch(P, Q):
    """Rigid 2x3 (rotation+translation, no scale, no reflection) mapping P->Q."""
    muP = P.mean(0); muQ = Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    M = np.zeros((2, 3)); M[:2, :2] = R; M[:, 2] = muQ - R @ muP
    return M

# ---- features ----
def holes_of(lbl):
    t = lbl > 0
    h = ndi.binary_fill_holes(t) & ~t
    lab, n = ndi.label(h)
    if n == 0: return np.zeros((0, 2))
    sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    cen = ndi.center_of_mass(h, lab, range(1, n + 1))
    return np.array([[c[1], c[0]] for c, s in zip(cen, sz) if s >= HOLE_MIN])

raw_les = [np.array(meta[i]['lesion_centroids'], float) if meta[i]['lesion_centroids']
           else np.zeros((0, 2)) for i in range(N)]

# ---- absolute isotropic pre-scale (anti-drift) ---- (flip-invariant, so done first)
med_nn = np.full(N, np.nan)
for i in range(N):
    if len(raw_les[i]) >= 3:
        d, _ = cKDTree(raw_les[i]).query(raw_les[i], k=2); med_nn[i] = np.median(d[:, 1])
rel = ~np.isnan(med_nn)
target = np.median(med_nn[rel])
scale = np.full(N, np.nan); scale[rel] = target / med_nn[rel]
diam = np.array([2 * np.sqrt(m['tissue_area'] / np.pi) for m in meta])
Dstar = np.median(diam[rel] * scale[rel]); scale[~rel] = Dstar / diam[~rel]
scale = np.array([np.median(scale[max(0, i-1):i+2]) for i in range(N)])
np.save(os.path.join(WORK, 'scale.npy'), scale)

# ---- automatic mounting-orientation (mirror) detection ----
# Sections placed on the slide MIRRORED cannot be aligned by any rigid (reflection-free)
# transform. For each consecutive pair we test whether slice i needs a REFLECTION relative
# to i-1 (does a flipped copy align far better?). A parity chain then yields each slice's
# handedness vs slice 0; we flip the minority so most slices keep their original orientation.
MATCH_R = 28.0   # canvas px: lesion/hole counted as matched within this radius
_DkD = np.array([[k, 0, 0], [0, k, 0]])
def _refl_mat(i, flip):
    H, W = labels[i].shape
    if flip == 'ud': return np.array([[1., 0, 0], [0, -1., H - 1]])
    if flip == 'lr': return np.array([[-1., 0, W - 1], [0, 1., 0]])
    return np.array([[1., 0, 0], [0, 1., 0]])
def _pre(i, flip):
    Fi = _refl_mat(i, flip)
    tc = apply_aff(Fi, [meta[i]['tissue_centroid']])[0]; s = scale[i]
    Pm = np.array([[s, 0, CW/2 - s*tc[0]], [0, s, CH/2 - s*tc[1]]])
    les = apply_aff(Fi, raw_les[i]) if len(raw_les[i]) else np.zeros((0, 2))
    lblf = labels[i] if flip is None else (labels[i][::-1] if flip == 'ud' else labels[i][:, ::-1])
    hol = holes_of(lblf)
    return (apply_aff(Pm, les) if len(les) else np.zeros((0, 2)),
            apply_aff(Pm, hol) if len(hol) else np.zeros((0, 2)), compose(Pm, Fi))
def _icp_frac(sL, sH, dL, dH):
    if len(sL)+len(sH) < 3 or len(dL)+len(dH) < 3: return 0.0, 0
    ctr = np.array([CW/2, CH/2]); best = None
    def match(M, gate):
        sp, dp = [], []
        for ss, ds in ((sL, dL), (sH, dH)):
            if len(ss) and len(ds):
                ts = apply_aff(M, ss); dist, jj = cKDTree(ds).query(ts)
                for a in range(len(ts)):
                    if dist[a] < gate: sp.append(ss[a]); dp.append(ds[jj[a]])
        return np.array(sp), np.array(dp)
    for deg in [0, 30, -30, 60, -60, 90, -90, 135, 180]:
        M = np.zeros((2, 3)); M[:2, :2] = rot2d(deg); M[:, 2] = ctr - M[:, :2] @ ctr
        for it in range(30):
            sp, dp = match(M, max(160*(0.88**it), 20))
            if len(sp) < 3: break
            est, inl = cv2.estimateAffinePartial2D(sp.astype(np.float32), dp.astype(np.float32),
                                                   method=cv2.RANSAC, ransacReprojThreshold=10.0)
            if est is None or inl is None: break
            m = inl.ravel().astype(bool)
            if m.sum() < 3: break
            Mn = kabsch(sp[m], dp[m])
            if np.allclose(Mn, M, atol=1e-3): M = Mn; break
            M = Mn
        sp, dp = match(M, 30)
        if len(sp) >= 3:
            key = (len(sp), -float(np.mean(np.linalg.norm(apply_aff(M, sp)-dp, axis=1))))
            if best is None or key > best[0]: best = (key, M.copy())
    if best is None: return 0.0, 0
    alls = np.vstack([p for p in (sL, sH) if len(p)]); alld = np.vstack([p for p in (dL, dH) if len(p)])
    dist, _ = cKDTree(alld).query(apply_aff(best[1], alls))
    return float((dist < MATCH_R).mean()), int((dist < MATCH_R).sum())
def _tis_ncc(i, PiS, j, PiD):
    sb = cv2.GaussianBlur(cv2.warpAffine(tstruct[i], compose(_DkD, PiS), (KS, KS)), (0, 0), 8).astype(np.float32)/255
    db = cv2.GaussianBlur(cv2.warpAffine(tstruct[j], compose(_DkD, PiD), (KS, KS)), (0, 0), 8).astype(np.float32)/255
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(db, sb, warp, cv2.MOTION_EUCLIDEAN,
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
        sb = cv2.warpAffine(sb, warp, (KS, KS))
    except cv2.error:
        pass
    m = (sb > 0.02) | (db > 0.02)
    if m.sum() < 50: return 0.0
    a = sb[m]-sb[m].mean(); b = db[m]-db[m].mean(); d = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d > 0 else 0.0

r = np.zeros(N, int)
print("mirror scan (i<-i-1): metric proper/flipped -> flip?")
for i in range(1, N):
    dL, dH, PiD = _pre(i-1, None)
    pL, pH, PiSp = _pre(i, None); pf, pn = _icp_frac(pL, pH, dL, dH)
    fL, fH, PiSf = _pre(i, 'ud'); ff, fn = _icp_frac(fL, fH, dL, dH)
    if max(pn, fn) >= 4:
        r[i] = 1 if (ff > pf + 0.15 and ff > 0.55) else 0
        tag = f"les {pf:.2f}/{ff:.2f}"
    else:
        npp = _tis_ncc(i, PiSp, i-1, PiD); nff = _tis_ncc(i, PiSf, i-1, PiD)
        r[i] = 1 if (nff > npp + 0.10 and nff > 0.55) else 0
        tag = f"tis {npp:.2f}/{nff:.2f}"
    print(f"  {i:2d}<-{i-1:2d}  {tag}  {'FLIP' if r[i] else ''}")
s = np.zeros(N, int)
for i in range(1, N): s[i] = s[i-1] ^ r[i]
if s.sum() > N/2: s = 1 - s                     # flip the minority, keep majority as-is
FLIP = {i: 'ud' for i in range(N) if s[i] == 1}

F = [_refl_mat(i, FLIP.get(i)) for i in range(N)]
def flip_img(i):
    f = FLIP.get(i)
    if f == 'ud': return labels[i][::-1].copy()
    if f == 'lr': return labels[i][:, ::-1].copy()
    return labels[i]
lesions = [apply_aff(F[i], raw_les[i]) if len(raw_les[i]) else np.zeros((0, 2)) for i in range(N)]
holes = [holes_of(flip_img(i)) for i in range(N)]
print("holes per slice:", [len(h) for h in holes], "| detected flips:", sorted(FLIP))

# pre-scale transform: scale about the (reflected) tissue centroid, centre at canvas centre.
# P[i] maps REFLECTED slice coords -> canvas (lesions/holes already live in that frame);
# Pimg[i] additionally reflects the original image pixels so renders/warps stay consistent.
P = []
for i in range(N):
    tx, ty = apply_aff(F[i], [meta[i]['tissue_centroid']])[0]; s = scale[i]
    P.append(np.array([[s, 0, CW/2 - s*tx], [0, s, CH/2 - s*ty]]))
Pimg = [compose(P[i], F[i]) for i in range(N)]
def ps(i, pts):
    return apply_aff(P[i], pts) if len(pts) else np.zeros((0, 2))

# small-canvas tissue render (for ECC + IoU); M is extra transform in canvas coords
Dk = np.array([[k, 0, 0], [0, k, 0]])
def render_small(i, M=None):
    T = compose(Dk, Pimg[i]) if M is None else compose(Dk, compose(M, Pimg[i]))
    return cv2.warpAffine(((labels[i] > 0)*255).astype(np.uint8), T, (KS, KS))

def render_struct(i, M=None):
    T = compose(Dk, Pimg[i]) if M is None else compose(Dk, compose(M, Pimg[i]))
    return cv2.warpAffine(tstruct[i], T, (KS, KS), flags=cv2.INTER_LINEAR)

def iou(a, b):
    A = a > 127; B = b > 127
    u = (A | B).sum()
    return (A & B).sum() / u if u else 0.0

def ncc(a, b):
    """Normalised cross-correlation of two tissue renders.  We blur heavily first so the
    *gross shape / boundary density* (which is consistent between adjacent sections)
    dominates over interior speckle (which is not), then correlate over the union."""
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 9)
    b = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 9)
    m = (a > 3) | (b > 3)
    if m.sum() < 50: return 0.0
    av = a[m] - a[m].mean(); bv = b[m] - b[m].mean()
    d = np.sqrt((av*av).sum() * (bv*bv).sum())
    return float((av*bv).sum() / d) if d > 0 else 0.0

MATCH_R = 28.0   # canvas px: lesion/hole counted as matched within this radius
def quality(M, i, j):
    """Selection score: primarily lesion+hole correspondence, secondarily tissue IoU.
    A round silhouette barely changes IoU under rotation, so lesion matching must lead."""
    iou_t = iou(render_small(i, M), render_small(j))
    nc = max(0.0, ncc(render_struct(i, M), render_struct(j)))   # tissue structural overlap
    res = []
    for s_set, d_set in ((ps(i, lesions[i]), ps(j, lesions[j])),
                         (ps(i, holes[i]),   ps(j, holes[j]))):
        if len(s_set) and len(d_set):
            ts = apply_aff(M, s_set)
            dist, _ = cKDTree(d_set).query(ts)
            res.extend(dist.tolist())
    if len(res) >= 4:
        res = np.array(res)
        frac = float((res < MATCH_R).mean())
        resid = float(np.minimum(res, MATCH_R).mean() / MATCH_R)   # 0 good .. 1 bad
        feat = 0.7 * frac + 0.3 * (1 - resid)
        # lesion correspondence leads; tissue IoU + NBTC structural NCC break ties and
        # rescue ambiguous lesion constellations (e.g. PVC3 slices 6-8).
        score = 0.45 * feat + 0.30 * iou_t + 0.25 * nc
        if iou_t < 0.4:            # implausible tissue overlap -> demote (e.g. reflection)
            score = min(score, iou_t)
        return score, iou_t, frac, nc
    # lesion-free slices: rely on tissue silhouette + structure
    return 0.5 * iou_t + 0.5 * nc, iou_t, -1.0, nc

# ---- candidate estimators ----
def ecc_affine(i, j):
    sb = cv2.GaussianBlur(render_small(i), (0, 0), 5).astype(np.float32)/255
    db = cv2.GaussianBlur(render_small(j), (0, 0), 5).astype(np.float32)/255
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 250, 1e-6)
    try:
        cv2.findTransformECC(db, sb, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        return None
    W = warp.astype(float)
    M = np.array([[W[0,0], W[0,1], W[0,2]/k], [W[1,0], W[1,1], W[1,2]/k]])  # small->canvas
    A = M[:, :2]; sv = np.linalg.svd(A, compute_uv=False)
    if np.linalg.det(A) <= 0 or not (0.75 < sv.min() and sv.max() < 1.3 and sv.max()/sv.min() < 1.5):
        return None
    return M

def tissue_ecc(i, j, init=None):
    """Refine pose using the NBTC tissue *structure* (edge energy) via ECC.
    `init` is an extra canvas transform to start from (e.g. the lesion ICP fit); the
    returned transform already includes it.  MOTION_EUCLIDEAN keeps it rigid (scale is
    pre-normalised) so texture/silhouette only nudges rotation+translation."""
    sb = cv2.GaussianBlur(render_struct(i, init), (0, 0), 8).astype(np.float32)/255
    db = cv2.GaussianBlur(render_struct(j),       (0, 0), 8).astype(np.float32)/255
    if sb.max() < 0.02 or db.max() < 0.02:
        return None
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cv2.findTransformECC(db, sb, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    except cv2.error:
        return None
    W = warp.astype(float)
    Mr = np.array([[W[0,0], W[0,1], W[0,2]/k], [W[1,0], W[1,1], W[1,2]/k]])  # residual, canvas
    A = Mr[:, :2]; sv = np.linalg.svd(A, compute_uv=False)
    if np.linalg.det(A) <= 0 or sv.max()/sv.min() > 1.05:   # rigid -> singular values ~1
        return None
    return Mr if init is None else compose(Mr, init)

def match_pairs(src_les, dst_les, src_hol, dst_hol, M, gate):
    sp, dp = [], []
    for s_set, d_set in ((src_les, dst_les), (src_hol, dst_hol)):
        if len(s_set) == 0 or len(d_set) == 0: continue
        ts = apply_aff(M, s_set); tree = cKDTree(d_set)
        dist, j = tree.query(ts)
        for a in range(len(ts)):
            if dist[a] < gate:
                sp.append(s_set[a]); dp.append(d_set[j[a]])
    return np.array(sp), np.array(dp)

def point_icp(i, j):
    """Multi-start similarity ICP on lesions+holes (rotation+scale+translation only).
    Similarity (not full affine) avoids reflections and degenerate shear, so it is the
    stable workhorse for lesion-to-lesion alignment; H/W anisotropy is handled by ECC."""
    sl, dl = ps(i, lesions[i]), ps(j, lesions[j])
    sh, dh = ps(i, holes[i]), ps(j, holes[j])
    if len(sl) + len(sh) < 3 or len(dl) + len(dh) < 3:
        return None
    ctr = np.array([CW/2, CH/2])
    best = None  # (count, -resid), M
    for deg in [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 135, 180]:
        M = np.zeros((2, 3)); M[:2, :2] = rot2d(deg)
        M[:, 2] = ctr - M[:, :2] @ ctr        # rotate about canvas centre
        for it in range(40):
            gate = max(160 * (0.88 ** it), 20)
            sp, dp = match_pairs(sl, dl, sh, dh, M, gate)
            if len(sp) < 3: break
            # RANSAC to find inliers, then a pure RIGID fit (scale already normalised)
            est, inl = cv2.estimateAffinePartial2D(sp.astype(np.float32), dp.astype(np.float32),
                                                   method=cv2.RANSAC, ransacReprojThreshold=10.0)
            if est is None or inl is None: break
            mask = inl.ravel().astype(bool)
            if mask.sum() < 3: break
            Mn = kabsch(sp[mask], dp[mask])
            if np.allclose(Mn, M, atol=1e-3): M = Mn; break
            M = Mn
        sp, dp = match_pairs(sl, dl, sh, dh, M, 30)
        if len(sp) >= 3:
            resid = float(np.mean(np.linalg.norm(apply_aff(M, sp) - dp, axis=1)))
            key = (len(sp), -resid)
            if best is None or key > best[0]:
                best = (key, M.copy())
    if best is None:
        return None
    # polish: trimmed least-squares rigid over ALL matches (drop worst 20%) so the
    # pose is balanced across the whole constellation -> removes systematic shift/tilt
    M = best[1]
    for _ in range(12):
        sp, dp = match_pairs(sl, dl, sh, dh, M, 42)
        if len(sp) < 3: break
        res = np.linalg.norm(apply_aff(M, sp) - dp, axis=1)
        kk = max(3, int(round(0.8 * len(sp))))
        idx = np.argsort(res)[:kk]
        Mn = kabsch(sp[idx], dp[idx])
        if np.allclose(Mn, M, atol=1e-4): M = Mn; break
        M = Mn
    return M

# ---- run pairwise, choose best by IoU ----
Q = [np.array([[1., 0, 0], [0, 1., 0]])]
log = []
for i in range(1, N):
    cands = [('identity', np.array([[1., 0, 0], [0, 1., 0]]))]
    Me = ecc_affine(i, i - 1)
    if Me is not None: cands.append(('ecc', Me))
    Mp = point_icp(i, i - 1)
    if Mp is not None: cands.append(('pts', Mp))
    Mt = tissue_ecc(i, i - 1)                 # tissue structure from identity
    if Mt is not None: cands.append(('tis', Mt))
    if Mp is not None:                        # tissue refine of the lesion fit
        Mpt = tissue_ecc(i, i - 1, init=Mp)
        if Mpt is not None: cands.append(('pts+tis', Mpt))
    scored = []
    for name, M in cands:
        q, iou_t, frac, nc = quality(M, i, i - 1)
        scored.append((q, iou_t, frac, nc, name, M))
    scored.sort(key=lambda x: -x[0])
    best_q, best_iou, best_frac, best_nc, best_name, best_M = scored[0]
    Q.append(compose(Q[i - 1], best_M))
    log.append((i, best_name, round(float(best_iou), 3),
                {n: round(float(s), 3) for s, _, _, _, n, _ in scored}))
    print(f"slice {i:2d}<-{i-1:2d}  use={best_name:8s} score={best_q:.3f} "
          f"IoU={best_iou:.3f} lesionMatch={best_frac:.2f} ncc={best_nc:.2f}   "
          f"cand={ {n: round(float(s),3) for s,_,_,_,n,_ in scored} }")

# final transforms + recentre (Pimg folds in the mounting-mirror reflection so the warp
# of the ORIGINAL label/tissue pixels is corrected automatically downstream)
final = [compose(Q[i], Pimg[i]) for i in range(N)]
cc = np.array([apply_aff(final[i], [meta[i]['tissue_centroid']])[0] for i in range(N)])
off = np.array([CW/2, CH/2]) - cc.mean(0)
T_off = np.array([[1, 0, off[0]], [0, 1, off[1]]], float)
final = [compose(T_off, f) for f in final]
np.save(os.path.join(WORK, 'transforms.npy'), np.array(final))
json.dump({'canvas': [CW, CH], 'log': log}, open(os.path.join(WORK, 'reg_log.json'), 'w'), indent=1)

# warp + overlays (lesions bright, holes outlined cyan, tissue dim)
PAL = np.array([[0,0,0],[0,160,0],[255,40,0],[0,0,255]], np.uint8)
for i in range(N):
    w = cv2.warpAffine(labels[i], final[i], (CW, CH), flags=cv2.INTER_NEAREST)
    np.save(os.path.join(OUT_DIR, f'al_{i:02d}.npy'), w)
    cv2.imwrite(os.path.join(DIAG_DIR, f'al_{i:02d}.png'), PAL[w])
for i in range(1, N):
    a = np.load(os.path.join(OUT_DIR, f'al_{i-1:02d}.npy'))
    b = np.load(os.path.join(OUT_DIR, f'al_{i:02d}.npy'))
    ov = np.zeros((CH, CW, 3), np.uint8)
    ov[:, :, 2] = np.maximum(((a==2)|(a==3))*255, (a==1)*80).astype(np.uint8)
    ov[:, :, 1] = np.maximum(((b==2)|(b==3))*255, (b==1)*80).astype(np.uint8)
    cv2.imwrite(os.path.join(DIAG_DIR, f'overlay_{i-1:02d}_{i:02d}.png'), cv2.resize(ov, (900, 900)))

# NBTC tissue overlays (prev=magenta, cur=green) using the SAME transforms -> shows
# whether the real tissue silhouette/structure agrees with the mask-driven alignment.
def fg(i):
    g = cv2.imread(os.path.join(TIS_DIR, f't_{i:02d}.png'), cv2.IMREAD_GRAYSCALE)
    if g is None:                       # no NBTC tissue scan for this dataset
        return np.zeros(labels[i].shape, np.uint8)
    return np.where(g < 225, (235 - g.astype(np.int16)).clip(0, 255), 0).astype(np.uint8)
for i in range(1, N):
    ga = cv2.warpAffine(fg(i-1), final[i-1], (CW, CH))
    gb = cv2.warpAffine(fg(i),   final[i],   (CW, CH))
    ov = np.zeros((CH, CW, 3), np.uint8)
    ov[:, :, 2] = ga; ov[:, :, 0] = ga; ov[:, :, 1] = gb   # magenta vs green
    cv2.imwrite(os.path.join(DIAG_DIR, f'tov_{i-1:02d}_{i:02d}.png'), cv2.resize(ov, (900, 900)))
print('done')
