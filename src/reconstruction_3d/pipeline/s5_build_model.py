"""Stage 5: build the 3D voxel model + interactive Plotly view + PLY export.

1. Use tracked lesion columns to define smooth centerlines.
2. Per slice, thin-plate-spline warp so each lesion snaps onto its column centerline
   (the explicit lesion-to-lesion registration), tissue deforms smoothly with it.
3. Stack into a voxel volume, marching-cubes surfaces, interactive Plotly model + PLY.
"""
import os, json
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.interpolate import RBFInterpolator
from skimage import measure
import plotly.graph_objects as go

WORK = os.environ.get('LSA_WORK') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')
# Consume the domain-cleaned, column-tracked maps (noise removed, lesions snapped to
# smooth centrelines, ablation enforced inside coagulation) -> no TPS needed here.
AL = os.path.join(WORK, 'aligned_clean')
PREFIX = 'alc_'
OUT = os.path.join(WORK, 'model')
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(os.path.join(WORK, 'meta.json')))
cc = json.load(open(os.path.join(WORK, 'columns_clean.json')))
K = cc['K']
N = len(meta)
CW = CH = 2200
S = 460
FZ = 4
SZ = 1.5

# ---- load cleaned maps, crop bbox, downsample ----
maps = [np.load(os.path.join(AL, f'{PREFIX}{i:02d}.npy')) for i in range(N)]
acc = np.zeros((CH, CW), bool)
for m in maps: acc |= (m > 0)
ys, xs = np.where(acc)
y0, x0 = max(0, ys.min()-30), max(0, xs.min()-30)
y1, x1 = min(CH, ys.max()+31), min(CW, xs.max()+31)
ch, cw = y1-y0, x1-x0
gw, gh = int(round(cw*S/max(ch,cw))), int(round(ch*S/max(ch,cw)))
sx, sy = gw/cw, gh/ch
def to_grid(p):
    p = np.asarray(p, float)
    return np.column_stack([(p[:,0]-x0)*sx, (p[:,1]-y0)*sy])
def down(m):
    return cv2.resize(m[y0:y1, x0:x1], (gw, gh), interpolation=cv2.INTER_NEAREST)
maps_g = [down(m) for m in maps]
print(f"grid {gw}x{gh}")

# Maps are already cleaned + column-tracked (lesions snapped to smooth centrelines),
# so no TPS lesion-snap is needed here -- just use the downsampled cleaned slices.
robust = list(range(K))
warped = maps_g
print(f"using {K} cleaned columns (no TPS needed)")

# ---- build volumes ----
from skimage import morphology
# tissue: regularize each slice (close + fill) so the outer shell is a clean cylinder
tis_slices = []
k = morphology.disk(5)
for w in warped:
    t = (w > 0).astype(np.uint8)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, k)
    t = cv2.morphologyEx(t, cv2.MORPH_OPEN, morphology.disk(3).astype(np.uint8))
    t = ndi.binary_fill_holes(t)
    lab, n = ndi.label(t)
    if n > 1:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n+1))
        t = lab == (np.argmax(sz)+1)
    tis_slices.append(t.astype(np.float32))
vt = np.stack(tis_slices)
vc = np.stack([((w==2)|(w==3)).astype(np.float32) for w in warped])
va = np.stack([(w==3).astype(np.float32) for w in warped])
def zint(v): return ndi.zoom(v, (FZ,1,1), order=1)
vt, vc, va = zint(vt), zint(vc), zint(va)
Z = vt.shape[0]

# remove small disconnected 3D blobs from lesion volumes (stray non-column coag).
# The 16 real columns are the large connected components; everything below the gap
# (~5000 vox) is a fragment -> drop it. Ablation may only live inside surviving coag.
def clean(v, min_vox):
    b = morphology.remove_small_objects(v > 0.5, min_size=min_vox)
    return (v * b).astype(np.float32)
def keep_surface(v, topz=10):
    # a lesion enters at the surface (z=0); drop any coag component whose shallowest
    # extent is mid-depth (a surface-disconnected tail left by columns merging up top).
    lab, n = ndi.label(v > 0.5)
    keep = np.zeros(v.shape, bool)
    for c in range(1, n + 1):
        m = lab == c
        if np.where(m)[0].min() <= topz: keep |= m
    return (v * keep).astype(np.float32)
vc = keep_surface(clean(vc, 5000), 10)
coag_support = ndi.binary_dilation(vc > 0.5, iterations=2)
va = clean(va, 150) * coag_support
# regularize tissue shell: z-median removes single-slice outliers (e.g. torn fragment)
vt = ndi.median_filter(vt, size=(9,1,1))
vt = ndi.median_filter(vt, size=(1,5,5))
np.savez_compressed(os.path.join(OUT,'volumes_tps.npz'), tissue=vt, coag=vc, abl=va, SZ=SZ)

vts = ndi.gaussian_filter(vt, (2.4,3.2,3.2))
vcs = ndi.gaussian_filter(vc, (1.1,1.2,1.2))
vas = ndi.gaussian_filter(va, (0.9,0.9,0.9))
# smoothing can pinch a thin tail into a tiny floating blob -> cull small components
# from the rendered (smoothed) volumes so only the real columns remain.
def cull(v, min_vox):
    return v * morphology.remove_small_objects(v > 0.5, min_size=min_vox)
vcs = keep_surface(cull(vcs, 3000), 10)
vas = cull(vas, 100) * ndi.binary_dilation(vcs > 0.5, iterations=2)

spacing=(SZ,1,1)
def mesh(v, level, color, op, name):
    try: verts,faces,_,_ = measure.marching_cubes(v, level, spacing=spacing)
    except Exception: return None
    z=(Z*SZ)-verts[:,0]
    return go.Mesh3d(x=verts[:,2],y=verts[:,1],z=z,i=faces[:,0],j=faces[:,1],k=faces[:,2],
        color=color,opacity=op,name=name,showlegend=True,flatshading=False,
        lighting=dict(ambient=0.5,diffuse=0.85,specular=0.2,roughness=0.85),hoverinfo='name')

traces=[t for t in [
    mesh(vts,0.5,'#27c727',0.10,'Live tissue'),
    mesh(vcs,0.5,'#1f4fff',0.42,'Coagulation zone'),
    mesh(vas,0.5,'#ff2222',0.96,'Ablation zone')] if t is not None]
fig=go.Figure(traces)
fig.update_layout(title=f'LSA 3D lesion reconstruction — {N} sections, {len(robust)} ablation columns',
    scene=dict(aspectmode='data', xaxis_title='x (px)', yaxis_title='y (px)',
        zaxis_title='depth  (slice 0 = surface)', bgcolor='white'),
    template='plotly_white', legend=dict(itemsizing='constant'))
fig.write_html(os.path.join(OUT,'lesion_3d_surfaces.html'), include_plotlyjs='cdn')
print('wrote lesion_3d_surfaces.html')

# ---- point cloud (Scatter3d) + PLY export ----
def pts_of(v, sub):
    zz, yy, xx = np.where(v > 0.5)
    if len(zz) > sub:
        sel = np.random.default_rng(0).choice(len(zz), sub, replace=False)
        zz, yy, xx = zz[sel], yy[sel], xx[sel]
    z = (Z*SZ) - zz*SZ
    return xx.astype(float), yy.astype(float), z
abl_b = va > 0.5
coa_only = (vc > 0.5) & (~abl_b)
ax_, ay_, az_ = pts_of(abl_b.astype(float), 60000)
cx_, cy_, cz_ = pts_of(coa_only.astype(float), 90000)
# tissue surface shell points only (erode then xor) to keep cloud light
tsurf = (vt > 0.5) & (~(ndi.binary_erosion(vt > 0.5, iterations=1)))
tx_, ty_, tz_ = pts_of(tsurf.astype(float), 60000)
pc = go.Figure([
    go.Scatter3d(x=tx_,y=ty_,z=tz_,mode='markers',name='Live tissue',
        marker=dict(size=1.3,color='#27c727',opacity=0.18)),
    go.Scatter3d(x=cx_,y=cy_,z=cz_,mode='markers',name='Coagulation',
        marker=dict(size=1.8,color='#1f4fff',opacity=0.5)),
    go.Scatter3d(x=ax_,y=ay_,z=az_,mode='markers',name='Ablation',
        marker=dict(size=2.0,color='#ff2222',opacity=0.85))])
pc.update_layout(title=f'LSA 3D lesion point cloud — {N} sections, {len(robust)} columns',
    scene=dict(aspectmode='data',zaxis_title='depth (slice 0 = surface)',bgcolor='white'),
    template='plotly_white', legend=dict(itemsizing='constant'))
pc.write_html(os.path.join(OUT,'lesion_3d_pointcloud.html'), include_plotlyjs='cdn')
print('wrote lesion_3d_pointcloud.html')

def write_ply(path, clouds):
    allpts = []
    for (x,y,z),(r,g,b) in clouds:
        for i in range(len(x)):
            allpts.append((x[i],y[i],z[i],r,g,b))
    with open(path,'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(allpts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in allpts:
            f.write(f"{p[0]:.2f} {p[1]:.2f} {p[2]:.2f} {p[3]} {p[4]} {p[5]}\n")
write_ply(os.path.join(OUT,'lesion_pointcloud.ply'),
          [((tx_,ty_,tz_),(39,199,39)),((cx_,cy_,cz_),(31,79,255)),((ax_,ay_,az_),(255,34,34))])
print('wrote lesion_pointcloud.ply')

# ---- depth profile analysis ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
px2_to_mm = None
zc = (np.arange(N))
red_a = np.array([m['n_red'] for m in meta], float)
blu_a = np.array([m['n_blue'] for m in meta], float)
grn_a = np.array([m['n_green'] for m in meta], float)
scale = np.load(os.path.join(WORK,'scale.npy'))
# normalize areas by per-slice scale^2 to common units
red_n = red_a/scale**2; blu_n=blu_a/scale**2; grn_n=grn_a/scale**2
fig2, ax = plt.subplots(1,2, figsize=(13,5))
ax[0].plot(zc, grn_n/1e3, '-o', color='#27a727', label='live tissue', lw=2, ms=3)
ax[0].plot(zc, blu_n/1e3, '-o', color='#1f4fff', label='coagulation', lw=2, ms=3)
ax[0].plot(zc, red_n/1e3, '-o', color='#ee2222', label='ablation', lw=2, ms=3)
ax[0].set_xlabel('depth (slice index, 0 = surface)'); ax[0].set_ylabel('cross-sectional area (10^3 norm. px)')
ax[0].set_title('Damage area vs depth'); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(zc, blu_n/1e3, '-o', color='#1f4fff', label='coagulation', lw=2, ms=3)
ax[1].plot(zc, red_n/1e3, '-o', color='#ee2222', label='ablation', lw=2, ms=3)
ax[1].set_xlabel('depth (slice index)'); ax[1].set_ylabel('cross-sectional area (10^3 norm. px)')
ax[1].set_title('Ablation ends shallower than coagulation'); ax[1].legend(); ax[1].grid(alpha=.3)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT,'depth_profile.png'), dpi=120)
print('wrote depth_profile.png')

# diagnostic montage of warped slices
cols=7; rows=(N+cols-1)//cols
cell=300; mont=np.zeros((rows*cell,cols*cell,3),np.uint8)
PAL=np.array([[0,0,0],[0,160,0],[255,40,0],[0,0,255]],np.uint8)
for i in range(N):
    im=cv2.resize(PAL[warped[i]],(cell,cell)); cv2.putText(im,str(i),(8,28),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
    r,c=divmod(i,cols); mont[r*cell:(r+1)*cell,c*cell:(c+1)*cell]=im
cv2.imwrite(os.path.join(WORK,'montage_warped.png'),mont)
print('done')
