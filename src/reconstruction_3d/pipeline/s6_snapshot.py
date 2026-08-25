"""Stage 6: render static PNG snapshots of the cleaned 3D model (work/model/)."""
import os, numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
import plotly.graph_objects as go
WORK=os.environ.get('LSA_WORK') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work'); OUT=os.path.join(WORK,'model')
d=np.load(os.path.join(OUT,'volumes_tps.npz'))
vt,vc,va,SZ=d['tissue'],d['coag'],d['abl'],float(d['SZ']); Z=vt.shape[0]
vts=ndi.gaussian_filter(vt,(2.4,3.2,3.2)); vcs=ndi.gaussian_filter(vc,(1.1,1.1,1.1)); vas=ndi.gaussian_filter(va,(.9,.9,.9))
def cull(v,m): return v*morphology.remove_small_objects(v>0.5,min_size=m)
def keep_surface(v,topz=10):
    lab,n=ndi.label(v>0.5); keep=np.zeros(v.shape,bool)
    for c in range(1,n+1):
        mm=lab==c
        if np.where(mm)[0].min()<=topz: keep|=mm
    return v*keep
vcs=keep_surface(cull(vcs,3000),10); vas=cull(vas,100)*ndi.binary_dilation(vcs>0.5,iterations=2)
spacing=(SZ,1,1)
def mesh(v,c,o,n):
    if (v>0.5).sum()==0: return None        # nothing to surface (thin/empty volume)
    verts,faces,_,_=measure.marching_cubes(v,0.5,spacing=spacing)
    z=(Z*SZ)-verts[:,0]
    return go.Mesh3d(x=verts[:,2],y=verts[:,1],z=z,i=faces[:,0],j=faces[:,1],k=faces[:,2],color=c,opacity=o,name=n,
        lighting=dict(ambient=0.5,diffuse=0.85,specular=0.2,roughness=0.85))
tt=mesh(vts,'#27c727',0.10,'t'); tc=mesh(vcs,'#1f4fff',0.45,'c'); ta=mesh(vas,'#ff2222',0.97,'a')
cams={'iso':dict(eye=dict(x=1.5,y=1.5,z=0.8)),'front':dict(eye=dict(x=0.0,y=2.5,z=0.0)),'side':dict(eye=dict(x=2.5,y=0.0,z=0.0))}
for tag,cam in cams.items():
    for layer in ['lesions','full']:
        data=[m for m in ([tc,ta] if layer=='lesions' else [tt,tc,ta]) if m is not None]
        fig=go.Figure(data)
        fig.update_layout(scene=dict(aspectmode='data',camera=cam,bgcolor='white',
            xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)),
            margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='white',width=900,height=950)
        fig.write_image(os.path.join(OUT,f'tps_{layer}_{tag}.png'))
print('done')
