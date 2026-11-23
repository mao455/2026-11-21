"""Public initial-RGB shape prior registered to current calibrated RGB masks.
Experimental diagnostic. The prior fixes shape, never current object pose.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation

def fit_shape(volume,obs,initial_info):
    _,frames=volume.frames(obs)
    if len(frames)<3:return None,dict(valid=False,reason='insufficient_current_views')
    initial_center=np.asarray(initial_info['center']);vertices=np.asarray(initial_info['collision_points'])
    local=vertices-initial_center
    hull=ConvexHull(local);axes=[np.arange(a,b+.00035,.0007) for a,b in zip(local.min(0),local.max(0))]
    grid=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
    points=grid[np.max(grid@hull.equations[:,:3].T+hull.equations[:,3],axis=1)<1e-10]
    points=np.r_[points,local]
    if len(points)>1800:points=points[np.linspace(0,len(points)-1,1800,dtype=int)]
    pixelsets={name:np.argwhere(f['mask'])[:,::-1].astype(float) for name,f in frames.items()}
    pixelsets={name:p[np.linspace(0,len(p)-1,min(160,len(p)),dtype=int)] for name,p in pixelsets.items()}
    def world(x):
        p=points.copy();p[:,2]*=x[3]
        rot=Rotation.from_euler('z',x[2]).as_matrix();p=p@rot.T
        p[:,2]+=initial_info['table_z']-p[:,2].min()+.0001
        p[:,:2]+=x[:2]
        return p
    def residual(x):
        p=world(x);r=[]
        for name,f in frames.items():
            uv,dist,valid=volume.project(name,p,f['camera'])
            depth=map_coordinates(f['robot_depth'],uv[:,::-1].T,order=0,mode='constant',cval=0.)
            visible=valid & (dist<depth-.0005) & color_projection_known(f,uv)
            sdf=map_coordinates(f['sdf'],uv[:,::-1].T,order=1,mode='constant',cval=20.)
            forward=np.maximum(sdf,0.)*visible
            reverse=cKDTree(uv[visible]).query(pixelsets[name])[0] if visible.any() else np.full(len(pixelsets[name]),20.)
            r.extend(forward/np.sqrt(len(points)));r.extend(reverse/np.sqrt(len(reverse)))
        return np.r_[r,(x[3]-1.)*.15]
    base=np.r_[initial_center[:2],0.,1.]
    lower=np.r_[initial_center[:2]-.025,-np.pi,.35];upper=np.r_[initial_center[:2]+.025,np.pi,.085/(initial_info['top_z']-initial_info['table_z'])]
    choices=[]
    for dx,dy,yaw,scale in [(0,0,0,1),(-.003,0,0,1),(.003,0,0,1),(0,-.003,0,1),(0,.003,0,1),(0,0,.35,1),(0,0,-.35,1),(0,0,0,.6),(0,0,0,1.5),(0,0,0,min(5.,upper[3]-.01))]:
        fit=least_squares(residual,base+[dx,dy,yaw,scale-1],bounds=(lower,upper),diff_step=1e-3,x_scale=[.005,.005,.3,.5],max_nfev=70)
        choices.append((float(np.sum(residual(fit.x)**2)),fit.x))
    choices.sort(key=lambda x:x[0]);cost,x=choices[0];p=world(x);audits={};passed=True
    for name,f in frames.items():
        uv,dist,valid=volume.project(name,p,f['camera']);depth=map_coordinates(f['robot_depth'],uv[:,::-1].T,order=0,mode='constant',cval=0.);visible=valid & (dist<depth-.0005) & color_projection_known(f,uv)
        reverse=cKDTree(uv[visible]).query(np.argwhere(f['mask'])[:,::-1])[0] if visible.any() else np.full(int(f['mask'].sum()),100.)
        sdf=map_coordinates(f['sdf'],uv[:,::-1].T,order=1,mode='constant',cval=100.)
        explained=float((reverse<=2.).mean());outside=float((sdf[visible]>1.).mean()) if visible.any() else 1.
        ok=explained>=.8 and outside<=.1;passed=passed and ok
        audits[name]=dict(explained_fraction=explained,reverse_p90_px=float(np.quantile(reverse,.9)),visible_points=int(visible.sum()),outside_mask_fraction=outside,passed=bool(ok))
    alternatives=[dict(cost=c,parameters=q.tolist()) for c,q in choices]
    near=[q for c,q in choices if c<=cost+.2]
    spread=float(np.ptp(np.array(near)[:,:2],axis=0).max())
    passed=passed and spread<.002
    rot=Rotation.from_euler('z',x[2]).as_matrix();newverts=local.copy();newverts[:,2]*=x[3];newverts=newverts@rot.T;newverts[:,2]+=initial_info['table_z']-newverts[:,2].min()+.0001;newverts[:,:2]+=x[:2]
    planar_axes=rot[:2,:2]@np.asarray(initial_info['axes']);loc=newverts[:,:2]@planar_axes;lo,hi=loc.min(0),loc.max(0)
    top=float(newverts[:,2].max());half=np.r_[(hi-lo)/2,(top-initial_info['table_z'])/2]
    report=dict(valid=bool(passed),method='initial_RGB_shape_current_multiview_registration',step=int(obs['step']),shape_prior_only=True,current_pose_from_fresh_RGB=True,truth_inputs=False,parameters=x.tolist(),cost=cost,views=audits,alternatives=alternatives,near_solution_center_spread_m=spread,reason=None if passed else 'current_image_explanation_or_pose_ambiguity',extent_m=(half*2).tolist(),current_center_m=x[:2].tolist())
    if not passed:return None,report
    info=dict(initial_info,axes=planar_axes,center=np.r_[planar_axes@((lo+hi)/2),initial_info['table_z']+half[2]],half=half,top_z=top,collision_points=newverts)
    volume.points=p;volume.boundary=p;volume.tree=cKDTree(p);volume.center=p.mean(0);volume.last_frames=frames;volume.last_report=report
    return info,report
