"""Offline primitive identity across tessellations; never a policy input.

Returns unresolved for unsupported shapes: a non-match is not proof that two
arbitrary meshes differ. Native size and shape up to uniform scale are separate.
"""
import itertools
import numpy as np

def _axes(mesh):
    v=np.asarray(mesh.vertices);normals=np.asarray(mesh.face_normals)
    candidates=list(np.eye(3))+list(np.linalg.eigh(np.cov((v-v.mean(0)).T))[1].T)
    canonical=normals.copy()
    for row in canonical:
        nz=np.flatnonzero(abs(row)>1e-8)
        if len(nz) and row[nz[0]]<0:row*=-1
    unique,inverse=np.unique(np.round(canonical,6),axis=0,return_inverse=True)
    areas=np.bincount(inverse,weights=mesh.area_faces);candidates.extend(unique[np.argsort(areas)[-6:]])
    result=[]
    for axis in candidates:
        axis=axis/np.linalg.norm(axis)
        if not any(abs(axis@old)>1-1e-9 for old in result):result.append(axis)
    return result

def cylinder_signature(mesh,radial_tolerance_m=1e-7,normal_tolerance=1e-5,minimum_side_normals=20):
    """Fit circular sides and two flat end planes, independently of tessellation.

    Low-sided prisms remain prisms. Sagitta bounds the polygon surface's
    approximation to an analytic cylinder; no exact polygon identity is claimed.
    """
    if not mesh.is_watertight:return None
    vertices=np.asarray(mesh.vertices);fits=[]
    for axis in _axes(mesh):
        dots=np.abs(mesh.face_normals@axis);side=dots<normal_tolerance;caps=dots>1-normal_tolerance
        if not side.any() or not caps.any() or mesh.area_faces[~(side|caps)].sum()>mesh.area*1e-6:continue
        count=len(np.unique(np.round(mesh.face_normals[side],6),axis=0))
        if count<minimum_side_normals:continue
        x=np.cross(axis,np.eye(3)[np.argmin(abs(axis))]);x/=np.linalg.norm(x);y=np.cross(axis,x)
        basis=np.stack([x,y,axis],axis=1);local=vertices@basis;uv=local[np.unique(mesh.faces[side]),:2]
        sol=np.linalg.lstsq(np.column_stack((2*uv,np.ones(len(uv)))),(uv**2).sum(1),rcond=None)[0];center2=sol[:2];r2=sol[2]+center2@center2
        if r2<=0:continue
        radius=float(np.sqrt(r2));error=float(np.max(abs(np.linalg.norm(uv-center2,axis=1)-radius)))
        low,high=local[:,2].min(),local[:,2].max();length=float(high-low);z=local[np.unique(mesh.faces[caps]),2]
        cap_error=float(np.minimum(abs(z-low),abs(z-high)).max())
        if error>radial_tolerance_m or cap_error>radial_tolerance_m:continue
        angles=np.unique(np.round(np.mod(np.arctan2(uv[:,1]-center2[1],uv[:,0]-center2[0]),2*np.pi),10));gap=float(np.diff(np.r_[angles,angles[0]+2*np.pi]).max())
        fits.append(dict(kind='circular_cylinder',radius_m=radius,length_m=length,aspect_ratio=length/(2*radius),axis=axis.tolist(),basis=basis.tolist(),center=(basis@np.r_[center2,(low+high)/2]).tolist(),side_normal_count=count,angular_ring_vertices=len(angles),max_radius_fit_error_m=error,max_cap_plane_error_m=cap_error,fit_tolerance_m=radial_tolerance_m,max_angular_gap_rad=gap,analytic_surface_error_upper_bound_m=radius*(1-np.cos(gap/2))+error+cap_error))
    return min(fits,key=lambda f:f['max_radius_fit_error_m']+f['max_cap_plane_error_m']) if fits else None

def box_signature(mesh,plane_tolerance_m=1e-7,normal_tolerance=1e-5):
    """Fit six support planes in an orthogonal frame, including subdivided faces."""
    if not mesh.is_watertight:return None
    vertices=np.asarray(mesh.vertices);normals=np.asarray(mesh.face_normals)
    for triple in itertools.combinations(_axes(mesh),3):
        basis=np.stack(triple,axis=1)
        if np.max(abs(basis.T@basis-np.eye(3)))>normal_tolerance:continue
        dots=abs(normals@basis)
        if np.any(dots.max(1)<1-normal_tolerance):continue
        local=vertices@basis;low=local.min(0);high=local.max(0)
        if np.minimum(abs(local-low),abs(local-high)).min(1).max()>plane_tolerance_m:continue
        group=np.argmax(dots,axis=1);centers=mesh.triangles_center@basis
        if any(not (group==i).any() for i in range(3)):continue
        residual=max(float(np.minimum(abs(centers[group==i,i]-low[i]),abs(centers[group==i,i]-high[i])).max()) for i in range(3))
        if residual>plane_tolerance_m:continue
        dims=np.sort(high-low)
        return dict(kind='box',dimensions_sorted_m=dims.tolist(),normalized_dimensions=(dims/dims[-1]).tolist(),max_plane_error_m=residual,fit_tolerance_m=plane_tolerance_m)
    return None

def primitive_signature(mesh):return cylinder_signature(mesh) or box_signature(mesh)

def compare_signatures(a,b,relative_tolerance=1e-5,absolute_tolerance_m=1e-8):
    if not a or not b or a['kind']!=b['kind']:return dict(resolved=False,same_native_geometry=None,same_up_to_uniform_scale=None)
    if a['kind']=='circular_cylinder':
        av=np.array([a['radius_m'],a['length_m']]);bv=np.array([b['radius_m'],b['length_m']]);an=av/av[0];bn=bv/bv[0]
        delta=dict(radius_m=abs(float(av[0]-bv[0])),length_m=abs(float(av[1]-bv[1])),relative_aspect=abs(a['aspect_ratio']-b['aspect_ratio'])/max(a['aspect_ratio'],b['aspect_ratio']))
    else:
        av=np.array(a['dimensions_sorted_m']);bv=np.array(b['dimensions_sorted_m']);an=av/av[-1];bn=bv/bv[-1];delta=dict(dimensions_m=abs(av-bv).tolist(),normalized_dimensions=abs(an-bn).tolist())
    return dict(resolved=True,kind=a['kind'],same_native_geometry=bool(np.allclose(av,bv,rtol=relative_tolerance,atol=absolute_tolerance_m)),same_up_to_uniform_scale=bool(np.allclose(an,bn,rtol=relative_tolerance,atol=0)),relative_tolerance=relative_tolerance,absolute_tolerance_m=absolute_tolerance_m,differences=delta)

def compare_meshes(a,b,**kwargs):
    sa,sb=primitive_signature(a),primitive_signature(b)
    return dict(a=sa,b=sb,**compare_signatures(sa,sb,**kwargs))
