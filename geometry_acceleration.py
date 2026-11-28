"""Exact extrema acceleration; original contact/intrusion semantics retained.

Only linear extrema use convex-hull vertices. FCL triangles, containment and
full-vertex interior tests are unchanged. This module belongs to evaluator side.
"""
import time
import numpy as np
import fcl
from scipy.spatial import ConvexHull

def install(verify=False):
    import geometry
    import convex_geometry as convex
    old_box=geometry.Audit.check;old_convex=convex.Audit.check
    stats=dict(calls=0,optimized_s=0.,original_s=0.,mismatches=0,max_metric_error_m=0.,separating_plane_prunes=0)
    def extrema(mesh):
        vertices=np.asarray(mesh.vertices)
        try:return vertices[ConvexHull(vertices).vertices]
        except Exception:return vertices
    def prepare(self):
        if not hasattr(self,'_s14_hulls'):
            self._s14_hulls={n:extrema(m) for n,m in self.meshes.items()}
    def box_check(self,body_poses,object_pose):
        prepare(self);obj=geometry.matrix(object_pose);center=obj[:3,:3]@self.bounds.mean(0)+obj[:3,3];half=(self.bounds[1]-self.bounds[0])/2;world_half=abs(obj[:3,:3])@half
        box=fcl.CollisionObject(fcl.Box(*(2*half-2*geometry.TOLERANCE)),fcl.Transform(obj[:3,:3],center));hits=[];table_min=float('inf')
        for name,mesh in self.meshes.items():
            t=geometry.matrix(body_poses[name]);v=self._s14_hulls[name]@t[:3,:3].T+t[:3,3];lo,hi=v.min(0),v.max(0);table_min=min(table_min,float(lo[2]))
            if np.any(hi<center-world_half) or np.any(lo>center+world_half):continue
            hand=fcl.CollisionObject(self.models[name],fcl.Transform(t[:3,:3],t[:3,3]));hit=bool(fcl.collide(hand,box,fcl.CollisionRequest(),fcl.CollisionResult()))
            if not hit and mesh.is_watertight:hit=bool(mesh.contains([t[:3,:3].T@(center-t[:3,3])])[0])
            if hit:hits.append(name)
        verts=self.vertices@obj[:3,:3].T+obj[:3,3];z=float(verts[:,2].min())
        return dict(hand_object_intrusion_links=hits,tolerance_m=geometry.TOLERANCE,hand_table_min_z=table_min,object_min_z=z,passed=not hits and table_min>=-geometry.TOLERANCE and z>=-geometry.TOLERANCE)
    def convex_check(self,body_poses,object_pose):
        if self.box is not None:return box_check(self.box,body_poses,object_pose)
        prepare(self);obj=geometry.matrix(object_pose);verts=self.vertices@obj[:3,:3].T+obj[:3,3];lo,hi=verts.min(0),verts.max(0);solid=fcl.CollisionObject(self.object_model,fcl.Transform(obj[:3,:3],obj[:3,3]));center=obj[:3,:3]@self.center+obj[:3,3];hits=[];table_min=float('inf')
        for name,mesh in self.meshes.items():
            t=geometry.matrix(body_poses[name]);h=self._s14_hulls[name]@t[:3,:3].T+t[:3,3];table_min=min(table_min,float(h[:,2].min()))
            if np.any(h.max(0)<lo) or np.any(h.min(0)>hi):continue
            # A strict positive lower support value on any object plane proves
            # the entire hand convex hull is outside the eroded solid. The
            # margin keeps boundary/tangent FCL semantics on the original path.
            local_hull=(h-obj[:3,3])@obj[:3,:3]
            if np.any(np.min(local_hull@self.planes[:,:3].T+self.planes[:,3],axis=0)>1e-12):
                stats['separating_plane_prunes']+=1
                continue
            hand=fcl.CollisionObject(self.models[name],fcl.Transform(t[:3,:3],t[:3,3]));hit=bool(fcl.collide(hand,solid,fcl.CollisionRequest(),fcl.CollisionResult()))
            if not hit and mesh.is_watertight:hit=bool(mesh.contains([t[:3,:3].T@(center-t[:3,3])])[0])
            if not hit:
                v=np.asarray(mesh.vertices)@t[:3,:3].T+t[:3,3];local=(v-obj[:3,3])@obj[:3,:3]
                # Every point inside the eroded convex solid is also inside
                # the original object's bounds. Cull only outside vertices;
                # keep the original strict halfspace test for every survivor.
                # Padding makes this broad phase conservative at roundoff.
                bounds=np.asarray(self.vertices)
                eligible=np.all((local>=bounds.min(0)-1e-12)&(local<=bounds.max(0)+1e-12),axis=1)
                local=local[eligible]
                hit=bool(np.any(np.all(local@self.planes[:,:3].T+self.planes[:,3]<0,axis=1)))
            if hit:hits.append(name)
        z=float(verts[:,2].min());return dict(hand_object_intrusion_links=hits,tolerance_m=geometry.TOLERANCE,hand_table_min_z=table_min,object_min_z=z,passed=not hits and table_min>=-geometry.TOLERANCE and z>=-geometry.TOLERANCE)
    def wrap(fast,original):
        def checked(self,*args):
            start=time.perf_counter();value=fast(self,*args);stats['optimized_s']+=time.perf_counter()-start;stats['calls']+=1
            if verify:
                start=time.perf_counter();reference=original(self,*args);stats['original_s']+=time.perf_counter()-start
                err=max(abs(value[k]-reference[k]) for k in ('hand_table_min_z','object_min_z'));stats['max_metric_error_m']=max(stats['max_metric_error_m'],err)
                if any(value[k]!=reference[k] for k in ('hand_object_intrusion_links','passed','tolerance_m')) or err>1e-12:
                    stats['mismatches']+=1;raise RuntimeError('S14 strict geometry equivalence failure')
            return value
        return checked
    # Original convex check calls self.box.check; leave it original during paired
    # verification so each reference truly uses the unchanged full vertex path.
    convex.Audit.check=wrap(convex_check,old_convex)
    if not verify:geometry.Audit.check=box_check
    return stats
