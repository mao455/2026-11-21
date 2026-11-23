"""Private convex-object extension; original box audit is called unchanged.

Same 0.5 mm inward surface offset and table tolerance as original. This is not
cooked-contact penetration. Concave objects deliberately raise, not convexify.
"""
import numpy as np
import trimesh
import fcl
from scipy.spatial import ConvexHull, HalfspaceIntersection
from geometry import Audit as BoxAudit, TOLERANCE, model, matrix

class Audit:
    def __init__(self, meshes, object_mesh):
        self.box = None
        if not object_mesh.is_watertight:
            raise ValueError('Object must be a watertight convex solid')
        if np.isclose(object_mesh.volume,np.prod(object_mesh.extents),rtol=1e-5):
            self.box=BoxAudit(meshes,object_mesh);return
        hull=ConvexHull(object_mesh.vertices)
        if not np.isclose(hull.volume,object_mesh.volume,rtol=1e-5):
            raise ValueError('Concave objects need a separately validated interior audit')
        planes=hull.equations.copy();planes[:,3]+=TOLERANCE
        interior=np.asarray(object_mesh.center_mass)
        if np.max(planes[:,:3]@interior+planes[:,3])>=0:
            raise ValueError('Erosion removed interior center')
        vertices=HalfspaceIntersection(planes,interior).intersections
        self.eroded=trimesh.convex.convex_hull(vertices)
        self.object_model=model(self.eroded);self.meshes=meshes
        self.models={n:model(m) for n,m in meshes.items()}
        self.vertices=np.asarray(object_mesh.vertices)
        self.center=interior
        self.planes=planes
    def check(self, body_poses, object_pose):
        if self.box is not None:return self.box.check(body_poses,object_pose)
        obj=matrix(object_pose);verts=self.vertices@obj[:3,:3].T+obj[:3,3]
        lower,upper=verts.min(0),verts.max(0)
        solid=fcl.CollisionObject(self.object_model,fcl.Transform(obj[:3,:3],obj[:3,3]))
        center=obj[:3,:3]@self.center+obj[:3,3];hits=[];table_min=float('inf')
        for name,mesh in self.meshes.items():
            t=matrix(body_poses[name]);v=mesh.vertices@t[:3,:3].T+t[:3,3]
            table_min=min(table_min,float(v[:,2].min()))
            if np.any(v.max(0)<lower) or np.any(v.min(0)>upper):continue
            hand=fcl.CollisionObject(self.models[name],fcl.Transform(t[:3,:3],t[:3,3]))
            hit=bool(fcl.collide(hand,solid,fcl.CollisionRequest(),fcl.CollisionResult()))
            if not hit and mesh.is_watertight:
                hit=bool(mesh.contains([t[:3,:3].T@(center-t[:3,3])])[0])
            # FCL BVH surfaces alone do not detect a hand fully inside the solid.
            if not hit:
                local=(v-obj[:3,3])@obj[:3,:3]
                hit=bool(np.any(np.all(local@self.planes[:,:3].T+self.planes[:,3]<0,axis=1)))
            if hit:hits.append(name)
        return {'hand_object_intrusion_links':hits,'tolerance_m':TOLERANCE,
                'hand_table_min_z':table_min,'object_min_z':float(verts[:,2].min()),
                'passed':not hits and table_min>=-TOLERANCE and verts[:,2].min()>=-TOLERANCE}
