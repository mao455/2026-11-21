"""Independent measured-state audit for the pinned solid cuboid experiment.

Intersect authored hand collision triangles with an eroded solid object box.
This is an interior intrusion test, NOT FCL triangle-contact penetration depth.
The erosion is a fixed contact tolerance. Cooked PhysX convex pieces may differ
from authored meshes; the scene manifest records that limitation explicitly.
"""
import numpy as np
import trimesh
import fcl
from scipy.spatial.transform import Rotation

TOLERANCE = 0.0005


def matrix(pose):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(np.asarray(pose)[[4, 5, 6, 3]]).as_matrix()
    out[:3, 3] = pose[:3]
    return out


def model(mesh):
    result = fcl.BVHModel()
    result.beginModel(len(mesh.vertices), len(mesh.faces))
    result.addSubModel(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
    result.endModel()
    return result


def intrudes(mesh, hand_transform, bounds, object_transform, tolerance=TOLERANCE):
    """Also detects complete containment in a closed hand mesh."""
    extents = bounds[1] - bounds[0] - 2 * tolerance
    if np.any(extents <= 0):
        raise ValueError('Tolerance erases object')
    center_world = object_transform[:3, :3] @ bounds.mean(axis=0) + object_transform[:3, 3]
    box = fcl.CollisionObject(fcl.Box(*extents), fcl.Transform(object_transform[:3, :3], center_world))
    hand = fcl.CollisionObject(model(mesh), fcl.Transform(hand_transform[:3, :3], hand_transform[:3, 3]))
    hit = bool(fcl.collide(hand, box, fcl.CollisionRequest(), fcl.CollisionResult()))
    if not hit and mesh.is_watertight:
        local_center = hand_transform[:3, :3].T @ (center_world - hand_transform[:3, 3])
        hit = bool(mesh.contains([local_center])[0])
    return hit


class Audit:
    def __init__(self, meshes, object_mesh):
        self.meshes = meshes
        self.bounds = object_mesh.bounds
        self.vertices = np.asarray(object_mesh.vertices)
        if not object_mesh.is_watertight or not np.isclose(object_mesh.volume, np.prod(object_mesh.extents), rtol=1e-5):
            raise ValueError('This evaluator requires the pinned axis-aligned solid cuboid')
        self.models = {name: model(mesh) for name, mesh in meshes.items()}

    def check(self, body_poses, object_pose):
        obj = matrix(object_pose)
        center = obj[:3, :3] @ self.bounds.mean(axis=0) + obj[:3, 3]
        half = (self.bounds[1] - self.bounds[0]) / 2
        world_half = np.abs(obj[:3, :3]) @ half
        box = fcl.CollisionObject(fcl.Box(*(2 * half - 2 * TOLERANCE)), fcl.Transform(obj[:3, :3], center))
        hits = []
        table_min = float('inf')
        for name, mesh in self.meshes.items():
            t = matrix(body_poses[name])
            verts = np.asarray(mesh.vertices) @ t[:3, :3].T + t[:3, 3]
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            table_min = min(table_min, float(lo[2]))
            if np.any(hi < center - world_half) or np.any(lo > center + world_half):
                continue
            hand = fcl.CollisionObject(self.models[name], fcl.Transform(t[:3, :3], t[:3, 3]))
            hit = bool(fcl.collide(hand, box, fcl.CollisionRequest(), fcl.CollisionResult()))
            if not hit and mesh.is_watertight:
                hit = bool(mesh.contains([t[:3, :3].T @ (center - t[:3, 3])])[0])
            if hit:
                hits.append(name)
        obj_vertices = self.vertices @ obj[:3, :3].T + obj[:3, 3]
        return {'hand_object_intrusion_links': hits, 'tolerance_m': TOLERANCE,
                'hand_table_min_z': table_min, 'object_min_z': float(obj_vertices[:, 2].min()),
                'passed': not hits and table_min >= -TOLERANCE and obj_vertices[:, 2].min() >= -TOLERANCE}


def usd_collision_meshes(stage, robot_path, body_names):
    """Read actual authored USD collisions in their simulator body frames."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    output, manifest = {}, []
    for name in body_names:
        body = stage.GetPrimAtPath(robot_path + '/' + name)
        if not body.IsValid():
            continue
        pieces = []
        body_world = np.asarray(cache.GetLocalToWorldTransform(body)).T
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            ancestor = prim
            collider = None
            while ancestor.IsValid() and ancestor != body:
                if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                    collider = ancestor
                    break
                ancestor = ancestor.GetParent()
            if collider is None or not UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get():
                continue
            if not prim.IsA(UsdGeom.Mesh):
                raise ValueError('Unsupported collision shape: ' + str(prim.GetPath()))
            usd = UsdGeom.Mesh(prim)
            counts = np.asarray(usd.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(usd.GetFaceVertexIndicesAttr().Get())
            faces, offset = [], 0
            for count in counts:
                poly = indices[offset:offset + count]
                faces.extend([[poly[0], poly[i], poly[i + 1]] for i in range(1, count - 1)])
                offset += count
            mesh = trimesh.Trimesh(np.asarray(usd.GetPointsAttr().Get()), faces, process=True)
            mesh.apply_transform(np.linalg.inv(body_world) @ np.asarray(cache.GetLocalToWorldTransform(prim)).T)
            pieces.append(mesh)
            approximation = UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
            manifest.append({'path': str(prim.GetPath()), 'approximation': str(approximation),
                             'vertices': len(mesh.vertices), 'faces': len(mesh.faces)})
        if pieces:
            output[name] = trimesh.util.concatenate(pieces)
    if not output:
        raise ValueError('No collision meshes discovered; fail closed')
    return output, manifest


if __name__ == '__main__':
    bounds = np.array([[-.01, -.01, -.01], [.01, .01, .01]])
    m = trimesh.creation.box(extents=[.01, .01, .01])
    t = np.eye(4)
    for x, expected in [(0.016, False), (.015, False), (.0148, False), (.014, True), (0., True)]:
        t[0, 3] = x
        assert intrudes(m, t, bounds, np.eye(4)) == expected, x
    assert intrudes(trimesh.creation.box(extents=[.1] * 3), np.eye(4), bounds, np.eye(4))
    print('PASS: separated, contact, tolerated overlap, penetration, both containment directions')
