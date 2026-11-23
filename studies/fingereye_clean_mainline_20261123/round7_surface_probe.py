"""Private, read-only live scene-query diagnostics; never controller inputs.

Ray comparisons sample actual query surfaces. They are neither a complete
export of cooked convex geometry nor a test of friction/contact stability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _json_value(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return 'Infinity' if value > 0 else '-Infinity' if value < 0 else 'NaN'
    if value is None or isinstance(value, (str, bool, float, int, np.integer)):
        return value
    try:
        return [_json_value(v) for v in value]
    except TypeError:
        return str(value)


def runtime_attributes(env):
    """Resolved runtime USD values and authoring state, not private solver state."""
    wanted = {
        'physxArticulation:enabledSelfCollisions',
        'physxArticulation:solverPositionIterationCount',
        'physxArticulation:solverVelocityIterationCount',
        'physxRigidBody:linearDamping', 'physxRigidBody:angularDamping',
        'physxRigidBody:solverPositionIterationCount',
        'physxRigidBody:solverVelocityIterationCount',
        'physxRigidBody:disableGravity', 'physxRigidBody:maxAngularVelocity',
        'physxRigidBody:maxLinearVelocity', 'physxRigidBody:maxDepenetrationVelocity',
        'physxCollision:contactOffset', 'physxCollision:restOffset',
        'physics:collisionEnabled', 'physics:rigidBodyEnabled',
    }
    actors = []
    for prim in env.sim.stage.Traverse():
        path = str(prim.GetPath())
        if not (path.startswith('/World/envs/env_0/Robot') or path.startswith('/World/envs/env_0/Object')):
            continue
        attrs = {}
        for attr in prim.GetAttributes():
            if attr.GetName() in wanted or attr.GetName().startswith('physxConvexDecompositionCollision:'):
                attrs[attr.GetName()] = {'value': _json_value(attr.Get()),
                                         'authored': attr.HasAuthoredValueOpinion()}
        if attrs:
            actors.append({'path': path, 'schemas': list(prim.GetAppliedSchemas()), 'attributes': attrs})
    scene = env.sim._physics_context._physx_scene_api.GetPrim()
    scene_attrs = {}
    for attr in scene.GetAttributes():
        if attr.GetName().startswith(('physxScene:', 'physics:')):
            scene_attrs[attr.GetName()] = {'value': _json_value(attr.Get()),
                                           'authored': attr.HasAuthoredValueOpinion()}
    return {'scope': 'Read-only resolved runtime USD attributes; authored False may be schema fallback',
            'scene_path': str(scene.GetPath()), 'scene_attributes': scene_attrs,
            'actors': actors, 'private_solver_internals_read': False}


def _authored_hit(mesh, origin, direction, maximum):
    locations, _, triangle_ids = mesh.ray.intersects_location(
        [origin], [direction], multiple_hits=True)
    distances = (locations - origin) @ direction
    eligible = np.flatnonzero((distances >= 0) & (distances <= maximum))
    if not len(eligible):
        return None
    k = eligible[np.argmin(distances[eligible])]
    return {'distance': float(distances[k]), 'position_local': locations[k].tolist(),
            'normal_local': mesh.face_normals[triangle_ids[k]].tolist()}


def _rays(meshes, transforms):
    """Use physical face planes plus a grid; never use triangle area as quality."""
    import trimesh
    definitions = [
        ('left_hand_index_link2', 'left_hand_index_camera_component3_link',
         np.array([.014525, 0., .063]), np.array([1., 0., 0.])),
        ('left_hand_thumb_link3', 'left_hand_thumb_camera_component3_link',
         np.array([0., -.063, .014525]), np.array([0., 0., 1.])),
    ]
    rays = []
    for distal, name, anchor_distal, normal_distal in definitions:
        if name not in meshes or distal not in transforms:
            continue
        relative = np.linalg.inv(transforms[name]) @ transforms[distal]
        anchor = relative[:3, :3] @ anchor_distal + relative[:3, 3]
        normal = relative[:3, :3] @ normal_distal
        normal /= np.linalg.norm(normal)
        mesh = meshes[name]
        mask = ((mesh.face_normals @ normal) > .995) & (np.abs(mesh.triangles_center @ normal - anchor @ normal) < .0005)
        ids = np.flatnonzero(mask)
        if len(ids):
            adjacency = mesh.face_adjacency
            adjacency = adjacency[mask[adjacency].all(axis=1)]
            components = trimesh.graph.connected_components(adjacency, nodes=ids, min_len=1)
            selected = min(components, key=lambda indices: np.linalg.norm(mesh.triangles_center[indices] - anchor, axis=1).min())
            mask[:] = False
            mask[selected] = True
        patch = {'area_m2': float(mesh.area_faces[mask].sum()),
                 'criterion': 'nearest connected planar patch within 0.5mm, normal dot > .995',
                 'area_is_surface_integral_not_contact_quality': True}
        samples = [('anchor', anchor)]
        if patch['area_m2'] > 0:
            center = np.average(mesh.triangles_center[mask], weights=mesh.area_faces[mask], axis=0)
            patch['centroid_local'] = center.tolist()
            samples.append(('patch_centroid', center))
        for label, point in samples:
            rays.append((name, label, point + normal * .02, -normal, .05, patch))
        # Reproduce the original physical-face grid in each component's frame.
        for cross in [-.008, 0., .008]:
            for longitudinal in [.025, .035, .045, .055, .060, .063]:
                if 'index' in name:
                    origin, direction = np.array([.03, cross, longitudinal]), np.array([-1., 0., 0.])
                else:
                    origin, direction = np.array([cross, -longitudinal, .03]), np.array([0., 0., -1.])
                rays.append((name, f'grid_{cross}_{longitudinal}', origin, direction, .05, None))
    # Axial samples cover every physical collision body, including palm and
    # phalanges that can create unwanted hand self contacts. Sparse rays still
    # do not constitute a complete hull audit.
    for name, mesh in meshes.items():
        center = mesh.bounds.mean(axis=0)
        for axis in range(3):
            for sign in [-1, 1]:
                normal = np.zeros(3)
                normal[axis] = sign
                origin = center + normal * (mesh.extents[axis] / 2 + .015)
                rays.append((name, f'axis_{axis}_{sign}', origin, -normal, float(mesh.extents[axis] + .03), None))
    return rays


def surface_probes(env, meshes):
    from geometry import matrix
    from omni.physx import get_physx_scene_query_interface

    query = get_physx_scene_query_interface()
    poses = np.c_[_array(env.hand.data.body_pos_w[0]), _array(env.hand.data.body_quat_w[0])]
    transforms = {name: matrix(pose) for name, pose in zip(env.hand.body_names, poses)}
    results = []
    for name, label, origin, direction, maximum, patch in _rays(meshes, transforms):
        tx = transforms[name]
        hits = []

        def callback(hit):
            hits.append({'rigid_body': str(hit.rigid_body), 'collision': str(hit.collision),
                         'distance': float(hit.distance), 'position_world': list(hit.position),
                         'normal_world': list(hit.normal)})
            return True

        row = {'body': name, 'label': label, 'origin_local': origin.tolist(),
               'direction_local': direction.tolist(), 'length_m': maximum, 'planar_patch': patch}
        try:
            row['authored_hit'] = _authored_hit(meshes[name], origin, direction, maximum)
            query.raycast_all((tx[:3, :3] @ origin + tx[:3, 3]).tolist(),
                              (tx[:3, :3] @ direction).tolist(), maximum, callback, bothSides=True)
            same = [h for h in hits if h['rigid_body'].endswith('/' + name)]
            cooked = min(same, key=lambda h: h['distance']) if same else None
            row.update(all_hit_count=len(hits), same_body_hit_count=len(same),
                       other_hit_bodies=sorted({h['rigid_body'] for h in hits if h not in same}),
                       live_query_hit=cooked, distance_difference_m=None)
            if cooked is not None:
                cooked['position_local'] = (tx[:3, :3].T @ (np.asarray(cooked['position_world']) - tx[:3, 3])).tolist()
                cooked['normal_local'] = (tx[:3, :3].T @ np.asarray(cooked['normal_world'])).tolist()
            authored = row['authored_hit']
            if authored is not None and cooked is not None:
                row['distance_difference_m'] = cooked['distance'] - authored['distance']
                row['normal_difference_rad'] = float(np.arccos(np.clip(np.dot(cooked['normal_local'], authored['normal_local']), -1, 1)))
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        results.append(row)
    differences = [abs(r['distance_difference_m']) for r in results if r.get('distance_difference_m') is not None]
    return {'rays': results,
            'validity': {'total_rays': len(results),
                         'rays_with_any_hit': sum(r.get('all_hit_count', 0) > 0 for r in results),
                         'rays_with_same_body_hit': sum(r.get('same_body_hit_count', 0) > 0 for r in results),
                         'surface_comparisons': len(differences), 'error_rays': sum('error' in r for r in results),
                         'max_absolute_distance_difference_m': max(differences) if differences else None},
            'limitations': ['Zero query hits do not establish collider absence or equality',
                            'Only sampled ray surfaces, not complete cooked convex topology',
                            'Ray surfaces exclude contactOffset, restOffset and friction patch behavior',
                            'Matching ray surfaces do not validate grasp contact quality or hardware geometry']}


def save_probes(env, meshes, out, label, step):
    """Log read-only queries at an existing state; do not step or mutate scene."""
    try:
        report = surface_probes(env, meshes)
    except Exception as exc:
        report = {'error': f'{type(exc).__name__}: {exc}',
                  'validity': {'surface_comparisons': 0},
                  'interpretation': 'Probe failed; no conclusion about collider surfaces is available'}
    report.update(label=label, step=step, physics_steps_added=0, physics_mutations=False)
    with (Path(out) / 'live_surface_probes.jsonl').open('a') as stream:
        stream.write(json.dumps(report, allow_nan=False) + '\n')
