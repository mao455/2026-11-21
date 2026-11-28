"""Read-only PhysX geometry, material, drive and per-contact observations."""

import json

from pathlib import Path

import numpy as np

import trimesh

from scipy.spatial import cKDTree

def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.asarray(value).copy()

class Contacts:
    def __init__(self, env):
        self.names = ['left_hand_index_camera_component3_link', 'left_hand_thumb_camera_component3_link', 'left_hand_index_link2']
        self.views = {name: env.hand._physics_sim_view.create_rigid_contact_view(
            '/World/envs/env_0/Robot/' + name, ['/World/envs/env_0/Object'], max_contact_data_count=256) for name in self.names}

    def read(self, dt):
        result = {}
        for name, view in self.views.items():
            forces, points, normals, separations, counts, starts = [array(x) for x in view.get_contact_data(dt)]
            count, start = int(counts.flat[0]), int(starts.flat[0])
            ids = slice(start, start + count)
            # Copy contact counts before querying friction: tensor buffers may be reused.
            contact = {'normal_forces_n': forces[ids].reshape(-1).tolist(), 'points_world': points[ids].tolist(),
                       'normals_world': normals[ids].tolist(), 'separations_m': separations[ids].reshape(-1).tolist(),
                       'normal_force_sum_n': float(forces[ids].sum()), 'count': count}
            try:
                friction, positions, cnt, st = [array(x) for x in view.get_friction_data(dt)]
                j, n = int(st.flat[0]), int(cnt.flat[0])
                contact.update(friction_forces_world=friction[j:j+n].tolist(), friction_points_world=positions[j:j+n].tolist())
            except Exception as exc:
                contact['friction_unavailable'] = str(exc)
            result[name] = contact
        return result

