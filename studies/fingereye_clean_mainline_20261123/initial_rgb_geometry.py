"""Explicit public RGB perception fixture; independent of robot calibration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import ConvexHull

SCHEMA = 'public_rgb_initial_geometry_v1'
PUBLIC_NAME = 'initial_rgb_geometry.json'
GEOMETRY_KEYS = ('center', 'half', 'axes', 'table_z', 'top_z', 'collision_points')
OVERHEAD_KEYS = ('intrinsics', 'camera_pose_world', 'width', 'height', 'capture_step')
PROVENANCE_KEYS = {'kind', 'source_run', 'source_step', 'source_note_sha256', 'initial_overhead',
    'exposures', 'static_camera_calibration_sha256', 'source_hand_description_sha256',
    'perception_controller_sha256', 'visual_hull'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def validate_reference(reference):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('initial_rgb_geometry_reference requires path and sha256')
    path = Path(reference['path'])
    digest = reference['sha256']
    if not path.is_absolute() or not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid public RGB geometry reference')
    return dict(path=str(path), sha256=digest)


def _finite_array(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('invalid public RGB geometry '+label)
    return array


def validate_payload(payload, table_z=None, overhead=None):
    if not isinstance(payload, dict) or set(payload) != {'schema', 'coordinate_frame', 'units', 'geometry', 'provenance'}:
        raise ValueError('unapproved public RGB geometry fields')
    if (payload['schema'], payload['coordinate_frame'], payload['units']) != (SCHEMA, 'world', 'm'):
        raise ValueError('public RGB geometry requires world coordinates in metres')
    geometry = payload['geometry']
    if not isinstance(geometry, dict) or set(geometry) != set(GEOMETRY_KEYS):
        raise ValueError('public RGB geometry requires the complete declared envelope')
    center = _finite_array(geometry['center'], (3,), 'center')
    half = _finite_array(geometry['half'], (3,), 'half')
    axes = _finite_array(geometry['axes'], (2, 2), 'axes')
    table = float(geometry['table_z']); top = float(geometry['top_z'])
    cloud = np.asarray(geometry['collision_points'], dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 4 or not np.isfinite(cloud).all():
        raise ValueError('invalid public RGB geometry collision_points')
    if not np.isfinite([table, top]).all() or np.any(half <= 0) or top <= table or not table <= center[2] <= top:
        raise ValueError('invalid public RGB geometry vertical extent')
    if not np.allclose(axes.T @ axes, np.eye(2), rtol=0., atol=1e-6):
        raise ValueError('invalid public RGB geometry planar axes')
    if table_z is not None and abs(table-float(table_z)) > 1e-6:
        raise ValueError('public RGB geometry world table differs from the task table')
    if cloud[:, 2].min() < table-1e-6:
        raise ValueError('public RGB geometry lies below its world table')
    try:
        hull = ConvexHull(cloud)
    except Exception as error:
        raise ValueError('public RGB geometry requires a finite three-dimensional hull') from error
    if not np.isfinite(hull.volume) or hull.volume <= 0:
        raise ValueError('public RGB geometry requires positive hull volume')
    provenance = payload['provenance']
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS or provenance.get('kind') != 'archived_public_rgb_perception':
        raise ValueError('public RGB geometry requires explicit observation-cache provenance')
    source = provenance.get('initial_overhead', {})
    metadata = source.get('metadata', {})
    if set(source) != {'rgb_sha256', 'metadata', 'metadata_sha256'} or set(metadata) != set(OVERHEAD_KEYS) or canonical_sha256(metadata) != source.get('metadata_sha256'):
        raise ValueError('invalid source overhead metadata provenance')
    if set(provenance.get('visual_hull', {})) != {'method', 'voxel_m', 'fused_frames', 'inflation_m', 'height_source'}:
        raise ValueError('unapproved source visual hull provenance fields')
    if not provenance.get('exposures') or not provenance.get('static_camera_calibration_sha256'):
        raise ValueError('public RGB geometry requires source exposures and static camera calibration hash')
    for exposure in provenance['exposures']:
        if set(exposure) != {'step', 'time_s', 'actual_q', 'actual_qvel', 'frames'}:
            raise ValueError('unapproved source public RGB exposure fields')
        q = np.asarray(exposure['actual_q'], float); qvel = np.asarray(exposure['actual_qvel'], float)
        if q.ndim != 1 or qvel.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(qvel).all() or not exposure.get('frames'):
            raise ValueError('invalid source public RGB exposure state')
        if any(frame['capture_step'] != exposure['step'] for frame in exposure['frames'].values()):
            raise ValueError('source RGB frame and exposure q are unpaired')
        if any(set(frame) != {'rgb_sha256', 'capture_step', 'width', 'height'} for frame in exposure['frames'].values()):
            raise ValueError('unapproved source public RGB frame fields')
    if overhead is not None:
        if sha256(overhead['rgb_path']) != source.get('rgb_sha256'):
            raise ValueError('public RGB geometry source overhead pixels differ')
        for key in OVERHEAD_KEYS:
            if not np.allclose(np.asarray(overhead[key]), np.asarray(metadata[key]), rtol=0., atol=1e-6):
                raise ValueError('public RGB geometry source overhead calibration differs: '+key)
    return payload


def publish_initial_rgb_geometry(case, public, overhead):
    reference = case.get('initial_rgb_geometry_reference')
    if reference is None:
        return None
    reference = validate_reference(reference)
    if sha256(reference['path']) != reference['sha256']:
        raise ValueError('public RGB geometry fixture hash changed')
    payload = json.loads(Path(reference['path']).read_text())
    validate_payload(payload, overhead=overhead)
    target = Path(public) / PUBLIC_NAME
    shutil.copyfile(reference['path'], target)
    return dict(path=str(target), sha256=reference['sha256'])


def read_public_geometry(declaration, public, table_z=None, overhead=None):
    declaration = validate_reference(declaration)
    path = Path(declaration['path']).resolve()
    if path.parent != Path(public).resolve() or path.name != PUBLIC_NAME:
        raise ValueError('RGB geometry fixture must be the declared public observation cache')
    if sha256(path) != declaration['sha256']:
        raise ValueError('public RGB geometry fixture hash changed')
    payload = json.loads(path.read_text())
    return deepcopy(validate_payload(payload, table_z, overhead))
