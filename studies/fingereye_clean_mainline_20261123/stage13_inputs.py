"""Host-only, hash-checked public RGB input for an exactly paired first view."""
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_KEYS = {'rgb_path', 'rgb_sha256', 'metadata_path', 'metadata_sha256'}


def validate_reference(reference):
    if not isinstance(reference, dict) or set(reference) != REFERENCE_KEYS:
        raise ValueError('initial_overhead_reference requires RGB and metadata paths and SHA256s')
    for key in ('rgb_path', 'metadata_path'):
        if not isinstance(reference[key], str) or not Path(reference[key]).is_absolute():
            raise ValueError('paired RGB reference paths must be absolute')
    for key in ('rgb_sha256', 'metadata_sha256'):
        value = reference[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid paired RGB reference SHA256')
    return dict(reference)


def paired_initial_rgb(case, live_rgb, live_metadata, public):
    """Use only a preacquired ordinary RGB plus its static camera calibration.

    The live capture is retained for audit. Neither this helper nor its public
    output reads simulator contacts or object pose.
    """
    live_rgb = np.asarray(live_rgb)
    reference = case.get('initial_overhead_reference')
    if reference is None:
        return live_rgb
    reference = validate_reference(reference)
    for prefix in ('rgb', 'metadata'):
        path = Path(reference[prefix + '_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference[prefix + '_sha256']:
            raise ValueError('paired initial RGB reference changed: ' + prefix)
    metadata = json.loads(Path(reference['metadata_path']).read_text())
    rgb = np.load(reference['rgb_path'], allow_pickle=False)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.shape != live_rgb.shape:
        raise ValueError('paired initial RGB shape/dtype differs from live camera')
    for key in ('width', 'height', 'capture_step'):
        if metadata[key] != live_metadata[key]:
            raise ValueError('paired initial RGB camera field differs: ' + key)
    for key in ('intrinsics', 'camera_pose_world'):
        if not np.allclose(metadata[key], live_metadata[key], rtol=0., atol=1e-6):
            raise ValueError('paired initial RGB static calibration differs: ' + key)
    public = Path(public)
    np.save(public.parent / 'overhead_live_initial.npy', live_rgb)
    audit = dict(reference=reference, live_pixels_sha256=hashlib.sha256(live_rgb.tobytes()).hexdigest(),
                 supplied_pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                 static_calibration_verified=True, capture_step=0,
                 input_kind='preacquired ordinary initial RGB; host-side pairing only')
    (public.parent / 'initial_overhead_input_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return rgb
