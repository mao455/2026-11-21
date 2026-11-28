from __future__ import annotations

import json
from pathlib import Path

from isaaclab.utils import configclass


@configclass
class FingerEyeCameraCfg:
    enabled: bool = False
    camera_names: tuple[str, ...] = ()
    width: int = 160
    height: int = 120
    focal_length: float = 1.0
    horizontal_aperture: float = 3.6
    fisheye_nominal_width: float = 640.0
    fisheye_nominal_height: float = 480.0
    fisheye_optical_centre_x: float = 320.0
    fisheye_optical_centre_y: float = 240.0
    fisheye_max_fov: float = 147.8
    fisheye_polynomial: tuple[float, ...] = (
        0.0,
        4.720752016383834e-3,
        -8.603927796940831e-6,
        1.215821809657514e-8,
        0.0,
        0.0,
    )
    back_fisheye_nominal_width: float = 320.0
    back_fisheye_nominal_height: float = 240.0
    back_fisheye_optical_centre_x: float = 160.0
    back_fisheye_optical_centre_y: float = 120.0
    back_fisheye_polynomial: tuple[float, ...] = (
        0.0,
        9.441504032767668e-3,
        -3.4415711187763324e-5,
        9.726574477260111e-8,
        0.0,
        0.0,
    )
    forward_offset: float = 0.0
    look_distance: float = 0.10
    depth_max: float = 0.45
    update_period: float = 0.0
    transparent_glass: bool = True
    camera_geometry_version: str = "xarm7_fisheye_polynomial_no_ray_offset_v1"


def load_fingereye_camera_cfg(path: str | Path) -> FingerEyeCameraCfg:
    """Load and validate one reusable FingerEye camera setup."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"FingerEye configuration must be a JSON object: {config_path}")
    values.pop("enabled", None)
    camera_cfg = FingerEyeCameraCfg()
    camera_cfg.from_dict(values)
    camera_cfg.camera_names = tuple(str(name) for name in camera_cfg.camera_names)
    if not camera_cfg.camera_names:
        raise ValueError(f"FingerEye configuration has no camera_names: {config_path}")
    if len(set(camera_cfg.camera_names)) != len(camera_cfg.camera_names):
        raise ValueError(f"FingerEye configuration contains duplicate camera_names: {config_path}")
    if camera_cfg.width <= 0 or camera_cfg.height <= 0:
        raise ValueError(
            f"FingerEye image size must be positive, got {camera_cfg.width}x{camera_cfg.height}."
        )
    return camera_cfg


def without_fingereye_camera_settings(values: dict, valid_fields: set[str]) -> dict:
    """Select environment fields while letting the run's camera JSON own camera settings."""
    return {
        name: value
        for name, value in values.items()
        if name in valid_fields and name != "fingereye_camera"
    }
