"""Authoritative XHand3cam mounts and FingerEye fisheye calibration.

The XHand asset already contains the camera installation frames.  This module
parses those fixed joints from the runtime URDF and only adds a child USD
``Camera`` prim at identity.  No camera extrinsic is duplicated in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib.util
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def _load_fingereye_camera_cfg():
    """Load the workspace-level camera configuration supplied for production."""

    config_path = next(
        (
            parent / "fingereye_camera_cfg.py"
            for parent in Path(__file__).resolve().parents
            if (parent / "fingereye_camera_cfg.py").is_file()
        ),
        None,
    )
    if config_path is None:
        raise FileNotFoundError("Workspace camera configuration fingereye_camera_cfg.py was not found")
    module_name = "_fingereye_workspace_camera_cfg"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load FingerEye camera configuration: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.FingerEyeCameraCfg(), config_path.resolve()


FINGEREYE_CAMERA_CFG, FINGEREYE_CAMERA_CFG_PATH = _load_fingereye_camera_cfg()
CALIBRATION_STATUS = "configured_fingereye_fisheye_polynomial"
DISTORTION_MODEL = "fisheyePolynomial"
OPTICAL_AXIS_CONVENTION = "opengl_-z_forward_+y_up"
EXTRINSIC_SOURCE = "urdf_fixed_joint_imported_into_runtime_usd"
CAMERA_WARNING = "CAMERA_USING_WORKSPACE_FINGEREYE_FISHEYE_CONFIG"

SUPPLIER_HFOV_DEG = 117.8
SUPPLIER_VFOV_DEG = 92.3
SUPPLIER_DFOV_DEG = 147.8
NOMINAL_FPS = 30.0

LOGICAL_CAMERA_MOUNTS = {
    "sensor_1": ("front1", 640, 480),
    "sensor_2": ("front2", 640, 480),
    "sensor_3_back": ("back", 320, 240),
}
SUPPORTED_FINGERS = ("index", "middle", "ring", "thumb")
ALL_XHAND_CAMERA_NAMES = tuple(
    f"{finger}_{mount_name}"
    for finger in SUPPORTED_FINGERS
    for mount_name in ("front1", "front2", "back")
)


def _triplet(text: str | None, *, field: str, joint_name: str) -> tuple[float, float, float]:
    values = tuple(float(value) for value in str(text or "").split())
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid {field} on URDF joint {joint_name!r}: {text!r}")
    return values


def ideal_intrinsic_matrix(width: int, height: int) -> tuple[tuple[float, float, float], ...]:
    """Return the requested supplier-FOV ideal pinhole matrix.

    The supplier publishes independent horizontal and vertical FOV values, so
    ``fx`` and ``fy`` intentionally differ.  Omniverse currently renders
    square pixels; that renderer limitation is reported separately.
    """

    fx = float(width) / (2.0 * math.tan(math.radians(SUPPLIER_HFOV_DEG) / 2.0))
    fy = float(height) / (2.0 * math.tan(math.radians(SUPPLIER_VFOV_DEG) / 2.0))
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return ((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0))


def renderer_intrinsic_matrix(width: int, height: int) -> tuple[tuple[float, float, float], ...]:
    """Return the actual first-version Omniverse pinhole matrix.

    Isaac Sim reports square pixels from the spawned focal length and
    horizontal aperture. The supplier ideal FOV matrix is separate metadata.
    """

    fx = float(width) * float(FINGEREYE_CAMERA_CFG.focal_length) / float(FINGEREYE_CAMERA_CFG.horizontal_aperture)
    return (
        (fx, 0.0, float(width) / 2.0),
        (0.0, fx, float(height) / 2.0),
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True)
class XHandCameraSpec:
    logical_name: str
    finger: str
    mount_name: str
    joint_name: str
    parent_link: str
    camera_link: str
    translation: tuple[float, float, float]
    rpy: tuple[float, float, float]
    width: int
    height: int
    urdf_path: str

    @property
    def intrinsic_matrix(self) -> tuple[tuple[float, float, float], ...]:
        return ideal_intrinsic_matrix(self.width, self.height)

    @property
    def renderer_intrinsic(self) -> tuple[tuple[float, float, float], ...]:
        return renderer_intrinsic_matrix(self.width, self.height)

    @property
    def camera_prim_path(self) -> str:
        return f"/World/envs/env_.*/Robot/{self.camera_link}/Camera"

    def metadata(self) -> dict:
        renderer_fx = self.renderer_intrinsic[0][0]
        renderer_vfov = math.degrees(
            2.0 * math.atan(float(self.height) / (2.0 * renderer_fx))
        )
        return {
            "logical_name": self.logical_name,
            "finger_module": self.finger,
            "mount_name": self.mount_name,
            "joint_name": self.joint_name,
            "parent_link": self.parent_link,
            "camera_link": self.camera_link,
            "camera_prim_path_pattern": self.camera_prim_path,
            "width": int(self.width),
            "height": int(self.height),
            "nominal_fps": NOMINAL_FPS,
            "hfov_deg": SUPPLIER_HFOV_DEG,
            "vfov_deg": SUPPLIER_VFOV_DEG,
            "dfov_deg": SUPPLIER_DFOV_DEG,
            "fx": float(self.intrinsic_matrix[0][0]),
            "fy": float(self.intrinsic_matrix[1][1]),
            "cx": float(self.intrinsic_matrix[0][2]),
            "cy": float(self.intrinsic_matrix[1][2]),
            "skew": 0.0,
            "intrinsic_matrix": [list(row) for row in self.intrinsic_matrix],
            "pixel_center_convention": "cx=(W-1)/2, cy=(H-1)/2",
            "renderer_intrinsic_matrix": [list(row) for row in self.renderer_intrinsic],
            "renderer_intrinsic_limitation": (
                "Omniverse pinhole rendering uses square pixels and image center W/2,H/2; "
                "the supplier-FOV ideal K above remains the requested calibration metadata."
            ),
            "renderer_hfov_deg": math.degrees(2.0 * math.atan(float(self.width) / (2.0 * renderer_fx))),
            "renderer_vfov_deg": float(renderer_vfov),
            "distortion_model": DISTORTION_MODEL,
            "distortion_coefficients": None,
            "calibration_status": CALIBRATION_STATUS,
            "translation": list(self.translation),
            "rpy": list(self.rpy),
            "extrinsic_source": EXTRINSIC_SOURCE,
            "extrinsic_urdf_path": self.urdf_path,
            "optical_axis_convention": OPTICAL_AXIS_CONVENTION,
            "image_transform": "none",
            "enabled": True,
            "color": True,
            "global_shutter_supplier_spec": True,
            "hardware_output_format_supplier_spec": "MJPEG",
            "dynamic_range_db_supplier_spec": 60.0,
            "focus_mode_supplier_spec": "fixed",
            "nominal_focus_distance_m_supplier_spec": 0.02,
            "depth_of_field_m_supplier_spec": [0.016, 0.06],
            "synchronization_supplier_spec": "three_stream_hardware_sync",
            "usb_interface_supplier_spec": "USB 2.0 shared",
        }


def _fisheye_parameters(spec: XHandCameraSpec) -> dict:
    """Resolve the shared front calibration or the dedicated back calibration."""

    is_back = spec.mount_name == "back"
    polynomial = tuple(
        float(value)
        for value in (
            FINGEREYE_CAMERA_CFG.back_fisheye_polynomial
            if is_back
            else FINGEREYE_CAMERA_CFG.fisheye_polynomial
        )
    )
    if len(polynomial) != 6:
        raise ValueError(f"FingerEye fisheye polynomial must contain six values, got {polynomial}")
    nominal_width = float(
        FINGEREYE_CAMERA_CFG.back_fisheye_nominal_width
        if is_back
        else FINGEREYE_CAMERA_CFG.fisheye_nominal_width
    )
    nominal_height = float(
        FINGEREYE_CAMERA_CFG.back_fisheye_nominal_height
        if is_back
        else FINGEREYE_CAMERA_CFG.fisheye_nominal_height
    )
    optical_centre_x = float(
        FINGEREYE_CAMERA_CFG.back_fisheye_optical_centre_x
        if is_back
        else FINGEREYE_CAMERA_CFG.fisheye_optical_centre_x
    )
    optical_centre_y = float(
        FINGEREYE_CAMERA_CFG.back_fisheye_optical_centre_y
        if is_back
        else FINGEREYE_CAMERA_CFG.fisheye_optical_centre_y
    )
    return {
        "nominal_width": nominal_width,
        "nominal_height": nominal_height,
        "optical_centre_x": optical_centre_x,
        "optical_centre_y": optical_centre_y,
        "max_fov": float(FINGEREYE_CAMERA_CFG.fisheye_max_fov),
        "polynomial": polynomial,
    }


def _fisheye_spawn_cfg(sim_utils, spec: XHandCameraSpec):
    params = _fisheye_parameters(spec)
    polynomial = params["polynomial"]
    horizontal_aperture = float(FINGEREYE_CAMERA_CFG.horizontal_aperture)
    return sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        focal_length=float(FINGEREYE_CAMERA_CFG.focal_length),
        focus_distance=float(FINGEREYE_CAMERA_CFG.look_distance),
        f_stop=0.0,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=horizontal_aperture
        * float(params["nominal_height"])
        / float(params["nominal_width"]),
        clipping_range=(0.005, float(FINGEREYE_CAMERA_CFG.depth_max)),
        fisheye_nominal_width=float(params["nominal_width"]),
        fisheye_nominal_height=float(params["nominal_height"]),
        fisheye_optical_centre_x=float(params["optical_centre_x"]),
        fisheye_optical_centre_y=float(params["optical_centre_y"]),
        fisheye_max_fov=float(params["max_fov"]),
        fisheye_polynomial_a=polynomial[0],
        fisheye_polynomial_b=polynomial[1],
        fisheye_polynomial_c=polynomial[2],
        fisheye_polynomial_d=polynomial[3],
        fisheye_polynomial_e=polynomial[4],
        fisheye_polynomial_f=polynomial[5],
    )


def _fisheye_metadata(spec: XHandCameraSpec, update_period: float) -> dict:
    params = _fisheye_parameters(spec)
    metadata = spec.metadata()
    metadata.update(
        {
            "projection_type": DISTORTION_MODEL,
            "distortion_model": DISTORTION_MODEL,
            "distortion_coefficients": list(params["polynomial"]),
            "calibration_status": CALIBRATION_STATUS,
            "fisheye_nominal_width": float(params["nominal_width"]),
            "fisheye_nominal_height": float(params["nominal_height"]),
            "fisheye_optical_centre_x": float(params["optical_centre_x"]),
            "fisheye_optical_centre_y": float(params["optical_centre_y"]),
            "fisheye_max_fov": float(params["max_fov"]),
            "fisheye_polynomial": list(params["polynomial"]),
            "camera_geometry_version": str(FINGEREYE_CAMERA_CFG.camera_geometry_version),
            "camera_config_path": str(FINGEREYE_CAMERA_CFG_PATH),
            "sensor_update_period_s": float(update_period),
        }
    )
    return metadata


def load_xhand_camera_specs(urdf_path: str | Path, finger: str) -> dict[str, XHandCameraSpec]:
    """Parse one physical three-camera module from the authoritative URDF."""

    finger = str(finger).strip().lower()
    if finger not in SUPPORTED_FINGERS:
        raise ValueError(
            f"Unsupported XHand camera finger {finger!r}; choose one of {SUPPORTED_FINGERS}."
        )
    urdf_path = Path(urdf_path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    robot_name = str(root.attrib.get("name", ""))
    if not robot_name.startswith("xhand3camFE_left"):
        raise ValueError(
            f"Refusing non-XHand3cam camera URDF {urdf_path}: robot={robot_name!r}"
        )

    joints = {str(joint.attrib.get("name", "")): joint for joint in root.findall("joint")}
    specs: dict[str, XHandCameraSpec] = {}
    for logical_name, (mount_name, width, height) in LOGICAL_CAMERA_MOUNTS.items():
        joint_name = f"{finger}_fingereye_camera_{mount_name}_joint"
        joint = joints.get(joint_name)
        if joint is None:
            raise ValueError(f"Camera joint {joint_name!r} is missing from {urdf_path}")
        if str(joint.attrib.get("type")) != "fixed":
            raise ValueError(f"Camera joint {joint_name!r} must be fixed")
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        if parent is None or child is None or origin is None:
            raise ValueError(f"Camera joint {joint_name!r} is incomplete")
        camera_link = str(child.attrib.get("link", ""))
        expected_link = f"{finger}_fingereye_camera_{mount_name}"
        if camera_link != expected_link:
            raise ValueError(
                f"Camera joint {joint_name!r} child mismatch: {camera_link!r} != {expected_link!r}"
            )
        specs[logical_name] = XHandCameraSpec(
            logical_name=logical_name,
            finger=finger,
            mount_name=mount_name,
            joint_name=joint_name,
            parent_link=str(parent.attrib.get("link", "")),
            camera_link=camera_link,
            translation=_triplet(origin.attrib.get("xyz"), field="xyz", joint_name=joint_name),
            rpy=_triplet(origin.attrib.get("rpy"), field="rpy", joint_name=joint_name),
            width=int(width),
            height=int(height),
            urdf_path=str(urdf_path),
        )
    return specs


def make_xhand_tiled_camera_cfgs(
    urdf_path: str | Path,
    finger: str,
    *,
    update_period: float = 1.0 / NOMINAL_FPS,
) -> tuple[dict[str, object], dict[str, dict]]:
    """Build Isaac Lab sensors while keeping all extrinsics in the USD parent Xform."""

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    specs = load_xhand_camera_specs(urdf_path, finger)
    configs: dict[str, object] = {}
    metadata: dict[str, dict] = {}
    for logical_name, spec in specs.items():
        configs[logical_name] = TiledCameraCfg(
            prim_path=spec.camera_prim_path,
            update_period=float(update_period),
            spawn=_fisheye_spawn_cfg(sim_utils, spec),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="opengl",
            ),
            data_types=["rgb"],
            width=int(spec.width),
            height=int(spec.height),
            update_latest_camera_pose=True,
        )
        metadata[logical_name] = _fisheye_metadata(spec, update_period)
    return configs, metadata


def make_xhand_all_tiled_camera_cfgs(
    urdf_path: str | Path,
    *,
    update_period: float = 1.0 / NOMINAL_FPS,
    resolution: tuple[int, int] | None = None,
) -> tuple[dict[str, object], dict[str, dict]]:
    """Build all 12 physical cameras on thumb/index/middle/ring modules.

    ``resolution`` is an optional recording-time override.  The URDF still
    owns every camera extrinsic; only the raster size (and the matching
    aperture/intrinsic metadata) is changed.  Keeping this override here is
    important because ``configure_xhand_cameras`` recreates the tiled-camera
    configs after the generic legacy camera helper has run.
    """

    if resolution is not None:
        if len(resolution) != 2:
            raise ValueError(f"XHand camera resolution must be (width, height), got {resolution!r}")
        requested_width, requested_height = (int(resolution[0]), int(resolution[1]))
        if requested_width <= 0 or requested_height <= 0:
            raise ValueError("XHand camera resolution must be positive")
    else:
        requested_width = requested_height = None

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    configs: dict[str, object] = {}
    metadata: dict[str, dict] = {}
    for finger in SUPPORTED_FINGERS:
        for spec in load_xhand_camera_specs(urdf_path, finger).values():
            logical_name = f"{finger}_{spec.mount_name}"
            spec = replace(spec, logical_name=logical_name)
            if requested_width is not None:
                spec = replace(spec, width=requested_width, height=requested_height)
            configs[logical_name] = TiledCameraCfg(
                prim_path=spec.camera_prim_path,
                update_period=float(update_period),
                spawn=_fisheye_spawn_cfg(sim_utils, spec),
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    convention="opengl",
                ),
                data_types=["rgb"],
                width=int(spec.width),
                height=int(spec.height),
                update_latest_camera_pose=True,
            )
            metadata[logical_name] = _fisheye_metadata(spec, update_period)
    if tuple(configs) != ALL_XHAND_CAMERA_NAMES:
        raise RuntimeError(f"Unexpected XHand camera order: {tuple(configs)}")
    return configs, metadata
