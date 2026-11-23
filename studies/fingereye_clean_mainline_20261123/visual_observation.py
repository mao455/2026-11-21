"""Strict execution-observation allowlist. Private diagnostics never cross it."""
from pathlib import Path
import numpy as np

CAMERAS = {f'{f}_{m}' for f in ('index','thumb') for m in ('front1','front2','back')}
OBS_KEYS = {'time_s','step','actual_q','actual_qvel','initial_overhead','fingereye'}
ROBOT_KEYS = {'urdf_path','joint_names','joint_limits','dt','table_z','action',
              'observation_profile','fingereye_calibration'}
FRAME_KEYS = {'rgb_path','width','height','capture_step','frame_id'}
TOP_KEYS = {'rgb_path','intrinsics','camera_pose_world','width','height','capture_step'}

def validate_response(response, public, initial=False):
    if set(response) != ({'ok','observation','robot_description'} if initial else {'ok','observation'}):
        raise ValueError('Response includes unapproved fields')
    obs=response['observation']; public=Path(public).resolve()
    if set(obs)!=OBS_KEYS:raise ValueError('Observation includes unapproved fields')
    if set(obs['fingereye'])!=CAMERAS:raise ValueError('FingerEye cameras missing or extra')
    for name in ('actual_q','actual_qvel'):
        q=np.asarray(obs[name],dtype=float)
        if q.shape!=(18,) or not np.isfinite(q).all():raise ValueError('Invalid joint state')
    top=obs['initial_overhead']
    if set(top)!=TOP_KEYS or top['capture_step']!=0:raise ValueError('Only initial overhead RGB allowed')
    for name,frame in [('overhead_initial',top),*obs['fingereye'].items()]:
        if name!='overhead_initial':
            if set(frame)!=FRAME_KEYS or frame['capture_step']!=obs['step']:
                raise ValueError('Invalid or stale FingerEye frame')
        path=Path(frame['rgb_path']).resolve()
        if path.parent!=public or path.suffix!='.npy':raise ValueError('Image outside public sensor directory')
        if name=='overhead_initial' and path.name!='overhead_initial.npy':
            raise ValueError('Overhead must be the fixed initial RGB')
        rgb=np.load(path,allow_pickle=False)
        if rgb.dtype!=np.uint8 or rgb.shape!=(frame['height'],frame['width'],3):
            raise ValueError('Only RGB uint8 images allowed')
    if initial:
        robot=response['robot_description']
        if set(robot)!=ROBOT_KEYS:raise ValueError('Unapproved robot-description fields')
        if set(robot['fingereye_calibration'])!=CAMERAS:raise ValueError('Calibration camera mismatch')
    allowed_prefixes=tuple(n+'_' for n in CAMERAS)
    for p in public.iterdir():
        if p.name in {'initial.json','overhead_initial.npy'}:continue
        if p.suffix!='.npy' or not p.name.startswith(allowed_prefixes):
            raise ValueError('Unapproved public file: '+p.name)

# Guard state belongs to one host runner, never to the submitted controller.
import hashlib
import json
import re

CALIBRATION_KEYS = set('logical_name finger_module mount_name joint_name parent_link camera_link camera_prim_path_pattern width height nominal_fps hfov_deg vfov_deg dfov_deg fx fy cx cy skew intrinsic_matrix pixel_center_convention renderer_intrinsic_matrix renderer_intrinsic_limitation renderer_hfov_deg renderer_vfov_deg distortion_model distortion_coefficients calibration_status translation rpy extrinsic_source extrinsic_urdf_path optical_axis_convention image_transform enabled color global_shutter_supplier_spec hardware_output_format_supplier_spec dynamic_range_db_supplier_spec focus_mode_supplier_spec nominal_focus_distance_m_supplier_spec depth_of_field_m_supplier_spec synchronization_supplier_spec usb_interface_supplier_spec projection_type fisheye_nominal_width fisheye_nominal_height fisheye_optical_centre_x fisheye_optical_centre_y fisheye_max_fov fisheye_polynomial camera_geometry_version camera_config_path sensor_update_period_s'.split())
_RGB_HEADERS={}
_base_validate_response=validate_response

def validate_response(response, public, initial=False):
    _base_validate_response(response,public,initial)
    if initial:
        for name,cal in response['robot_description']['fingereye_calibration'].items():
            if set(cal)!=CALIBRATION_KEYS:raise ValueError('Unapproved calibration fields')
            if cal['logical_name']!=name or cal['projection_type']!='fisheyePolynomial':
                raise ValueError('Unexpected camera calibration')
            for key,count in [('translation',3),('rpy',3),('fisheye_polynomial',6)]:
                a=np.asarray(cal[key],dtype=float)
                if a.shape!=(count,) or not np.isfinite(a).all():raise ValueError('Invalid calibration values')
    for name,frame in response['observation']['fingereye'].items():
        if Path(frame['rgb_path']).name!=f"{name}_{frame['frame_id']:05d}.npy":
            raise ValueError('Unexpected camera filename')
    # Even unreferenced public arrays must be RGB; cached stat/header validation
    # lives in ObservationGuard for normal execution.
    for p in Path(public).glob('*.npy'):
        if p.name=='overhead_initial.npy':continue
        if not any(re.fullmatch(re.escape(n)+r'_\d{5}\.npy',p.name) for n in CAMERAS):
            raise ValueError('Unexpected public array filename')
        stat=p.stat();signature=(stat.st_size,stat.st_mtime_ns)
        if _RGB_HEADERS.get(str(p))==signature:continue
        a=np.load(p,allow_pickle=False,mmap_mode='r')
        if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3:raise ValueError('Non-RGB public array')
        _RGB_HEADERS[str(p)]=signature


class ObservationGuard:
    """Freeze initial overhead and initial.json; reject any unissued public file."""
    def __init__(self, public):
        self.public=Path(public).resolve();self.top=None;self.top_hash=None
        self.ready=None;self.issued=set();self.last_step=-1;self.last_frame=-1

    def validate(self,response,initial=False):
        validate_response(response,self.public,initial)
        obs=response['observation'];top=json.dumps(obs['initial_overhead'],sort_keys=True)
        digest=hashlib.sha256(Path(obs['initial_overhead']['rgb_path']).read_bytes()).hexdigest()
        if self.top is not None and (top!=self.top or digest!=self.top_hash):
            raise ValueError('Initial overhead was modified')
        if obs['step']<self.last_step:raise ValueError('Joint/image time moved backwards')
        ids={v['frame_id'] for v in obs['fingereye'].values()}
        if len(ids)!=1:raise ValueError('FingerEye streams not synchronized')
        frame=next(iter(ids))
        if obs['step']>self.last_step and frame<=self.last_frame:raise ValueError('Reused frame at newer physics step')
        issued=self.issued|{Path(v['rgb_path']).name for v in obs['fingereye'].values()}
        allowed=issued|{'overhead_initial.npy','initial.json'}
        if {p.name for p in self.public.iterdir()}-allowed:raise ValueError('Unissued public file')
        initial_path=self.public/'initial.json'
        if self.ready is not None and initial_path.exists():
            if json.loads(initial_path.read_text())!=self.ready:raise ValueError('Initial protocol file changed')
        if initial:
            if self.ready is not None:raise ValueError('Repeated initialization')
            self.ready=json.loads(json.dumps(response))
        self.top,self.top_hash=top,digest
        self.issued,self.last_step,self.last_frame=issued,obs['step'],frame
