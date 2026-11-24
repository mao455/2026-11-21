"""Formal H/S v2 two-finger strategy: call main(ctx) through run_stage17.py.

Maintained source, extracted from the physically tested Stage17 controller.
No controller concatenation or runtime algorithm overlays. Static hand paths are
replaced by the existing worker sandbox using ctx.robot_description.
See README.md for the public input contract and verified execution command.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.ndimage import label
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import cv2
import time
from scipy.ndimage import distance_transform_edt, binary_erosion, map_coordinates
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from scipy.ndimage import label as _s3_label
import functools as _s6_functools
from dataclasses import dataclass
try:
    from hand_description import PadCalibration, HandDescription
except ImportError:
    from mainline.hand_description import PadCalibration, HandDescription

def color_reliability(rgb):
    """Photometric validity only; dark/clipped pixels cannot certify free space."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    value = rgb.max(axis=-1)
    white_clipped = (value >= 254) & (rgb.min(axis=-1) >= 245)
    return (value > 8) & ~white_clipped

def color_foreground(rgb, hue=None, initial=False):
    """One H/S foreground rule for initial cue, mapping, retry and lift proof."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    reliable = color_reliability(rgb)
    foreground = (hsv[..., 1] > (65 if hue is None else 60)) & reliable
    if hue is not None:
        delta = abs(hsv[..., 0].astype(float) - hue)
        foreground &= np.minimum(delta, 180 - delta) < 12
    return foreground

def color_projection_known(frame, uv):
    """Sample trustworthy image domain; unknown pixels give no positive/negative vote."""
    if 'reliable' not in frame:
        return np.ones(len(uv), dtype=bool)
    return map_coordinates(frame['reliable'].astype(np.uint8), uv[:, ::-1].T, order=0, mode='constant', cval=0).astype(bool)
URDF = '/home/CNS2025331827/docs/grasp_small_obj/FingerEyeLab/assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf'
MIDDLE_CONTACT = dict(link='left_hand_middle_link2', point=[0.01175, 0.0, 0.047], normal=[1.0, 0.0, 0.0], source='static identity-fixed component3 front face; no measured material or drive calibration')
STAGE3_RELEASE_TICKET = None
S5_LOCAL_RECOVERY = False
S6_UPDATE_MM = 0.0008
S5_AXIS = 'auto'
S5_CLOSURE = 'servo'
S5_CAP = 0.0045
STAGE3_REFINEMENT_ATTITUDE_LIMIT_RAD = 0.12
MODES = ('simultaneous', 'thumb_first', 'index_first', 'release_reload')
S6_FAST = True
STAGE3_RECOVERY_VERSION = 'recovery_v4_map_consistency'
STAGE3_ATTEMPT = 1
STAGE3_RECOVERY_ATTITUDE_LIMIT_RAD = 0.02
_S6_IK_CACHE = {}
NAMES = ['base_x', 'base_y', 'base_z', 'base_yaw', 'base_pitch', 'base_roll', 'left_hand_thumb_joint1', 'left_hand_thumb_joint2', 'left_hand_thumb_joint3', 'left_hand_index_spread_joint', 'left_hand_index_joint1', 'left_hand_index_joint2', 'left_hand_middle_joint1', 'left_hand_middle_joint2', 'left_hand_ring_joint1', 'left_hand_ring_joint2', 'left_hand_pinky_joint1', 'left_hand_pinky_joint2']
STAGE2_DELTA_M = 0.00025
STAGE3_SUPPORT_CAMERAS = ('index_front1', 'index_front2', 'index_back', 'thumb_front1', 'thumb_front2', 'thumb_back')
STAGE3_SUPPORT_VERSION = 'support_v3'
STAGE3_TOTAL_STEPS = 24000
STAGE3_RESERVE_ATTEMPT_STEPS = 6000
CAMERAS = ('index_front1', 'index_front2', 'index_back', 'thumb_front1', 'thumb_front2', 'thumb_back')

class Robot:
    """Planning geometry and actuation use the runner's single hand model."""
    def __init__(self, urdf=None):
        data=_ACTIVE_DESCRIPTION
        self.urdf_path=data['urdf_path']
        self.names=tuple(data['joint_names']);self.palm_link=data['palm_link']
        self.description=HandDescription(self.urdf_path,joint_names=self.names,
            cameras=data['fingereye_calibration'],base_joint_names=data['base_joint_names'])
        self.joints=self.description.joints
        self.lo=self.description.lower.copy();self.hi=self.description.upper.copy()
        self.base_indices=np.array([self.description.index[n] for n in data['base_joint_names']],int)
        self.all_finger_indices=np.array([i for i,n in enumerate(self.names) if n not in data['base_joint_names']],int)
        self.catalog={row['finger']:row for row in data['pads']}
        if len(self.catalog)<2:
            raise RuntimeError('Selected URDF has fewer than two mesh-supported contact patches: '+str(list(self.catalog)))
        self.surfaces={};self.meshes=[]
        for entry in self.description.collision_meshes:
            mesh=trimesh.load(entry['path'],force='mesh',process=True)
            mesh.apply_scale(entry['scale']);mesh.apply_transform(entry['origin'])
            name=entry['link']
            self.surfaces[name]=trimesh.util.concatenate([self.surfaces[name],mesh]) if name in self.surfaces else mesh
        for name,mesh in self.surfaces.items():
            vertices=np.unique(np.asarray(mesh.vertices),axis=0)
            try:vertices=vertices[ConvexHull(vertices).vertices]
            except Exception:pass
            self.meshes.append((name,vertices))
        self.select_pair(tuple(self.catalog)[:2])

    def select_pair(self,pair,region_indices=None):
        self.finger_names=tuple(pair)
        rows=[self.catalog[name] for name in pair]
        self.contact_links=tuple(r['link'] for r in rows)
        self.contact_points=np.array([r['point_m'] for r in rows])
        self.contact_normals=np.array([r['outward_normal'] for r in rows])
        self.contact_distals=np.array([r['distal_direction'] for r in rows])
        self.active_finger_indices=np.array(sorted({self.description.index[n] for r in rows for n in r['joint_names']}),int)
        self.unused_finger_indices=np.array(sorted(set(self.all_finger_indices)-set(self.active_finger_indices)),int)
        self.patch_options=[[dict(r,point=np.array(r['point']),normal=np.array(r['normal'])) for r in row['regions']] for row in rows]
        default=[int(np.argmin([np.linalg.norm(r['point']-row['point_m']) for r in options])) for row,options in zip(rows,self.patch_options)]
        self.select_regions(default if region_indices is None else region_indices)

    def select_regions(self,indices):
        self.contact_region_indices=tuple(indices)
        selected=[options[i] for options,i in zip(self.patch_options,indices)]
        self.contact_points=np.array([r['point'] for r in selected])
        self.contact_normals=np.array([r['normal'] for r in selected])
        self.description.pads=tuple(PadCalibration(name,link,tuple(point),tuple(normal),tuple(self.catalog[name]['joint_names']),self.catalog[name]['support_radius_m']) for name,link,point,normal in zip(self.finger_names,self.contact_links,self.contact_points,self.contact_normals))

    def configure_contacts(self,count):
        if count!=2:raise ValueError('This frozen task uses two contact pads')

    def fk(self,q):
        return self.description.fk(q)

    def tips(self,q):
        poses=self.fk(q)
        points=np.array([poses[n][:3,:3]@p+poses[n][:3,3] for n,p in zip(self.contact_links,self.contact_points)])
        normals=np.array([poses[n][:3,:3]@p for n,p in zip(self.contact_links,self.contact_normals)])
        distal=np.array([poses[n][:3,:3]@p for n,p in zip(self.contact_links,self.contact_distals)])
        return points,normals,distal,poses

    def vertices(self,poses):
        return np.concatenate([v@poses[n][:3,:3].T+poses[n][:3,3] for n,v in self.meshes])

def _S3_ESTIMATE(observation):
    info = _R13_ESTIMATE(observation)
    cam = observation['initial_overhead']
    rgb = np.load(cam['rgb_path']).astype(float) / 255
    mask = color_foreground(np.uint8(np.rint(rgb * 255)), initial=True)
    yy, xx = np.indices(mask.shape)
    mask &= (abs(xx - mask.shape[1] / 2) < mask.shape[1] * 0.25) & (abs(yy - mask.shape[0] / 2) < mask.shape[0] * 0.3)
    components, count = label(mask)
    pixels = max([np.argwhere(components == k) for k in range(1, count + 1)], key=len)
    uv = pixels[:, [1, 0]].astype(float) + 0.5
    K = np.asarray(cam['intrinsics'])
    pose = np.asarray(cam['camera_pose_world'])
    rot = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    rays = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] @ rot.T
    xy = (pose[:3] + rays * (-pose[2] / rays[:, 2])[:, None])[:, :2]
    xy = xy[ConvexHull(xy).vertices]
    pix = info['pixel_m'] * 0.5
    xy = np.concatenate([xy + [a, b] for a in [-pix, pix] for b in [-pix, pix]])
    xy = xy[ConvexHull(xy).vertices]
    info['collision_points'] = np.r_[np.c_[xy, np.zeros(len(xy))], np.c_[xy, np.full(len(xy), info['top_z'])]]
    return info

class _S3_VOLUME:

    def __init__(self, robot, calibration, info, resolution=0.001):
        self.robot, self.calibration, self.info = (robot, calibration, info)
        self.resolution = resolution
        self.visibility = RobotVisibility(robot.urdf_path)
        color = np.uint8(np.asarray(info['color']) * 255).reshape(1, 1, 3)
        self.hue = float(cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0, 0, 0])
        self.lookups, self.rays = ({}, {})
        self.size = (80, 60)
        for name, c in calibration.items():
            rho = np.linspace(0, np.hypot(c['fisheye_nominal_width'], c['fisheye_nominal_height']), 4096)
            theta = np.polynomial.polynomial.polyval(rho, c['fisheye_polynomial'])
            valid = np.r_[True, np.diff(theta) > 0]
            self.lookups[name] = (theta[valid], rho[valid])
            yy, xx = np.indices((self.size[1], self.size[0]))
            xy = np.c_[(xx.ravel() + 0.5) * c['fisheye_nominal_width'] / self.size[0] - c['fisheye_optical_centre_x'], (yy.ravel() + 0.5) * c['fisheye_nominal_height'] / self.size[1] - c['fisheye_optical_centre_y']]
            radius = np.linalg.norm(xy, axis=1)
            angle = np.polynomial.polynomial.polyval(radius, c['fisheye_polynomial'])
            self.rays[name] = np.c_[np.sin(angle)[:, None] * xy / np.maximum(radius[:, None], 1e-10), -np.cos(angle)]
            self.rays[name][:, 1] *= -1
        radius = max(info['half'][:2]) + 0.008
        self.lower = np.r_[np.asarray(info['center'])[:2] - radius, info['table_z'] + 0.0005]
        self.axes = [np.arange(self.lower[j], self.lower[j] + 2 * radius + resolution / 2, resolution) for j in range(2)]
        self.axes.append(np.arange(self.lower[2], info['table_z'] + 0.045, resolution))
        self.shape = tuple((len(a) for a in self.axes))
        self.grid = np.stack(np.meshgrid(*self.axes, indexing='ij'), -1).reshape(-1, 3)
        self.points = None
        self.boundary = None
        self.reference_points = None
        self.state = None
        self.started = False
        self.last_frames = None

    def project(self, name, world, camera):
        c = self.calibration[name]
        local = (world - camera[:3, 3]) @ camera[:3, :3]
        radius = np.linalg.norm(local[:, :2], axis=1)
        angle = np.arctan2(radius, -local[:, 2])
        rho = np.interp(angle, *self.lookups[name])
        uv = local[:, :2] / np.maximum(radius[:, None], 1e-12) * rho[:, None]
        uv[:, 1] *= -1
        uv += [c['fisheye_optical_centre_x'], c['fisheye_optical_centre_y']]
        uv *= [self.size[0] / c['fisheye_nominal_width'], self.size[1] / c['fisheye_nominal_height']]
        valid = (angle < np.deg2rad(c['fisheye_max_fov'] / 2)) & (uv[:, 0] >= 0) & (uv[:, 0] < self.size[0]) & (uv[:, 1] >= 0) & (uv[:, 1] < self.size[1])
        return (uv - 0.5, np.linalg.norm(local, axis=1), valid)

    def frames(self, obs):
        poses = self.robot.fk(np.asarray(obs['actual_q']))
        cameras, frames = ({}, {})
        for name, f in obs['fingereye'].items():
            rgb = np.load(f['rgb_path'], allow_pickle=False)
            rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            d = abs(hsv[:, :, 0].astype(float) - self.hue)
            mask = color_foreground(rgb, self.hue).astype('uint8')
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            if count < 2 or max(stats[1:, 4]) < 5:
                continue
            mask = labels == 1 + np.argmax(stats[1:, 4])
            camera = poses[self.calibration[name]['camera_link']]
            cameras[name] = (camera, self.rays[name])
            sdf = distance_transform_edt(~mask) - distance_transform_edt(mask)
            frames[name] = {'mask': mask, 'reliable': color_reliability(rgb), 'sdf': sdf, 'camera': camera}
        depths = self.visibility.depth(poses, cameras)
        for name, frame in frames.items():
            frame['robot_depth'] = depths[name].reshape(self.size[::-1])
        return (poses, frames)

    def carve(self, obs, fuse=False):
        start = time.perf_counter()
        poses, frames = self.frames(obs)
        support = np.zeros(len(self.grid), dtype=int)
        free = np.zeros(len(self.grid), dtype=int)
        for name, frame in frames.items():
            uv, dist, valid = self.project(name, self.grid, frame['camera'])
            sdf = map_coordinates(frame['sdf'], uv[:, ::-1].T, order=1, mode='constant', cval=100.0)
            depth = map_coordinates(frame['robot_depth'], uv[:, ::-1].T, order=0, mode='constant', cval=0.0)
            visible = valid & (dist < depth - 0.0005) & color_projection_known(frame, uv)
            support += visible & (sdf <= 1.0)
            free += visible & (sdf > 1.0)
        top = obs['initial_overhead']
        rgb = np.load(top['rgb_path'], allow_pickle=False)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        delta = abs(hsv[:, :, 0].astype(float) - self.hue)
        mask = color_foreground(rgb, self.hue)
        sdf_top = distance_transform_edt(~mask) - distance_transform_edt(mask)
        pose = np.asarray(top['camera_pose_world'])
        R = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        local = (self.grid - pose[:3]) @ R
        K = np.asarray(top['intrinsics'])
        uv = local[:, :2] / np.maximum(local[:, 2:3], 1e-09) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]] - 0.5
        overhead = map_coordinates(sdf_top, uv[:, ::-1].T, order=1, mode='constant', cval=100.0) <= 0.5
        keep = (support >= 2) & (free == 0) & overhead
        single_center = self.grid[keep].mean(0) if keep.any() else None
        if fuse:
            if not hasattr(self, 'map_support'):
                self.map_support = np.zeros(len(keep), dtype=bool)
                self.map_free = np.zeros(len(keep), dtype=bool)
                self.map_frames = 0
            self.map_support |= support >= 2
            self.map_free |= free > 0
            self.map_frames += 1
            keep = self.map_support & ~self.map_free & overhead
        volume = keep.reshape(self.shape)
        points = self.grid[keep]
        report = {'method': 'occlusion_aware_rgb_visual_hull', 'step': obs['step'], 'usable_views': len(frames), 'voxel_m': self.resolution, 'occupied_voxels': int(keep.sum()), 'unknown_voxels': int(((support < 2) & (free == 0) & overhead).sum()), 'fused_frames': getattr(self, 'map_frames', 1), 'single_frame_center_world_m': None if single_center is None else single_center.tolist(), 'valid': False, 'compute_s': time.perf_counter() - start}
        if len(points) < 25:
            report['reason'] = 'insufficient_multiview_support'
            return report
        boundary = volume & ~binary_erosion(volume)
        truncated = bool(volume[:, :, -1].any() or volume[0].any() or volume[-1].any() or volume[:, 0].any() or volume[:, -1].any())
        self.points, self.boundary = (points, self.grid[boundary.ravel()])
        self.tree = cKDTree(self.boundary)
        self.center = points.mean(0)
        self.last_frames = frames
        report.update(valid=not truncated and len(frames) >= 3, truncated=truncated, center_world_m=self.center.tolist(), bounds_world_m=[points.min(0).tolist(), points.max(0).tolist()], volume_mm3=float(len(points) * (1000 * self.resolution) ** 3), compute_s=time.perf_counter() - start)
        self.last_report = report
        return report

    def support(self, q):
        """Closest full component3 surface to RGB boundary, never contact truth."""
        poses = self.robot.fk(q)
        answers = []
        for digit in self.robot.finger_names:
            name = self.robot.catalog[digit]['surface_link']
            mesh = self.robot.surfaces[name]
            if not hasattr(self, 'surface_samples'):
                self.surface_samples = {}
            if name not in self.surface_samples:
                import trimesh
                tri = np.asarray(mesh.triangles)
                pts = np.concatenate([tri[:, 0] * a + tri[:, 1] * b + tri[:, 2] * (1 - a - b) for a, b in ((0, 0), (1, 0), (0, 1), (0.5, 0.5), (0.5, 0), (0, 0.5), (1 / 3, 1 / 3))])
                self.surface_samples[name] = np.unique(np.round(pts, 7), axis=0)
            local = self.surface_samples[name]
            pose = poses[name]
            world = local @ pose[:3, :3].T + pose[:3, 3]
            distances, ids = self.tree.query(world)
            minimum = float(distances.min())
            near = distances <= minimum + 0.0003
            weights = np.exp(-(distances[near] - minimum) / 0.00015)
            p = np.average(world[near], axis=0, weights=weights)
            target = np.average(self.boundary[ids[near]], axis=0, weights=weights)
            answers.append({'digit': digit, 'shell_point_world_m': p.tolist(), 'volume_point_world_m': target.tolist(), 'gap_m': minimum, 'shell_point_local_m': (pose[:3, :3].T @ (p - pose[:3, 3])).tolist(), 'near_samples': int(near.sum())})
        return answers

    def begin_lift(self):
        self.started = True
        self.reference_points = None

    def update(self, obs):
        """Track a frozen coarse point volume by multiview contour registration.

        Both directions of image distance prevent shrink-to-interior solutions.
        Robot-occluded model points are unknown, not background violations.
        """
        started = time.perf_counter()
        poses, frames = self.frames(obs)
        palm = poses[self.robot.palm_link]
        result = {'valid': False, 'views': {n: {'visible': True, 'pixels': int(f['mask'].sum())} for n, f in frames.items()}, 'usable_views': len(frames), 'method': 'rgb_point_volume_registration', 'relative_center_m': None, 'translation_uncertainty_m': None, 'orientation_palm_rotvec': None}
        if self.points is None or len(frames) < 3:
            return result
        if self.reference_points is None:
            points = self.points[::max(1, len(self.points) // 2200)]
            self.reference_points = points - self.center
            self.state = np.r_[palm[:3, :3].T @ (self.center - palm[:3, 3]), Rotation.from_matrix(palm[:3, :3].T).as_rotvec()]
        base = self.state.copy()
        masks = {}
        for name, f in frames.items():
            pixels = np.argwhere(f['mask'])[:, ::-1].astype(float)
            masks[name] = pixels[::max(1, len(pixels) // 80)][:80]

        def residual(x):
            pp = self.reference_points @ Rotation.from_rotvec(x[3:]).as_matrix().T + x[:3]
            world = pp @ palm[:3, :3].T + palm[:3, 3]
            residuals = []
            for name, f in frames.items():
                uv, dist, valid = self.project(name, world, f['camera'])
                depth = map_coordinates(f['robot_depth'], uv[:, ::-1].T, order=0, mode='constant', cval=0.0)
                visible = valid & (dist <= depth + 0.0005)
                sdf = map_coordinates(f['sdf'], uv[:, ::-1].T, order=1, mode='constant', cval=15.0)
                residuals.append(np.where(visible, np.maximum(0, sdf), 0) / np.sqrt(len(uv)))
                if visible.sum() >= 3:
                    distances, _ = cKDTree(uv[visible]).query(masks[name])
                    residuals.append(distances / np.sqrt(len(distances)))
                else:
                    residuals.append(np.full(len(masks[name]), 15.0 / np.sqrt(len(masks[name]))))
            return np.concatenate(residuals)
        lo = base - [0.012, 0.012, 0.012, 0.4, 0.4, 0.4]
        hi = base + [0.012, 0.012, 0.012, 0.4, 0.4, 0.4]
        fit = least_squares(residual, base, bounds=(lo, hi), diff_step=0.001, x_scale=[0.01, 0.01, 0.01, 0.2, 0.2, 0.2], max_nfev=18, loss='soft_l1', f_scale=0.3)
        error = float(np.linalg.norm(residual(fit.x)) / np.sqrt(len(frames)))
        self.state = fit.x
        translation_jac = fit.jac[:, :3]
        rotation_jac = fit.jac[:, 3:]
        marginal = translation_jac - rotation_jac @ np.linalg.lstsq(rotation_jac, translation_jac, rcond=1e-08)[0]
        eigen = np.linalg.eigvalsh(marginal.T @ marginal)
        uncertainty = 1.0 / np.sqrt(max(eigen[0], 1e-12))
        result.update(valid=bool(error < 3.0 and uncertainty < 0.006), relative_center_m=fit.x[:3].tolist(), orientation_palm_rotvec=fit.x[3:].tolist(), translation_uncertainty_m=float(uncertainty), image_distance_px=error, minimum_information=float(eigen[0]), compute_s=time.perf_counter() - started)
        return result

def initial_hue(rgb):
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    mask = color_foreground(rgb)
    yy, xx = np.indices((h, w))
    mask &= (abs(xx - w / 2) < w * 0.25) & (abs(yy - h / 2) < h * 0.3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n < 2:
        raise ValueError('No initial color foreground')
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return float(np.median(hsv[:, :, 0][labels == k]))

def _s3_steps(delta, minimum=120, translation=0.06, joint=0.8):
    delta = np.abs(np.asarray(delta))
    seconds = 1.875 * max(float(delta[:3].max()) / translation, float(delta[3:].max()) / joint)
    return int(max(minimum, np.ceil(seconds * 120 / 24) * 24))

def stable_joint_response(actual_qvel, hand, threshold_rad_s=0.06):
    """V6 settling predicate parameterized by semantic finger groups, not slices."""
    indices = sorted({i for group in hand.finger_joint_indices.values() for i in group})
    if not indices:
        raise ValueError('No calibrated active finger group')
    velocity = np.asarray(actual_qvel, dtype=float)
    if velocity.shape != (len(hand.joint_names),):
        raise ValueError('Joint velocity does not match hand descriptor')
    maximum = float(np.max(np.abs(velocity[indices])))
    return (maximum < threshold_rad_s, maximum)

def smooth_segment(start, end, steps, phase):
    assert steps % 6 == 0
    return [[(start + (end - start) * (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)).tolist(), 6, phase] for t in np.arange(1, steps // 6 + 1) / (steps // 6)]

def capture_preload_snapshot(observation, robot, static):
    """Capture all calibrated/live view slots, raw actual q/qvel and robot FK.

    static is RGBVolume or {calibration, color, optional finger_joint_indices,
    wrist_joint_indices, depth_bounds_m}. color must come from initial RGB.
    Unknown-depth bounds are assumptions, explicitly exposed in the evidence.
    """
    settings = _s2v_settings(static)
    calibration = settings['calibration']
    actual = np.asarray(observation['actual_q'], dtype=float)
    velocity = np.asarray(observation.get('actual_qvel', []), dtype=float)
    if actual.ndim != 1 or not len(actual) or (not np.isfinite(actual).all()):
        raise ValueError('Expected raw finite actual_q')
    if velocity.shape != actual.shape or not np.isfinite(velocity).all():
        raise ValueError('Expected finite actual_qvel synchronized with actual_q')
    color = np.asarray(settings['color'], dtype=float)
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError('Initial-RGB color cue must have three finite channels')
    hue = float(cv2.cvtColor(np.uint8(np.clip(color * 255, 0, 255)).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0, 0])
    poses = robot.fk(actual)
    groups, wrist, prismatic = _s2v_groups(robot, calibration, settings, len(actual))
    bounds = np.asarray(settings.get('depth_bounds_m', [0.002, 0.35]), dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or (not 0 < bounds[0] < bounds[1]):
        raise ValueError('Expected positive increasing ray-distance bounds')
    views, records = ({}, observation['fingereye'])
    for name in sorted(set(calibration) | set(records)):
        row = calibration.get(name, {})
        view = dict(finger=str(row.get('finger_module', name)), visible=False, status='missing_rgb', area_px=0, centroid_px=None)
        views[name] = view
        if name not in calibration:
            view['status'] = 'uncalibrated_view'
            continue
        if name not in records:
            continue
        record = records[name]
        if record.get('capture_step', observation.get('step')) != observation.get('step'):
            view['status'] = 'unsynchronized_rgb'
            continue
        try:
            image = _s2v_read_rgb(record)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            difference = abs(hsv[:, :, 0].astype(float) - hue)
            mask = color_foreground(image, hue).astype(np.uint8)
            area = int(mask.sum())
            camera = np.asarray(poses[row['camera_link']], dtype=float)
            if camera.shape != (4, 4) or not np.isfinite(camera).all():
                raise ValueError('Camera FK must be finite 4x4')
            view.update(status='ok', area_px=area, visible=area >= 12, centroid_px=np.argwhere(mask).mean(axis=0)[::-1].tolist() if area else None, gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask=mask, camera=camera.copy(), calibration=_s2v_calibration(row), size=(image.shape[1], image.shape[0]))
        except (OSError, ValueError, KeyError, cv2.error) as error:
            view.update(status='invalid_rgb_or_calibration', error=str(error))
    return dict(step=observation.get('step'), actual_q=actual.copy(), actual_qvel=velocity.copy(), views=views, groups=groups, wrist_joint_indices=wrist, prismatic_indices=prismatic, depth_bounds_m=bounds.tolist())

def stage2_response_action(evidence, round_index, mode):
    """Convert uncertain response into a finite thumb/index displacement."""
    if mode in ('motor_sham', 'fixed_motor'):
        return (np.full(2, STAGE2_DELTA_M if round_index == 0 else 0.00035), 'motor_sham_sensor_independent_schedule')
    if mode == 'fixed':
        return (np.full(2, STAGE2_DELTA_M), 'fixed_sensor_independent_schedule')
    if round_index == 0:
        return (np.full(2, STAGE2_DELTA_M), 'common_light_diagnostic_probe')
    if not isinstance(evidence, dict):
        return (np.zeros(2), 'unknown_response_hold')
    action = evidence.get('advice', 'hold')
    confidence = float(evidence.get('confidence', 0.0))
    threshold = 0.25 if action == 'bounded_squeeze' else 0.5
    if not np.isfinite(confidence) or confidence < threshold:
        return (np.zeros(2), 'uncertain_response_hold')
    if action in ('release', 'backoff', 'reduce'):
        return (np.full(2, -STAGE2_DELTA_M), 'response_backoff')
    if action == 'bounded_squeeze':
        return (np.full(2, STAGE2_DELTA_M), 'response_bilateral_close')
    if action == 'rebalance':
        finger = evidence.get('rebalance_finger')
        if finger in ('thumb', 'index'):
            delta = np.zeros(2)
            delta[('thumb', 'index').index(finger)] = STAGE2_DELTA_M
            return (delta, 'response_close_' + finger)
    return (np.zeros(2), 'response_hold_' + str(action))

def compare_visual_response(before, after, commanded_delta=None):
    """Return bounded, JSON-safe evidence; never equate no motion with grasp.

    commanded_delta is the controller's endpoint target minus BEFORE actual q,
    not the change between two target positions. This preserves servo lag.
    """
    if after['actual_q'].shape != before['actual_q'].shape:
        raise ValueError('Snapshots describe different robot joint inventories')
    motor = _s2v_motor(before, after, commanded_delta)
    difference = after['actual_q'] - before['actual_q']
    wrist_indices = before['wrist_joint_indices']
    prismatic = set(before['prismatic_indices'])
    wrist_fixed = bool(wrist_indices) and all((abs(difference[index]) <= (0.0008 if index in prismatic else 0.015) for index in wrist_indices))
    views, fingers, uncertainty = ({}, {}, ['RGB is not force/contact sensing or success evidence'])
    if not wrist_fixed:
        uncertainty.append('wrist moved or URDF wrist-chain inference unavailable')
    if before.get('step') is not None and after.get('step') is not None:
        if after['step'] <= before['step']:
            raise ValueError('Preload snapshots must advance in time')
    bounds = [min(before['depth_bounds_m'][0], after['depth_bounds_m'][0]), max(before['depth_bounds_m'][1], after['depth_bounds_m'][1])]
    for name in sorted(set(before['views']) | set(after['views'])):
        first, last = (before['views'].get(name), after['views'].get(name))
        finger = (first or last)['finger']
        row = dict(finger=finger, visible_before=bool(first and first['visible']), visible_after=bool(last and last['visible']), confidence=0.0, registered=False, motion_evidence=False, signed_response_px=[None, None])
        views[name] = row
        group = fingers.setdefault(finger, dict(visible_before=0, visible_after=0, registered_views=0, motion_views=0, confidence=0.0, views=[]))
        group['views'].append(name)
        group['visible_before'] += row['visible_before']
        group['visible_after'] += row['visible_after']
        if not first or not last or first['status'] != 'ok' or (last['status'] != 'ok'):
            row['status'] = 'missing_or_invalid_view'
            continue
        row.update(status='ok', area_before_px=first['area_px'], area_after_px=last['area_px'], area_ratio=float(last['area_px'] / max(first['area_px'], 1)), centroid_shift_px=(np.asarray(last['centroid_px']) - first['centroid_px']).tolist() if row['visible_before'] and row['visible_after'] else None)
        if first['size'] != last['size']:
            row['status'] = 'changed_native_dimensions'
            continue
        if not row['visible_before'] or not row['visible_after']:
            row['uncertainty'] = 'not visible; absence is not zero contact'
            continue
        row.update(_s2v_response(first, last, bounds))
        group['registered_views'] += row['registered']
        group['motion_views'] += row['motion_evidence']
        group['confidence'] = max(group['confidence'], row['confidence'])
    visible = [name for name, row in fingers.items() if row['visible_after']]
    lost = [name for name, row in fingers.items() if row['visible_before'] and (not row['visible_after'])]
    moving = [name for name, row in fingers.items() if row['motion_views']]
    registered = [name for name, row in fingers.items() if row['registered_views']]
    asymmetric = bool(len(moving) == 1 and len(registered) >= 2)
    confidence = min([fingers[name]['confidence'] for name in visible], default=0.0)
    advice, reason = ('hold', 'no evidence warrants more squeeze; no-motion is not contact')
    if not wrist_fixed:
        advice, reason = ('reobserve', 'preload comparison requires approximately fixed wrist')
    elif lost:
        advice, reason = ('replan', 'finger lost object visibility; do not squeeze blindly')
    elif len(visible) < 2:
        advice, reason = ('reobserve', 'fewer than two finger groups see the object')
    elif asymmetric:
        advice, reason = ('rebalance', 'one-sided registered image response; hold moving side, bounded probe other side')
        confidence = fingers[moving[0]]['confidence']
    elif moving:
        advice, reason = ('hold', 'registered image response beyond camera-only depth envelope; no squeeze escalation')
    elif len(registered) < 2:
        advice, reason = ('hold', 'texture/depth uncertainty prevents registered motion sign')
    else:
        driven = [row for row in motor['fingers'].values() if row['drive_deficit'] is not None]
        if len(driven) >= 2 and all((row['settled'] and -0.25 <= row['drive_deficit'] < 0.55 for row in driven)):
            advice, reason = ('bounded_squeeze', 'bilateral visibility with settled tracked closure; one small bounded probe only')
        elif any((row['settled'] is False for row in driven)):
            advice, reason = ('hold', 'motor still settling; do not confuse lag with compliance')
    classification = {'replan': 'visibility_loss_not_resolved_by_blind_squeeze', 'reobserve': 'insufficient_observability', 'rebalance': 'asymmetric_registered_image_response', 'bounded_squeeze': 'bounded_probe_candidate_not_contact_proof'}.get(advice, 'motion_risk_hold' if moving else 'contact_geometry_unresolved')
    return dict(schema=1, step_before=before.get('step'), step_after=after.get('step'), wrist_fixed=wrist_fixed, wrist_joint_indices=wrist_indices, depth_bounds_m=bounds, views=views, fingers=fingers, motor=motor, visible_fingers=visible, lost_fingers=lost, moving_fingers=moving, asymmetric_motion=asymmetric, confidence=float(confidence), rebalance_finger=next((name for name in visible if name not in moving), None) if asymmetric else None, advice=advice, reason=reason, uncertainty=uncertainty, classification=classification, object_displacement_m=None, force_estimate_n=None)

def stage3_support_snapshot(observation, robot, static):
    """Read only native FingerEye RGB, synchronized actual q/qvel and static FK.

    static: {calibration|fingereye_calibration, color|hue, palm_link?,
             depth_bounds_m?}. Color is the immutable initial-RGB cue in [0,1].
    The default 2..350 mm ray-distance bound is an exposed modeling assumption.
    All six slots survive missing/occluded/invalid frames as unknown entries.
    """
    if not isinstance(static, dict):
        static = dict(calibration=static.calibration, color=static.info['color'])
    calibration = static.get('calibration', static.get('fingereye_calibration'))
    if calibration is None:
        raise ValueError('Static camera calibration required')
    observation=s9_exposure(observation)
    q = np.asarray(observation['actual_q'], dtype=float)
    qvel = np.asarray(observation['actual_qvel'], dtype=float)
    if q.ndim != 1 or not len(q) or qvel.shape != q.shape or (not np.isfinite(np.r_[q, qvel]).all()):
        raise ValueError('Raw synchronized finite actual q/qvel required')
    if 'hue' in static:
        hue = float(static['hue'])
    else:
        color = np.asarray(static['color'], dtype=float)
        if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError('Initial RGB color must be three channels in [0,1]')
        hue = float(cv2.cvtColor(np.uint8(color * 255).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0, 0])
    if not np.isfinite(hue) or not 0 <= hue < 180:
        raise ValueError('Finite initial RGB hue required')
    bounds = np.asarray(static.get('depth_bounds_m', [0.002, 0.35]), dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or (not 0 < bounds[0] < bounds[1]):
        raise ValueError('Positive increasing ray-distance bounds required')
    poses = robot.fk(q)
    palm = np.asarray(poses[static.get('palm_link', robot.palm_link)], dtype=float)
    if palm.shape != (4, 4) or not np.isfinite(palm).all():
        raise ValueError('Finite palm FK required')
    step = observation['step']
    views = {}
    for name in STAGE3_SUPPORT_CAMERAS:
        row = dict(status='missing_rgb', visible=False, finger=calibration[name]['finger_module'], area_px=0)
        views[name] = row
        record = observation.get('fingereye', {}).get(name)
        if record is None or name not in calibration:
            continue
        if record.get('capture_step', step) != step:
            row['status'] = 'unsynchronized_rgb'
            continue
        try:
            rgb = np.asarray(record['rgb']) if 'rgb' in record else np.load(record['rgb_path'], allow_pickle=False)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError('Native uint8 RGB required')
            if record.get('width', rgb.shape[1]) != rgb.shape[1] or record.get('height', rgb.shape[0]) != rgb.shape[0]:
                raise ValueError('Native RGB dimension mismatch')
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            delta = abs(hsv[:, :, 0].astype(float) - hue)
            mask = color_foreground(rgb, hue).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            areas = np.sort(stats[1:, cv2.CC_STAT_AREA]) if count > 1 else []
            ambiguous = bool(len(areas) > 1 and areas[-2] > 0.3 * areas[-1])
            if count > 1:
                mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
            area = int(mask.sum())
            border = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
            camera = np.asarray(poses[calibration[name]['camera_link']], dtype=float)
            if camera.shape != (4, 4) or not np.isfinite(camera).all():
                raise ValueError('Finite camera FK required')
            row.update(status='ok', visible=area >= 20, area_px=area, mask=mask, gray=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), centroid_px=np.argwhere(mask).mean(0)[::-1] if area else None, border=border, ambiguous=ambiguous, camera=camera.copy(), calibration=_s3s_calibration(calibration[name]), size=(rgb.shape[1], rgb.shape[0]), frame_id=record.get('frame_id'))
        except (KeyError, ValueError, OSError, cv2.error) as error:
            row.update(status='invalid_rgb_or_calibration', error=str(error))
    return dict(selected_fingers=list(robot.finger_names), version=STAGE3_SUPPORT_VERSION, step=int(step), actual_q=q.copy(), actual_qvel=qvel.copy(), palm=palm.copy(), views=views, depth_bounds_m=bounds.tolist())

def stage3_support_pair(before, after, mode='lift'):
    """Compare two snapshots under carried/stationary-world lift hypotheses.

    Neither silhouette centroid nor a quiet color mask is object-pose truth.
    A 3 mm proof commonly remains unknown under the broad depth envelope.
    During hold, only ACTUAL camera-stationary views can flag image slip/spin.
    """
    if mode not in ('lift', 'hold'):
        raise ValueError('mode must be lift or hold')
    step_delta = int(after['step'] - before['step'])
    palm_delta = after['palm'][:3, 3] - before['palm'][:3, 3]
    attitude = _s3s_rotation(before['palm'][:3, :3], after['palm'][:3, :3])
    rows = []
    for name in STAGE3_SUPPORT_CAMERAS:
        a, b = (before['views'][name], after['views'][name])
        row = dict(camera=name, finger=a['finger'], classification='unknown')
        rows.append(row)
        stale = a.get('frame_id') is not None and b.get('frame_id') is not None and (b['frame_id'] <= a['frame_id'])
        if step_delta <= 0 or stale or (not a['visible']) or (not b['visible']) or (a['status'] != 'ok') or (b['status'] != 'ok'):
            row['reason'] = 'missing_invisible_or_nonincreasing_frames'
            continue
        if a['size'] != b['size'] or a.get('ambiguous') or b.get('ambiguous') or a.get('border') or b.get('border'):
            row['reason'] = 'ambiguous_or_truncated_silhouette'
            continue
        shift = b['centroid_px'] - a['centroid_px']
        union = int(np.logical_or(a['mask'], b['mask']).sum())
        iou = float(np.logical_and(a['mask'], b['mask']).sum() / max(union, 1))
        log_area = float(abs(np.log(b['area_px'] / a['area_px'])))
        camera_translation = float(np.linalg.norm(b['camera'][:3, 3] - a['camera'][:3, 3]))
        camera_rotation = _s3s_rotation(a['camera'][:3, :3], b['camera'][:3, :3])
        settled = np.max(abs(before['actual_qvel'][:3])) < 0.001 and np.max(abs(after['actual_qvel'][:3])) < 0.001 and (np.max(abs(before['actual_qvel'][3:])) < 0.02) and (np.max(abs(after['actual_qvel'][3:])) < 0.02)
        stationary = camera_translation <= 0.0001 and camera_rotation <= 0.003 and settled
        row.update(shift_px=float(np.linalg.norm(shift)), log_area=log_area, iou=iou, camera_translation_m=camera_translation, camera_rotation_rad=camera_rotation, camera_stationary=bool(stationary))
        if mode == 'hold':
            if not stationary:
                row['reason'] = 'actual_camera_motion'
                continue
            angle, tracks = _s3s_texture_rotation(a, b)
            row.update(image_rotation_rad=angle, texture_tracks=tracks)
            risk = np.linalg.norm(shift) > 2.5 or log_area > 0.18 or iou < 0.65
            if angle is not None and abs(angle) > 0.035:
                risk = True
            row['classification'] = 'image_motion_risk' if risk else 'image_consistent'
        else:
            moved = cv2.warpAffine(a['mask'], np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]]), a['size'], flags=cv2.INTER_NEAREST)
            shape_iou = float(np.logical_and(moved, b['mask']).sum() / max(np.logical_or(moved, b['mask']).sum(), 1))
            row['translated_shape_iou'] = shape_iou
            relative_a = np.linalg.solve(before['palm'], a['camera'])
            relative_b = np.linalg.solve(after['palm'], b['camera'])
            rigid_camera = np.linalg.norm(relative_a[:3, 3] - relative_b[:3, 3]) <= 0.00015 and _s3s_rotation(relative_a[:3, :3], relative_b[:3, :3]) <= 0.003
            row['camera_rigid_to_palm'] = bool(rigid_camera)
            if log_area > 0.35 or shape_iou < 0.65:
                if rigid_camera and (log_area > 0.35 or shape_iou < 0.6):
                    row['classification'] = 'relative_motion_risk'
                row['reason'] = 'changing_contour_area_or_occlusion'
                continue
            bounds = before['depth_bounds_m']
            if bounds != after['depth_bounds_m']:
                row['reason'] = 'depth_assumption_changed'
                continue
            depths = np.geomspace(bounds[0], bounds[1], 256)
            carried = _s3s_hypothesis(a, b, before['palm'], after['palm'], depths, 'carried')
            grounded = _s3s_hypothesis(a, b, before['palm'], after['palm'], depths, 'grounded')
            row.update(carried=carried, grounded=grounded)
            if carried['possible'] is True and grounded['possible'] is False:
                row['classification'] = 'carrying_evidence'
            elif grounded['possible'] is True and carried['possible'] is False:
                row['classification'] = 'grounded_evidence'
            else:
                row['reason'] = 'hypotheses_overlap_or_neither_fits'

    def bilateral(label):
        return set(before['selected_fingers']).issubset({r['finger'] for r in rows if r['classification'] == label})
    lift_valid = step_delta > 0 and palm_delta[2] >= 0.002 and (np.linalg.norm(palm_delta[:2]) <= 0.0005) and (attitude <= 0.01)
    carry, ground = (bilateral('carrying_evidence'), bilateral('grounded_evidence'))
    risk = set(before['selected_fingers']).issubset({r['finger'] for r in rows if r['classification'] in ('grounded_evidence', 'relative_motion_risk')})
    state = 'unknown'
    if mode == 'lift' and lift_valid and (carry != ground):
        state = 'carrying_evidence' if carry else 'grounded_evidence'
    elif mode == 'lift' and lift_valid and risk and (not carry):
        state = 'carrying_inconsistent'
    if mode == 'hold' and bilateral('image_motion_risk'):
        state = 'image_motion_risk'
    elif mode == 'hold' and bilateral('image_consistent'):
        state = 'image_consistent'
    return dict(version=STAGE3_SUPPORT_VERSION, mode=mode, state=state, before_step=before['step'], step=after['step'], step_delta=step_delta, actual_palm_delta_m=palm_delta.tolist(), actual_palm_rotation_rad=attitude, lift_excitation_valid=bool(lift_valid), views=rows, depth_bounds_m=before['depth_bounds_m'], motor_contact_used=False, native_object_omega_observable=False, private_success_certified=False)

def stage3_support_decision(lift_reports, hold_reports, *, attempt=1, max_attempts=3, remaining_steps=24000, remaining_wall_s=2400.0, motor_scores=None, reobservations=0, feedback=True, phase='final', recovery_reserve_steps=2400, next_attempt_reserve_steps=6000, retry_reserve_wall_s=600.0, use_active_lift_risk=True):
    """Finite policy for proof continuation, observation, actual retry or stop.

    Root enforces total budgets and original guards. Retry reserves are admission
    estimates, not timing guarantees; root must check every command and replan.
    Motor scores are recorded only: never sufficient to continue/stop/retry.
    """
    if phase not in ('proof', 'middle', 'final'):
        raise ValueError('phase must be proof, middle or final')
    if max_attempts not in (1, 2, 3) or not 1 <= attempt <= max_attempts:
        raise ValueError('Physical attempt limit must be 1..3')
    lift = [r for r in lift_reports if r.get('mode') == 'lift']
    hold = [r for r in hold_reports if r.get('mode') == 'hold']
    recent = hold[-10:]
    chronological = all((a.get('step') == b.get('before_step') for a, b in zip(recent, recent[1:])))
    held_steps = sum((max(0, int(r.get('step_delta', 0))) for r in recent))
    consistent = len(recent) >= 10 and held_steps >= 240 and chronological and all((r['state'] == 'image_consistent' for r in recent))
    motion_risk = len(hold) >= 3 and all((r['state'] == 'image_motion_risk' for r in hold[-3:])) and all((a.get('step') == b.get('before_step') for a, b in zip(hold[-3:], hold[-2:])))
    last_carry = max((i for i, r in enumerate(lift) if r['state'] == 'carrying_evidence'), default=-1)
    unresolved = lift[last_carry + 1:]
    active_motion_risk = bool(use_active_lift_risk) and any((a['state'] in ('grounded_evidence', 'carrying_inconsistent') and b['state'] in ('grounded_evidence', 'carrying_inconsistent') and (b.get('before_step', -1) >= a.get('step', 0)) for a, b in zip(unresolved, unresolved[1:])))
    carried = bool(lift and lift[-1]['state'] == 'carrying_evidence')
    grounded = bool(lift and lift[-1]['state'] == 'grounded_evidence')
    scores = None if motor_scores is None else [float(x) if np.isfinite(x) else None for x in motor_scores]
    result = dict(action=None, reason=None, support_state='unknown', attempt=int(attempt), max_attempts=int(max_attempts), phase=phase, feedback=bool(feedback), motor_contact_hypothesis=scores, motor_contact_used=False, carrying_evidence=carried, grounded_evidence=grounded, persistent_image_motion_risk=motion_risk, repeated_lift_image_risk=active_motion_risk, hold_consistent=bool(consistent), use_active_lift_risk=bool(use_active_lift_risk), remaining_steps=int(remaining_steps), remaining_wall_s=float(remaining_wall_s), private_success_certified=False, native_object_omega_observable=False)

    def finish(action, reason, support='unknown'):
        result.update(action=action, reason=reason, support_state=support)
        return result
    if remaining_steps < 24 or remaining_wall_s <= 0:
        return finish('stop_budget', 'episode_budget_exhausted')
    if phase != 'final' and (not feedback or not (grounded or motion_risk or active_motion_risk)):
        return finish('continue_lift', 'bounded_larger_proof_needed' if not carried else 'continue_original_lift')
    if feedback and carried and consistent and (not motion_risk) and (not active_motion_risk):
        return finish('stop_supported', 'carrying_and_stationary_image_evidence_uncertified', 'carrying_evidence')
    failure = grounded or motion_risk or active_motion_risk
    if feedback and (not failure):
        if phase == 'final' and reobservations < 1 and (remaining_steps >= 240) and (remaining_wall_s >= 30):
            return finish('reobserve', 'one_bounded_stationary_observation_window')
        return finish('stop_unknown', 'unresolved_RGB_do_not_destroy_possible_carried_grasp')
    if attempt >= max_attempts:
        return finish('stop_limit', 'physical_attempt_limit', 'failure_evidence' if failure else 'unknown')
    reserve = int(recovery_reserve_steps + next_attempt_reserve_steps)
    if remaining_steps < reserve or remaining_wall_s < retry_reserve_wall_s:
        return finish('stop_budget', 'insufficient_recovery_and_next_attempt_reserve', 'failure_evidence' if failure else 'unknown')
    return finish('retry', 'observed_failure' if feedback else 'blind_registered_attempt_schedule', 'failure_evidence' if failure else 'unknown')

def _S9_LIFT(s, height=0.03, hold=360):
    ctx = s['ctx']
    before = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    target = s['last'].copy()
    target[2] = s['c']['q'][2] + height
    n = 240 if height <= 0.0031 else 300
    recent = []
    previous = before
    s['lift_veto'] = False
    for q, count, label in smooth_segment(s['last'].copy(), target, n, 's5_proof' if height <= 0.0031 else 's5_lift'):
        s5_step(s, q, count, label)
        if abs(float(s['obs']['actual_q'][2]) - float(previous['actual_q'][2])) < 0.0022:
            continue
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        response = stage3_support_pair(previous, snap, 'lift')
        previous = snap
        recent.append(response)
        s5_note(ctx, 's5_lift_in_progress', response)
        if len(recent) >= 2 and all((x['state'] == 'grounded_evidence' for x in recent[-2:])):
            s['lift_veto'] = True
            s5_note(ctx, 's5_lift_veto', dict(step=s['obs']['step'], reason='two_current_RGB_intervals_grounded', commanded_height_m=height))
            break
    after = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    r = stage3_support_pair(before, after, 'lift')
    s['lift_reports'].append(r)
    s5_note(ctx, 's5_lift_response', dict(step=s['obs']['step'], height_m=height, report=r))
    for _ in range(0 if s['lift_veto'] else hold // 24):
        s5_step(s, s['last'], 24, 's5_hold')
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        r = stage3_support_pair(after, snap, 'hold')
        s['hold_reports'].append(r)
        after = snap
    if hold and (not s['lift_veto']):
        s5_note(ctx, 's5_hold_response', dict(step=s['obs']['step'], reports=s['hold_reports'][-hold // 24:]))
    return r

def grasp_seed(robot, prior, targets, inward, fraction=None):
    q=np.asarray(prior,float).copy()
    q[robot.base_indices]=0.
    if fraction is not None:
        ids=robot.active_finger_indices
        q[ids]=robot.lo[ids]+fraction*(robot.hi[ids]-robot.lo[ids])
    p,n,d,_=robot.tips(q)
    u=n[0]-n[1]
    if np.linalg.norm(u)<1e-5:u=p[1]-p[0]
    u/=np.linalg.norm(u)
    v=d.mean(0);v-=u*(v@u)
    if np.linalg.norm(v)<1e-5:v=np.cross(u,np.eye(3)[np.argmin(abs(u))])
    v/=np.linalg.norm(v)
    target_u=np.asarray(inward[0])-inward[1];target_u/=np.linalg.norm(target_u)
    target_v=np.array([0.,0.,-1.]);target_v-=target_u*(target_v@target_u);target_v/=np.linalg.norm(target_v)
    source=np.column_stack([u,v,np.cross(u,v)])
    destination=np.column_stack([target_u,target_v,np.cross(target_u,target_v)])
    rotation=destination@source.T
    q[3:6]=Rotation.from_matrix(rotation).as_euler('ZYX')
    q[:3]=np.mean(targets,axis=0)-rotation@p.mean(0)
    return np.clip(q,robot.lo+2e-6,robot.hi-2e-6)


def solve_target_ik(robot, prior, targets, inward, normal_weight=12.0, seed_index=0):
    """The same bounded geometry search for every declared hand DOF."""
    prior=np.asarray(prior,float);targets=np.asarray(targets);inward=np.asarray(inward)
    if prior.shape!=robot.lo.shape:raise ValueError('IK prior must follow declared joint order')
    started=time.perf_counter();lo=robot.lo+1e-6;hi=robot.hi-1e-6
    finger_scale=np.sqrt(6/max(1,len(robot.all_finger_indices)))
    seeds=[]
    if np.max(np.linalg.norm(robot.tips(prior)[0]-targets,axis=1))<.01:
        seeds.append(np.clip(prior,lo+1e-7,hi-1e-7))
    seeds.extend(grasp_seed(robot,prior,targets,inward,f) for f in (None,.15,.4,.7))
    attempts=[];best=None
    for number,seed in enumerate(seeds):
        def residual(q, collision=False):
            p,n,_,poses=robot.tips(q);verts=robot.vertices(poses)
            rotation=Rotation.from_euler('ZYX',q[3:6]).as_matrix()@Rotation.from_euler('ZYX',seed[3:6]).as_matrix().T
            clearance=(robot.ik_checker.optimization_clearance(poses)*3000 if collision else
                np.minimum(verts[:,2]-robot.ik_checker.table_z-.001,0)*5000/np.sqrt(len(verts)))
            return np.r_[(p-targets).ravel()*1500*np.sqrt(2/len(targets)),
                (n-inward).ravel()*normal_weight*np.sqrt(2/len(targets)),clearance,
                Rotation.from_matrix(rotation).as_rotvec()*.25,
                (q[robot.all_finger_indices]-seed[robot.all_finger_indices])*.08*finger_scale]
        fit=least_squares(residual,seed,bounds=(lo,hi),x_scale=np.r_[[.01]*3,[.3]*(len(seed)-3)],
            max_nfev=120,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        q=fit.x;p,n,_,poses=robot.tips(q)
        errors=np.linalg.norm(p-targets,axis=1);angles=np.arccos(np.clip(np.sum(n*inward,axis=1),-1,1))
        source_clearance=robot.ik_checker.check(q)
        geometry_cost=float(fit.cost);geometry_nfev=int(fit.nfev);collision_refined=False
        # A source-mesh intersection is a discontinuous objective far from a
        # grasp. First reach the contact manifold, then refine nearby collisions.
        if errors.max()<.002 and angles.max()<.7 and not source_clearance['passed']:
            fit=least_squares(lambda x:residual(x,True),q,bounds=(lo,hi),
                x_scale=np.r_[[.01]*3,[.3]*(len(seed)-3)],diff_step=1e-4,
                max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
            q=fit.x;p,n,_,poses=robot.tips(q)
            errors=np.linalg.norm(p-targets,axis=1);angles=np.arccos(np.clip(np.sum(n*inward,axis=1),-1,1))
            source_clearance=robot.ik_checker.check(q);collision_refined=True
        clearance=float(robot.vertices(poses)[:,2].min()-robot.ik_checker.table_z)
        contact_feasible=bool(errors.max()<.0008 and angles.max()<.5 and clearance>=.00065)
        feasible=contact_feasible and source_clearance['passed']
        meta=dict(seed=number,normal_weight=normal_weight,
            cost=float(fit.cost),nfev=geometry_nfev+(int(fit.nfev) if collision_refined else 0),solver_success=bool(fit.success),error_m=float(errors.max()),
            per_finger_error_m=errors.tolist(),fingers=list(robot.finger_names),normal_error_rad=angles.tolist(),
            table_clearance_m=clearance,wall_s=time.perf_counter()-started,q=q.tolist(),targets=targets.tolist(),
            inward=inward.tolist(),joint_count=len(q),finger_regularizer_normalization=finger_scale,
            geometry_cost=geometry_cost,geometry_nfev=geometry_nfev,collision_refined=collision_refined,
            objective_phase='source_mesh_refinement' if collision_refined else 'contact_geometry',
            source_clearance=source_clearance,
            seed_source='current measured joint state and limit-normalized geometry search',solver_prior=seed.tolist())
        attempts.append(dict(seed=number,feasible=feasible,contact_feasible=contact_feasible,error_m=meta['error_m'],normal_error_rad=angles.tolist(),table_clearance_m=clearance,cost=meta['cost'],collision_refined=collision_refined,source_clearance=source_clearance))
        score=(not feasible,not contact_feasible,max(0.,.00065-clearance)*5000+errors.max()*1500+angles.max(),fit.cost)
        if best is None or score<best[0]:best=(score,q.copy(),meta)
        if feasible:break
    meta=best[2];meta['seed_attempts']=attempts
    return best[1],meta

def _s3_attitude_delta(first, second):
    a = Rotation.from_euler('ZYX', np.asarray(first)).as_matrix()
    b = Rotation.from_euler('ZYX', np.asarray(second)).as_matrix()
    return float((Rotation.from_matrix(a).inv() * Rotation.from_matrix(b)).magnitude())

def stage3_approach(robot, q, points, inward, initial=None):
    acts = _S3_APPROACH(robot, q, points, inward, initial)
    if STAGE3_ATTEMPT == 1:
        return acts
    return _s3_retime_approach(acts, initial)

def branch(robot, q, points, inward, prefix, mode='simultaneous', preload=0.0015):
    acts = list(prefix)
    start = np.array(acts[-1][0])
    last = start.copy()
    for k in range(1, 11):
        t = k / 10
        if mode == 'simultaneous':
            p = np.full(len(points), 0.0005 + (preload - 0.0005) * min(t * 2, 1))
        elif mode == 'slow':
            p = np.full(len(points), 0.0005 + (preload - 0.0005) * t)
        elif mode == 'thumb_first':
            p = 0.0005 + (preload - 0.0005) * np.r_[min(2 * t, 1), np.full(len(points) - 1, max(2 * t - 1, 0))]
        elif mode == 'index_first':
            p = 0.0005 + (preload - 0.0005) * np.r_[max(2 * t - 1, 0), np.full(len(points) - 1, min(2 * t, 1))]
        elif mode == 'release_reload':
            p = np.full(len(points), 0.0005 * (1 - 2 * t) if t <= 0.5 else preload * (2 * t - 1))
        else:
            raise ValueError(mode)
        last = close_ik(robot, q, points + inward * p[:, None])
        acts.append([last.tolist(), 24, mode])
    lifted = last.copy()
    lifted[2] += 0.03
    acts += smooth_segment(last, lifted, 600, 'lift')
    acts.append([lifted.tolist(), 360, 'hold'])
    return acts

def _S6_V3_RETIME(acts, initial):
    last = np.asarray(initial, dtype=float).copy()
    out, i = ([], 0)
    while i < len(acts):
        phase = acts[i][2]
        j = i
        while j + 1 < len(acts) and acts[j + 1][2] == phase:
            j += 1
        if phase in ('close', 'probe_hold') or phase.startswith('route_'):
            out += acts[i:j + 1]
        else:
            target = np.asarray(acts[j][0])
            n = _s3_steps(target - last, sum((x[1] for x in acts[i:j + 1])), 0.08, 0.9)
            out += smooth_segment(last, target, n, phase)
        last = np.asarray(acts[j][0])
        i = j + 1
    return out

def _s3_note(ctx, key, value):
    ctx.note(json.dumps({key: value}, allow_nan=False))

class _Stage3Mover:

    def __init__(self, ctx, state):
        self.ctx, self.state = (ctx, state)
        self.robot, self.checker = (state['robot'], state['checker'])
        self.obs = ctx.observe()
        self.last = np.asarray(state['last'], dtype=float).copy()
        self.wrist = self.last[3:6].copy()
        self.actual_wrist_reference = np.asarray(self.obs['actual_q'])[3:6].copy()
        self.max_actual_attitude_delta = 0.0
        self.history = [(int(self.obs['step']), np.asarray(self.obs['actual_q']).copy())]

    def settle_evidence(self):
        span = self.history[-1][0] - self.history[0][0]
        samples = np.asarray([q for _, q in self.history])
        ranges = np.ptp(samples, axis=0)
        qvel = abs(np.asarray(self.obs['actual_qvel']))
        settled = bool(span >= 48 and ranges[:3].max() <= 0.0005 and (ranges[3:].max() <= 0.01) and (qvel[:3].max() <= 0.005) and (qvel[3:].max() <= 0.04))
        return {'settled': settled, 'window_steps': span, 'sample_stride_steps': 6, 'actual_q_range': ranges.tolist(), 'actual_qvel': qvel.tolist(), 'translation_range_limit_m': 0.0005, 'joint_range_limit_rad': 0.01}

    def audit(self, phase, started_step, initial_actual):
        _s3_note(self.ctx, 'stage3_recovery_phase', {'phase': phase, 'started_step': started_step, 'step': int(self.obs['step']), 'actual_base_before': np.asarray(initial_actual)[:6].tolist(), 'actual_base_after': np.asarray(self.obs['actual_q'])[:6].tolist(), 'command_base_after': self.last[:6].tolist(), 'command_wrist_attitude_delta_rad': _s3_attitude_delta(self.wrist, self.last[3:6]), 'sampled_max_actual_wrist_attitude_delta_rad': self.max_actual_attitude_delta, 'actual_attitude_limit_rad': STAGE3_RECOVERY_ATTITUDE_LIMIT_RAD, 'sample_stride_steps': 6, 'object_contact_clearance_certified': False})

    def move(self, target, minimum, phase, object_check=False):
        target = np.asarray(target, dtype=float)
        n = _s3_steps(target - self.last, minimum)
        if self.obs['step'] + n > STAGE3_TOTAL_STEPS - STAGE3_RESERVE_ATTEMPT_STEPS:
            _s3_note(self.ctx, 'stage3_recovery_reject', {'phase': phase, 'reason': 'step_reserve'})
            return False
        if not np.allclose(target[3:6], self.wrist, rtol=0, atol=1e-10):
            _s3_note(self.ctx, 'stage3_recovery_reject', {'phase': phase, 'reason': 'wrist_target_change_before_release_ticket'})
            return False
        started_step = int(self.obs['step'])
        initial_actual = np.asarray(self.obs['actual_q']).copy()
        path = smooth_segment(self.last, target, n, phase)
        for q, count, _ in path:
            q = np.asarray(q)
            if np.any(q < self.robot.lo) or np.any(q > self.robot.hi) or (not self.checker.check(q, object_check=object_check)['passed']):
                _s3_note(self.ctx, 'stage3_recovery_reject', {'phase': phase, 'reason': 'original_geometry_or_limits'})
                return False
        for q, count, name in path:
            self.obs = self.ctx.step(np.asarray(q), steps=count, phase=name)
            self.last = np.asarray(q).copy()
            actual = np.asarray(self.obs['actual_q'])
            self.history.append((int(self.obs['step']), actual.copy()))
            self.history = self.history[-9:]
            self.max_actual_attitude_delta = max(self.max_actual_attitude_delta, _s3_attitude_delta(self.actual_wrist_reference, actual[3:6]))
            if self.max_actual_attitude_delta > STAGE3_RECOVERY_ATTITUDE_LIMIT_RAD:
                self.audit(phase, started_step, initial_actual)
                _s3_note(self.ctx, 'stage3_recovery_reject', {'phase': phase, 'reason': 'measured_wrist_attitude_drift'})
                self.sync()
                return False
            v = abs(np.asarray(self.obs['actual_qvel']))
            if v[:3].max() > 0.15 or v[3:].max() > 1.5:
                self.audit(phase, started_step, initial_actual)
                _s3_note(self.ctx, 'stage3_recovery_reject', {'phase': phase, 'reason': 'observed_native_speed'})
                self.sync()
                return False
        self.audit(phase, started_step, initial_actual)
        self.sync()
        return True

    def sync(self):
        self.state.update(obs=self.obs, last=self.last.copy())

def _s3_tip_frame(robot, q):
    q = np.asarray(q)
    rotation = Rotation.from_euler('ZYX', q[3:6]).as_matrix()
    return (robot.tips(q)[0] - q[:3]) @ rotation

def vector(node, key, default='0 0 0'):
    return np.fromstring(node.get(key, default), sep=' ')

def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, :3] = Rotation.from_euler('xyz', vector(node, 'rpy')).as_matrix()
        t[:3, 3] = vector(node, 'xyz')
    return t

def _R13_ESTIMATE(observation):
    """RGB tabletop homography. Height is an explicit uncertain shape prior."""
    cam = observation['initial_overhead']
    rgb = np.load(cam['rgb_path']).astype(float) / 255
    saturation = rgb.max(-1) - rgb.min(-1)
    mask = color_foreground(np.uint8(np.rint(rgb * 255)), initial=True)
    yy, xx = np.indices(mask.shape)
    mask &= (abs(xx - mask.shape[1] / 2) < mask.shape[1] * 0.25) & (abs(yy - mask.shape[0] / 2) < mask.shape[0] * 0.3)
    components, count = label(mask)
    parts = []
    for k in range(1, count + 1):
        pixels = np.argwhere(components == k)
        if len(pixels) >= 8:
            parts.append(pixels)
    if not parts:
        raise RuntimeError('No confident RGB foreground in initial overhead')
    pixels = max(parts, key=len)
    uv = pixels[:, [1, 0]].astype(float) + 0.5
    K = np.asarray(cam['intrinsics'])
    pose = np.asarray(cam['camera_pose_world'])
    R = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    rays = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] @ R.T
    world = pose[:3] + rays * (-pose[2] / rays[:, 2])[:, None]
    center = world[:, :2].mean(0)
    _, axes = np.linalg.eigh(np.cov(world[:, :2].T))
    axes = axes[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    local = (world[:, :2] - center) @ axes
    pixel = pose[2] / K[0, 0]
    lo, hi = (local.min(0), local.max(0))
    half = (hi - lo) / 2 + pixel * 0.5
    height = float(np.clip(2 * max(half), 0.008, 0.03))
    center += axes @ ((lo + hi) / 2)
    color = np.median(rgb[pixels[:, 0], pixels[:, 1]], axis=0)
    return {'center': np.r_[center, height / 2], 'half': np.r_[half, height / 2], 'axes': axes, 'table_z': 0.0, 'top_z': height, 'pixel_m': float(pixel), 'color': color, 'height_source': 'uncertain footprint-scale prior; not measured'}

class RobotVisibility:

    def __init__(self, urdf):
        self.parts = []
        self.query_count = 0
        grouped = {}
        for link in ET.parse(urdf).getroot().findall('link'):
            name = link.get('name')
            if 'fingereye_camera_' in name:
                continue
            for visual in link.findall('visual'):
                mesh = visual.find('geometry/mesh')
                if mesh is None:
                    continue
                path = Path(mesh.get('filename'))
                if not path.is_absolute():
                    path = Path(urdf).parent / path
                loaded = trimesh.load(path, force='mesh', process=False)
                loaded.apply_scale(vector(mesh, 'scale', '1 1 1'))
                loaded.apply_transform(origin(visual.find('origin')))
                grouped.setdefault(name, []).append(loaded)
        for name, meshes in grouped.items():
            mesh = trimesh.util.concatenate(meshes)
            intersector = RayMeshIntersector(mesh)
            _ = intersector._scene
            self.parts.append((name, mesh.bounds.copy(), intersector))

    def depth(self, poses, cameras):
        result = {}
        for name, (camera, rays) in cameras.items():
            depth = np.full(len(rays), 1.0)
            for link, bounds, intersect in self.parts:
                link_pose = poses[link]
                relative_r = link_pose[:3, :3].T @ camera[:3, :3]
                relative_t = (camera[:3, 3] - link_pose[:3, 3]) @ link_pose[:3, :3]
                directions = rays @ relative_r.T
                safe = np.where(abs(directions) < 1e-15, 1e-15, directions)
                a = (bounds[0] - relative_t) / safe
                b = (bounds[1] - relative_t) / safe
                enter = np.minimum(a, b).max(axis=1)
                leave = np.maximum(a, b).min(axis=1)
                keep = np.flatnonzero((leave >= np.maximum(enter, 0.005)) & (enter < depth))
                if len(keep) == 0:
                    continue
                query_d = directions[keep]
                origins = np.repeat(relative_t[None], len(keep), axis=0)
                locations, indices, _ = intersect.intersects_location(origins + query_d * 0.005, query_d, multiple_hits=False)
                self.query_count += 1
                if len(indices):
                    distances = np.linalg.norm(locations - origins[indices], axis=1)
                    indices = keep[indices]
                    depth[indices] = np.minimum(depth[indices], distances)
            result[name] = depth
        return result



class _R13_CHECKER:
    """Reuse full-mesh BVHs and derive only structural assembly exclusions."""

    def __init__(self, robot, info):
        import itertools
        import fcl
        import trimesh
        self.fcl = fcl
        self.robot = robot
        tree = ET.parse(robot.urdf_path).getroot()
        physical = sorted(n.get('name') for n in tree.findall('link') if n.find('inertial') is not None)
        self.excluded = robot.description.structural_collision_exclusions(physical)
        self.physical_body_count = len(physical)
        self.retained_physical_pair_count = len(physical) * (len(physical) - 1) // 2 - len(self.excluded)
        self.names = sorted(robot.surfaces)
        self.hulls = {name: v for name, v in robot.meshes}
        self.pairs = np.asarray([(i, j) for i, j in itertools.combinations(range(len(self.names)), 2) if (self.names[i], self.names[j]) not in self.excluded], dtype=int)

        def model(mesh):
            result = fcl.BVHModel()
            result.beginModel(len(mesh.vertices), len(mesh.faces))
            result.addSubModel(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
            result.endModel()
            return result
        # Refining the RGB object envelope does not change the hand meshes.
        # Keep one full-resolution BVH per link for this Robot's lifetime;
        # duplicating these large models can exhaust the worker at first refine.
        if not hasattr(robot, '_fcl_surface_models'):
            robot._fcl_surface_models = {name: model(robot.surfaces[name]) for name in self.names}
        self.models = robot._fcl_surface_models
        self.table_z = float(info['table_z'])
        rotation = np.eye(3)
        rotation[:2, :2] = np.asarray(info['axes'])
        center = np.r_[np.asarray(info['center'])[:2], self.table_z + float(info['half'][2])]
        half = np.maximum(np.asarray(info['half']) - 0.00065, 0.0005)
        box = trimesh.creation.box(extents=2 * half)
        self.object = fcl.CollisionObject(model(box), fcl.Transform(rotation, center))
        self.object_bounds = np.array([center - abs(rotation) @ half, center + abs(rotation) @ half])
        self.nodes_checked = 0

    def check(self, q, object_check=True):
        return _S6_BASE_CHECK(self, q, object_check)

def _S6_BASE_CHECK(self, q, object_check=True):
    fcl = self.fcl
    poses = self.robot.fk(q)
    bounds = np.asarray([[np.min(v @ poses[n][:3, :3].T + poses[n][:3, 3], axis=0), np.max(v @ poses[n][:3, :3].T + poses[n][:3, 3], axis=0)] for n in self.names for v in (self.hulls[n],)])
    a, b = (self.pairs[:, 0], self.pairs[:, 1])
    overlap = np.all(bounds[a, 1] >= bounds[b, 0], axis=1) & np.all(bounds[b, 1] >= bounds[a, 0], axis=1)
    objects = {}

    def obj(i):
        if i not in objects:
            pose = poses[self.names[i]]
            objects[i] = fcl.CollisionObject(self.models[self.names[i]], fcl.Transform(pose[:3, :3], pose[:3, 3]))
        return objects[i]
    self_hits = []
    for i, j in self.pairs[overlap]:
        if fcl.collide(obj(i), obj(j), fcl.CollisionRequest(), fcl.CollisionResult()):
            self_hits.append([self.names[i], self.names[j]])
    object_hits = []
    if object_check:
        overlap = np.all(bounds[:, 1] >= self.object_bounds[0], axis=1) & np.all(bounds[:, 0] <= self.object_bounds[1], axis=1)
        for i in np.flatnonzero(overlap):
            if fcl.collide(obj(i), self.object, fcl.CollisionRequest(), fcl.CollisionResult()):
                object_hits.append(self.names[i])
    self.nodes_checked += 1
    clearance = float(np.min(bounds[:, 0, 2]) - self.table_z)
    return {'table_clearance_m': clearance, 'table_clearance_link': self.names[int(np.argmin(bounds[:, 0, 2]))], 'thresholds': {'table_clearance_m': 0.00065, 'self_overlap_m': 0.0, 'object_overlap_m': 0.0}, 'self_collisions': self_hits, 'object_collisions': object_hits, 'passed': bool(clearance >= 0.00065 and (not self_hits) and (not object_hits))}

def _s2v_settings(static):
    if isinstance(static, dict):
        return dict(static)
    return dict(calibration=static.calibration, color=static.info['color'])

def _s2v_groups(robot, calibration, settings, count):
    parents = {}
    prismatic = set()
    for joint in getattr(robot, 'joints', []):
        if isinstance(joint, tuple) and len(joint) >= 6:
            parent, child, _, _, kind, index = joint[:6]
            parents[child] = (parent, index)
            if index is not None and kind == 'prismatic':
                prismatic.add(int(index))
    chains = {}
    for name, row in calibration.items():
        indices, seen = (set(), set())
        link = row.get('camera_link')
        while link in parents and link not in seen:
            seen.add(link)
            link, index = parents[link]
            if index is not None:
                indices.add(int(index))
        chains[name] = indices
    nonempty = [indices for indices in chains.values() if indices]
    common = set.intersection(*nonempty) if nonempty else set()
    wrist = sorted(settings.get('wrist_joint_indices', common))
    groups = {}
    for name, row in calibration.items():
        finger = str(row.get('finger_module', name))
        groups.setdefault(finger, set()).update(chains[name] - set(wrist))
    if 'finger_joint_indices' in settings:
        groups = {name: set(indices) for name, indices in settings['finger_joint_indices'].items()}
    groups = {name: sorted((index for index in indices if 0 <= index < count)) for name, indices in groups.items()}
    return (groups, wrist, sorted(prismatic))

def _s2v_read_rgb(record):
    image = np.asarray(record['rgb']) if 'rgb' in record else np.load(record['rgb_path'], allow_pickle=False)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('FingerEye requires native uint8 RGB, not depth/RGBA')
    if record.get('width', image.shape[1]) != image.shape[1] or record.get('height', image.shape[0]) != image.shape[0]:
        raise ValueError('RGB dimensions disagree with the native record')
    return image

def _s2v_calibration(row):
    result = dict(row)
    if result.get('projection_type', 'fisheyePolynomial') == 'fisheyePolynomial':
        radius = np.linspace(0, np.hypot(row['fisheye_nominal_width'], row['fisheye_nominal_height']), 2048)
        angle = np.polynomial.polynomial.polyval(radius, row['fisheye_polynomial'])
        increasing = np.r_[True, np.diff(angle) > 0]
        angle, radius = (angle[increasing], radius[increasing])
        if len(angle) < 2 or not np.isfinite(angle).all() or np.any(np.diff(angle) <= 0):
            raise ValueError('Noninvertible fisheye calibration')
        result['_angles'], result['_radii'] = (angle, radius)
    elif result['projection_type'] != 'pinhole':
        raise ValueError('Unsupported calibrated camera projection')
    return result

def _s2v_motor(before, after, command):
    measured = after['actual_q'] - before['actual_q']
    result = dict(measured_delta=measured.tolist(), commanded_delta=None, tracking_residual=None, fingers={}, force_observable=False)
    if command is None:
        result['uncertainty'] = 'own command delta unavailable; no tracking/compliance inference'
        return result
    command = np.asarray(command, dtype=float)
    if command.shape != measured.shape or not np.isfinite(command).all():
        raise ValueError('commanded_delta must match raw actual_q and be finite')
    residual = command - measured
    result.update(commanded_delta=command.tolist(), tracking_residual=residual.tolist())
    for finger, indices in before['groups'].items():
        active = [index for index in indices if abs(command[index]) > 1e-06]
        row = dict(joint_indices=indices, active_joint_indices=active, tracking_ratio=None, drive_deficit=None, settled=None)
        if active:
            denominator = float(command[active] @ command[active])
            ratio = float(measured[active] @ command[active] / denominator)
            speed = float(np.max(np.abs(after['actual_qvel'][active])))
            row.update(tracking_ratio=ratio, drive_deficit=1.0 - ratio, settled=speed < 0.06, max_abs_qvel=speed)
        result['fingers'][finger] = row
    result['uncertainty'] = 'tracking deficit is a compliance/servo-lag proxy, not contact force; command starts at before.actual_q'
    return result

def _s2v_response(before, after, bounds):
    source, target, detected = _s2v_tracks(before, after)
    result = dict(track_count=len(source), detected_tracks=detected, registered=False, signed_response_px=[None, None], response_interval_px=None, confidence=0.0, motion_evidence=False)
    if len(source) < 4:
        result['uncertainty'] = 'insufficient_tracked_texture; silhouette motion is not object motion'
        return result
    rays = _s2v_rays(source, before['calibration'], before['size'])
    distances = np.geomspace(*bounds, 33)
    world = rays[:, None, :] * distances[None, :, None]
    world = world @ before['camera'][:3, :3].T + before['camera'][:3, 3]
    local = (world - after['camera'][:3, 3]) @ after['camera'][:3, :3]
    predicted, valid = _s2v_project(local, after['calibration'], after['size'])
    if not np.all(valid):
        result['uncertainty'] = 'unknown-depth static projections leave calibrated FOV'
        return result
    residual = target[:, None, :] - predicted
    sampling_pad = np.max(np.abs(np.diff(predicted, axis=1)), axis=(0, 1))
    lower = np.median(residual.min(axis=1), axis=0) - 1.25 - sampling_pad
    upper = np.median(residual.max(axis=1), axis=0) + 1.25 + sampling_pad
    signed = np.where(lower > 0, lower, np.where(upper < 0, upper, 0.0))
    spread = np.median(np.linalg.norm(target - source - np.median(target - source, axis=0), axis=1))
    reliability = min(1.0, len(source) / 12.0) * min(1.0, len(source) / max(detected, 1))
    reliability *= float(np.exp(-spread / 3.0))
    amplitude = float(np.linalg.norm(signed))
    result.update(registered=True, signed_response_px=signed.tolist(), observed_flow_px=np.median(target - source, axis=0).tolist(), response_interval_px=[lower.tolist(), upper.tolist()], camera_translation_m=float(np.linalg.norm(after['camera'][:3, 3] - before['camera'][:3, 3])), track_dispersion_px=float(spread), confidence=float(reliability * min(1.0, amplitude / 2.0)) if amplitude else float(0.5 * reliability), motion_evidence=bool(amplitude > 1.0 and reliability >= 0.35), uncertainty='conditional on color association, LK correspondences, calibration and stated depth bounds')
    return result

def _s3s_calibration(row):
    result = dict(row)
    if row.get('projection_type', 'fisheyePolynomial') == 'fisheyePolynomial':
        radius = np.linspace(0.0, np.hypot(row['fisheye_nominal_width'], row['fisheye_nominal_height']), 4096)
        angle = np.polynomial.polynomial.polyval(radius, row['fisheye_polynomial'])
        increasing = np.r_[True, np.diff(angle) > 0]
        angle, radius = (angle[increasing], radius[increasing])
        if len(angle) < 2 or not np.isfinite(angle).all() or np.any(np.diff(angle) <= 0):
            raise ValueError('Invalid fisheye calibration')
        result['_angles'], result['_radii'] = (angle, radius)
    elif row['projection_type'] != 'pinhole':
        raise ValueError('Unsupported calibrated camera projection')
    return result

def _s3s_rotation(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))

def _s3s_texture_rotation(a, b):
    """Track foreground texture only; return None when rotation is unobservable."""
    points = cv2.goodFeaturesToTrack(a['gray'], maxCorners=60, qualityLevel=0.025, minDistance=4, mask=cv2.erode(a['mask'], np.ones((3, 3), np.uint8)) * 255)
    if points is None or len(points) < 6:
        return (None, 0)
    opts = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))
    forward, ok, _ = cv2.calcOpticalFlowPyrLK(a['gray'], b['gray'], points, None, **opts)
    if forward is None:
        return (None, 0)
    backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(b['gray'], a['gray'], forward, None, **opts)
    if backward is None:
        return (None, 0)
    p, q = (points[:, 0], forward[:, 0])
    keep = ok[:, 0].astype(bool) & back_ok[:, 0].astype(bool)
    keep &= np.linalg.norm(backward[:, 0] - p, axis=1) < 0.75
    xy = np.rint(np.nan_to_num(q)).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < b['size'][0]) & (xy[:, 1] >= 0) & (xy[:, 1] < b['size'][1])
    ids = np.flatnonzero(inside)
    inside[ids] &= b['mask'][xy[ids, 1], xy[ids, 0]].astype(bool)
    keep &= inside & np.isfinite(q).all(1)
    if keep.sum() < 6 or np.linalg.norm(np.ptp(p[keep], axis=0)) < 8:
        return (None, int(keep.sum()))
    affine, inlier = cv2.estimateAffinePartial2D(p[keep], q[keep], method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=200)
    if affine is None or inlier.sum() < 6 or inlier.mean() < 0.7:
        return (None, int(keep.sum()))
    return (float(np.arctan2(affine[1, 0], affine[0, 0])), int(inlier.sum()))

def _s3s_hypothesis(a, b, before_palm, after_palm, depths, hypothesis):
    ray = _s3s_ray(a['centroid_px'], a['calibration'], a['size'])
    world = ray[None] * depths[:, None] @ a['camera'][:3, :3].T + a['camera'][:3, 3]
    if hypothesis == 'carried':
        relative = (world - before_palm[:3, 3]) @ before_palm[:3, :3]
        world = relative @ after_palm[:3, :3].T + after_palm[:3, 3]
    local = (world - b['camera'][:3, 3]) @ b['camera'][:3, :3]
    uv, valid = _s3s_project(local, b['calibration'], b['size'])
    predicted = uv[valid] - a['centroid_px']
    if not len(predicted):
        return dict(possible=None, interval_px=None)
    observed = b['centroid_px'] - a['centroid_px']
    lo, hi = (predicted.min(0) - 2.0, predicted.max(0) + 2.0)
    return dict(possible=bool(np.all((observed >= lo) & (observed <= hi))), interval_px=[lo.tolist(), hi.tolist()], valid_depth_fraction=float(valid.mean()), best_sample_error_px=float(np.linalg.norm(predicted - observed, axis=1).min()))

def _S3_APPROACH(robot, q, points, inward, initial=None):
    try:
        from joint_path import plan_finger_rrt_connect
    except ImportError:
        from mainline.joint_path import plan_finger_rrt_connect
    initial = np.asarray(_ACTIVE_DESCRIPTION['initial_q']) if initial is None else np.array(initial)
    checker=robot.ik_checker
    raised=initial.copy();raised[2]=.35
    robot.approach_search=[];fallback=None
    for opening in (.006,.003,.0015):
        opened=close_ik(robot,q,points-inward*opening)
        high=opened.copy();high[2]=.35
        near=opened.copy();near[2]+=.045
        report=dict(opening_m=opening,opened_endpoint=checker.check(opened))
        robot.approach_search.append(report)
        # Keep a conventional failed path for the original diagnostic gate.
        if fallback is None:
            fallback=[]
            for a,b,n,label in ((initial,raised,120,'raise'),(raised,high,600,'orient'),(high,near,360,'translate'),(near,opened,360,'descend')):
                fallback+=smooth_segment(a,b,n,label)
            for amount in np.linspace(-opening,.0005,30):
                fallback.append([close_ik(robot,q,points+inward*amount).tolist(),24,'close'])
            fallback.append([fallback[-1][0],240,'probe_hold'])
        if not report['opened_endpoint']['passed']:continue
        finger_goal=raised.copy();finger_goal[robot.all_finger_indices]=opened[robot.all_finger_indices]
        route,stats=plan_finger_rrt_connect(robot,checker,raised,finger_goal,
            seed=17001,max_extensions=1000,max_collision_checks=5000,object_check=True)
        report['joint_path']=stats
        if route is None:continue
        acts=smooth_segment(initial,raised,120,'raise');cur=raised.copy()
        for finger_q in route[1:]:
            nxt=cur.copy();nxt[robot.all_finger_indices]=finger_q
            acts+=smooth_segment(cur,nxt,12,'route_fingers');cur=nxt
        for a,b,n,label in ((cur,high,600,'orient'),(high,near,360,'translate'),(near,opened,360,'descend')):
            acts+=smooth_segment(a,b,n,label)
        for amount in np.linspace(-opening,.0005,30):
            acts.append([close_ik(robot,q,points+inward*amount).tolist(),24,'close'])
        acts.append([acts[-1][0],240,'probe_hold'])
        report['full_path']=_s12_path_gate(robot,checker,initial,acts)
        if report['full_path']['passed']:
            report['selected']=True
            return acts
    return fallback

class _geometry_Stage3LiveVolume(_S3_VOLUME):
    """Current six-view visual hull; initial RGB supplies COLOR only.

    A finite outer hull is not a measured contact surface. Missing/robot-hidden
    pixels stay unknown; a second camera on one digit is not an opposing view.
    Never fuse masks across a moving release, or clip by the initial silhouette.
    """

    def __init__(self, robot, calibration, info, resolution=0.001):
        super().__init__(robot, calibration, info, resolution)
        center = np.asarray(info['center'])[:2]
        radius = 0.045
        table = float(info['table_z'])
        self.axes = [np.arange(v - radius, v + radius + resolution / 2, resolution) for v in center]
        self.axes += [np.arange(table + 0.0005, table + 0.0851, resolution)]
        self.lower = np.array([a[0] for a in self.axes])
        self.shape = tuple((len(a) for a in self.axes))
        self.grid = np.stack(np.meshgrid(*self.axes, indexing='ij'), -1).reshape(-1, 3)

def _s9_frames(self, obs):
    obs=s9_exposure(obs)
    poses = self.robot.fk(np.asarray(obs['actual_q'], dtype=float))
    cameras, frames = ({}, {})
    self.slot_report = {}
    for name in CAMERAS:
        slot = obs.get('fingereye', {}).get(name)
        self.slot_report[name] = {'present': slot is not None, 'usable': False}
        if slot is None or name not in self.calibration:
            self.slot_report[name]['reason'] = 'missing_slot_or_static_calibration'
            continue
        capture_step = slot.get('capture_step')
        self.slot_report[name].update(capture_step=capture_step, frame_id=slot.get('frame_id'), camera_link=self.calibration[name]['camera_link'], frame_age_steps=None if capture_step is None else int(obs['step'] - capture_step), capture_timestamp_available=capture_step is not None, pose_at_capture_measured=False, camera_pose_world_4x4=poses[self.calibration[name]['camera_link']].tolist())
        if capture_step is not None and (capture_step > obs['step'] or obs['step'] - capture_step > 24):
            self.slot_report[name]['reason'] = 'stale_or_future_frame'
            continue
        rgb = np.load(slot['rgb_path'], allow_pickle=False)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            continue
        rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        d = abs(hsv[:, :, 0].astype(float) - self.hue)
        mask = color_foreground(rgb, self.hue).astype('uint8')
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        ids = [i for i in range(1, count) if stats[i, 4] >= 5]
        if not ids:
            continue
        mask = np.isin(labels, ids)
        camera = poses[self.calibration[name]['camera_link']]
        cameras[name] = (camera, self.rays[name])
        frames[name] = {'mask': mask, 'reliable': color_reliability(rgb), 'sdf': distance_transform_edt(~mask) - distance_transform_edt(mask), 'camera': camera}
        self.slot_report[name].update(usable=True, pixels=int(mask.sum()))
    depths = self.visibility.depth(poses, cameras) if cameras else {}
    for name, frame in frames.items():
        frame['robot_depth'] = depths[name].reshape(self.size[::-1])
    return (poses, frames)

def _vector(node, name, default='0 0 0'):
    return np.fromstring(node.get(name, default), sep=' ')

def _origin(node):
    result = np.eye(4)
    if node is not None:
        result[:3, :3] = Rotation.from_euler('xyz', _vector(node, 'rpy')).as_matrix()
        result[:3, 3] = _vector(node, 'xyz')
    return result

def resolve_mesh(urdf_path, filename):
    """Resolve ordinary paths and local ROS package exports; never fetch assets."""
    urdf_path = Path(urdf_path).resolve()
    if filename.startswith('package://'):
        package, relative = filename[len('package://'):].split('/', 1)
        for parent in urdf_path.parents:
            if parent.name == package:
                return parent / relative
            candidate = parent / package / relative
            if candidate.exists():
                return candidate
        return urdf_path.parent.parent / relative
    return (urdf_path.parent / filename).resolve()

def _s2v_tracks(before, after):
    mask = cv2.erode(before['mask'], np.ones((3, 3), np.uint8)) * 255
    points = cv2.goodFeaturesToTrack(before['gray'], maxCorners=36, qualityLevel=0.025, minDistance=4, mask=mask, blockSize=3)
    if points is None:
        return (np.empty((0, 2)), np.empty((0, 2)), 0)
    options = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))
    forward, status, error = cv2.calcOpticalFlowPyrLK(before['gray'], after['gray'], points, None, **options)
    if forward is None:
        return (np.empty((0, 2)), np.empty((0, 2)), len(points))
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(after['gray'], before['gray'], forward, None, **options)
    if backward is None:
        return (np.empty((0, 2)), np.empty((0, 2)), len(points))
    source, target = (points.reshape(-1, 2), forward.reshape(-1, 2))
    finite = np.isfinite(target).all(axis=1)
    nearest = np.rint(np.where(finite[:, None], target, 0)).astype(int)
    inside = finite & (nearest[:, 0] >= 0) & (nearest[:, 0] < after['size'][0])
    inside &= (nearest[:, 1] >= 0) & (nearest[:, 1] < after['size'][1])
    foreground = np.zeros(len(source), dtype=bool)
    foreground[inside] = after['mask'][nearest[inside, 1], nearest[inside, 0]] > 0
    valid = (status.ravel() > 0) & (status_back.ravel() > 0) & foreground
    valid &= np.linalg.norm(backward.reshape(-1, 2) - source, axis=1) < 0.8
    valid &= error.ravel() < 20
    return (source[valid], target[valid], len(points))

def _s2v_rays(uv, row, size):
    if row.get('projection_type', 'fisheyePolynomial') == 'pinhole':
        rays = np.column_stack(((uv[:, 0] - row['cx']) / row['fx'], -(uv[:, 1] - row['cy']) / row['fy'], -np.ones(len(uv))))
        return rays / np.linalg.norm(rays, axis=1)[:, None]
    xy = (uv + 0.5) * [row['fisheye_nominal_width'] / size[0], row['fisheye_nominal_height'] / size[1]] - [row['fisheye_optical_centre_x'], row['fisheye_optical_centre_y']]
    radius = np.linalg.norm(xy, axis=1)
    angle = np.polynomial.polynomial.polyval(radius, row['fisheye_polynomial'])
    rays = np.column_stack((np.sin(angle)[:, None] * xy / np.maximum(radius[:, None], 1e-12), -np.cos(angle)))
    rays[:, 1] *= -1
    return rays

def _s2v_project(local, row, size):
    distance = np.linalg.norm(local, axis=-1)
    if row.get('projection_type', 'fisheyePolynomial') == 'pinhole':
        depth = -local[..., 2]
        uv = np.stack((row['cx'] + row['fx'] * local[..., 0] / np.maximum(depth, 1e-12), row['cy'] - row['fy'] * local[..., 1] / np.maximum(depth, 1e-12)), axis=-1)
        valid = depth > 1e-08
    else:
        radius = np.linalg.norm(local[..., :2], axis=-1)
        angle = np.arctan2(radius, -local[..., 2])
        rho = np.interp(angle, row['_angles'], row['_radii'])
        uv = local[..., :2] / np.maximum(radius[..., None], 1e-12) * rho[..., None]
        uv[..., 1] *= -1
        uv += [row['fisheye_optical_centre_x'], row['fisheye_optical_centre_y']]
        uv *= [size[0] / row['fisheye_nominal_width'], size[1] / row['fisheye_nominal_height']]
        uv -= 0.5
        valid = angle < np.deg2rad(row['fisheye_max_fov'] / 2)
    valid &= (distance > 1e-08) & np.isfinite(uv).all(axis=-1)
    return (uv, valid)

def _s3s_ray(uv, row, size):
    if row.get('projection_type', 'fisheyePolynomial') == 'pinhole':
        ray = np.array([(uv[0] - row['cx']) / row['fx'], -(uv[1] - row['cy']) / row['fy'], -1.0])
        return ray / np.linalg.norm(ray)
    xy = (uv + 0.5) * [row['fisheye_nominal_width'] / size[0], row['fisheye_nominal_height'] / size[1]] - [row['fisheye_optical_centre_x'], row['fisheye_optical_centre_y']]
    radius = np.linalg.norm(xy)
    angle = np.polynomial.polynomial.polyval(radius, row['fisheye_polynomial'])
    ray = np.r_[np.sin(angle) * xy / max(radius, 1e-12), -np.cos(angle)]
    ray[1] *= -1
    return ray

def _s3s_project(local, row, size):
    if row.get('projection_type', 'fisheyePolynomial') == 'pinhole':
        depth = -local[:, 2]
        uv = np.c_[row['cx'] + row['fx'] * local[:, 0] / np.maximum(depth, 1e-12), row['cy'] - row['fy'] * local[:, 1] / np.maximum(depth, 1e-12)]
        valid = depth > 1e-08
    else:
        radius = np.linalg.norm(local[:, :2], axis=1)
        angle = np.arctan2(radius, -local[:, 2])
        rho = np.interp(angle, row['_angles'], row['_radii'])
        uv = local[:, :2] / np.maximum(radius[:, None], 1e-12) * rho[:, None]
        uv[:, 1] *= -1
        uv += [row['fisheye_optical_centre_x'], row['fisheye_optical_centre_y']]
        uv *= [size[0] / row['fisheye_nominal_width'], size[1] / row['fisheye_nominal_height']]
        uv -= 0.5
        valid = angle < np.deg2rad(row['fisheye_max_fov'] / 2)
    return (uv, valid & np.isfinite(uv).all(axis=1))

class SelfAwareGeometry(_R13_CHECKER):
    """Reuse full-mesh BVHs and derive only structural assembly exclusions."""

    def __init__(self, robot, info):
        super().__init__(robot, info)
        self.envelope_source = 'legacy_box'
        if 'collision_points' not in info:
            return
        import fcl, trimesh
        points = np.asarray(info['collision_points'])
        h = ConvexHull(points)
        planes = np.unique(np.round(h.equations, 12), axis=0)
        planes[:, 3] += 0.00065
        fit = linprog([0, 0, 0, -1], A_ub=np.c_[planes[:, :3], np.ones(len(planes))], b_ub=-planes[:, 3], bounds=[(None, None)] * 3 + [(0, None)], method='highs')
        if not fit.success or fit.x[3] < 1e-06:
            return
        verts = HalfspaceIntersection(planes, fit.x[:3]).intersections
        mesh = trimesh.convex.convex_hull(verts)
        b = fcl.BVHModel()
        b.beginModel(len(mesh.vertices), len(mesh.faces))
        b.addSubModel(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32))
        b.endModel()
        self.object = fcl.CollisionObject(b)
        self.object_bounds = mesh.bounds
        self.envelope_source = 'RGB_convex_envelope'

    def optimization_clearance(self, poses, object_check=True):
        """Metric source-mesh penalties; exact boolean gates remain decisive."""
        vertices=[self.hulls[name]@poses[name][:3,:3].T+poses[name][:3,3] for name in self.names]
        lower=np.array([v.min(0) for v in vertices]);upper=np.array([v.max(0) for v in vertices]);objects={}
        def obj(i):
            if i not in objects:
                pose=poses[self.names[i]]
                objects[i]=self.fcl.CollisionObject(self.models[self.names[i]],self.fcl.Transform(pose[:3,:3],pose[:3,3]))
            return objects[i]
        def overlap(a,b):
            result=self.fcl.CollisionResult()
            hit=self.fcl.collide(a,b,self.fcl.CollisionRequest(enable_contact=True,num_max_contacts=16),result)
            return max([0.]+[float(c.penetration_depth) for c in result.contacts])+.0001 if hit else 0.
        pair_loss=np.zeros(len(self.pairs))
        near=np.all(upper[self.pairs[:,0]]>=lower[self.pairs[:,1]],axis=1)&np.all(upper[self.pairs[:,1]]>=lower[self.pairs[:,0]],axis=1)
        for k in np.flatnonzero(near):
            i,j=self.pairs[k];pair_loss[k]=overlap(obj(i),obj(j))
        object_loss=np.zeros(len(vertices))
        if object_check:
            near=np.all(upper>=self.object_bounds[0],axis=1)&np.all(lower<=self.object_bounds[1],axis=1)
            for i in np.flatnonzero(near):object_loss[i]=overlap(obj(i),self.object)
        return np.r_[np.minimum(lower[:,2]-self.table_z-.001,0),pair_loss,object_loss]

    def check(self, q, object_check=True):
        cache = self.__dict__.setdefault('_s6_exact_checks', {})
        key = (np.asarray(q, dtype=float).tobytes(), bool(object_check))
        if key not in cache:
            cache[key] = _S6_BASE_CHECK(self, q, object_check)
        return cache[key]

    def diagnose(self, q, object_check=True):
        """Metric witnesses for the original triangle/clearance guard.

        This is called at candidate endpoints and the first refused path node,
        never used as an approximate replacement for ``check``. Intersecting
        triangle meshes can report zero penetration depth: the exact overlap
        boolean, rather than that depth, remains the rejection criterion.
        """
        report = dict(self.check(q, object_check))
        fcl, poses = (self.fcl, self.robot.fk(q))
        objects = {name: fcl.CollisionObject(self.models[name], fcl.Transform(poses[name][:3, :3], poses[name][:3, 3])) for name in self.names}
        witnesses = [dict(kind='table', links=[report['table_clearance_link'], 'table'], clearance_m=report['table_clearance_m'], threshold_m=0.00065, passed=report['table_clearance_m'] >= 0.00065, measurement='transformed full collision-mesh minimum z')]
        pairs = [('self_collision', pair) for pair in report['self_collisions']]
        pairs += [('object_collision', [name, 'RGB_envelope']) for name in report['object_collisions']]
        for kind, pair in pairs:
            other = self.object if kind == 'object_collision' else objects[pair[1]]
            result = fcl.CollisionResult()
            fcl.collide(objects[pair[0]], other, fcl.CollisionRequest(enable_contact=True, num_max_contacts=32), result)
            depths = [float(contact.penetration_depth) for contact in result.contacts]
            witnesses.append(dict(kind=kind, links=list(pair), clearance_m=-max([0.0] + depths), threshold_m=0.0, passed=False, exact_mesh_overlap=True, contacts_returned=len(depths), measurement='FCL triangle contact penetration; zero depth can still overlap'))
        report['clearance_witnesses'] = witnesses
        report['first_failed_guard'] = next((item for item in witnesses if not item['passed']), None)
        return report

class Stage3LiveVolume(_geometry_Stage3LiveVolume):

    def carve(self, obs, fuse=False):
        info, report = fit_shape(self, s9_exposure(obs), _S14_SHAPE_PRIOR)
        if info is None:
            self.points = None
            self.boundary = None
            self.tree = None
        return report

    def frames(self, obs):
        return _s9_frames(self, s9_exposure(obs))

def _s12_geometry_detail(checker, q, check=None, object_check=True):
    check = checker.check(q, object_check=object_check) if check is None else check
    if hasattr(checker, 'diagnose'):
        return checker.diagnose(q, object_check=object_check)
    detail = dict(check)
    detail.setdefault('thresholds', dict(table_clearance_m=0.00065, self_overlap_m=0.0, object_overlap_m=0.0))
    return detail

def _s12_endpoint_gate(robot, q, meta, checker, initial, refinement=False):
    """One ordered feasibility audit for two and three contacts."""
    checks = []

    def gate(name, passed, detail):
        checks.append(dict(gate=name, passed=bool(passed), **detail))
    gate('public_contact_patch_bounds', meta.get('region_patch_valid', True), dict(source='RGB convex-envelope surface hypothesis', regions=meta.get('contact_regions'), measurements=meta.get('region_patch_measurements')))
    fingers = list(robot.finger_names)
    errors = np.asarray(meta.get('per_finger_error_m', [meta['error_m']] * len(fingers)))
    normals = np.asarray(meta['normal_error_rad'])
    gate('per_finger_position', np.all(errors < 0.0008), dict(unit='m', threshold=0.0008, comparison='<', per_finger=[dict(finger=finger, link=robot.contact_links[i], value=float(value)) for i, (finger, value) in enumerate(zip(fingers, errors))]))
    gate('per_finger_normal', np.all(normals < 0.5), dict(unit='rad', threshold=0.5, comparison='<', per_finger=[dict(finger=finger, link=robot.contact_links[i], value=float(value)) for i, (finger, value) in enumerate(zip(fingers, normals))]))
    bad = np.flatnonzero((q < robot.lo) | (q > robot.hi))
    gate('joint_limits', not len(bad), dict(joints=[dict(index=int(i), name=NAMES[i], value=float(q[i]), lower=float(robot.lo[i]), upper=float(robot.hi[i])) for i in bad]))
    endpoint = checker.check(q)
    meta['endpoint'] = _s12_geometry_detail(checker, q, endpoint)
    gate('endpoint_clearance_collision', endpoint['passed'], dict(geometry=meta['endpoint']))
    if refinement:
        angle = _s3_attitude_delta(np.asarray(initial)[3:6], q[3:6])
        meta['refinement_attitude_delta_rad'] = angle
        gate('large_near_object_refinement_attitude_change', angle <= STAGE3_REFINEMENT_ATTITUDE_LIMIT_RAD, dict(value=angle, threshold=STAGE3_REFINEMENT_ATTITUDE_LIMIT_RAD, unit='rad'))
    meta['feasibility_gates'] = checks
    failure = next((check for check in checks if not check['passed']), None)
    if failure is not None:
        meta.update(stage3_reject=failure['gate'], first_rejection=failure['gate'], first_rejection_detail=failure, feasibility_passed=False)
    return failure is None

def _s12_path_gate(robot, checker, initial, acts, *, object_check=True, rate_gate=True):
    """Audit each existing command-path node in temporal order, with witnesses.

    The interpolation schedule and native 120Hz denominator are unchanged.
    A close/probe intentionally retains the original object-overlap exemption;
    self, table, joint and velocity guards still apply there.
    """
    total_nodes = len(acts)
    checks, failures, previous = ([], [], np.asarray(initial))
    peak_translation, peak_joint = (0.0, 0.0)
    for index, (target, steps, phase) in enumerate(acts):
        q = np.asarray(target)
        inspect_object = bool(object_check and phase not in ('close', 'probe_hold'))
        geometry = checker.check(q, object_check=inspect_object)
        rate = abs(q - previous) / (steps / 120.0)
        peak_translation = max(peak_translation, float(rate[:3].max()))
        peak_joint = max(peak_joint, float(rate[3:].max()))
        bad = np.flatnonzero((q < robot.lo) | (q > robot.hi))
        failure = None
        if not geometry['passed']:
            failure = dict(gate='clearance_collision', geometry=geometry if failures else _s12_geometry_detail(checker, q, geometry, object_check=inspect_object))
        elif len(bad):
            failure = dict(gate='joint_limits', joints=[dict(index=int(i), name=NAMES[i], value=float(q[i]), lower=float(robot.lo[i]), upper=float(robot.hi[i])) for i in bad])
        elif rate_gate and (rate[:3].max() >= 0.15 or rate[3:].max() >= 1.5):
            failure = dict(gate='commanded_velocity', peak_translation_m_s=float(rate[:3].max()), peak_joint_rad_s=float(rate[3:].max()), translation_threshold_m_s=0.15, joint_threshold_rad_s=1.5, comparison='<')
        if failure is not None:
            failures.append(dict(node=index, phase=phase, steps=int(steps), object_guard_enabled=inspect_object, commanded_q=q.tolist(), **failure))
        checks.append(geometry)
        previous = q
        # Refusal is decisive; an accepted path still checks every node.
        if failure is not None:
            break
    audit_complete = len(checks) == total_nodes
    return dict(passed=not failures and audit_complete, audit_complete=audit_complete, total_nodes=total_nodes, checked_nodes=len(checks), first_rejection=failures[0] if failures else None, failed_path_nodes=[failure['node'] for failure in failures], min_table_clearance_m=min((check['table_clearance_m'] for check in checks), default=float('inf')), commanded_peak_translation_m_s=peak_translation, commanded_peak_joint_rad_s=peak_joint, native_rate_hz=120, object_guard_scope='original close/probe exemption retained')

def _s12_complete_candidate(robot, checker, initial, q, inward, meta, retime):
    points = robot.tips(q)[0]
    prefix = stage3_approach(robot, q, points, inward, initial)
    if retime:
        prefix = _s3_retime_approach(prefix, initial)
    path = _s12_path_gate(robot, checker, initial, prefix)
    meta['approach_search']=getattr(robot,'approach_search',[])
    meta.update(path_passed=path['passed'], path_audit=path, failed_path_nodes=path['failed_path_nodes'], commanded_peak_translation_m_s=path['commanded_peak_translation_m_s'], commanded_peak_joint_rad_s=path['commanded_peak_joint_rad_s'], first_wrist_change_phase=next((phase for target, _, phase in prefix if _s3_attitude_delta(np.asarray(initial)[3:6], np.asarray(target)[3:6]) > 0.02), None))
    if not path['passed']:
        meta.update(stage3_reject='approach_close_path', first_rejection='approach_close_path', first_rejection_detail=path['first_rejection'], feasibility_passed=False)
        return None
    modes, audits = ([], {})
    for mode in MODES:
        tail = branch(robot, q, points, inward, prefix, mode)[len(prefix):]
        audit = _s12_path_gate(robot, checker, prefix[-1][0] if prefix else initial, tail, object_check=False, rate_gate=False)
        audits[mode] = audit
        if audit['passed']:
            modes.append(mode)
    meta['lift_tail_audits'] = audits
    if not modes:
        meta.update(stage3_reject='original_lift_tail_modes', first_rejection='original_lift_tail_modes', first_rejection_detail=dict(gate='original_lift_tail_modes', modes=audits), feasibility_passed=False)
        return None
    meta.update(feasibility_passed=True, verified_min_table_clearance_m=min([path['min_table_clearance_m']] + [audits[mode]['min_table_clearance_m'] for mode in modes]), feasibility_scope='public surface; individual IK; exact mesh endpoint; all original approach/close and lift nodes')
    candidate = dict(q=q, points=points, inward=inward, prefix=prefix, meta=meta, modes=modes)
    _s12_candidate_identity(candidate)
    return candidate

def _s12_candidate_identity(candidate):
    """Bind a plan to its geometry and commands, independent of ranking index."""
    import hashlib
    payload = dict(q=np.asarray(candidate['q']).tolist(), points=np.asarray(candidate['points']).tolist(), inward=np.asarray(candidate['inward']).tolist(), targets=candidate['meta'].get('targets'), prefix=[[np.asarray(q).tolist(), int(steps), phase] for q, steps, phase in candidate['prefix']], modes=list(candidate.get('modes', ())))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    candidate['meta'].update(planner_candidate_id='contacts' + str(len(candidate['points'])) + '-' + digest[:20], command_plan_sha256=digest)
    return digest

def _s12_rgb_contact_regions(info, axis, height_fractions, tangent_angles):
    """Ray/facet intersections of the public RGB envelope, not object truth.

    The initial overhead height is a hypothesis; only live FingerEye geometry
    warrants a visibility claim. Surface normals are piecewise hull normals.
    """
    cloud = np.asarray(info['collision_points'], dtype=float)
    hull = ConvexHull(cloud)
    planes = hull.equations
    center = np.asarray(info['center'], dtype=float).copy()
    axis = np.asarray(axis, dtype=float)
    tangent = np.cross([0.0, 0.0, 1.0], axis)
    table = float(info['table_z'])
    top = float(cloud[:, 2].max())
    regions = []
    for finger, h, angle in zip(('thumb', 'index', 'middle'), height_fractions, tangent_angles):
        sign = -1.0 if finger == 'thumb' else 1.0
        direction = sign * np.cos(angle) * axis + np.sin(angle) * tangent
        origin = center.copy()
        origin[2] = table + max(0.004, h * (top - table))
        denom = planes[:, :3] @ direction
        distance = -(planes[:, :3] @ origin + planes[:, 3]) / np.where(abs(denom) > 1e-12, denom, np.nan)
        eligible = np.flatnonzero((denom > 1e-08) & (distance >= 0))
        if not len(eligible):
            return []
        facet = int(eligible[np.argmin(distance[eligible])])
        normal = planes[facet, :3]
        point = origin + direction * distance[facet] + normal * 0.0003
        regions.append(dict(finger=finger, point_world_m=point.tolist(), outward_normal_world=normal.tolist(), visibility='RGB envelope hypothesis; side visibility not certified', position_uncertainty_m=0.0015, normal_uncertainty_rad=0.15, height_fraction=float(h), angular_offset_rad=float(angle), facet=facet, height_source='public RGB envelope; initial overhead height may be footprint prior'))
    return regions

def axis_contact_candidates(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, retime=False, refinement=False):
    results, rejected = ([], [])
    for axis_index, axis in enumerate(axes):
        axis = np.asarray(axis, dtype=float).copy()
        if axis[0] < 0:
            axis = -axis
        tangent = np.cross([0.0, 0.0, 1.0], axis)
        pts = np.asarray(info['collision_points'])
        projections = pts[:, :2] @ np.array([axis[:2], tangent[:2]]).T
        lo, hi = (projections.min(0), projections.max(0))
        center = axis * ((lo[0] + hi[0]) / 2) + tangent * ((lo[1] + hi[1]) / 2)
        half = (hi - lo) / 2
        rotation = Rotation.from_euler('z', np.arctan2(axis[1], axis[0])).as_matrix()
        seed = np.asarray(prior).copy()
        seed[robot.unused_finger_indices] = initial[robot.unused_finger_indices]
        for h in heights:
            goals = center + np.array([-axis, axis]) * (half[0] + 0.0003) + tangent * shift
            goals[:, 2] = float(info['table_z']) + max(0.004, h * 2 * info['half'][2])
            inward = np.array([axis, -axis])
            q, meta = optimize(robot, seed, goals, inward)
            meta.update(height_fraction=h, along_fraction=shift / max(2 * half[1], 1e-06), stage3_axis_world=axis.tolist(), axis_index=axis_index, stage3_shift_m=shift, geometry_source=geometry_source)
            _s12_candidate_identity(dict(q=q, points=robot.tips(q)[0], inward=inward, prefix=[], meta=meta))
            if not _s12_endpoint_gate(robot, q, meta, checker, initial, refinement):
                rejected.append(meta)
                continue
            candidate = _s12_complete_candidate(robot, checker, initial, q, inward, meta, retime)
            if candidate is not None:
                results.append(candidate)
                if len(results) >= 2:
                    return (results, rejected)
            else:
                rejected.append(meta)
        if results:
            break
    return (results, rejected)

def _s3_retime_approach(acts, initial):
    if not S6_FAST:
        return _S6_V3_RETIME(acts, initial)
    last = np.asarray(initial, dtype=float).copy()
    out, i = ([], 0)
    while i < len(acts):
        phase = acts[i][2]
        j = i
        while j + 1 < len(acts) and acts[j + 1][2] == phase:
            j += 1
        target = np.asarray(acts[j][0], dtype=float)
        if phase in ('close', 'probe_hold') or phase.startswith('route_'):
            out += acts[i:j + 1]
        elif np.max(np.abs(target - last)) > 1e-10:
            near = phase == 'descend'
            n = _s3_steps(target - last, 360 if near else 24, 0.045 if near else 0.1, 0.7 if near else 1.0)
            out += smooth_segment(last, target, n, phase)
        last = target.copy()
        i = j + 1
    return out

def stage3_recover_release(ctx, state, local_probe=None):
    """Finite physical set-down/open/retreat; no stale overhead geometry reuse.

    Unknown object clearance is retained as uncertainty. The conservative
    fallback lowers to4mm above the previous grasp, then opens in1mm increments.
    This cannot certify clearance of an occluded rotated object; host judges it.
    """
    global STAGE3_RELEASE_TICKET
    STAGE3_RELEASE_TICKET = None
    if not all((k in state for k in ('robot', 'checker', 'last', 'c', 'volume', 'info'))):
        _s3_note(ctx, 'stage3_recovery_stop', {'reason': 'no_executable_attempt_state'})
        return None
    mover = _Stage3Mover(ctx, state)
    info = state['newinfo'] if state.get('refined') else state['info']
    c = state['c']
    _s3_note(ctx, 'stage3_recovery_begin', {'version': STAGE3_RECOVERY_VERSION, 'attempt': STAGE3_ATTEMPT, 'step': int(mover.obs['step']), 'actual_base': np.asarray(mover.obs['actual_q'])[:6].tolist(), 'command_base': mover.last[:6].tolist(), 'reorientation_permitted': False, 'current_object_pose_known': False})
    base_z = float(c['q'][2])
    target_z = min(float(mover.last[2]), base_z + 0.004)
    while mover.last[2] > target_z + 1e-08:
        lowered = mover.last.copy()
        lowered[2] = max(target_z, lowered[2] - 0.002)
        if not mover.move(lowered, 48, 's3_lower'):
            return None
    _s3_note(ctx, 'stage3_setdown', {'step': int(mover.obs['step']), 'height_above_previous_grasp_m': float(mover.last[2] - base_z), 'actual_setdown_certified': False, 'wrist_attitude_target_delta': 0.0})
    actual = np.asarray(mover.obs['actual_q'])
    tips = mover.robot.tips(actual)[0]
    before_open_local = _s3_tip_frame(mover.robot, actual)
    base_rotation = Rotation.from_euler('ZYX', actual[3:6]).as_matrix()
    outward_local = -np.asarray(c['inward']) @ base_rotation
    for opening in (0.001, 0.002, 0.003, 0.004, 0.005, 0.006):
        targets = tips - np.asarray(c['inward']) * opening
        opened = close_ik(mover.robot, mover.last, targets)
        opened[:6] = mover.last[:6]
        if not mover.move(opened, 48, 's3_open'):
            return None
        if local_probe is not None and opening >= 0.002 and local_probe(mover, state):
            return state
    opened_actual = np.asarray(mover.obs['actual_q'])
    opening = ((_s3_tip_frame(mover.robot, opened_actual) - before_open_local) * outward_local).sum(1)
    opening_passed = bool(np.all(opening >= 0.003))
    _s3_note(ctx, 'stage3_release', {'step': int(mover.obs['step']), 'finger_opening_target_m': 0.006, 'actual_opening_projected_m': opening.tolist(), 'minimum_measured_opening_m': 0.003, 'nominal_opening_passed': opening_passed, 'wrist_attitude_target_delta': 0.0, 'actual_object_separation_certified': False})
    if not opening_passed:
        _s3_note(ctx, 'stage3_recovery_stop', {'reason': 'insufficient_measured_finger_opening'})
        return None
    opened_step = int(mover.obs['step'])
    opened_base_z = float(opened_actual[2])
    live = Stage3LiveVolume(mover.robot, state['volume'].calibration, info)
    current, report = stage3_live_geometry(live, mover.obs, info)
    last_report = report
    accepted_step = int(mover.obs['step']) if current is not None else None
    retreat_start = mover.last.copy()
    for dz in (0.004, 0.008, 0.012, 0.02, 0.03, 0.045):
        target = retreat_start.copy()
        target[2] += dz
        if not mover.move(target, 48, 's3_retreat'):
            return None
        if not mover.move(mover.last.copy(), 48, 's3_settle'):
            return None
        settle = mover.settle_evidence()
        if not settle['settled']:
            continue
        candidate, newreport = stage3_live_geometry(live, mover.obs, info)
        last_report = newreport
        if candidate is not None:
            current, report, accepted_step = (candidate, newreport, int(mover.obs['step']))
        if candidate is not None and local_probe is not None and local_probe(mover, state):
            return state
        if candidate is not None and dz >= 0.008:
            break
    accepted = current is not None and accepted_step == int(mover.obs['step'])
    if not accepted:
        for dx, dy in ((-0.006, 0.0), (0.006, 0.0), (0.0, -0.006), (0.0, 0.006), (0.0, 0.0)):
            target = retreat_start.copy()
            target[:3] += [dx, dy, 0.012]
            if not mover.move(target, 48, 's14_release_view'):
                return None
            if not mover.move(mover.last.copy(), 48, 's3_settle'):
                return None
            settle = mover.settle_evidence()
            if not settle['settled']:
                continue
            candidate, newreport = stage3_live_geometry(live, mover.obs, info)
            last_report = newreport
            _s3_note(ctx, 'stage14_release_view', dict(step=int(mover.obs['step']), actual_q=mover.obs['actual_q'], offset_m=[dx, dy, 0.012], accepted=candidate is not None, report=newreport))
            if candidate is not None:
                current, report, accepted_step = (candidate, newreport, int(mover.obs['step']))
                accepted = True
                break
    _s3_note(ctx, 'stage3_relocalize', {'step': int(mover.obs['step']), 'accepted': bool(accepted), 'report': report if accepted else last_report, 'source': 'current_live_fingereye', 'initial_silhouette_used': False})
    if not accepted:
        _s3_note(ctx, 'stage3_recovery_stop', {'reason': 'no_fresh_released_geometry'})
        return None
    settle = mover.settle_evidence()
    measured_retreat = float(np.asarray(mover.obs['actual_q'])[2] - opened_base_z)
    ready = bool(opening_passed and measured_retreat >= 0.006 and settle['settled'] and (mover.max_actual_attitude_delta <= STAGE3_RECOVERY_ATTITUDE_LIMIT_RAD))
    ticket = {'ready': ready, 'source_attempt': STAGE3_ATTEMPT, 'opened_step': opened_step, 'retreat_completed_step': int(mover.obs['step']), 'fresh_map_step': accepted_step, 'actual_retreat_m': measured_retreat, 'minimum_actual_retreat_m': 0.006, 'actual_opening_projected_m': opening.tolist(), 'sampled_max_actual_wrist_attitude_delta_rad': mover.max_actual_attitude_delta, 'settle': settle, 'object_separation_certified': False, 'reorientation_phase': 'released_high_pregrasp_only'}
    _s3_note(ctx, 'stage3_release_retreat_ticket', ticket)
    if not ready:
        _s3_note(ctx, 'stage3_recovery_stop', {'reason': 'open_retreat_settle_gate_failed'})
        return None
    STAGE3_RELEASE_TICKET = ticket
    state.update(info=current, newinfo=current, refined=False, volume=live, checker=SelfAwareGeometry(mover.robot, current), recovered=True)
    return state

def stage3_recover(ctx, state, local_probe=None):
    global STAGE3_RELEASE_TICKET
    recovered = stage3_recover_release(ctx, state, local_probe)
    if recovered is None:
        return None
    if recovered.get('local_retry_ready'):
        return recovered
    old_ticket = dict(STAGE3_RELEASE_TICKET)
    mover = _Stage3Mover(ctx, recovered)
    start = mover.last.copy()
    for extra in (0.0, 0.004, 0.008, 0.012, 0.016):
        if extra:
            target = start.copy()
            target[2] += extra
            if not mover.move(target, 48, 's7_release_clearance_retreat'):
                return None
            if not mover.move(mover.last.copy(), 48, 's7_release_clearance_settle'):
                return None
            info, report = stage3_live_geometry(recovered['volume'], mover.obs, recovered['info'])
            if info is None:
                s5_note(ctx, 's7_release_clearance', dict(step=mover.obs['step'], extra_retreat_m=extra, ready=False, reason='fresh_map_refused', report=report))
                continue
            checker = SelfAwareGeometry(mover.robot, info)
        else:
            info, checker = (recovered['info'], recovered['checker'])
        measured = checker.check(np.asarray(mover.obs['actual_q']))
        commanded = checker.check(mover.last)
        settle = mover.settle_evidence()
        ready = measured['passed'] and commanded['passed'] and (extra == 0.0 or settle['settled'])
        s5_note(ctx, 's7_release_clearance', dict(step=mover.obs['step'], extra_retreat_m=extra, ready=bool(ready), measured=measured, commanded=commanded, reason='fresh_envelope_and_hand_clear' if ready else 'fresh_envelope_overlaps_released_hand'))
        if ready:
            recovered.update(info=info, newinfo=info, checker=checker, obs=mover.obs, last=mover.last.copy())
            old_ticket.update(ready=True, fresh_map_step=int(mover.obs['step']), retreat_completed_step=int(mover.obs['step']), actual_retreat_m=old_ticket['actual_retreat_m'] + float(mover.obs['actual_q'][2] - start[2]), extra_clearance_retreat_m=extra)
            STAGE3_RELEASE_TICKET = old_ticket
            return recovered
        STAGE3_RELEASE_TICKET = dict(old_ticket, ready=False)
        if measured['self_collisions'] or commanded['self_collisions']:
            break
    s5_note(ctx, 'stage3_recovery_stop', dict(reason='released_hand_not_clear_of_fresh_envelope'))
    return None

def solve_close_ik(robot, q, targets, normal_weight=1.0):
    base = q.copy()
    desired = robot.tips(q)[1]
    selected = robot.active_finger_indices

    def residual(x):
        cur = base.copy()
        cur[selected] = x
        p, n, _, _ = robot.tips(cur)
        return np.r_[(p - targets).ravel() * 1500, (n - desired).ravel() * normal_weight, (x - base[selected]) * 0.03]
    lo = robot.lo[selected].copy() + 1e-07
    hi = robot.hi[selected].copy() - 1e-07
    fit = least_squares(residual, np.clip(base[selected], lo + 1e-07, hi - 1e-07), bounds=(lo, hi), max_nfev=45)
    base[selected] = fit.x
    return base

def close_ik(robot, q, targets, normal_weight=1.0):
    key = (id(robot), tuple(robot.contact_links), robot.contact_points.tobytes(), robot.contact_normals.tobytes(), np.asarray(q, dtype=float).tobytes(), np.asarray(targets, dtype=float).tobytes(), float(normal_weight))
    if key not in _S6_IK_CACHE:
        _S6_IK_CACHE[key] = solve_close_ik(robot, q, targets, normal_weight)
    return _S6_IK_CACHE[key].copy()

class S9FrameInbox:

    def __init__(self, max_age_s=0.1):
        self.latest = None
        self.max_age_s = max_age_s
        self.counts = dict(new=0, duplicate=0, late=0, missing=0, future=0, stale_stop=0)

    def select(self, observation):
        o = dict(observation)
        c = o.get('image_state')
        frames = o.get('fingereye')
        valid = c is not None and frames and (len({v.get('frame_id') for v in frames.values()}) == 1)
        if not valid:
            self.counts['missing'] += 1
        elif c['step'] > o['step']:
            self.counts['future'] += 1
        elif self.latest is not None and c['step'] < self.latest[0]['step']:
            self.counts['late'] += 1
        elif self.latest is not None and c['step'] == self.latest[0]['step']:
            self.counts['duplicate'] += 1
        else:
            if any((f['capture_step'] != c['step'] for f in frames.values())):
                raise ValueError('unpaired image state')
            self.latest = (c, frames)
            self.counts['new'] += 1
        if self.latest is None or o['time_s'] - self.latest[0]['time_s'] > self.max_age_s + 1e-10:
            self.counts['stale_stop'] += 1
            raise RuntimeError('no valid image within maximum age; safe stop')
        o['image_state'], o['fingereye'] = self.latest
        return o

def s5_note(ctx, key, value):
    ctx.note(json.dumps({key: value}, allow_nan=False))

def initialize_state(ctx):
    obs = ctx.observe()
    robot = Robot(ctx.robot_description.get('urdf_path', URDF))
    mode = 'two'
    robot.configure_contacts(2)
    info = _S3_ESTIMATE(obs)
    checker = SelfAwareGeometry(robot, info)
    initial = np.clip(np.asarray(obs['actual_q']), robot.lo + 1e-7, robot.hi - 1e-7)
    if not checker.check(initial)['passed']:
        raise RuntimeError('declared initial hand pose fails original geometry guard')
    before = np.asarray(obs['actual_q'])
    obs = ctx.step(initial, steps=24, phase='hand_initialization')
    s5_note(ctx, 'cross_hand_initialization', dict(commanded_q=initial.tolist(), before_q=before.tolist(), actual_q=obs['actual_q'], actual_qvel=obs['actual_qvel'], step=obs['step'], check=checker.check(np.asarray(obs['actual_q'])), grasp_attempt=False))
    volume = _S3_VOLUME(robot, ctx.robot_description['fingereye_calibration'], info)
    rgb = np.load(obs['initial_overhead']['rgb_path'])
    hue = initial_hue(rgb)
    return dict(hand=robot.description, ctx=ctx, obs=obs, robot=robot, info=info, volume=volume, checker=checker, last=np.asarray(obs['actual_q']).copy(), hue=hue, attempt=1, squeeze=np.full(len(robot.contact_points), 0.0005), lift_reports=[], hold_reports=[], started=time.monotonic())

def s5_static(s):
    return dict(s['ctx'].robot_description, volume=s['volume'], info=s['info'], hue=s['hue'], calibration=s['volume'].calibration, color=s['info']['color'], finger_joint_indices=s['robot'].description.finger_joint_indices)

def dispatch_checked_step(s, q, n, phase, object_check=False):
    if not 1 <= n <= 24:
        raise ValueError('each dispatch must contain 1..24 physical steps')
    if s['obs']['step'] + n > 24000 or time.monotonic() - s['started'] > 2300:
        raise RuntimeError('finite episode budget')
    q = np.asarray(q)
    robot = s['robot']
    last = s['last']
    check = s['checker'].check(q, object_check=object_check)
    rate = np.abs(q - last) / (n / 120)
    if not check['passed'] or np.any(q < robot.lo) or np.any(q > robot.hi) or (rate[:3].max() > 0.14) or (rate[3:].max() > 1.4):
        s5_note(s['ctx'], 's5_guard', dict(phase=phase, check=check, rate=rate.tolist(), step=s['obs']['step']))
        raise RuntimeError('original geometry/joint/rate guard: ' + phase)
    while n:
        count = min(24, n)
        s['obs'] = s['ctx'].step(q, steps=count, phase=phase)
        n -= count
        if True and phase in ('s5_close', 's5_preload_adjust', 's5_preload_hold', 's13_preload_recovery'):
            tracking = q - np.asarray(s['obs']['actual_q'])
            speed = np.asarray(s['obs']['actual_qvel'])
            s.setdefault('servo_samples', []).append(dict(step=s['obs']['step'], phase=phase, frame_id=s9_frame(s['obs']), target_q=q.tolist(), actual_q=s['obs']['actual_q'], actual_qvel=s['obs']['actual_qvel'], tracking_error=tracking.tolist()))
            if abs(speed[:3]).max() > 0.15 or abs(speed[3:]).max() > 1.5:
                s5_note(s['ctx'], 's11_servo_stop', dict(reason='measured_native_robot_speed', step=s['obs']['step']))
                raise RuntimeError('measured closure velocity guard')
    s['last'] = q.copy()
    return s['obs']

def s5_move(s, target, n, phase, object_check=False):
    target = np.asarray(target)
    if phase == 's5_close':
        delta = abs(target - s['last'])
        duration = 1.875 * max(float(delta[:3].max()) / 0.06, float(delta[3:].max()) / 0.8)
        n = max(12, int(np.ceil(duration * 120 / 6)) * 6)
    elif phase == 's5_preload_adjust':
        n = _s3_steps(target - s['last'], minimum=24, translation=0.06, joint=0.8)
        s5_note(s['ctx'], 's8_preload_motion', dict(step=s['obs']['step'], old_minimum_steps=60, new_steps=n))
    elif phase in ('s5_close_observe', 's5_preload_hold'):
        total = 0
        stable_count = 0
        maximum = 120 if phase == 's5_close_observe' else 60
        while total < maximum:
            count = 6 if phase == 's5_preload_hold' or S9_EARLY_SETTLE else 24
            s5_move_segment(s, target, count, phase, object_check)
            total += count
            settled, velocity = stable_joint_response(s['obs']['actual_qvel'], s['hand'])
            stable_count = stable_count + 1 if settled else 0
            if total >= 24 and settled and (phase == 's5_close_observe' or stable_count >= 2):
                break
        return
    else:
        n = max(n, _s3_steps(target - s['last'], minimum=n, translation=0.06, joint=0.8))
    robot = s['robot']
    path = smooth_segment(s['last'].copy(), target, n, phase)
    if phase == 's5_close' and True:
        start_check = time.perf_counter()
        previous = s['last'].copy()
        for index, (node, count, label) in enumerate(path):
            node = np.asarray(node)
            rate = abs(node - previous) / (count / 120)
            check = s['checker'].check(node, object_check=object_check)
            if not check['passed'] or np.any(node < robot.lo) or np.any(node > robot.hi) or (rate[:3].max() > 0.14) or (rate[3:].max() > 1.4):
                s5_note(s['ctx'], 's11_close_refused', dict(node=index, check=check))
                raise RuntimeError('closure preflight original guard')
            previous = node
        s5_note(s['ctx'], 's11_close_preflight', dict(nodes=len(path), compute_s=time.perf_counter() - start_check))
    for q, count, label in path:
        s5_step(s, q, count, label, object_check)

def s11_contact_quality(robot, candidate, initial):
    meta = candidate['meta']
    errors = np.asarray(meta.get('per_finger_error_m', [meta['error_m']] * len(candidate['points'])))
    normals = np.asarray(meta['normal_error_rad'])
    points = candidate['points']
    axis = candidate['inward'][0]
    tangent = np.cross([0.0, 0.0, 1.0], axis)
    span = float(np.ptp(points @ tangent))
    time_s = sum((n for _, n, _ in candidate['prefix'])) / 120
    clearance = float(meta['endpoint']['table_clearance_m'])
    terms = dict(mean_position=float(np.mean(errors / 0.0008)), mean_normal=float(np.mean(normals / 0.5)), uncertainty_m=0.0015, tangent_span_m=span, min_table_clearance_m=clearance, planned_motion_s=time_s, per_finger_error_m=errors.tolist(), all_fingers_passed=bool(np.all(errors < 0.0008) and np.all(normals < 0.5)), support_scope='geometric spread only; friction and forces not measured')
    score = terms['mean_position'] + terms['mean_normal'] + 0.02 * time_s + 0.001 / max(clearance, 0.00065) - min(span / 0.01, 1.0) * 0.2
    return (score, terms)

def s12_task_quality(robot, candidate, info):
    """Same world-frame tasks/COM assumption for both groups, no private inputs."""
    from general_grasp_contacts import task_wrench_margin
    q = candidate['q']
    count = len(candidate['points'])
    indices = robot.active_finger_indices
    jacobian = np.zeros((3 * count, len(indices)))
    for column, index in enumerate(indices):
        plus = q.copy()
        minus = q.copy()
        plus[index] += 1e-06
        minus[index] -= 1e-06
        jacobian[:, column] = ((robot.tips(plus)[0] - robot.tips(minus)[0]) / 2e-06).ravel()
    limits = {joint.get('name'): float(joint.find('limit').get('effort', 'inf')) for joint in ET.parse(robot.urdf_path).getroot().findall('joint') if joint.find('limit') is not None}
    effort = np.array([limits.get(NAMES[i], np.inf) for i in indices])
    points = np.asarray(candidate['meta']['targets'])
    parameters = dict(S11_CONFIG['task_wrench'])
    sigma = parameters.pop('position_sigma_m', 0.00025)
    parameters.pop('margin_tolerance', None)
    common = dict(reference=np.asarray(info['center']), torque_map=jacobian.T, effort_limits=effort, **parameters)
    nominal = task_wrench_margin(points, -np.asarray(candidate['inward']), **common)
    robust = task_wrench_margin(points, -np.asarray(candidate['inward']), covariances=np.tile(np.eye(3) * sigma ** 2, (count, 1, 1)), **common)
    return dict(nominal=nominal, robust=robust, position_sigma_m=sigma, classification='conditional_load_model_not_physical_certificate', measured_force_available=False)

def s12_select_contacts(s, candidate):
    count = len(candidate['points'])
    s['robot'].select_pair(candidate['meta']['contact_fingers'],candidate['meta']['hand_contact_regions'])
    s['robot'].configure_contacts(count)
    s['hand'] = s['robot'].description
    s['squeeze'] = np.full(count, 0.0005)
    s5_note(s['ctx'], 's12_candidate_command_binding', dict(attempt=s['attempt'], step=s['obs']['step'], frame_id=s9_frame(s['obs']), contact_count=count, candidate_id=candidate['meta'].get('candidate_id'), actual_q=s['obs']['actual_q'], planned_q=candidate['q'].tolist(), task_quality=candidate['meta'].get('task_quality'), approach_first_command=np.asarray(candidate['prefix'][0][0]).tolist() if candidate['prefix'] else None))

def s12_order_candidates(candidates, tolerance=0.1):
    """Compare conditional margins; overlapping error intervals use motion.

    Nominal versus sampled-uncertainty margins are not confidence intervals.
    Relative tolerance prevents insignificant alpha differences determining
    finger count. Unsupported grasps remain an explicit empirical fallback.
    """
    remaining = list(candidates)
    ordered = []
    while remaining:
        supported = [c for c in remaining if c['meta']['task_quality']['nominal']['gravity_feasible']]
        pool = supported or remaining
        best = max(pool, key=lambda c: c['meta']['task_quality']['robust']['minimum_task_margin'])
        bt = best['meta']['task_quality']
        br = bt['robust']['minimum_task_margin']
        bn = bt['nominal']['minimum_task_margin']
        tied = []
        for c in pool:
            t = c['meta']['task_quality']
            r = t['robust']['minimum_task_margin']
            n = t['nominal']['minimum_task_margin']
            threshold = max(abs(n - r), abs(bn - br), tolerance * max(1.0, abs(br), abs(r)))
            if br - r <= threshold:
                tied.append(c)

        def motion(c):
            terms = c['meta']['quality_terms']
            return (terms['planned_motion_s'], -terms['min_table_clearance_m'], terms['uncertainty_m'], c['meta']['quality_score'])
        chosen = min(tied, key=motion)
        chosen['meta']['rank_reason'] = dict(best_robust_margin=br, relative_tolerance=tolerance, tied_candidate_ids=[c['meta']['candidate_id'] for c in tied], rule='uncertainty-overlap or relative margin tie: shortest motion, clearance, observation, normalized error', load_gate='nominal_gravity_supported' if supported else 'uncertified_empirical_fallback')
        ordered.append(chosen)
        remaining = [c for c in remaining if c is not chosen]
    return ordered

def rank_contact_candidates(s, robot, info, prior, checker, initial, axes, heights, shift, geometry_source, **kwargs):
    robot.ik_checker=checker
    settings = S11_CONFIG
    mode = 'two'
    if True and False:
        robot.configure_contacts(2)
        return _s3_candidates_on_axes(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, **kwargs)
    modes = (2,)
    if kwargs.get('refinement') and s.get('c') is not None:
        modes = (len(s['c']['points']),)
    accepted = []
    rejected = []
    qualities = []
    import itertools
    pairs=[robot.finger_names] if kwargs.get('refinement') else list(itertools.combinations(robot.catalog,2))
    for pair in pairs:
        pair_started=time.process_time()
        count=2
        robot.select_pair(pair)
        robot.configure_contacts(count)
        region_combinations=list(itertools.product(*(range(len(options)) for options in robot.patch_options)))
        region_combinations.sort(key=lambda ids:-sum(float(robot.patch_options[i][k]['point']@robot.contact_distals[i]) for i,k in enumerate(ids)))
        if kwargs.get('refinement'):
            region_combinations=[tuple(s['c']['meta']['hand_contact_regions'])]
        for region_number,regions in enumerate(region_combinations):
            if region_number and time.process_time()-pair_started>=90.:
                s5_note(s['ctx'],'cross_pair_search_budget',dict(pair=list(pair),cpu_s=time.process_time()-pair_started,limit_cpu_s=90.,regions_checked=region_number,regions_total=len(region_combinations),reason='return existing feasible plans within the common finite worker budget'))
                break
            robot.select_regions(regions)
            choices, failures = _s3_candidates_on_axes(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, **kwargs)
            # Persist the first refusal now; a planning timeout can prevent
            # the final aggregate rejection note from ever being written.
            first_failures = []
            for failure in failures:
                detail = failure.get('first_rejection_detail') or {}
                geometry = detail.get('geometry') or failure.get('endpoint') or {}
                witness = geometry.get('first_failed_guard') or {}
                links = list(witness.get('links') or [])
                if not links:
                    links = [row['link'] for row in detail.get('per_finger', [])
                             if 'link' in row and row.get('value', 0.) >= detail.get('threshold', float('inf'))]
                if not links:
                    links = sorted({link for pair in geometry.get('self_collisions', []) for link in pair})
                    links += list(geometry.get('object_collisions', []))
                path_audit = failure.get('path_audit') or {}
                first_failures.append(dict(
                    first_rejection=failure.get('first_rejection'),
                    phase=detail.get('phase', 'endpoint'), node=detail.get('node'),
                    links=links, commanded_q=detail.get('commanded_q', failure.get('q')),
                    path_audit={key: path_audit[key] for key in ('audit_complete', 'total_nodes', 'checked_nodes') if key in path_audit}))
            s5_note(s['ctx'],'cross_region_search',dict(pair=list(pair),regions=list(regions),accepted=len(choices),rejections=[f.get('first_rejection') for f in failures],first_failures=first_failures))
            for failure in failures:
                failure.update(contact_count=count,hand_contact_regions=list(regions))
            rejected += failures
            for index, candidate in enumerate(choices):
                repeated = any((len(old['points']) == len(candidate['points']) and np.mean(np.linalg.norm(np.asarray(old['points']) - candidate['points'], axis=1)) < 0.001 for old in s.get('tried_contacts', [])))
                if repeated:
                    rejected.append(dict(contact_count=count, reason='already_physically_tried_placement', targets=candidate['points'].tolist()))
                    continue
                score, terms = s11_contact_quality(robot, candidate, initial)
                task = s12_task_quality(robot, candidate, info)
                candidate['meta'].update(hand_contact_regions=list(regions),contact_fingers=list(pair), contact_count=count, quality_score=score, quality_terms=terms, task_quality=task, candidate_id=str(s['attempt']) + '-' + '+'.join(pair) + '-' + str(index))
                accepted.append(candidate)
                qualities.append(dict(candidate_id=candidate['meta']['candidate_id'], contact_count=count, score=score, terms=terms, task_quality=task))
            if choices:break
    accepted = s12_order_candidates(accepted, S11_CONFIG['task_wrench'].get('margin_tolerance', 0.1))
    if accepted:robot.select_pair(accepted[0]['meta']['contact_fingers'],accepted[0]['meta']['hand_contact_regions'])
    robot.configure_contacts(2)
    s5_note(s['ctx'], 's11_finger_selection', dict(step=s['obs']['step'], frame_id=s9_frame(s['obs']), mode='two', selected=len(accepted[0]['points']) if accepted else None, selection_scope='both_groups_geometry_and_path_feasible' if {len(c['points']) for c in accepted} == {2, 3} else 'only_one_group_feasible' if accepted else 'no_feasible_group', ordered_candidate_ids=[c['meta']['candidate_id'] for c in accepted], candidates=qualities, rejected=rejected, actual_attempts=s['attempt'] - 1, candidate_rejections=len(rejected), selection_reason='nominal gravity; robust task reserve; overlapping uncertainty or relative tolerance uses motion/clearance', rank_reasons=[dict(candidate_id=c['meta']['candidate_id'], **c['meta']['rank_reason']) for c in accepted], middle_calibration=MIDDLE_CONTACT))
    return (accepted, rejected)

def s5_plan(s, axis='short', height=None, shift=0.0):
    if s['attempt'] >= 3 and True and False:
        previous = np.asarray(s['c']['inward'][0])
        choices = []
        for angle in (np.pi / 4, -np.pi / 4):
            axis = Rotation.from_euler('z', angle).apply(previous)
            width = 2 * float(np.sum(abs(np.asarray(s['info']['axes']).T @ axis[:2]) * np.asarray(s['info']['half'])[:2]))
            choices.append((width, axis))
        width, axis = min(choices, key=lambda item: item[0])
        s5_note(s['ctx'], 's8_retry_geometry', dict(step=int(s['obs']['step']), attempt=s['attempt'], previous_failure=s.get('s8_previous_decision', {}), chosen_axis_world=axis.tolist(), projected_envelope_width_m=width, rule='observed_failure_then_new_diagonal_minimum_envelope_width', uncertainty='RGB/encoder evidence does not certify contact force'))
        axis = axis.tolist()
    info = s['info']
    robot = s['robot']
    side = int(np.argmin(info['half'][:2]))
    direction = np.r_[info['axes'][:, side], 0.0]
    if max(info['half'][:2]) / min(info['half'][:2]) < 1.15:
        direction = np.array([1.0, 0.0, 0.0])
    if axis == 'auto':
        ratio = float(max(info['half'][:2]) / min(info['half'][:2]))
        axis = 'orthogonal' if ratio >= 1.45 else 'short'
        s5_note(s['ctx'], 's5_direction_choice', dict(step=s['obs']['step'], aspect_ratio=ratio, threshold=1.45, chosen=axis, scope='two_development_shapes_not_universal'))
    if axis == 'orthogonal':
        direction = np.cross([0.0, 0.0, 1.0], direction)
    if axis == 'previous_orthogonal':
        direction = np.cross([0.0, 0.0, 1.0], s['c']['inward'][0])
    if isinstance(axis, (list, tuple)):
        direction = np.asarray(axis)
    heights = (0.85, 0.6, 0.4) if height is None else (float(height),)
    initial = s13_legal_start(s)
    if initial is None:
        return False
    if s['attempt'] > 1:
        ticket = STAGE3_RELEASE_TICKET
        if not ticket or not ticket.get('ready') or ticket.get('source_attempt') != s['attempt'] - 1:
            s5_note(s['ctx'], 's5_orientation_refusal', dict(reason='no_completed_release_retreat_ticket'))
            return False
    if s['attempt'] > 1 and S5_LOCAL_RECOVERY:
        candidates, rejected = s5_local_candidates(s, direction, heights, shift, initial)
        for scan in range(2):
            if candidates:
                break
            s5_note(s['ctx'], 's5_local_refusal_scan', dict(step=s['obs']['step'], scan=scan, rejected=rejected))
            if not s5_scan_release(s):
                return False
            info = s['info']
            candidates, failures = s5_local_candidates(s, direction, heights, shift, s['last'])
            rejected += failures
    else:
        candidates, rejected = s11_contact_candidates(s, robot, info, np.asarray(s['obs']['actual_q']), s['checker'], initial, [direction], heights, shift, 's5_current_rgb' if s['attempt'] > 1 else 's5_initial_rgb', retime=True)
    s5_note(s['ctx'], 's5_plan', dict(attempt=s['attempt'], step=s['obs']['step'], axis_world=direction.tolist(), accepted=len(candidates), rejected=rejected, info={k: np.asarray(info[k]).tolist() for k in ('center', 'half', 'axes')}, candidates=[dict(meta=c['meta'], q=c['q'].tolist(), points=c['points'].tolist(), inward=c['inward'].tolist()) for c in candidates]))
    if not candidates:
        return False
    s['c'] = candidates[0]
    s12_select_contacts(s, s['c'])
    s['axis'] = direction
    s['lift_reports'] = []
    s['hold_reports'] = []
    s['refined'] = False
    return True

def s5_local_candidates(s, direction, heights, shift, initial):
    pts = s['volume'].points
    axis = np.asarray(direction, dtype=float).copy()
    axis /= np.linalg.norm(axis)
    tangent = np.cross([0.0, 0.0, 1.0], axis)
    along = pts @ tangent
    center = float(np.median(along)) + shift
    results, rejected = ([], [])
    for h in heights:
        z = s['info']['table_z'] + max(0.004, h * 2 * s['info']['half'][2])
        band = (abs(pts[:, 2] - z) <= 0.0016) & (abs(along - center) <= 0.0016)
        selected = pts[band]
        if len(selected) < 25:
            rejected.append(dict(reason='insufficient_local_multiview_surface', height_m=z, voxels=len(selected)))
            continue
        across = selected @ axis
        lo, hi = (float(across.min()), float(across.max()))
        if hi - lo < 0.003:
            rejected.append(dict(reason='no_resolved_local_chord', height_m=z, width_m=hi - lo))
            continue
        ends = np.array([axis * lo + tangent * center, axis * hi + tangent * center])
        ends[:, 2] = z
        local = dict(s['info'], collision_points=ends)
        choices, failures = s11_contact_candidates(s, s['robot'], local, np.asarray(s['obs']['actual_q']), s['checker'], initial, [axis], (h,), 0.0, 'current_rgb_local_height_chord', retime=True)
        for c in choices:
            c['meta']['local_surface_selection'] = dict(height_m=z, along_world_m=center, band_radius_m=0.0016, voxels=len(selected), chord_m=[lo, hi], full_collision_map_retained=True, surface_normal_measured=False)
        results += choices
        rejected += failures
    return (results, rejected)

def s5_scan_release(s):
    if not STAGE3_RELEASE_TICKET or not STAGE3_RELEASE_TICKET.get('ready'):
        return False
    target = s['last'].copy()
    target[2] += 0.006
    s5_move(s, target, 144, 's5_release_view_retreat', True)
    s5_move(s, target, 48, 's5_release_view_settle', True)
    current, report = stage3_live_geometry(s['volume'], s['obs'], s['info'])
    qvel = np.abs(s['obs']['actual_qvel'])
    settled = bool(qvel[:3].max() < 0.005 and qvel[3:].max() < 0.05)
    s5_note(s['ctx'], 's5_release_view_map', dict(step=s['obs']['step'], settled=settled, report=report))
    if current is None or not settled:
        STAGE3_RELEASE_TICKET['ready'] = False
        return False
    s['info'] = current
    s['checker'] = SelfAwareGeometry(s['robot'], current)
    STAGE3_RELEASE_TICKET.update(fresh_map_step=int(s['obs']['step']), retreat_completed_step=int(s['obs']['step']))
    return True

def s5_approach(s):
    prefix = s['c']['prefix']
    total = sum((n for _, n, phase in prefix if phase == 'descend'))
    descended = 0
    next_fraction = 0.9
    checked = 0
    for q, n, phase in prefix:
        if phase in ('close', 'probe_hold'):
            break
        s5_step(s, q, n, 's5_' + phase, object_check=True)
        if phase == 'descend':
            descended += n
            if checked < 1 and descended >= next_fraction * total:
                before = int(s['obs']['step'])
                checked += 1
                next_fraction = 0.9
                changed = s5_refine(s)
                s5_note(s['ctx'], 's5_approach_observation', dict(step=s['obs']['step'], observation_step=before, descent_steps_completed=descended, descent_steps_planned=total, remaining_prefix_replaced=changed))
                if changed:
                    break
    s5_note(s['ctx'], 's5_arrived', dict(step=s['obs']['step'], actual_q=s['obs']['actual_q'], actual_pad_m=s['robot'].tips(np.asarray(s['obs']['actual_q']))[0].tolist(), planned_points_m=s['c']['points'].tolist()))

def s5_refine(s):
    report = s['volume'].carve(s['obs'])
    s5_note(s['ctx'], 's5_near_map', report)
    if not report.get('valid'):
        return False
    info = s['info']
    points = s['volume'].points
    local = points[:, :2] @ info['axes']
    lo, hi = (local.min(0), local.max(0))
    refined = dict(info, center=np.r_[info['axes'] @ ((lo + hi) / 2), points[:, 2].max() / 2], half=np.r_[(hi - lo) / 2, points[:, 2].max() / 2], collision_points=np.concatenate([points + np.array([a, b, c]) for a in [-0.0005, 0.0005] for b in [-0.0005, 0.0005] for c in [-0.0005, 0.0005]]))
    change = max(float(np.linalg.norm(refined['center'] - info['center'])), float(np.max(abs(refined['half'] - info['half']))))
    s5_note(s['ctx'], 's6_target_change', dict(step=s['obs']['step'], change_m=change, threshold_m=S6_UPDATE_MM, material=change >= S6_UPDATE_MM))
    if change < S6_UPDATE_MM:
        return False
    checker = SelfAwareGeometry(s['robot'], refined)
    actual = np.asarray(s['obs']['actual_q'])
    legal_actual = np.clip(actual, s['robot'].lo + 1e-07, s['robot'].hi - 1e-07)
    projection = float(np.max(abs(legal_actual - actual)))
    if projection > 1e-05:
        s5_note(s['ctx'], 's6_refinement_refused', dict(reason='measured_joint_limit_violation', projection=projection))
        return False
    choices, rejected = s11_contact_candidates(s, s['robot'], refined, np.asarray(s['obs']['actual_q']), checker, legal_actual, [s['c']['inward'][0]], (s['c']['meta']['height_fraction'],), 0.0, 's6_current_rgb_actual_joints', retime=True, refinement=True)
    s5_note(s['ctx'], 's5_refinement_gate', dict(step=s['obs']['step'], accepted=len(choices), rejected=rejected, planning_from='projected_actual_joints', limit_projection=projection, actual_minus_command=(actual - s['last']).tolist()))
    if not choices:
        return False
    c = choices[0]
    opened = close_ik(s['robot'], c['q'], c['points'] - c['inward'] * 0.006)
    if not checker.check(legal_actual)['passed']:
        return False
    n = _s3_steps(opened - s['last'], 240, 0.045, 0.7)
    path = smooth_segment(s['last'], opened, n, 's5_rgb_realign')
    failed = [i for i, (q, _, _) in enumerate(path) if not checker.check(np.asarray(q))['passed']]
    if failed:
        s5_note(s['ctx'], 's5_refinement_connection_refused', dict(step=s['obs']['step'], failed_nodes=failed))
        return False
    s['checker'] = checker
    s5_move(s, opened, n, 's5_rgb_realign', True)
    s['c'] = c
    s12_select_contacts(s, c)
    s['info'] = refined
    s['refined'] = True
    return True

def s11_motor_snapshot(s):
    hand = s['hand']
    groups = {p.finger: [hand.index[n] for n in p.joint_names] for p in hand.pads}
    return dict(actual_q=np.asarray(s['obs']['actual_q']), actual_qvel=np.asarray(s['obs']['actual_qvel']), groups=groups)

def s5_close(s, mode='fixed', cap=0.0025):
    c = s['c']
    robot = s['robot']
    ctx = s['ctx']
    fast = True
    displacements = [0.0005] if fast else np.linspace(-0.006, 0.0005, 30)
    s5_note(ctx, 's11_close_begin', dict(step=s['obs']['step'], fast=fast, frame_id=s9_frame(s['obs']), capture_step=s['obs']['image_state']['step'], actual_q=s['obs']['actual_q'], last_command=s['last'].tolist(), points=c['points'].tolist()))
    for displacement in displacements:
        target = close_ik(robot, c['q'], c['points'] + c['inward'] * displacement)
        s5_move(s, target, 24, 's5_close')
    s5_move(s, s['last'], 120, 's5_close_observe')
    squeeze = np.full(len(c['points']), 0.0005)
    evidence = None
    contact_at = np.full(len(c['points']), np.nan)
    last_increment = np.full(len(c['points']), 0.00025)
    for i in range(15 if mode == 'servo' else 6):
        previous_visual = evidence
        visual_round = not fast or i % 2 == 0 or evidence is None or bool(evidence.get('moving_fingers')) or (not all((m.get('settled') for m in evidence.get('motor', {}).get('fingers', {}).values())))
        before = capture_preload_snapshot(s['obs'], robot, s5_static(s)) if visual_round else s11_motor_snapshot(s)
        delta = np.full(len(c['points']), 0.00025 if i == 0 else 0.00035)
        reason = 'fixed_schedule'
        if mode in ('adaptive', 'servo') and evidence is None:
            visible_views = [row for row in before.get('views', {}).values()
                             if row.get('status') == 'ok' and row.get('visible', False)]
            if len(visible_views) < 2:
                delta[:] = 0.0
                reason = 'missing_object_rgb_diagnostic_probe_hold'
        if mode == 'feedback' and evidence is not None:
            delta, reason = stage2_response_action(evidence, i, 'preload')
        if mode in ('adaptive', 'servo') and evidence is not None:
            motors = evidence['motor']['fingers']
            object_observation = evidence.get('object_observation', {})
            object_associated = object_observation.get('associated', False)
            delta = np.full(len(c['points']), 0.00035)
            reason = 'bounded_motor_probe_with_whole_object_association'
            for k, finger in enumerate(robot.finger_names):
                motor = motors.get(finger, {})
                if not object_associated or not motor.get('settled', False):
                    delta[k] = 0.0
                    reason = 'object_association_or_motor_settle_hold'
                deficit = motor.get('drive_deficit')
                if last_increment[k] > 0.0001 and deficit is not None and (deficit >= 0.45) and np.isnan(contact_at[k]):
                    contact_at[k] = squeeze[k]
                allowance = 0.00105 if mode == 'servo' else 0.0007
                if not np.isnan(contact_at[k]):
                    delta[k] = min(delta[k], max(0.0, contact_at[k] + allowance - squeeze[k]))
            if object_observation.get('registered_motion_veto', False) or (evidence.get('moving_fingers') and evidence.get('confidence', 0.0) >= 0.5):
                delta[:] = 0.0
                reason = 'registered_object_motion_hold'
        proposed = np.clip(squeeze + delta, 0.0005, cap)
        target = close_ik(robot, c['q'], c['points'] + c['inward'] * proposed[:, None])
        s5_note(ctx, 's5_closure_action', dict(step=s['obs']['step'], round=i, mode=mode, reason=reason, response_used=evidence if mode in ('feedback', 'adaptive', 'servo') else None, contact_proxy_at_m=[float(x) if np.isfinite(x) else None for x in contact_at], squeeze_m=proposed.tolist(), target_q=target.tolist()))
        s5_move(s, target, 60, 's5_preload_adjust')
        s5_move(s, target, 60, 's5_preload_hold')
        after = capture_preload_snapshot(s['obs'], robot, s5_static(s)) if visual_round else s11_motor_snapshot(s)
        if visual_round:
            evidence = compare_preload(before, after, commanded_delta=target - before['actual_q'])
        else:
            evidence = dict(previous_visual, motor=_s2v_motor(before, after, target - before['actual_q']))
            evidence.update(confidence=0.0, moving_fingers=[], visual_evidence_reused_for_state_only=True)
        s5_note(ctx, 's11_probe_tracking', dict(step=s['obs']['step'], frame_id=s9_frame(s['obs']), visual_processed=visual_round, target_q=target.tolist(), actual_q=s['obs']['actual_q'], actual_qvel=s['obs']['actual_qvel'], tracking_error=(target - np.asarray(s['obs']['actual_q'])).tolist(), measured_tip_m=robot.tips(np.asarray(s['obs']['actual_q']))[0].tolist(), cap_m=cap, force_known=False))
        s5_note(ctx, 's5_closure_response', dict(step=s['obs']['step'], round=i, features=evidence))
        last_increment = proposed - squeeze
        squeeze = proposed
        if mode == 'servo' and np.all(last_increment < 1e-07):
            break
    s['squeeze'] = squeeze
    s['closure_evidence'] = evidence or {}
    s['closure_evidence_step'] = (evidence or {}).get('step_after')
    s['last_closure_cap'] = cap
    if fast:
        s5_note(ctx, 's11_servo_audit', dict(attempt=s['attempt'], samples=s.pop('servo_samples', []), rules=SENSOR_RULES, position_preload_cap_m=cap, force_estimate_available=False))

def lift_and_hold(s, height=0.03, hold=360, _full=False):
    if S9_EARLY_SETTLE and height <= 0.0031 and (not _full):
        return early_proof(s, height, hold)
    ctx = s['ctx']
    before = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    target = s['last'].copy()
    target[2] = s['c']['q'][2] + height
    n = 240 if height <= 0.0031 else 300
    recent = []
    previous = before
    s['lift_veto'] = False
    for q, count, label in smooth_segment(s['last'].copy(), target, n, 's5_proof' if height <= 0.0031 else 's5_lift'):
        s5_step(s, q, count, label)
        if abs(float(s['obs']['actual_q'][2]) - float(previous['actual_q'][2])) < 0.0022:
            continue
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        response = stage3_support_pair(previous, snap, 'lift')
        previous = snap
        recent.append(response)
        s5_note(ctx, 's5_lift_in_progress', response)
        if len(recent) >= 2 and all((x['state'] == 'grounded_evidence' for x in recent[-2:])):
            s['lift_veto'] = True
            s5_note(ctx, 's5_lift_veto', dict(step=s['obs']['step'], reason='two_current_RGB_intervals_grounded', commanded_height_m=height))
            break
    after = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    r = stage3_support_pair(before, after, 'lift')
    s['lift_reports'].append(r)
    s5_note(ctx, 's5_lift_response', dict(step=s['obs']['step'], height_m=height, report=r))
    for _ in range(0 if s['lift_veto'] else hold // 24):
        s5_step(s, s['last'], 24, 's5_hold')
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        r = stage3_support_pair(after, snap, 'hold')
        s['hold_reports'].append(r)
        after = snap
    if hold and (not s['lift_veto']):
        s5_note(ctx, 's5_hold_response', dict(step=s['obs']['step'], reports=s['hold_reports'][-hold // 24:]))
    return r

def s5_decide(s, phase, reobservations=0):
    result = stage3_support_decision(s['lift_reports'], s['hold_reports'], attempt=s['attempt'], max_attempts=S5_ATTEMPTS, phase=phase, reobservations=reobservations, remaining_steps=24000 - int(s['obs']['step']), remaining_wall_s=2300 - (time.monotonic() - s['started']))
    if s.get('lift_veto'):
        result.update(action='retry' if s['attempt'] < S5_ATTEMPTS else 'stop_limit', reason='in_progress_grounded_RGB_veto', support_state='failure_evidence')
    s5_note(s['ctx'], 's5_' + phase + '_decision', result)
    s['s8_previous_decision'] = dict(result)
    return result

def s5_reobserve(s):
    previous = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    for _ in range(10):
        s5_step(s, s['last'], 24, 's5_reobserve')
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        s['hold_reports'].append(stage3_support_pair(previous, snap, 'hold'))
        previous = snap
    s5_note(s['ctx'], 's5_reobservation_completed', dict(step=s['obs']['step'], reports=s['hold_reports'][-10:]))
SENSOR_RULES = {'coarse': 'initial ordinary overhead RGB', 'approach': 'new exposure-paired RGB at descent checkpoint or changed geometry', 'close': 'encoder feedback every command; new RGB at entry and anomaly', 'preload': 'new RGB on even probes/anomaly; encoders on every probe', 'lift': 'new exposure-paired RGB for every independent support interval', 'recover': 'new RGB after each bounded opening/retreat before local planning'}

def early_proof(s, height=0.03, hold=360):
    if not S9_EARLY_SETTLE or height > 0.0031:
        return _S9_LIFT(s, height, hold)
    before = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    start = int(s['obs']['step'])
    target = s['last'].copy()
    target[2] = s['c']['q'][2] + height
    for q, count, label in smooth_segment(s['last'].copy(), target, 240, 's5_proof'):
        s5_step(s, q, count, label)
        c = s['obs']['image_state']
        if float(c['actual_q'][2]) - float(before['actual_q'][2]) < 0.0022:
            continue
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        r = stage3_support_pair(before, snap, 'lift')
        if r['state'] == 'grounded_evidence':
            decision = stage3_support_decision(s['lift_reports'] + [r], s['hold_reports'], attempt=s['attempt'], max_attempts=S5_ATTEMPTS, phase='proof')
            if decision['action'] in ('retry', 'stop_limit'):
                s['lift_reports'].append(r)
                s['lift_veto'] = False
                event = dict(stage='s5_proof', attempt=s['attempt'], trigger_frame=s9_frame(s['obs']), trigger_step=s['obs']['step'], capture_step=c['step'], legacy_decision_step=start + 240 + hold, saved_s=(start + 240 + hold - int(s['obs']['step'])) / 120, predicate='existing V7 grounded evidence and proof decision', report=r, decision=decision)
                _S9_IO['early'].append(event)
                s5_note(s['ctx'], 's9_early_proof', event)
                return r
    after = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    r = stage3_support_pair(before, after, 'lift')
    s['lift_reports'].append(r)
    s['lift_veto'] = False
    s5_note(s['ctx'], 's5_lift_response', dict(step=s['obs']['step'], height_m=height, report=r))
    for _ in range(hold // 24):
        s5_step(s, s['last'], 24, 's5_hold')
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        s['hold_reports'].append(stage3_support_pair(after, snap, 'hold'))
        after = snap
    return r

def s5_move_segment(s, target, n, phase, object_check=False):
    n = max(n, _s3_steps(target - s['last'], minimum=n, translation=0.06, joint=0.8))
    for q, count, label in smooth_segment(s['last'].copy(), target, n, phase):
        s5_step(s, q, count, label, object_check)
_S13_ORIGINAL_STEP = dispatch_checked_step

def s13_log(s, event, **fields):
    s5_note(s['ctx'], 's13_recovery', dict(event=event, step=int(s['obs']['step']), frame_id=s9_frame(s['obs']), actual_grasps=s.get('actual_grasps', 0), candidate_id=s.get('c', {}).get('meta', {}).get('planner_candidate_id'), **fields))

def s13_legal_start(s, checker=None):
    actual = np.asarray(s['obs']['actual_q'], dtype=float)
    robot = s['robot']
    legal = np.clip(actual, robot.lo + 1e-07, robot.hi - 1e-07)
    projection = float(np.max(abs(legal - actual)))
    if not np.isfinite(actual).all() or projection > 1e-05:
        s13_log(s, 'guard_refused', branch='planning', reason='measured_joint_limit', projection=projection)
        return None
    checker = checker or s['checker']
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        check = checker.check(actual + (legal - actual) * fraction)
        if not check['passed']:
            s13_log(s, 'guard_refused', branch='planning', reason='actual_to_legal_connection', fraction=fraction, check=check)
            return None
    s13_log(s, 'legal_command_seed', actual_q=actual.tolist(), legal_q=legal.tolist(), projection=projection)
    return legal

def s5_step(s, q, n, phase, object_check=False):
    pending = s.get('s13_pending_grasp')
    closure = phase in ('s5_close', 's13_preload_recovery')
    if pending and closure and (s.get('actual_grasps', 0) >= S5_ATTEMPTS):
        s13_log(s, 'guard_refused', branch=pending['branch'], reason='actual_grasp_limit')
        raise RuntimeError('actual grasp budget')
    before = int(s['obs']['step'])
    value = _S13_ORIGINAL_STEP(s, q, n, phase, object_check)
    if pending and closure and (int(s['obs']['step']) > before):
        s['actual_grasps'] = s.get('actual_grasps', 0) + 1
        s['attempt'] = s['actual_grasps']
        s13_log(s, 'grasp_dispatched', branch=pending['branch'], started_step=before, target_q=np.asarray(q).tolist(), actual_q=s['obs']['actual_q'], contact_count=len(s['c']['points']), points=s['c']['points'].tolist(), command_plan_sha256=s['c']['meta'].get('command_plan_sha256'), planned_q=s['c']['q'].tolist(), recovery_id=pending.get('recovery_id'))
        s.pop('s13_pending_grasp', None)
    return value

def s13_preflight(s, path, checker=None, object_check=False):
    checker = checker or s['checker']
    previous = s['last'].copy()
    robot = s['robot']
    for i, (node, count, label) in enumerate(path):
        node = np.asarray(node)
        check = checker.check(node, object_check=object_check)
        rate = abs(node - previous) / (count / 120)
        if not check['passed'] or np.any(node < robot.lo) or np.any(node > robot.hi) or (rate[:3].max() > 0.14) or (rate[3:].max() > 1.4):
            return dict(passed=False, node=i, phase=label, check=check, rate=rate.tolist())
        previous = node
    return dict(passed=True, nodes=len(path))

def s13_local_fit(robot, actual, goals, inward):
    indices = np.r_[np.arange(3), robot.active_finger_indices]
    start = np.asarray(actual).copy()
    lower = robot.lo[indices] + 1e-07
    upper = robot.hi[indices] - 1e-07
    lower[:3] = np.maximum(lower[:3], start[:3] - [0.002, 0.002, 0.008])
    upper[:3] = np.minimum(upper[:3], start[:3] + [0.002, 0.002, 0.002])

    def residual(x):
        q = start.copy()
        q[indices] = x
        points, normals, _, _ = robot.tips(q)
        return np.r_[(points - goals).ravel() * 1500, (normals - inward).ravel() * 4, (q[:3] - start[:3]) * 2, (q[robot.active_finger_indices] - start[robot.active_finger_indices]) * 0.03]
    fit = least_squares(residual, np.clip(start[indices], lower + 1e-09, upper - 1e-09), bounds=(lower, upper), max_nfev=45)
    q = start.copy()
    q[indices] = fit.x
    return q

def local_pose_probe(mover, state):
    """Called after actual opening; new map and all active pads required."""
    mover.sync()
    state.update(obs=mover.obs, last=mover.last.copy(), ctx=mover.ctx)
    if state.get('s13_local_calls', 0) >= 2:
        return False
    state['s13_local_calls'] = state.get('s13_local_calls', 0) + 1
    rid = state.setdefault('recovery_serial', 0) + 1
    state['recovery_serial'] = rid
    s13_log(state, 'selected', branch='local_pose', recovery_id=rid, opening_already_executed=True)
    volume = Stage3LiveVolume(mover.robot, state['volume'].calibration, state['info'])
    info, report = stage3_live_geometry(volume, mover.obs, state['info'])
    if info is None:
        s13_log(state, 'guard_refused', branch='local_pose', recovery_id=rid, reason='fresh_RGB_map', report=report)
        return False
    checker = SelfAwareGeometry(mover.robot, info)
    legal = s13_legal_start(state, checker)
    if legal is None:
        return False
    if not checker.check(mover.last)['passed']:
        s13_log(state, 'guard_refused', branch='local_pose', reason='commanded_start_clearance')
        return False
    previous = state['c']
    old = np.asarray(previous['points'])
    inward = np.asarray(previous['inward'])
    pts = volume.points
    tangent = np.cross([0.0, 0.0, 1.0], inward[0])
    rejected = []
    for dz, shift in ((0.002, 0.0), (-0.002, 0.0), (0.0, 0.0015), (0.0, -0.0015)):
        guessed = old + np.array([0.0, 0.0, dz]) + tangent * shift
        goals = []
        for goal, normal in zip(guessed, inward):
            near = pts[(abs(pts[:, 2] - goal[2]) < 0.0016) & (np.linalg.norm((pts - goal) @ np.array([tangent, [0.0, 0.0, 1.0]]).T, axis=1) < 0.0025)]
            if len(near) < 12:
                break
            projection = near @ normal
            surface = near[projection <= projection.min() + 0.0005]
            goals.append(surface[np.argmin(np.linalg.norm(surface - goal, axis=1))])
        if len(goals) != len(old):
            rejected.append(dict(reason='unresolved_all_pad_RGB_patch', dz=dz, shift=shift))
            continue
        goals = np.asarray(goals)
        if np.any(goals[:, 2] < info['table_z'] + 0.003) or np.any(goals[:, 2] > info['top_z'] - 0.001):
            continue
        q = s13_local_fit(mover.robot, legal, goals, inward)
        q[3:6] = mover.last[3:6]
        points, normals, _, _ = mover.robot.tips(q)
        error = np.linalg.norm(points - goals, axis=1)
        angle = np.arccos(np.clip(np.sum(normals * inward, axis=1), -1, 1))
        novelty = float(np.mean(np.linalg.norm(points - old, axis=1)))
        if error.max() > 0.0008 or angle.max() > 0.5 or (not 0.001 <= novelty <= 0.004):
            rejected.append(dict(reason='IK_or_bounded_novelty', error_m=error.tolist(), normal_rad=angle.tolist(), novelty_m=novelty))
            continue
        opened = close_ik(mover.robot, q, goals - inward * 0.006)
        opened[3:6] = mover.last[3:6]
        path = smooth_segment(mover.last, opened, _s3_steps(opened - mover.last, 48, 0.045, 0.7), 's13_local_reposition')
        audit = s13_preflight(state, path, checker, True)
        closed = close_ik(mover.robot, q, points + inward * 0.0005)
        close_path = smooth_segment(opened, closed, _s3_steps(closed - opened, 24, 0.06, 0.8), 's5_close')
        close_audit = s13_preflight(dict(state, last=opened), close_path, checker, False)
        if not audit['passed'] or not close_audit['passed']:
            rejected.append(dict(reason='local_or_close_path', path=audit, close=close_audit))
            continue
        s13_log(state, 'guard', branch='local_pose', recovery_id=rid, audit=audit, close_audit=close_audit, new_RGB=report)
        mover.checker = checker
        if not mover.move(opened, sum((n for _, n, _ in path)), 's13_local_reposition', True):
            s13_log(state, 'guard_refused', branch='local_pose', reason='runtime_guard')
            return False
        meta = dict(previous['meta'], targets=goals.tolist(), error_m=float(error.max()), per_finger_error_m=error.tolist(), normal_error_rad=angle.tolist(), local_novelty_m=novelty, local_dz_m=dz, local_shift_m=shift, local_close_path_nodes=len(close_path))
        candidate = dict(previous, q=q, points=points, prefix=path, meta=meta)
        meta['task_quality'] = s12_task_quality(mover.robot, candidate, info)
        _s12_candidate_identity(candidate)
        state.update(info=info, newinfo=info, volume=volume, checker=checker, c=candidate, local_retry_ready=True, refined=True, s13_recovery_branch='local_pose', s13_recovery_id=rid)
        mover.sync()
        s13_log(state, 'executed', branch='local_pose', recovery_id=rid, novelty_m=novelty, dz_m=dz, shift_m=shift)
        return True
    s13_log(state, 'guard_refused', branch='local_pose', recovery_id=rid, reason='bounded_candidates_exhausted', rejected=rejected)
    return False

def release_and_resample(s):
    global STAGE3_ATTEMPT
    STAGE3_ATTEMPT = s['actual_grasps']
    state = dict(s, newinfo=s['info'], refined=False)
    s13_log(s, 'considered', branch='local_pose', reason='fresh_RGB_and_local_path_required', contact_count=len(s['c']['points']))
    s13_log(s, 'selected', branch='release_resample', recovery_id=s.get('recovery_serial', 0) + 1, conditional_fallback=True)
    recovered = stage3_recover(s['ctx'], state, s13_local_probe)
    s.update({k: state[k] for k in ('obs', 'last', 'recovery_serial', 's13_local_calls') if k in state})
    if recovered is None:
        s13_log(s, 'guard_refused', branch='release_resample', reason='original_release_retreat_or_fresh_map_gate')
        return False
    for key in ('obs', 'last', 'info', 'volume', 'checker', 'c', 'refined', 'local_retry_ready', 's13_recovery_branch', 's13_recovery_id'):
        if key in recovered:
            s[key] = recovered[key]
    s['attempt'] = s['actual_grasps'] + 1
    s['squeeze'] = np.full(len(s['c']['points']), 0.0005)
    if recovered.get('local_retry_ready'):
        s['next_branch'] = 'local_pose'
        return True
    s['next_branch'] = 'release_resample'
    s13_log(s, 'executed', branch='release_resample', ticket=STAGE3_RELEASE_TICKET)
    return True
_S9_IO = {'cycles': [], 'consumers': [], 'early': [], 'duplicate_cache_hits': 0, 'invalid_frames': 0}
_S9_CACHE = {}

def s9_frame(obs):
    c = obs['image_state']
    frames = obs['fingereye']
    ids = {f['frame_id'] for f in frames.values()}
    if len(ids) != 1 or c['step'] > obs['step'] or obs['time_s'] - c['time_s'] > S9_MAX_FRAME_AGE:
        _S9_IO['invalid_frames'] += 1
        raise RuntimeError('no fresh valid RGB; safe stop')
    return next(iter(ids))

def s9_exposure(obs):
    s9_frame(obs)
    c = obs['image_state']
    return dict(obs, step=c['step'], time_s=c['time_s'], actual_q=c['actual_q'], actual_qvel=c['actual_qvel'])

def s9_visual(name, fn):

    def wrapped(obs, *a, **kw):
        fid = s9_frame(obs)
        key = (name, fid, id(a[0]) if a else None)
        if key in _S9_CACHE:
            _S9_IO['duplicate_cache_hits'] += 1
            return _S9_CACHE[key]
        start = time.perf_counter()
        c = obs['image_state']
        result = fn(s9_exposure(obs), *a, **kw)
        _S9_IO['consumers'].append(dict(function=name, control_step=obs['step'], frame_id=fid, capture_step=c['step'], capture_q=c['actual_q'], current_q=obs['actual_q'], consume_wall_s=start, capture_to_use_wall_s=start - c['capture_done_wall_s'], frame_age_s=obs['time_s'] - c['time_s'], compute_s=time.perf_counter() - start))
        _S9_CACHE[key] = result
        if len(_S9_CACHE) > 30:
            for k in list(_S9_CACHE):
                if k[1] < fid - 2:
                    del _S9_CACHE[k]
        return result
    return wrapped
capture_preload_snapshot = s9_visual('capture_preload_snapshot', capture_preload_snapshot)
stage3_support_snapshot = s9_visual('stage3_support_snapshot', stage3_support_snapshot)

class _S9Context:

    def __init__(self, ctx):
        self.ctx = ctx
        self.robot_description = ctx.robot_description
        self.last_return = time.perf_counter()
        self.inbox = S9FrameInbox(S9_MAX_FRAME_AGE)
        _S9_IO['inbox'] = self.inbox.counts

    def step(self, q, steps=1, phase=''):
        if steps % 6:
            raise ValueError('20Hz commands require multiples of6 physical steps')
        for _ in range(steps // 6):
            start = time.perf_counter()
            compute = start - self.last_return
            obs = self.inbox.select(self.ctx.step(q, steps=6, phase=phase))
            done = time.perf_counter()
            s9_frame(obs)
            _S9_IO['cycles'].append(dict(end_step=obs['step'], phase=phase, compute_between_dispatch_s=compute, components_s=dict(_S9_WORK), dispatch_rpc_s=done - start, frame_id=s9_frame(obs), capture_step=obs['image_state']['step'], frame_age_s=obs['time_s'] - obs['image_state']['time_s']))
            _S9_WORK.clear()
            self.last_return = time.perf_counter()
        return obs

    def observe(self):
        obs = self.inbox.select(self.ctx.observe())
        s9_frame(obs)
        self.last_return = time.perf_counter()
        return obs

    def note(self, text):
        return self.ctx.note(text)
_S9_WORK = {}
_S9_STACK = []
_S9_SAMPLES = {}

def s9_timed(category, fn):

    def call(*a, **kw):
        start = time.perf_counter()
        frame = [start, 0.0]
        _S9_STACK.append(frame)
        try:
            return fn(*a, **kw)
        finally:
            elapsed = time.perf_counter() - start
            _S9_STACK.pop()
            exclusive = elapsed - frame[1]
            _S9_WORK[category] = _S9_WORK.get(category, 0.0) + exclusive
            _S9_SAMPLES.setdefault(category, []).append(exclusive)
            if _S9_STACK:
                _S9_STACK[-1][1] += elapsed
    return call
optimize = s9_timed('ik', solve_target_ik)
close_ik = s9_timed('ik', close_ik)
capture_preload_snapshot = s9_timed('vision', capture_preload_snapshot)
stage3_support_snapshot = s9_timed('vision', stage3_support_snapshot)
stage3_support_pair = s9_timed('vision', stage3_support_pair)
compare_preload = s9_timed('vision', compare_visual_response)
_S3_ESTIMATE = s9_timed('vision', _S3_ESTIMATE)
s5_decide = s9_timed('fsm', s5_decide)
SelfAwareGeometry.check = s9_timed('path_checks', SelfAwareGeometry.check)

def main(ctx):
    global _ACTIVE_DESCRIPTION, URDF, NAMES, CAMERAS, STAGE3_SUPPORT_CAMERAS
    _ACTIVE_DESCRIPTION=ctx.robot_description
    URDF=_ACTIVE_DESCRIPTION['urdf_path']
    NAMES=list(_ACTIVE_DESCRIPTION['joint_names'])
    CAMERAS=tuple(_ACTIVE_DESCRIPTION['fingereye_calibration'])
    STAGE3_SUPPORT_CAMERAS=CAMERAS
    start = time.perf_counter()
    try:
        return policy_main(_S9Context(ctx))
    finally:
        _S9_IO['component_samples_s'] = _S9_SAMPLES
        _S9_IO.update(total_wall_s=time.perf_counter() - start, early_settle=S9_EARLY_SETTLE, physics_hz=120, control_hz=20, camera_fps=30)
        ctx.note(json.dumps({'s9_io_audit': _S9_IO}, allow_nan=False))
S9_MAX_FRAME_AGE = 0.1
S9_EARLY_SETTLE = True
S5_ATTEMPTS = 3

def fit_shape(volume, obs, initial_info):
    _, frames = volume.frames(obs)
    if len(frames) < 3:
        return (None, dict(valid=False, reason='insufficient_current_views'))
    initial_center = np.asarray(initial_info['center'])
    vertices = np.asarray(initial_info['collision_points'])
    local = vertices - initial_center
    hull = ConvexHull(local)
    axes = [np.arange(a, b + 0.00035, 0.0007) for a, b in zip(local.min(0), local.max(0))]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    points = grid[np.max(grid @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1) < 1e-10]
    points = np.r_[points, local]
    if len(points) > 1800:
        points = points[np.linspace(0, len(points) - 1, 1800, dtype=int)]
    pixelsets = {name: np.argwhere(f['mask'])[:, ::-1].astype(float) for name, f in frames.items()}
    pixelsets = {name: p[np.linspace(0, len(p) - 1, min(160, len(p)), dtype=int)] for name, p in pixelsets.items()}

    def world(x):
        p = points.copy()
        p[:, 2] *= x[3]
        rot = Rotation.from_euler('z', x[2]).as_matrix()
        p = p @ rot.T
        p[:, 2] += initial_info['table_z'] - p[:, 2].min() + 0.0001
        p[:, :2] += x[:2]
        return p

    def residual(x):
        p = world(x)
        r = []
        for name, f in frames.items():
            uv, dist, valid = volume.project(name, p, f['camera'])
            depth = map_coordinates(f['robot_depth'], uv[:, ::-1].T, order=0, mode='constant', cval=0.0)
            visible = valid & (dist < depth - 0.0005) & color_projection_known(f, uv)
            sdf = map_coordinates(f['sdf'], uv[:, ::-1].T, order=1, mode='constant', cval=20.0)
            forward = np.maximum(sdf, 0.0) * visible
            reverse = cKDTree(uv[visible]).query(pixelsets[name])[0] if visible.any() else np.full(len(pixelsets[name]), 20.0)
            r.extend(forward / np.sqrt(len(points)))
            r.extend(reverse / np.sqrt(len(reverse)))
        return np.r_[r, (x[3] - 1.0) * 0.15]
    base = np.r_[initial_center[:2], 0.0, 1.0]
    lower = np.r_[initial_center[:2] - 0.025, -np.pi, 0.35]
    upper = np.r_[initial_center[:2] + 0.025, np.pi, 0.085 / (initial_info['top_z'] - initial_info['table_z'])]
    choices = []
    for dx, dy, yaw, scale in [(0, 0, 0, 1), (-0.003, 0, 0, 1), (0.003, 0, 0, 1), (0, -0.003, 0, 1), (0, 0.003, 0, 1), (0, 0, 0.35, 1), (0, 0, -0.35, 1), (0, 0, 0, 0.6), (0, 0, 0, 1.5), (0, 0, 0, min(5.0, upper[3] - 0.01))]:
        fit = least_squares(residual, base + [dx, dy, yaw, scale - 1], bounds=(lower, upper), diff_step=0.001, x_scale=[0.005, 0.005, 0.3, 0.5], max_nfev=70)
        choices.append((float(np.sum(residual(fit.x) ** 2)), fit.x))
    choices.sort(key=lambda x: x[0])
    cost, x = choices[0]
    p = world(x)
    audits = {}
    passed = True
    for name, f in frames.items():
        uv, dist, valid = volume.project(name, p, f['camera'])
        depth = map_coordinates(f['robot_depth'], uv[:, ::-1].T, order=0, mode='constant', cval=0.0)
        visible = valid & (dist < depth - 0.0005) & color_projection_known(f, uv)
        reverse = cKDTree(uv[visible]).query(np.argwhere(f['mask'])[:, ::-1])[0] if visible.any() else np.full(int(f['mask'].sum()), 100.0)
        sdf = map_coordinates(f['sdf'], uv[:, ::-1].T, order=1, mode='constant', cval=100.0)
        explained = float((reverse <= 2.0).mean())
        outside = float((sdf[visible] > 1.0).mean()) if visible.any() else 1.0
        ok = explained >= 0.8 and outside <= 0.1
        passed = passed and ok
        audits[name] = dict(explained_fraction=explained, reverse_p90_px=float(np.quantile(reverse, 0.9)), visible_points=int(visible.sum()), outside_mask_fraction=outside, passed=bool(ok))
    alternatives = [dict(cost=c, parameters=q.tolist()) for c, q in choices]
    near = [q for c, q in choices if c <= cost + 0.2]
    spread = float(np.ptp(np.array(near)[:, :2], axis=0).max())
    passed = passed and spread < 0.002
    rot = Rotation.from_euler('z', x[2]).as_matrix()
    newverts = local.copy()
    newverts[:, 2] *= x[3]
    newverts = newverts @ rot.T
    newverts[:, 2] += initial_info['table_z'] - newverts[:, 2].min() + 0.0001
    newverts[:, :2] += x[:2]
    planar_axes = rot[:2, :2] @ np.asarray(initial_info['axes'])
    loc = newverts[:, :2] @ planar_axes
    lo, hi = (loc.min(0), loc.max(0))
    top = float(newverts[:, 2].max())
    half = np.r_[(hi - lo) / 2, (top - initial_info['table_z']) / 2]
    report = dict(valid=bool(passed), method='initial_RGB_shape_current_multiview_registration', step=int(obs['step']), shape_prior_only=True, current_pose_from_fresh_RGB=True, truth_inputs=False, parameters=x.tolist(), cost=cost, views=audits, alternatives=alternatives, near_solution_center_spread_m=spread, reason=None if passed else 'current_image_explanation_or_pose_ambiguity', extent_m=(half * 2).tolist(), current_center_m=x[:2].tolist())
    if not passed:
        return (None, report)
    info = dict(initial_info, axes=planar_axes, center=np.r_[planar_axes @ ((lo + hi) / 2), initial_info['table_z'] + half[2]], half=half, top_z=top, collision_points=newverts)
    volume.points = p
    volume.boundary = p
    volume.tree = cKDTree(p)
    volume.center = p.mean(0)
    volume.last_frames = frames
    volume.last_report = report
    return (info, report)
_S14_INIT = initialize_state
_S14_SHAPE_PRIOR = None

def s5_init(ctx):
    global _S14_SHAPE_PRIOR
    s = _S14_INIT(ctx)
    _S14_SHAPE_PRIOR = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in s['info'].items()}
    return s

def stage3_live_geometry(volume, obs, previous_info):
    info, report = fit_shape(volume, s9_exposure(obs), _S14_SHAPE_PRIOR)
    return (info, report)
_S14_EXTREMA_CANDIDATES = axis_contact_candidates
_S14_SURFACE_ACTIVE = False

def _s14_surface_candidates(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, retime=False, refinement=False):
    results, rejected = ([], [])
    for axis_index, axis in enumerate(axes):
        axis = np.asarray(axis, dtype=float).copy()
        if axis[0] < 0:
            axis = -axis
        tangent = np.cross([0.0, 0.0, 1.0], axis)
        pts = np.asarray(info['collision_points'])
        projections = pts[:, :2] @ np.array([axis[:2], tangent[:2]]).T
        lo, hi = (projections.min(0), projections.max(0))
        center = axis * ((lo[0] + hi[0]) / 2) + tangent * ((lo[1] + hi[1]) / 2)
        half = (hi - lo) / 2
        rotation = Rotation.from_euler('z', np.arctan2(axis[1], axis[0])).as_matrix()
        seed = np.asarray(prior).copy()
        seed[robot.unused_finger_indices] = initial[robot.unused_finger_indices]
        for h in heights:
            shifted = dict(info, center=np.asarray(info['center']) + tangent * shift)
            regions = _s12_rgb_contact_regions(shifted, axis, (h, h), (0.0, 0.0))
            if not regions:
                rejected.append(dict(first_rejection='no_public_surface_intersection', height_fraction=h))
                continue
            goals = np.array([r['point_world_m'] for r in regions])
            inward = -np.array([r['outward_normal_world'] for r in regions])
            q, meta = optimize(robot, seed, goals, inward)
            meta.update(contact_regions=regions, contact_target_method='ray_to_current_RGB_convex_surface', previous_method='global_axis_extrema_at_unrelated_tangent_coordinate')
            meta.update(height_fraction=h, along_fraction=shift / max(2 * half[1], 1e-06), stage3_axis_world=axis.tolist(), axis_index=axis_index, stage3_shift_m=shift, geometry_source=geometry_source)
            _s12_candidate_identity(dict(q=q, points=robot.tips(q)[0], inward=inward, prefix=[], meta=meta))
            if not _s12_endpoint_gate(robot, q, meta, checker, initial, refinement):
                rejected.append(meta)
                continue
            candidate = _s12_complete_candidate(robot, checker, initial, q, inward, meta, retime)
            if candidate is not None:
                results.append(candidate)
                if len(results) >= 2:
                    return (results, rejected)
            else:
                rejected.append(meta)
        if results:
            break
    return (results, rejected)

def _s3_candidates_on_axes(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, retime=False, refinement=False):
    if len(robot.contact_points) != 2 or not _S14_SURFACE_ACTIVE:
        return _S14_EXTREMA_CANDIDATES(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, retime, refinement)
    return _s14_surface_candidates(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, retime, refinement)
_S14_CONTACT_CANDIDATES = rank_contact_candidates

def s11_contact_candidates(s, *args, **kwargs):
    global _S14_SURFACE_ACTIVE
    previous = _S14_SURFACE_ACTIVE
    _S14_SURFACE_ACTIVE = bool(s.get('actual_grasps', 0) >= 1)
    try:
        return _S14_CONTACT_CANDIDATES(s, *args, **kwargs)
    finally:
        _S14_SURFACE_ACTIVE = previous
def closure_object_observation(evidence):
    views = evidence.get('views', {})
    associated = sorted(name for name, row in views.items()
                        if row.get('object_identity_association', False))
    moving = sorted(name for name, row in views.items()
                    if row.get('registered', False)
                    and row.get('motion_evidence', False)
                    and row.get('confidence', 0.0) >= 0.5)
    return dict(
        associated=bool(evidence.get('wrist_fixed', False) and len(associated) >= 2),
        associated_views=associated,
        registered_motion_veto=bool(moving),
        motion_views=moving,
        scope='whole-object color association across named calibrated views; no contact-face or camera-owner requirement',
        contact_observable=False,
    )

_S15_COMPARE_PRELOAD = compare_preload

def compare_preload(before, after, commanded_delta=None):
    result = _S15_COMPARE_PRELOAD(before, after, commanded_delta)
    result['exposure_steps'] = [before.get('step'), after.get('step')]
    result['bounded_probe_association'] = {}
    for name, row in result['views'].items():
        first, last = (before['views'].get(name), after['views'].get(name))
        if not first or not last or first['status'] != 'ok' or (last['status'] != 'ok'):
            continue
        if not first['visible'] or not last['visible'] or first['size'] != last['size']:
            continue
        camera_translation = float(np.linalg.norm(last['camera'][:3, 3] - first['camera'][:3, 3]))
        camera_rotation = _s3s_rotation(first['camera'][:3, :3], last['camera'][:3, :3])
        intersection = int(np.logical_and(first['mask'], last['mask']).sum())
        union = int(np.logical_or(first['mask'], last['mask']).sum())
        iou = intersection / max(1, union)
        shift = float(np.linalg.norm(np.asarray(last['centroid_px']) - first['centroid_px']))
        # Silhouette identity does not assert a stationary camera or object.
        # Exposure FK and the calibrated response above still decide motion.
        identity = bool(min(first['area_px'], last['area_px']) >= 24 and iou >= 0.8 and shift <= 1.0)
        associated = bool(identity and camera_translation <= 0.0001 and camera_rotation <= 0.002)
        row.update(association_iou=iou, centroid_displacement_px=shift, exposure_camera_translation_m=camera_translation, exposure_camera_rotation_rad=camera_rotation, bounded_probe_association=associated, object_identity_association=identity)
        if associated:
            result['bounded_probe_association'].setdefault(row['finger'], []).append(name)
    result['probe_association_thresholds'] = dict(min_area_px=24, min_mask_iou=0.8, max_centroid_displacement_px=1.0, max_camera_translation_m=0.0001, max_camera_rotation_rad=0.002, evidence_scope='same-color silhouette association; not contact')
    result['object_observation'] = closure_object_observation(result)
    return result

def s13_preload_proposal(s, evidence):
    names = s['robot'].finger_names
    q = np.asarray(s['obs']['actual_q'], dtype=float)
    qvel = np.asarray(s['obs']['actual_qvel'], dtype=float)
    error = np.asarray(s['last']) - q
    groups = s5_static(s)['finger_joint_indices']
    errors = {name: float(np.max(abs(error[list(groups[name])]))) for name in names}
    speeds = {name: float(np.max(abs(qvel[list(groups[name])]))) for name in names}
    fresh = bool(evidence.get('step_after') == s.get('closure_evidence_step') and 0 <= int(s['obs']['step']) - int(evidence.get('step_after', -1000)) <= 12 and evidence.get('wrist_fixed') and (not evidence.get('visual_evidence_reused_for_state_only')))
    associated = all((evidence.get('bounded_probe_association', {}).get(name) for name in names))
    settled = all((speeds[name] < 0.06 for name in names))
    tracking = all((errors[name] <= 0.03 for name in names))
    grounded = bool(s.get('s8_previous_decision', {}).get('grounded_evidence'))
    squeeze = np.asarray(s['squeeze'])
    cap = float(s.get('last_closure_cap', S5_CAP))
    reason = dict(eligible=False, fresh=fresh, bilateral_association=associated, settled=settled, absolute_tracking_valid=tracking, grounded_evidence=grounded, max_abs_tracking_error_rad=errors, max_abs_qvel_rad_s=speeds, squeeze_m=squeeze.tolist(), cap_m=cap, thresholds=dict(max_q_error_rad=0.03, max_qvel_rad_s=0.06, probe_increment_m=0.00025, max_evidence_age_steps=12), force_observable=False)
    return (None, dict(reason, reason='preload_disabled_for_paired_control'))

def s13_preload(s):
    before = capture_preload_snapshot(s['obs'], s['robot'], s5_static(s))
    s5_move(s, s['last'], 24, 's13_recovery_observe')
    after = capture_preload_snapshot(s['obs'], s['robot'], s5_static(s))
    evidence = compare_preload(before, after, commanded_delta=s['last'] - before['actual_q'])
    s['closure_evidence_step'] = evidence.get('step_after')
    proposal, reason = s13_preload_proposal(s, evidence)
    s13_log(s, 'fresh_preload_evidence', evidence=evidence)
    s13_log(s, 'considered', branch='preload', **reason)
    return False
_S15_LIFT = lift_and_hold

def s5_lift(s, height=0.03, hold=360, _full=False):
    if height <= 0.0031 and s.pop('s15_pending_reproof', False):
        height = max(height, float(s['obs']['actual_q'][2] - s['c']['q'][2]) + 0.003)
        s['s15_preload_height_floor'] = height
        s13_log(s, 'preload_reproof', target_height_from_original_m=height, additional_wrist_lift_m=0.003, force_known=False)
    if 's15_preload_height_floor' in s:
        height = max(height, s['s15_preload_height_floor'])
    return _S15_LIFT(s, height, hold, _full)
_S15_LOCAL_PROBE = local_pose_probe

def s13_local_probe(mover, state):
    if state.get('s15_local_retry_executed', False):
        s13_log(state, 'guard_refused', branch='local_pose', reason='failed_local_retry_requires_full_release_resample')
        return False
    accepted = _S15_LOCAL_PROBE(mover, state)
    if accepted:
        state['s15_local_retry_executed'] = True
    return accepted
_S15_RELEASE_RECOVER = release_and_resample

def s13_release_recover(s):
    recovered = _S15_RELEASE_RECOVER(s)
    if recovered and s.get('next_branch') == 'local_pose':
        s['s15_local_retry_executed'] = True
    return recovered

def s13_evaluate(s):
    for phase, height, hold in (('proof', 0.003, 48), ('middle', 0.0165, 0), ('final', 0.03, 360)):
        s5_lift(s, height, hold)
        decision = s5_decide(s, phase)
        if decision['action'] == 'reobserve':
            s5_reobserve(s)
            decision = s5_decide(s, phase, reobservations=1)
        if phase == 'final' and decision['action'] == 'stop_unknown':
            target = max(0.03, float(s['obs']['actual_q'][2] - s['c']['q'][2])) + 0.002
            if target <= 0.035 and (not s.get('s15_final_excitation_used', False)):
                s['s15_final_excitation_used'] = True
                s13_log(s, 'unknown_final_excitation', hypothesis='possible_carried_grasp_with_unresolved_RGB', trigger=decision, additional_commanded_lift_m=0.002, target_height_from_grasp_m=target, outcome='requires_independent_trajectory_verification')
                s5_lift(s, target, 360)
                decision = s5_decide(s, phase, reobservations=1)
        if decision['action'] != 'continue_lift':
            return decision
    return decision

def policy_main(ctx):
    s = s5_init(ctx)
    s['actual_grasps'] = 0
    s['next_branch'] = 'initial'
    while s['actual_grasps'] < S5_ATTEMPTS or s.get('preload_ready', False):
        if not s.pop('preload_ready', False):
            local = s.pop('local_retry_ready', False)
            if not local:
                axis = S5_AXIS if not s['actual_grasps'] else 'previous_orthogonal'
                if not s5_plan(s, axis):
                    s13_log(s, 'stopped', reason='no_feasible_next_grasp')
                    return
                s5_approach(s)
                if not s.get('refined'):
                    s5_refine(s)
            s.setdefault('tried_contacts', []).append(dict(points=s['c']['points'].tolist()))
            s['s13_pending_grasp'] = dict(branch=s['next_branch'], recovery_id=s.get('s13_recovery_id'))
            s5_close(s, S5_CLOSURE, S5_CAP)
        decision = s13_evaluate(s)
        s13_log(s, 'failure_decision', decision=decision)
        if decision['action'] != 'retry' or s['actual_grasps'] >= S5_ATTEMPTS:
            return
        if not s.get('s15_preload_used', False) and s13_preload(s):
            s['s15_preload_used'] = True
            s['preload_ready'] = True
            continue
        if s.get('s15_preload_used', False):
            s13_log(s, 'guard_refused', branch='preload', reason='one_bounded_probe_per_episode')
        s.pop('s15_preload_height_floor', None)
        if not s13_release_recover(s):
            return
S11_CONFIG = {'fast_close': True, 'local_retry': False, 'profile': 'fast', 'fingers': 'two', 'task_ranking': True, 'evidence_recovery': True, 'task_wrench': {'mass_range_kg': [0.005, 0.015], 'friction': 0.3, 'normal_uncertainty': 0.15, 'max_normal_force': 1.0, 'lateral_force_n': 0.02, 'torque_nm': 0.0001, 'position_sigma_m': 0.0015, 'margin_tolerance': 0.1, 'pad_radius_m': 0.0012}}
S15_PRELOAD = False

def s16_ik_terms(robot, q, prior, targets, inward, normal_weight):
    q = np.asarray(q)
    prior = np.asarray(prior)
    targets = np.asarray(targets)
    inward = np.asarray(inward)
    p, n, _, poses = robot.tips(q)
    vertices = robot.vertices(poses)
    count = len(targets)
    scale = float(np.sqrt(2 / count))
    errors = np.linalg.norm(p - targets, axis=1)
    angles = np.arccos(np.clip(np.sum(n * inward, axis=1), -1, 1))
    position = (p - targets) * 1500 * scale
    normal = (n - inward) * normal_weight * scale
    table = np.minimum(vertices[:, 2] - robot.ik_checker.table_z - 0.001, 0) * 5000 / np.sqrt(len(vertices))
    wrist = Rotation.from_matrix(Rotation.from_euler('ZYX',q[3:6]).as_matrix() @ Rotation.from_euler('ZYX',prior[3:6]).as_matrix().T).as_rotvec() * .25
    fingers = (q[robot.all_finger_indices] - prior[robot.all_finger_indices]) * .08 * np.sqrt(6/len(robot.all_finger_indices))
    components = dict(position=float(np.sum(position ** 2)), normal=float(np.sum(normal ** 2)), table=float(np.sum(table ** 2)), wrist_regularizer=float(np.sum(wrist ** 2)), finger_regularizer=float(np.sum(fingers ** 2)))
    joint_margin = np.minimum(q - robot.lo, robot.hi - q)
    return dict(contact_count=count, units=dict(position='m', angle='rad', wrist_translation='m', wrist_rotation_and_finger_joints='rad', residual='dimensionless weighted mixed objective'), position_weight_per_m=1500.0, normal_chord_weight=float(normal_weight), contact_count_normalization=scale, per_finger_position_error_m=errors.tolist(), max_position_error_m=float(errors.max()), rms_position_error_m=float(np.sqrt(np.mean(errors ** 2))), per_finger_normal_error_rad=angles.tolist(), max_normal_error_rad=float(angles.max()), rms_normal_error_rad=float(np.sqrt(np.mean(angles ** 2))), per_finger_position_squared_residual=np.sum(position ** 2, axis=1).tolist(), per_finger_normal_squared_residual=np.sum(normal ** 2, axis=1).tolist(), squared_residual_contributions=components, reconstructed_scipy_cost=0.5 * sum(components.values()), per_coordinate_limit_margin=joint_margin.tolist(), minimum_translation_limit_margin_m=float(joint_margin[:3].min()), minimum_angular_limit_margin_rad=float(joint_margin[3:].min()), table_clearance_m=float(vertices[:, 2].min()), q=q.tolist(), prior_q=prior.tolist(), targets_world_m=targets.tolist(), desired_inward_world=inward.tolist(), actual_pad_points_world_m=p.tolist(), actual_pad_normals_world=n.tolist(), hard_gate_scope='each finger position and normal plus full endpoint/path checker; RMS never replaces these gates')
_S16_ORIGINAL_OPTIMIZE = optimize

def optimize(robot, prior, targets, inward, normal_weight=12.0, seed_index=0):
    requested = float(normal_weight)
    common = None
    used = requested if common is None else float(common)
    q, meta = _S16_ORIGINAL_OPTIMIZE(robot, prior, targets, inward, normal_weight=used, seed_index=seed_index)
    meta['s16_numerics'] = s16_ik_terms(robot, q, np.asarray(meta.get('solver_prior',prior)), targets, inward, used)
    if meta.get('collision_refined'):
        terms=meta['s16_numerics']
        contributions=terms['squared_residual_contributions']
        contributions.pop('table')
        collision=robot.ik_checker.optimization_clearance(robot.fk(q))*3000
        contributions['source_mesh_clearance']=float(collision@collision)
        terms['reconstructed_scipy_cost']=.5*sum(contributions.values())
    meta['s16_numerics'].update(objective_phase=meta.get('objective_phase'),requested_normal_weight=requested, used_normal_weight=used, nfev=meta['nfev'], max_nfev=120, solver_success=meta['solver_success'], termination_detail_available=False, search_scope='final fixed-target IK only; region IK is a distinct preceding solver')
    return (q, meta)
S17_COLOR_MODE = 'hs'
