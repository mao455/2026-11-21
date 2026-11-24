"""Shared rooted hand topology and FK, extracted unchanged from the clean mainline.

The runner and policy use this module with the declared command order. URDF
and meshes are read-only inputs; runtime conversion lives in each episode.
"""
from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


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

@dataclass(frozen=True)
class PadCalibration:
    finger: str
    link: str
    point_m: tuple[float, float, float]
    outward_normal: tuple[float, float, float]
    joint_names: tuple[str, ...]
    support_radius_m: float | None = None

class HandDescription:
    """Order-independent topology with explicit actuator and optical semantics.

    A different hand needs new PadCalibration/camera calibration, not a change
    to FK, contact transforms or joint-group discovery. The runner declares
    actuator order and dimension for the selected URDF.
    """

    def __init__(self, urdf_path, joint_names=None, pads=(), cameras=None, measured_response=None, base_joint_names=()):
        self.urdf_path = str(Path(urdf_path).resolve())
        tree = ET.parse(self.urdf_path).getroot()
        self.robot_name = tree.get('name')
        self.links = tuple((link.get('name') for link in tree.findall('link')))
        active = {j.get('name'): j for j in tree.findall('joint') if j.get('type') != 'fixed'}
        self.joint_names = tuple(joint_names or active)
        if len(set(self.joint_names)) != len(self.joint_names) or set(self.joint_names) != set(active):
            raise ValueError('The command joint order must contain each active URDF joint exactly once')
        self.index = {name: i for i, name in enumerate(self.joint_names)}
        self.base_joint_names = tuple(base_joint_names)
        self.pads = tuple(pads)
        self.cameras = dict(cameras or {})
        self.measured_response = dict(measured_response or {})
        self.lower = np.full(len(self.joint_names), -np.inf)
        self.upper = np.full(len(self.joint_names), np.inf)
        self.urdf_velocity = np.full(len(self.joint_names), np.nan)
        self.joint_types = [None] * len(self.joint_names)
        joints, children = ([], set())
        for j in tree.findall('joint'):
            name, kind = (j.get('name'), j.get('type'))
            if kind not in ('fixed', 'prismatic', 'revolute', 'continuous'):
                raise ValueError('Unsupported URDF joint type: ' + str(kind))
            parent, child = (j.find('parent').get('link'), j.find('child').get('link'))
            index = self.index.get(name)
            axis = _vector(j.find('axis'), 'xyz') if j.find('axis') is not None else np.array([1.0, 0.0, 0.0])
            if index is not None:
                self.joint_types[index] = kind
                limit = j.find('limit')
                if limit is not None:
                    self.lower[index] = float(limit.get('lower', '-inf'))
                    self.upper[index] = float(limit.get('upper', 'inf'))
                    self.urdf_velocity[index] = float(limit.get('velocity', 'nan'))
            joints.append((parent, child, _origin(j.find('origin')), axis, kind, index))
            children.add(child)
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError('Expected one rooted hand tree')
        self.root = roots.pop()
        self.joints, known, pending = ([], {self.root}, joints)
        while pending:
            ready = [j for j in pending if j[0] in known]
            if not ready:
                raise ValueError('Disconnected or cyclic URDF')
            for joint in ready:
                self.joints.append(joint)
                known.add(joint[1])
                pending.remove(joint)
        self.collision_meshes = []
        self.collision_count = len(tree.findall('.//collision'))
        for link in tree.findall('link'):
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                if mesh is not None:
                    self.collision_meshes.append(dict(link=link.get('name'), path=str(resolve_mesh(self.urdf_path, mesh.get('filename'))), scale=_vector(mesh, 'scale', '1 1 1').tolist(), origin=_origin(collision.find('origin')).tolist()))
        self.camera_mount_candidates = tuple((name for name in self.links if 'cam' in name.lower() or 'optical' in name.lower()))
        for pad in self.pads:
            if pad.link not in self.links or not set(pad.joint_names).issubset(self.index):
                raise ValueError('Pad calibration does not match the URDF')
            if not np.isclose(np.linalg.norm(pad.outward_normal), 1.0):
                raise ValueError('Pad outward normal must be unit length')

    def structural_collision_exclusions(self, links=None):
        """Exclude one rigid assembly or direct movable neighbours only.

        The planner and simulator share this topology rule; distant links and
        all inter-finger pairs remain subject to physical collision checks.
        """
        import itertools
        roots = {name: name for name in self.links}
        def root(name):
            while roots[name] != name:
                roots[name] = roots[roots[name]]
                name = roots[name]
            return name
        for parent, child, _, _, kind, _ in self.joints:
            if kind == 'fixed':
                roots[root(child)] = root(parent)
        adjacent = {tuple(sorted((root(parent), root(child))))
                    for parent, child, _, _, kind, _ in self.joints if kind != 'fixed'}
        return {tuple(sorted((a, b))) for a, b in itertools.combinations(links or self.links, 2)
                if root(a) == root(b) or tuple(sorted((root(a), root(b)))) in adjacent}

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.joint_names),) or not np.all(np.isfinite(q)):
            raise ValueError('Invalid actual joint vector')
        poses = {self.root: np.eye(4)}
        for parent, child, fixed, axis, kind, index in self.joints:
            transform = fixed.copy()
            if index is not None:
                if kind in ('revolute', 'continuous'):
                    transform[:3, :3] = fixed[:3, :3] @ Rotation.from_rotvec(axis * q[index]).as_matrix()
                elif kind == 'prismatic':
                    transform[:3, 3] += fixed[:3, :3] @ (axis * q[index])
            poses[child] = poses[parent] @ transform
        return poses

    def pads_world(self, q):
        poses = self.fk(q)
        points = np.array([poses[p.link][:3, :3] @ p.point_m + poses[p.link][:3, 3] for p in self.pads])
        normals = np.array([poses[p.link][:3, :3] @ p.outward_normal for p in self.pads])
        return (points, normals)

    @property
    def finger_joint_indices(self):
        return {p.finger: tuple((self.index[name] for name in p.joint_names)) for p in self.pads}

    def deployment_gaps(self, required_cameras=2, compatible_action_dimension=None):
        gaps = []
        missing = sorted({m['path'] for m in self.collision_meshes if not Path(m['path']).is_file()})
        if missing:
            gaps.append(dict(kind='missing_collision_mesh', count=len(missing), paths=missing))
        if len(self.pads) < 2:
            gaps.append(dict(kind='pad_contact_frames_unavailable', required='two calibrated opposed pads'))
        if len(self.cameras) < required_cameras:
            gaps.append(dict(kind='camera_calibration_unavailable', supplied=len(self.cameras), required=required_cameras, mount_candidates=list(self.camera_mount_candidates), fields='logical view/group, optical transform/convention, intrinsics/distortion, resolution'))
        for name, camera in self.cameras.items():
            if camera.get('camera_link') not in self.links or not camera.get('optical_axis_convention'):
                gaps.append(dict(kind='incomplete_camera_frame', camera=name))
        if not self.measured_response:
            gaps.append(dict(kind='actuator_response_uncalibrated', required='tracking response, bounded closure margin, measured speed'))
        if compatible_action_dimension is not None and len(self.joint_names) != compatible_action_dimension:
            gaps.append(dict(kind='host_action_dimension', actual=len(self.joint_names), compatible=compatible_action_dimension))
        return gaps

    def summary(self):
        return dict(robot_name=self.robot_name, urdf_path=self.urdf_path, root=self.root, action_dimension=len(self.joint_names), joint_names=list(self.joint_names), joint_types=self.joint_types, collision_count=self.collision_count, collision_mesh_references=len(self.collision_meshes), missing_collision_meshes=sorted({m['path'] for m in self.collision_meshes if not Path(m['path']).is_file()}), camera_mount_candidates=list(self.camera_mount_candidates), calibrated_camera_count=len(self.cameras), finger_joint_indices=self.finger_joint_indices, deployment_gaps=self.deployment_gaps())
