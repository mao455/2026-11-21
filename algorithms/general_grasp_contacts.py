"""CPU contact proposals from robot URDF, actual q and provenance-bearing patches.

All poses are in the URDF root frame, lengths in metres and angles in radians.
Only robot collision meshes referenced by the URDF are loaded. Nothing here
reads object identity/assets, evaluator state, contact sensors or simulator state.
Candidates are conditional hypotheses, not collision/contact/success certificates.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation


def _vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'Expected finite vector of length {size}')
    return result


def _unit(value):
    result = _vector(value)
    length = np.linalg.norm(result)
    if length < 1e-12:
        raise ValueError('A known normal/axis must be nonzero')
    return result / length


def _origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = _vector(np.fromstring(element.get('xyz', '0 0 0'), sep=' '))
        transform[:3, :3] = Rotation.from_euler(
            'xyz', _vector(np.fromstring(element.get('rpy', '0 0 0'), sep=' '))).as_matrix()
    return transform


def _tangents(normal, hints=()):
    for hint in hints:
        projection = hint - normal * np.dot(hint, normal)
        if np.linalg.norm(projection) > 1e-10:
            tangent = _unit(projection)
            return tangent, np.cross(normal, tangent)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = _unit(np.cross(normal, axis))
    return tangent, np.cross(normal, tangent)


@dataclass
class ObjectPatch:
    """A local visual hypothesis; inferred depth is unbounded unless explicit.

    covariance describes metric position uncertainty, normal_uncertainty is an
    angular *bound*, not a calibrated probability. Neither is learned here.
    """

    point: object
    normal: object
    covariance: object
    provenance: str
    normal_uncertainty: float = 0.25
    friction_lower: float = 0.3
    depth_bounded: object = None
    support: tuple = ()
    label: str = ''
    unknown_reasons: tuple = ()
    covariance_caveat: str = ''

    def __post_init__(self):
        if self.provenance not in ('measured', 'inferred', 'unknown'):
            raise ValueError('provenance must be measured, inferred or unknown')
        self.point = None if self.point is None else _vector(self.point)
        self.normal = None if self.normal is None else _unit(self.normal)
        if self.covariance is not None:
            self.covariance = np.asarray(self.covariance, dtype=float)
            if (self.covariance.shape != (3, 3) or not np.all(np.isfinite(self.covariance))
                    or not np.allclose(self.covariance, self.covariance.T, atol=1e-12)
                    or np.linalg.eigvalsh(self.covariance).min() < -1e-12):
                raise ValueError('covariance must be finite, symmetric positive semidefinite')
        if not np.isfinite(self.normal_uncertainty) or not 0 <= self.normal_uncertainty <= np.pi:
            raise ValueError('normal_uncertainty must be an angular bound in [0, pi]')
        if not np.isfinite(self.friction_lower) or self.friction_lower < 0:
            raise ValueError('friction_lower is an explicit nonnegative assumption')
        if self.depth_bounded is None:
            self.depth_bounded = self.provenance == 'measured'
        self.support = tuple(self.support)
        self.unknown_reasons = tuple(self.unknown_reasons)

    @classmethod
    def from_mapping(cls, patch):
        """Accept the perception adapter's position/outward_normal spelling too."""
        return cls(point=patch.get('point', patch.get('position')),
                   normal=patch.get('normal', patch.get('outward_normal')),
                   covariance=patch.get('covariance'), provenance=patch.get('provenance', 'unknown'),
                   normal_uncertainty=patch.get('normal_uncertainty', 0.25),
                   friction_lower=patch.get('friction_lower', 0.3),
                   depth_bounded=patch.get('depth_bounded', None if 'depth_unknown' not in patch else not patch['depth_unknown']),
                   support=patch.get('support', patch.get('source_views', ())),
                   label=patch.get('label', patch.get('method', '')),
                   unknown_reasons=patch.get('unknown_reasons', ()),
                   covariance_caveat=patch.get('covariance_caveat', ''))

    def rejection_reason(self, sigma=2., max_position_uncertainty=0.01):
        if self.provenance == 'unknown':
            return 'unknown_geometry'
        if not self.depth_bounded:
            return 'unbounded_depth'
        if self.point is None or self.normal is None or self.covariance is None:
            return 'missing_metric_position_normal_or_covariance'
        if sigma * np.sqrt(max(0., np.linalg.eigvalsh(self.covariance).max())) > max_position_uncertainty:
            return 'position_uncertainty_exceeds_budget'
        if self.normal_uncertainty >= np.arctan(self.friction_lower):
            return 'normal_uncertainty_erases_robust_friction_cone'
        return None


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    velocity: float
    effort: float
    mimic: object = None


@dataclass
class RobotAnchor:
    link: str
    point: np.ndarray
    normal: np.ndarray
    normal_reliable: bool


@dataclass
class FingerChain:
    name: str
    joint_names: tuple
    distal_links: tuple
    anchors: list = field(default_factory=list)


class URDFHand:
    """Small independent URDF tree implementation, including linear mimic joints."""

    @classmethod
    def from_urdf(cls, path, joint_order=None, package_roots=None, anchors_per_link=32):
        if anchors_per_link < 1:
            raise ValueError('anchors_per_link must be positive')
        hand = cls()
        hand.path = Path(path).resolve()
        tree = ET.parse(hand.path).getroot()
        links = tree.findall('link')
        hand.links = tuple(link.get('name') for link in links)
        if len(set(hand.links)) != len(hand.links):
            raise ValueError('Duplicate link names')
        hand.joints, hand.children, hand.parent_joint = {}, {name: [] for name in hand.links}, {}
        for element in tree.findall('joint'):
            kind = element.get('type')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError(f'Unsupported joint type {kind}: provide a fixed-base URDF')
            limit, mimic = element.find('limit'), element.find('mimic')
            if kind in ('revolute', 'prismatic') and (limit is None or 'lower' not in limit.attrib or 'upper' not in limit.attrib):
                raise ValueError(f'Missing joint bounds for {element.get("name")}')
            joint = Joint(element.get('name'), kind, element.find('parent').get('link'),
                          element.find('child').get('link'), _origin(element.find('origin')),
                          np.zeros(3) if kind == 'fixed' else _unit(np.fromstring(
                              element.find('axis').get('xyz', '1 0 0') if element.find('axis') is not None else '1 0 0', sep=' ')),
                          float(limit.get('lower')) if kind in ('revolute', 'prismatic') else -np.inf,
                          float(limit.get('upper')) if kind in ('revolute', 'prismatic') else np.inf,
                          float(limit.get('velocity', 'inf')) if limit is not None else np.inf,
                          float(limit.get('effort', 'inf')) if limit is not None else np.inf,
                          None if mimic is None else (mimic.get('joint'), float(mimic.get('multiplier', '1')), float(mimic.get('offset', '0'))))
            if joint.name in hand.joints or joint.child in hand.parent_joint:
                raise ValueError('Duplicate joint or multiple parents')
            if joint.parent not in hand.children or joint.child not in hand.children:
                raise ValueError('Joint references an unknown link')
            if (np.isnan(joint.lower) or np.isnan(joint.upper) or joint.lower > joint.upper
                    or joint.velocity <= 0 or np.isnan(joint.velocity) or joint.effort <= 0 or np.isnan(joint.effort)):
                raise ValueError('Invalid URDF limit')
            hand.joints[joint.name] = joint
            hand.children[joint.parent].append(joint)
            hand.parent_joint[joint.child] = joint
        roots = set(hand.links) - set(hand.parent_joint)
        if len(roots) != 1:
            raise ValueError('URDF must have exactly one root')
        hand.root = next(iter(roots))
        hand.topology, visited = [], {hand.root}
        frontier = [hand.root]
        while frontier:
            parent = frontier.pop(0)
            for joint in hand.children[parent]:
                if joint.child in visited:
                    raise ValueError('Cyclic URDF')
                visited.add(joint.child)
                hand.topology.append(joint)
                frontier.append(joint.child)
        if visited != set(hand.links):
            raise ValueError('Disconnected or cyclic URDF')
        independent = tuple(joint.name for joint in hand.joints.values() if joint.kind != 'fixed' and joint.mimic is None)
        hand.joint_order = independent if joint_order is None else tuple(joint_order)
        if len(set(hand.joint_order)) != len(independent) or set(hand.joint_order) != set(independent):
            raise ValueError('joint_order must list every independent movable URDF joint exactly once')
        hand.index = {name: index for index, name in enumerate(hand.joint_order)}
        hand.lower = np.full(len(hand.index), -np.inf)
        hand.upper = np.full(len(hand.index), np.inf)
        hand.velocity = np.full(len(hand.index), np.inf)
        hand.effort = np.array([hand.joints[name].effort for name in hand.joint_order])
        hand.dependencies = {}
        for joint in hand.topology:
            if joint.kind == 'fixed':
                continue
            source, multiplier, offset = hand._dependency(joint.name, set())
            hand.dependencies[joint.name] = source, multiplier, offset
            source_index = hand.index[source]
            if abs(multiplier) < 1e-14:
                if not joint.lower <= offset <= joint.upper:
                    raise ValueError('Constant mimic is outside its joint bounds')
            else:
                limits = sorted(((joint.lower - offset) / multiplier, (joint.upper - offset) / multiplier))
                hand.lower[source_index] = max(hand.lower[source_index], limits[0])
                hand.upper[source_index] = min(hand.upper[source_index], limits[1])
                hand.velocity[source_index] = min(hand.velocity[source_index], joint.velocity / abs(multiplier))
        if np.any(hand.lower > hand.upper):
            raise ValueError('Inconsistent mimic/independent limits')
        hand.geometry_warnings, hand.collision_meshes = [], {}
        for link in links:
            pieces = []
            for collision in link.findall('collision'):
                try:
                    mesh = hand._load_collision(collision.find('geometry'), package_roots or {})
                    mesh.apply_transform(_origin(collision.find('origin')))
                    pieces.append(mesh)
                except (ValueError, OSError, ImportError, KeyError) as error:
                    hand.geometry_warnings.append(f'{link.get("name")}: {error}')
            if pieces:
                import trimesh
                hand.collision_meshes[link.get('name')] = trimesh.util.concatenate(pieces)
        hand.fingers = hand._discover_fingers(anchors_per_link)
        return hand

    def _dependency(self, name, seen):
        if name in seen or name not in self.joints or self.joints[name].kind == 'fixed':
            raise ValueError('Cyclic/missing/fixed mimic source')
        joint = self.joints[name]
        if joint.mimic is None:
            return name, 1., 0.
        parent, multiplier, offset = joint.mimic
        if not np.isfinite(multiplier) or not np.isfinite(offset):
            raise ValueError('Nonfinite mimic relation')
        source, parent_scale, parent_offset = self._dependency(parent, seen | {name})
        return source, multiplier * parent_scale, multiplier * parent_offset + offset

    def _load_collision(self, geometry, package_roots):
        import trimesh
        if geometry is None or len(geometry) != 1:
            raise ValueError('Missing/ambiguous collision geometry')
        shape = geometry[0]
        if shape.tag == 'mesh':
            filename = shape.get('filename', '')
            if filename.startswith('package://'):
                package, relative = filename[len('package://'):].split('/', 1)
                if package not in package_roots:
                    raise ValueError(f'Unknown mesh package {package}; supply package_roots')
                path = Path(package_roots[package]) / relative
            elif filename.startswith('file://'):
                path = Path(filename[len('file://'):])
            elif '://' in filename:
                raise ValueError('Remote mesh URIs are not supported')
            else:
                path = self.path.parent / filename
            mesh = trimesh.load(path, force='mesh', process=True)
            scale = _vector(np.fromstring(shape.get('scale', '1 1 1'), sep=' '))
            if np.any(scale <= 0):
                raise ValueError('Collision scale must be positive')
            mesh.apply_scale(scale)
        elif shape.tag == 'box':
            extents = _vector(np.fromstring(shape.get('size', ''), sep=' '))
            if np.any(extents <= 0):
                raise ValueError('Box extents must be positive')
            mesh = trimesh.creation.box(extents=extents)
        elif shape.tag == 'sphere':
            radius = float(shape.get('radius'))
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError('Sphere radius must be positive')
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        elif shape.tag == 'cylinder':
            radius, height = float(shape.get('radius')), float(shape.get('length'))
            if not np.all(np.isfinite([radius, height])) or min(radius, height) <= 0:
                raise ValueError('Cylinder dimensions must be positive')
            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
        else:
            raise ValueError(f'Unsupported collision geometry {shape.tag}')
        if not len(mesh.faces) or not np.all(np.isfinite(mesh.vertices)):
            raise ValueError('Empty/nonfinite collision mesh')
        mesh.fix_normals(multibody=True)
        return mesh

    def _discover_fingers(self, anchors_per_link):
        movable_below = {}
        for joint in reversed(self.topology):
            movable_below[joint.child] = any(child.kind != 'fixed' or movable_below[child.child]
                                             for child in self.children[joint.child])
        fingers = []
        for distal in self.topology:
            if distal.kind == 'fixed' or movable_below[distal.child]:
                continue
            distal_links, frontier = [], [distal.child]
            while frontier:
                link = frontier.pop(0)
                distal_links.append(link)
                frontier.extend(joint.child for joint in self.children[link] if joint.kind == 'fixed')
            ancestor, joints = distal.child, []
            while ancestor in self.parent_joint:
                joint = self.parent_joint[ancestor]
                if joint.kind != 'fixed':
                    joints.append(joint.name)
                ancestor = joint.parent
            finger = FingerChain(distal.child, tuple(reversed(joints)), tuple(distal_links))
            for link in distal_links:
                mesh = self.collision_meshes.get(link)
                if mesh is None:
                    continue
                points, normals = np.asarray(mesh.triangles_center), np.asarray(mesh.face_normals)
                valid = np.flatnonzero(np.asarray(mesh.area_faces) > 1e-14)
                if not len(valid):
                    continue
                spread = max(float(np.linalg.norm(mesh.extents)), 1e-9)
                features = np.column_stack((points[valid] / spread, normals[valid] * 0.4))
                selected, nearest = [0], np.full(len(valid), np.inf)
                for _ in range(min(anchors_per_link, len(valid)) - 1):
                    nearest = np.minimum(nearest, np.sum((features - features[selected[-1]]) ** 2, axis=1))
                    nearest[selected] = -np.inf
                    selected.append(int(np.argmax(nearest)))
                reliable = bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)
                if not reliable:
                    self.geometry_warnings.append(f'{link}: open/inconsistent robot surface; anchors not actionable')
                finger.anchors.extend(RobotAnchor(link, points[index].copy(), normals[index].copy(), reliable)
                                      for index in valid[selected])
            if not finger.anchors:
                self.geometry_warnings.append(f'{finger.name}: no collision-surface anchors')
            fingers.append(finger)
        return fingers

    def q_vector(self, q):
        values = np.array([q[name] for name in self.joint_order]) if isinstance(q, Mapping) else q
        return _vector(values, len(self.joint_order))

    def bound_violations(self, q, tolerance=1e-9):
        values = self.q_vector(q)
        return [dict(joint=name, measured=float(values[index]), lower=float(self.lower[index]),
                     upper=float(self.upper[index]), excess=float(max(self.lower[index] - values[index],
                                                                   values[index] - self.upper[index], 0.)))
                for index, name in enumerate(self.joint_order)
                if values[index] < self.lower[index] - tolerance or values[index] > self.upper[index] + tolerance]

    def fk(self, q, with_joint_frames=False):
        values = self.q_vector(q)
        poses, frames = {self.root: np.eye(4)}, {}
        for joint in self.topology:
            frame = poses[joint.parent] @ joint.origin
            frames[joint.name] = frame
            motion = np.eye(4)
            if joint.kind != 'fixed':
                source, multiplier, offset = self.dependencies[joint.name]
                position = multiplier * values[self.index[source]] + offset
                if joint.kind == 'prismatic':
                    motion[:3, 3] = joint.axis * position
                else:
                    motion[:3, :3] = Rotation.from_rotvec(joint.axis * position).as_matrix()
            poses[joint.child] = frame @ motion
        return (poses, frames) if with_joint_frames else poses

    def contact_jacobian(self, q, link, point):
        poses, frames = self.fk(q, with_joint_frames=True)
        world_point = poses[link][:3, :3] @ point + poses[link][:3, 3]
        jacobian = np.zeros((3, len(self.joint_order)))
        ancestor = link
        while ancestor in self.parent_joint:
            joint = self.parent_joint[ancestor]
            if joint.kind != 'fixed':
                source, multiplier, _ = self.dependencies[joint.name]
                axis = frames[joint.name][:3, :3] @ joint.axis
                derivative = axis if joint.kind == 'prismatic' else np.cross(axis, world_point - frames[joint.name][:3, 3])
                jacobian[:, self.index[source]] += multiplier * derivative
            ancestor = joint.parent
        return jacobian

    def summary(self):
        return dict(root=self.root, joint_order=list(self.joint_order),
                    fingers=[dict(name=finger.name, joints=list(finger.joint_names),
                                  distal_links=list(finger.distal_links), anchors=len(finger.anchors),
                                  reliable_anchors=sum(anchor.normal_reliable for anchor in finger.anchors))
                             for finger in self.fingers], geometry_warnings=list(self.geometry_warnings),
                    collision_links=len(self.collision_meshes))


def friction_wrench_feasibility(points, normals, *, friction=0.3, normal_uncertainty=0.,
                                covariances=None, sigma=2., max_normal_force=2.,
                                external_wrenches=None, reference=None, cone_edges=8,
                                torque_map=None, effort_limits=None, check_uncertainty=True,
                                gravity_wrench=None, minimum_normal_fraction=.01,
                                load_scale_cap=10., contact_groups=None):
    """Bounded point-contact LP with conservative inscribed friction pyramids.

    Wrenches are [force, torque] expressed about reference. Default tests are
    +/-1 N along each base axis, zero torque: NOT six-dimensional force closure.
    Position scenarios perturb one patch/eigenaxis at a time; passing is not a
    certificate over the full joint covariance ellipsoid. Forces are planned,
    never measured. URDF effort bounds can constrain a supplied J.T map.
    With gravity_wrench supplied, gravity stays fixed while only the task
    disturbance is scaled: W f + gravity + alpha * disturbance = 0. A separate
    alpha=0 solve checks support before any disturbance reserve is reported.
    contact_groups can bind multiple footprint sites to one shared normal-force
    limit. Their identical covariance is applied as a rigid patch translation.
    """
    points, normals = np.asarray(points, dtype=float), np.asarray(normals, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or normals.shape != points.shape:
        raise ValueError('Need two or more point/normal pairs')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError('Nonfinite contact geometry')
    normals = np.array([_unit(normal) for normal in normals])
    count = len(points)
    friction = np.broadcast_to(friction, (count,)).astype(float)
    uncertainty = np.broadcast_to(normal_uncertainty, (count,)).astype(float)
    limits = np.broadcast_to(max_normal_force, (count,)).astype(float)
    groups = np.arange(count) if contact_groups is None else np.asarray(contact_groups)
    if groups.shape != (count,) or groups.dtype.kind not in 'iu' or np.any(groups < 0):
        raise ValueError('contact_groups must contain one nonnegative integer per site')
    group_ids = np.unique(groups)
    group_limits = np.array([limits[np.flatnonzero(groups == group)[0]] for group in group_ids])
    if any(not np.all(limits[groups == group] == limit) for group, limit in zip(group_ids, group_limits)):
        raise ValueError('Sites in one contact group must declare the same shared normal-force limit')
    if (cone_edges < 4 or not np.all(np.isfinite([*friction, *uncertainty, *limits, sigma]))
            or np.any(friction < 0) or np.any(uncertainty < 0) or np.any(limits <= 0) or sigma < 0):
        raise ValueError('Invalid friction, uncertainty, load, sigma or cone edge budget')
    if not np.isfinite(minimum_normal_fraction) or not 0 <= minimum_normal_fraction <= 1:
        raise ValueError('minimum_normal_fraction must be in [0, 1]')
    if not np.isfinite(load_scale_cap) or load_scale_cap <= 0:
        raise ValueError('load_scale_cap must be finite and positive')
    gravity = np.zeros(6) if gravity_wrench is None else _vector(gravity_wrench, 6)
    angle = np.arctan(friction) - uncertainty
    if np.any(angle <= 0):
        return dict(feasible=False, reason='normal_uncertainty_erases_robust_friction_cone', minimum_load_scale=0.)
    robust_friction = np.tan(angle)
    center = points.mean(axis=0) if reference is None else _vector(reference)
    length_scale = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-3)
    loads = np.column_stack((np.vstack((np.eye(3), -np.eye(3))), np.zeros((6, 3)))) if external_wrenches is None else np.atleast_2d(external_wrenches).astype(float)
    if loads.shape[1] != 6 or not len(loads) or not np.all(np.isfinite(loads)) or np.any(np.linalg.norm(loads, axis=1) < 1e-12):
        raise ValueError('Supply nonzero finite external wrenches about reference')
    blocks = []
    basis_hints = [*(points - center), *normals, *loads[:, :3], *loads[:, 3:], gravity[:3], gravity[3:]]
    for normal, coefficient in zip(normals, robust_friction):
        inward = -normal
        tangent, second = _tangents(inward, basis_hints)
        blocks.append(np.array([inward + coefficient * (np.cos(angle) * tangent + np.sin(angle) * second)
                                for angle in np.linspace(0, 2 * np.pi, cone_edges, endpoint=False)]).T)
    force_map = np.zeros((3 * count, count * cone_edges))
    normal_map = np.zeros((len(group_ids), count * cone_edges))
    for index, block in enumerate(blocks):
        force_map[3 * index:3 * index + 3, cone_edges * index:cone_edges * (index + 1)] = block
        group_row = int(np.flatnonzero(group_ids == groups[index])[0])
        normal_map[group_row, cone_edges * index:cone_edges * (index + 1)] = 1.
    inequality = [normal_map, -normal_map]
    bounds = [group_limits, -group_limits * minimum_normal_fraction]
    if torque_map is not None:
        torque_map, effort_limits = np.asarray(torque_map, dtype=float), np.asarray(effort_limits, dtype=float)
        if torque_map.ndim != 2 or torque_map.shape[1] != count * 3 or effort_limits.shape != (torque_map.shape[0],):
            raise ValueError('Invalid joint effort map/limits')
        if not np.all(np.isfinite(torque_map)) or np.any(effort_limits <= 0) or np.any(np.isnan(effort_limits)):
            raise ValueError('Nonfinite torque map or invalid effort limits')
        finite = np.isfinite(effort_limits)
        actuator_map = torque_map[finite] @ force_map
        inequality.extend((actuator_map, -actuator_map))
        bounds.extend((effort_limits[finite], effort_limits[finite]))
    inequality = np.column_stack((np.vstack(inequality), np.zeros(sum(len(bound) for bound in bounds))))
    bounds = np.concatenate(bounds)
    scenarios = [points]
    if covariances is not None:
        covariances = np.asarray(covariances, dtype=float)
        if covariances.shape != (count, 3, 3) or not np.all(np.isfinite(covariances)):
            raise ValueError('Expected finite covariance per contact')
        for group in group_ids:
            members = np.flatnonzero(groups == group)
            covariance = covariances[members[0]]
            if not all(np.allclose(covariances[index], covariance, rtol=0., atol=1e-15) for index in members):
                raise ValueError('Footprint sites in one contact group must share covariance')
            if not np.allclose(covariance, covariance.T, atol=1e-12):
                raise ValueError('Covariance must be symmetric')
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if eigenvalues.min() < -1e-12:
                raise ValueError('Covariance must be positive semidefinite')
            if check_uncertainty:
                for axis in np.flatnonzero(eigenvalues > 1e-16):
                    for sign in (-1., 1.):
                        scenario = points.copy()
                        scenario[members] += sign * sigma * np.sqrt(eigenvalues[axis]) * eigenvectors[:, axis]
                        scenarios.append(scenario)
    minimum_scale, worst_residual, witness, nominal_rank = np.inf, 0., None, None
    solver_statuses = set()
    gravity_supported = True
    gravity_failures = []
    gravity_checks = []
    solve_count = 0
    per_load = [dict(external_wrench=load.tolist(), minimum_load_scale=float('inf'), scenario=None)
                for load in loads]
    for scenario_index, scenario in enumerate(scenarios):
        wrench_map = np.zeros((6, count * cone_edges))
        for index, block in enumerate(blocks):
            wrench_map[:3, index * cone_edges:(index + 1) * cone_edges] = block
            wrench_map[3:, index * cone_edges:(index + 1) * cone_edges] = np.cross(scenario[index] - center, block.T).T / length_scale
        if scenario_index == 0:
            nominal_rank = int(np.linalg.matrix_rank(wrench_map))
        scaled_gravity = gravity.copy()
        scaled_gravity[3:] /= length_scale
        if gravity_wrench is not None:
            support = linprog(np.full(count * cone_edges, 1e-8), A_ub=inequality[:, :-1],
                              b_ub=bounds, A_eq=wrench_map, b_eq=-scaled_gravity,
                              bounds=[(0., None)] * (count * cone_edges), method='highs')
            solve_count += 1
            solver_statuses.add(int(support.status))
            gravity_checks.append(dict(scenario=scenario_index, feasible=bool(support.success),
                                       solver_status=int(support.status),
                                       contact_forces=None if not support.success else (force_map @ support.x).reshape(-1, 3).tolist()))
            if not support.success:
                gravity_supported = False
                gravity_failures.append(dict(scenario=scenario_index, solver_status=int(support.status)))
                # Disturbance support is undefined if the unperturbed lift
                # load is infeasible. A helping disturbance must not masquerade
                # as reserve available while maintaining gravity equilibrium.
                for item in per_load:
                    item.update(minimum_load_scale=0., scenario=scenario_index)
                minimum_scale = 0.
                worst_residual = float('inf')
                witness = dict(scenario=scenario_index, reason='gravity_not_supported',
                               external_wrench=None, contact_forces=None, normal_forces=None)
                continue
        for load_index, load in enumerate(loads):
            scaled_load = load.copy()
            scaled_load[3:] /= length_scale
            equality = np.column_stack((wrench_map, scaled_load))
            objective = np.r_[np.full(count * cone_edges, 1e-8), -1.]
            solution = linprog(objective, A_ub=inequality, b_ub=bounds, A_eq=equality,
                               b_eq=-scaled_gravity, bounds=[(0., None)] * (count * cone_edges) + [(0., load_scale_cap)],
                               method='highs')
            solve_count += 1
            solver_statuses.add(int(solution.status))
            scale = float(solution.x[-1]) if solution.success else 0.
            residual = float(np.max(np.abs(equality @ solution.x + scaled_gravity))) if solution.success else float('inf')
            worst_residual = max(worst_residual, residual)
            if scale < per_load[load_index]['minimum_load_scale']:
                per_load[load_index].update(minimum_load_scale=scale, scenario=scenario_index)
            if scale < minimum_scale:
                minimum_scale = scale
                witness = dict(scenario=scenario_index, external_wrench=load.tolist(),
                               contact_forces=None if not solution.success else (force_map @ solution.x[:-1]).reshape(-1, 3).tolist(),
                               normal_forces=None if not solution.success else (normal_map @ solution.x[:-1]).tolist())
    return dict(feasible=bool(gravity_supported and minimum_scale >= 1. - 1e-6 and worst_residual <= 1e-6),
                minimum_load_scale=float(max(0., minimum_scale)), scale_cap=float(load_scale_cap),
                normalized_residual=worst_residual if np.isfinite(worst_residual) else None,
                nominal_wrench_rank=nominal_rank, full_force_closure_certified=False,
                reference=center.tolist(), external_wrenches=loads.tolist(), robust_friction=robust_friction.tolist(),
                max_normal_force=group_limits.tolist(), contact_groups=groups.tolist(),
                normal_force_scope='shared sum over footprint sites within each contact group',
                uncertainty_scenarios=len(scenarios),
                uncertainty_scope='one-patch principal-axis scenarios; not joint uncertainty certification',
                solver_statuses=sorted(solver_statuses), worst_witness=witness,
                gravity_feasible=gravity_supported, gravity_wrench=gravity.tolist(),
                gravity_failure_scenarios=gravity_failures, gravity_checks=gravity_checks,
                lp_solve_count=solve_count, per_load=per_load,
                load_semantics='fixed gravity plus scaled disturbance' if gravity_wrench is not None else 'scaled entire external wrench',
                force_semantics='hypothetical LP forces at the reported load scale, not measurements',
                effort_semantics='quasistatic contact effort; hand own gravity/dynamics reserve unavailable')


def task_wrench_margin(points, normals, *, reference, mass_range_kg=(.01, .05),
                       friction=.3, normal_uncertainty=.1, covariances=None,
                       max_normal_force=2., torque_map=None, effort_limits=None,
                       gravity_m_s2=(0., 0., -9.81), lift_acceleration_m_s2=(0., 0., .5),
                       lateral_force_n=.05, torque_nm=.0005, sigma=2.,
                       cone_edges=8, remove_third=True, load_scale_cap=100., pad_radius_m=0.):
    """Common, explicitly conditional lift/rotation reserve for any finger group.

    The COM is *assumed* to be ``reference`` (a public geometry estimate), never
    read from simulation. Both mass endpoints are tested. Force/torque tests use
    that same reference and world/root axes for every candidate. Contact normal
    bounds shrink the friction cones; supplied metric covariance adds the same
    one-contact principal-axis scenarios as friction_wrench_feasibility.

    ``lift_feasible`` requires weight plus commanded lift acceleration and the
    specified horizontal perturbation. ``task_feasible`` additionally requires
    all three torque axes. These are separate: two ideal point contacts cannot
    resist torque about their connecting line, even if they can lift. No soft
    finger torsion or measured contact force/pressure is inferred.
    Missing force/mass/friction calibration is made explicit in the result.
    An optional finite pad is represented by four rim sites sharing one total
    normal-force bound. Moments arise only from distributed Coulomb forces,
    with no independent torsion parameter. Its radius must come from static
    pad calibration and is conditional on that footprint actually contacting.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or normals.shape != points.shape:
        raise ValueError('Need two or more contact point/normal pairs')
    radii = np.broadcast_to(pad_radius_m, (len(points),)).astype(float)
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError('pad_radius_m must contain finite nonnegative radii')
    sites, site_normals, group_indices = [], [], []
    for index, (point, normal, radius) in enumerate(zip(points, normals, radii)):
        unit_normal = _unit(normal)
        tangent, second = _tangents(unit_normal, (np.array([0., 0., 1.]),))
        offsets = [np.zeros(3)] if radius == 0 else [radius * tangent, -radius * tangent,
                                                               radius * second, -radius * second]
        for offset in offsets:
            sites.append(point + offset)
            site_normals.append(unit_normal)
            group_indices.append(index)
    sites, site_normals = np.array(sites), np.array(site_normals)
    group_indices = np.array(group_indices, dtype=int)
    site_friction = np.broadcast_to(friction, (len(points),))[group_indices]
    site_normal_uncertainty = np.broadcast_to(normal_uncertainty, (len(points),))[group_indices]
    site_force_limits = np.broadcast_to(max_normal_force, (len(points),))[group_indices]
    if covariances is not None:
        covariances = np.asarray(covariances)
        if covariances.shape != (len(points), 3, 3):
            raise ValueError('Expected one covariance per original pad contact')
    site_covariances = None if covariances is None else covariances[group_indices]
    site_torque_map = None
    if torque_map is not None:
        torque_map = np.asarray(torque_map, float)
        if torque_map.ndim != 2 or torque_map.shape[1] != 3 * len(points):
            raise ValueError('torque_map must contain three force columns per original contact')
        site_torque_map = np.column_stack([torque_map[:, 3 * index:3 * index + 3] for index in group_indices])
    center = _vector(reference)
    masses = np.asarray(mass_range_kg, float)
    if (masses.shape != (2,) or not np.all(np.isfinite(masses)) or
            masses[0] <= 0 or masses[1] < masses[0]):
        raise ValueError('mass_range_kg must contain ordered positive endpoints')
    if not np.isfinite(lateral_force_n) or not np.isfinite(torque_nm) or lateral_force_n <= 0 or torque_nm <= 0:
        raise ValueError('Task force and torque disturbances must be positive')
    gravity = _vector(gravity_m_s2)
    acceleration = _vector(lift_acceleration_m_s2)
    # D'Alembert load in the accelerating hand frame: m * (g - a).
    effective_gravity = gravity - acceleration
    if np.linalg.norm(effective_gravity) < 1e-9:
        raise ValueError('Lift load must have nonzero effective gravity')
    force_axes = np.eye(3)[:2] * lateral_force_n
    force_loads = np.c_[np.vstack((force_axes, -force_axes)), np.zeros((4, 3))]
    torque_axes = np.eye(3) * torque_nm
    torque_loads = np.c_[np.zeros((6, 3)), np.vstack((torque_axes, -torque_axes))]
    loads = np.vstack((force_loads, torque_loads))
    names = ['force_x+', 'force_y+', 'force_x-', 'force_y-',
             'torque_x+', 'torque_y+', 'torque_z+', 'torque_x-', 'torque_y-', 'torque_z-']
    trials = []
    for mass in np.unique(masses):
        result = friction_wrench_feasibility(
            sites, site_normals, friction=site_friction, normal_uncertainty=site_normal_uncertainty,
            covariances=site_covariances, sigma=sigma, max_normal_force=site_force_limits,
            external_wrenches=loads, reference=center, cone_edges=cone_edges,
            torque_map=site_torque_map, effort_limits=effort_limits,
            gravity_wrench=np.r_[mass * effective_gravity, np.zeros(3)],
            minimum_normal_fraction=0., load_scale_cap=load_scale_cap, contact_groups=group_indices)
        trials.append(dict(mass_kg=float(mass), **result))
    per_load = []
    for index, name in enumerate(names):
        records = [dict(mass_kg=trial['mass_kg'], **trial['per_load'][index])
                   for trial in trials if 'per_load' in trial]
        worst = min(records, key=lambda item: item['minimum_load_scale']) if records else dict(minimum_load_scale=0.)
        per_load.append(dict(name=name, **worst))
    lateral_margin = min(item['minimum_load_scale'] for item in per_load[:4])
    torque_margin = min(item['minimum_load_scale'] for item in per_load[4:])
    gravity_ok = all(trial.get('gravity_feasible', False) for trial in trials)
    result = dict(
        gravity_feasible=gravity_ok,
        lift_feasible=bool(gravity_ok and lateral_margin >= 1. - 1e-6),
        task_feasible=bool(gravity_ok and min(lateral_margin, torque_margin) >= 1. - 1e-6),
        minimum_task_margin=float(min(lateral_margin, torque_margin)),
        lateral_margin=float(lateral_margin), torque_margin=float(torque_margin),
        margin_scale_cap=float(load_scale_cap),
        reference=center.tolist(), contact_points=points.tolist(), outward_normals=normals.tolist(),
        footprint_points=sites.tolist(), footprint_contact_groups=group_indices.tolist(),
        per_load=per_load, mass_trials=trials,
        assumptions=dict(mass_range_kg=masses.tolist(), friction_lower=np.asarray(friction).tolist(),
                         normal_uncertainty_rad=np.asarray(normal_uncertainty).tolist(),
                         max_normal_force_n=np.asarray(max_normal_force).tolist(),
                         gravity_m_s2=gravity.tolist(), lift_acceleration_m_s2=acceleration.tolist(),
                         horizontal_disturbance_n=float(lateral_force_n), torque_disturbance_nm=float(torque_nm),
                         com='assumed at common public geometry reference',
                         calibration='mass, friction, normal-force limits and acceleration are declared assumptions, not measurements',
                         pad_radius_m=radii.tolist(),
                         force_model='four rim Coulomb sites per finite pad; shared per-pad normal-force sum; no intrinsic torsion' if np.any(radii) else 'bounded point contacts; no pad torsional friction',
                         footprint='assumed planar contacting footprint within static calibrated pad; not observed contact area',
                         jacobian='center-contact Jacobian repeated at rim sites; local kinematic approximation',
                         effort='URDF incremental contact effort only; actuator gravity/dynamics reserve unavailable',
                         uncertainty='normal bound plus one-contact covariance axes; not a full joint uncertainty certificate'),
        input_scope='public contact geometry, candidate-q Jacobians, static hand limits; no object truth',
        ranking_scope='feasibility gates first; compare common task margin and lateral/torque terms; retain uncertainty caveats')
    if remove_third and len(points) == 3:
        arguments = dict(reference=center, mass_range_kg=masses, gravity_m_s2=gravity,
                         lift_acceleration_m_s2=acceleration, lateral_force_n=lateral_force_n,
                         torque_nm=torque_nm, sigma=sigma, cone_edges=cone_edges, remove_third=False,
                         load_scale_cap=load_scale_cap, pad_radius_m=radii[:2])
        for key, value in [('friction', friction), ('normal_uncertainty', normal_uncertainty),
                           ('max_normal_force', max_normal_force)]:
            arguments[key] = np.broadcast_to(value, (3,))[:2]
        if covariances is not None:
            arguments['covariances'] = np.asarray(covariances)[:2]
        if torque_map is not None:
            arguments['torque_map'] = np.asarray(torque_map)[:, :6]
            arguments['effort_limits'] = effort_limits
        baseline = task_wrench_margin(points[:2], normals[:2], **arguments)
        result['without_third_contact'] = {key: baseline[key] for key in
                                           ('gravity_feasible', 'lift_feasible', 'task_feasible',
                                            'minimum_task_margin', 'lateral_margin', 'torque_margin')}
        result['third_contact_increment'] = {key: result[key] - baseline[key] for key in
                                             ('minimum_task_margin', 'lateral_margin', 'torque_margin')}
    return result


@dataclass
class ContactTarget:
    finger: str
    anchor: RobotAnchor
    patch_index: int
    patch: ObjectPatch


@dataclass
class IKResult:
    q: np.ndarray
    feasible: bool
    position_errors: np.ndarray
    normal_errors: np.ndarray
    evaluations: int


@dataclass
class PlanResult:
    q: object
    contacts: list
    feasible: bool
    diagnostics: dict

    def as_dict(self):
        return dict(q=None if self.q is None else self.q.tolist(), feasible=self.feasible,
                    contacts=[dict(finger=contact.finger, link=contact.anchor.link,
                                   robot_local_anchor=contact.anchor.point.tolist(),
                                   robot_local_normal=contact.anchor.normal.tolist(),
                                   target_point=contact.patch.point.tolist(), target_outward_normal=contact.patch.normal.tolist(),
                                   patch_index=contact.patch_index, provenance=contact.patch.provenance,
                                   covariance=contact.patch.covariance.tolist(),
                                   normal_uncertainty=contact.patch.normal_uncertainty,
                                   support=list(contact.patch.support),
                                   unknown_reasons=list(contact.patch.unknown_reasons),
                                   covariance_caveat=contact.patch.covariance_caveat) for contact in self.contacts],
                    diagnostics=self.diagnostics)


class ContactPlanner:
    def __init__(self, hand, *, position_tolerance=0.002, normal_tolerance=0.6,
                 max_position_uncertainty=0.01, max_normal_force=2., beam_width=24,
                 anchors_per_pair=2, candidates_per_count=6, max_evaluations=70):
        self.hand = hand
        if (min(position_tolerance, normal_tolerance, max_position_uncertainty, max_normal_force) <= 0
                or min(beam_width, anchors_per_pair, candidates_per_count, max_evaluations) < 1):
            raise ValueError('Positive planner tolerances/budgets are required')
        self.position_tolerance, self.normal_tolerance = position_tolerance, normal_tolerance
        self.max_position_uncertainty, self.max_normal_force = max_position_uncertainty, max_normal_force
        self.beam_width, self.anchors_per_pair = beam_width, anchors_per_pair
        self.candidates_per_count, self.max_evaluations = candidates_per_count, max_evaluations

    def _bounds(self, q, max_delta=None, time_horizon=None):
        violations = self.hand.bound_violations(q)
        if violations:
            raise ValueError(f'Actual q violates URDF bounds: {violations}')
        lower, upper = self.hand.lower.copy(), self.hand.upper.copy()
        for index, name in enumerate(self.hand.joint_order):
            if self.hand.joints[name].kind == 'continuous':
                lower[index] = max(lower[index], q[index] - np.pi)
                upper[index] = min(upper[index], q[index] + np.pi)
        if max_delta is not None:
            delta = np.broadcast_to(max_delta, q.shape).astype(float)
            if np.any(delta < 0) or not np.all(np.isfinite(delta)):
                raise ValueError('max_delta must be finite and nonnegative')
            lower, upper = np.maximum(lower, q - delta), np.minimum(upper, q + delta)
        if time_horizon is not None:
            if not np.isfinite(time_horizon) or time_horizon <= 0:
                raise ValueError('time_horizon must be positive')
            delta = self.hand.velocity * time_horizon
            lower, upper = np.maximum(lower, q - delta), np.minimum(upper, q + delta)
        return lower, upper

    def solve_ik(self, q, contacts, *, initial_q=None, max_delta=None, time_horizon=None):
        actual = self.hand.q_vector(q)
        lower, upper = self._bounds(actual, max_delta, time_horizon)
        active = set()
        for contact in contacts:
            ancestor = contact.anchor.link
            while ancestor in self.hand.parent_joint:
                joint = self.hand.parent_joint[ancestor]
                if joint.kind != 'fixed':
                    source, multiplier, _ = self.hand.dependencies[joint.name]
                    if abs(multiplier) > 1e-14:
                        active.add(self.hand.index[source])
                ancestor = joint.parent
        active = np.array(sorted(index for index in active if upper[index] - lower[index] > 1e-12), dtype=int)
        scale = np.maximum(upper[active] - lower[active], 1e-6)

        def errors(values, regularize):
            candidate = actual.copy()
            candidate[active] = values
            poses, residual = self.hand.fk(candidate), []
            for contact in contacts:
                transform = poses[contact.anchor.link]
                position = transform[:3, :3] @ contact.anchor.point + transform[:3, 3]
                normal = transform[:3, :3] @ contact.anchor.normal
                residual.extend((position - contact.patch.point) / self.position_tolerance)
                residual.extend((normal + contact.patch.normal) / self.normal_tolerance)
            if regularize:
                residual.extend(0.02 * (values - actual[active]) / scale)
            return np.asarray(residual)

        candidate = actual.copy()
        evaluations = 0
        if len(active):
            seed = actual if initial_q is None else self.hand.q_vector(initial_q)
            solution = least_squares(lambda values: errors(values, True), np.clip(seed[active], lower[active], upper[active]),
                                     bounds=(lower[active], upper[active]), max_nfev=self.max_evaluations,
                                     ftol=1e-7, xtol=1e-7, gtol=1e-7)
            candidate[active] = solution.x
            evaluations = int(solution.nfev)
        residual = errors(candidate[active], False).reshape(-1, 6)
        position_errors = np.linalg.norm(residual[:, :3], axis=1) * self.position_tolerance
        chord = np.linalg.norm(residual[:, 3:], axis=1) * self.normal_tolerance
        normal_errors = 2. * np.arcsin(np.clip(chord / 2., 0., 1.))
        return IKResult(candidate, bool(np.all(position_errors <= self.position_tolerance) and
                                       np.all(normal_errors <= self.normal_tolerance)),
                        position_errors, normal_errors, evaluations)

    def _wrench(self, q, contacts, external_wrenches, reference, check_uncertainty):
        torque_map = np.column_stack([self.hand.contact_jacobian(q, contact.anchor.link, contact.anchor.point).T
                                      for contact in contacts])
        return friction_wrench_feasibility(
            [contact.patch.point for contact in contacts], [contact.patch.normal for contact in contacts],
            friction=[contact.patch.friction_lower for contact in contacts],
            normal_uncertainty=[contact.patch.normal_uncertainty for contact in contacts],
            covariances=[contact.patch.covariance for contact in contacts],
            max_normal_force=self.max_normal_force, external_wrenches=external_wrenches, reference=reference,
            torque_map=torque_map, effort_limits=self.hand.effort, check_uncertainty=check_uncertainty)

    def candidate_quality(self, actual_q, contacts, solution, wrench):
        """Comparable per-finger quality; lower scores rank first.

        This estimates command travel time from URDF velocity limits. It is a
        lower bound, not an actuator settling prediction or a contact proof.
        Each finger still has to pass solve_ik's individual error thresholds.
        """
        actual = self.hand.q_vector(actual_q)
        delta = np.abs(solution.q - actual)
        velocity = self.hand.velocity
        travel_s = float(np.max(np.divide(delta, velocity, out=np.zeros_like(delta),
                                         where=np.isfinite(velocity) & (velocity > 0))))
        position = float(np.mean(solution.position_errors / self.position_tolerance))
        normal = float(np.mean(solution.normal_errors / self.normal_tolerance))
        uncertainty = float(np.mean([
            np.sqrt(np.trace(contact.patch.covariance)) / self.max_position_uncertainty
            + .1 * (contact.patch.provenance == 'inferred') for contact in contacts]))
        support = float(min(wrench['minimum_load_scale'], 5.))
        return dict(position_error_per_finger=position, normal_error_per_finger=normal,
                    uncertainty_per_finger=uncertainty, minimum_command_travel_s=travel_s,
                    support_load_scale=support,
                    score=position + normal + uncertainty + .05 * travel_s - .2 * support)

    def plan(self, q, patches, *, external_wrenches=None, reference=None, max_delta=None,
             time_horizon=None, safety_check=None, contact_counts=None, finger_groups=None):
        """Return a bounded proposal, never an action or an unknown-space proof.

        safety_check is the existing legal robot/table safety adapter, accepting
        q in joint_order and returning bool or {'passed': bool, ...}. It must not
        consult private object state. All live images remain the caller's inputs.
        contact_counts=(2,), (3,) or (2, 3) selects fixed/automatic counts;
        finger_groups optionally restricts legal combinations by discovered name.
        The same IK, individual error gates, safety callback and wrench checks
        apply to every permitted count. No mode changes the execution guards.
        """
        actual = self.hand.q_vector(q)
        self._bounds(actual, max_delta, time_horizon)
        known_fingers = {finger.name for finger in self.hand.fingers}
        groups = None if finger_groups is None else tuple(frozenset(group) for group in finger_groups)
        if groups is not None and (not groups or any(len(group) < 2 or not group <= known_fingers for group in groups)):
            raise ValueError('finger_groups must contain legal groups of at least two discovered fingers')
        counts = None if contact_counts is None else tuple(contact_counts)
        if counts is not None and (not counts or any(isinstance(count, bool) or not isinstance(count, (int, np.integer))
                                                    or count < 2 for count in counts)):
            raise ValueError('contact_counts must contain integers of at least two')
        counts = None if counts is None else set(map(int, counts))
        if groups is not None and counts is not None:
            groups = tuple(group for group in groups if len(group) in counts)
            if not groups:
                raise ValueError('finger_groups and contact_counts have no common supported group')
        patches = [ObjectPatch.from_mapping(patch) if isinstance(patch, Mapping) else patch for patch in patches]
        rejected, usable = [], []
        for index, patch in enumerate(patches):
            reason = patch.rejection_reason(max_position_uncertainty=self.max_position_uncertainty)
            if reason:
                rejected.append(dict(patch=index, reason=reason, provenance=patch.provenance))
            else:
                usable.append((index, patch))
        diagnostics = dict(joint_order=list(self.hand.joint_order), rejected_patches=rejected,
                           discovered_fingers=len(self.hand.fingers), geometry_warnings=list(self.hand.geometry_warnings),
                           collision_certified=False, physical_contact='unknown', strict_success='not_evaluated',
                           robot_anchor_semantics='collision surface samples, not validated contact pads or exposed assembly surfaces',
                           unknown_space='not free; unseen-object/swept-volume safety requires the execution adapter',
                           input_scope='URDF robot geometry, actual q and visual local patches only',
                           bounded_search=dict(beam_width=self.beam_width, anchors_per_pair=self.anchors_per_pair,
                                               candidates_per_count=self.candidates_per_count),
                           requested_contact_counts=None if counts is None else sorted(counts),
                           requested_finger_groups=None if groups is None else [sorted(group) for group in groups],
                           ranking_semantics='mean position/normal error and uncertainty per finger; all fingers pass; command-time lower bound and support quality',
                           candidate_qualities=[], tested_contact_counts=[], evaluated_combinations=0,
                           combination_failures=dict(ik=0, safety=0, nominal_wrench=0, uncertain_wrench=0))
        if len(usable) < 2:
            diagnostics['reason'] = 'fewer_than_two_actionable_patches'
            return PlanResult(None, [], False, diagnostics)
        poses, options = self.hand.fk(actual), []
        for finger_index, finger in enumerate(self.hand.fingers):
            anchors = [anchor for anchor in finger.anchors if anchor.normal_reliable]
            for patch_index, patch in usable:
                ranked = []
                for anchor in anchors:
                    transform = poses[anchor.link]
                    distance = np.linalg.norm(transform[:3, :3] @ anchor.point + transform[:3, 3] - patch.point)
                    opposition = np.linalg.norm(transform[:3, :3] @ anchor.normal + patch.normal)
                    ranked.append((float(distance / self.position_tolerance + opposition), anchor))
                for _, anchor in sorted(ranked, key=lambda item: item[0])[:self.anchors_per_pair]:
                    contact = ContactTarget(finger.name, anchor, patch_index, patch)
                    solution = self.solve_ik(actual, [contact], max_delta=max_delta, time_horizon=time_horizon)
                    if solution.feasible:
                        cost = float(np.linalg.norm(solution.q - actual) + solution.position_errors.sum() / self.position_tolerance)
                        options.append((finger_index, patch_index, contact, solution.q, cost))
        diagnostics['reachable_finger_patch_options'] = len(options)
        beams, finalists = [([], -1, set(), 0.)], []
        maximum = min(len(self.hand.fingers), len(usable))
        if counts is not None:
            maximum = min(maximum, max(counts))
        if groups is not None:
            maximum = min(maximum, max(map(len, groups)))
        for count in range(1, maximum + 1):
            expanded = []
            for choices, last_finger, used_patches, score in beams:
                for option in options:
                    finger_index, patch_index, contact, _, cost = option
                    if finger_index <= last_finger or patch_index in used_patches:
                        continue
                    selected = frozenset([choice[2].finger for choice in choices] + [contact.finger])
                    if groups is not None and not any(selected <= group for group in groups):
                        continue
                    if any(np.linalg.norm(contact.patch.point - previous[2].patch.point) < 1e-5 for previous in choices):
                        continue
                    expanded.append((choices + [option], finger_index, used_patches | {patch_index}, score + cost))
            beams = sorted(expanded, key=lambda beam: beam[3])[:self.beam_width]
            if count < 2 or not beams or (counts is not None and count not in counts):
                continue
            diagnostics['tested_contact_counts'].append(count)
            for choices, _, _, cost in beams[:self.candidates_per_count]:
                contacts = [choice[2] for choice in choices]
                if groups is not None and frozenset(contact.finger for contact in contacts) not in groups:
                    continue
                seed = np.mean([choice[3] for choice in choices], axis=0)
                solution = self.solve_ik(actual, contacts, initial_q=seed, max_delta=max_delta, time_horizon=time_horizon)
                diagnostics['evaluated_combinations'] += 1
                if not solution.feasible:
                    diagnostics['combination_failures']['ik'] += 1
                    continue
                safety = None if safety_check is None else safety_check(solution.q.copy())
                if safety is not None and not (safety.get('passed', False) if isinstance(safety, Mapping) else bool(safety)):
                    diagnostics['combination_failures']['safety'] += 1
                    continue
                wrench = self._wrench(solution.q, contacts, external_wrenches, reference, False)
                if not wrench['feasible']:
                    diagnostics['combination_failures']['nominal_wrench'] += 1
                    continue
                quality = self.candidate_quality(actual, contacts, solution, wrench)
                diagnostics['candidate_qualities'].append(dict(fingers=[contact.finger for contact in contacts], **quality))
                finalists.append((quality['score'], contacts, solution, safety, quality))
        for score, contacts, solution, safety, quality in sorted(finalists, key=lambda item: item[0]):
            wrench = self._wrench(solution.q, contacts, external_wrenches, reference, True)
            if not wrench['feasible']:
                diagnostics['combination_failures']['uncertain_wrench'] += 1
                continue
            diagnostics.update(reason='conditional_candidate', wrench=wrench, score=score, quality=quality,
                               position_errors=solution.position_errors.tolist(), normal_errors=solution.normal_errors.tolist(),
                               safety_adapter=safety, safety_adapter_checked=safety_check is not None,
                               execution_ready=False, requires_closed_loop_validation=True)
            return PlanResult(solution.q, contacts, True, diagnostics)
        diagnostics.update(reason='bounded_search_found_no_feasible_combination', execution_ready=False)
        return PlanResult(None, [], False, diagnostics)


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('urdf', type=Path)
    args = parser.parse_args()
    print(json.dumps(URDFHand.from_urdf(args.urdf).summary(), indent=2))
