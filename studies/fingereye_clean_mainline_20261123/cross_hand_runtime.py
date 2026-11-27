"""URDF selection and immutable runtime description shared by host and backend.

Only generated derivatives are written. Source URDF/STL bytes are fingerprinted.
Six declared world-to-palm coordinates are an interface actuator, not hand DOFs.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'mainline'))
from hand_description import HandDescription, resolve_mesh

BASE_NAMES = ('base_x', 'base_y', 'base_z', 'base_yaw', 'base_pitch', 'base_roll')


def normalize_virtual_palm(root):
    """Reparameterize a geometry-free 3P+3R prefix as the public XYZ/ZYX chart.

    No physical link or finger joint is changed. Original roll/pitch/yaw chart
    coordinates are related by equal rotation matrices, not by renaming angles.
    """
    links = {link.get('name'): link for link in root.findall('link')}
    joints = root.findall('joint')
    if set(BASE_NAMES) <= {j.get('name') for j in joints}:
        return None
    children = {j.find('child').get('link') for j in joints}
    current = next(iter(set(links) - children))
    prefix = []
    while len(prefix) < 6:
        link = links[current]
        if link.find('visual') is not None or link.find('collision') is not None:
            return None
        downstream = [j for j in joints if j.find('parent').get('link') == current]
        if len(downstream) != 1 or downstream[0].get('type') == 'fixed':
            return None
        joint = downstream[0]
        origin = joint.find('origin')
        if origin is not None and not np.allclose(np.r_[np.fromstring(origin.get('xyz', '0 0 0'), sep=' '), np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')], 0):
            return None
        prefix.append(joint)
        current = joint.find('child').get('link')
    kinds = [j.get('type') for j in prefix]
    axes = [np.fromstring(j.find('axis').get('xyz'), sep=' ') for j in prefix]
    if kinds != ['prismatic'] * 3 + ['revolute'] * 3 or not np.allclose(axes[:3], np.eye(3)):
        return None
    rotations = {int(np.argmax(axis)): j for axis, j in zip(axes[3:], prefix[3:])
                 if np.allclose(axis, np.eye(3)[np.argmax(axis)])}
    if set(rotations) != {0, 1, 2}:
        return None
    original = [dict(name=j.get('name'), axis=axis.tolist(), kind=j.get('type')) for j, axis in zip(prefix, axes)]
    ordered = prefix[:3] + [rotations[2], rotations[1], rotations[0]]
    # The original empty links are retained; only their virtual coordinate chart changes.
    parents = [prefix[0].find('parent').get('link')] + [j.find('child').get('link') for j in prefix[:-1]]
    targets = parents[1:] + [current]
    names = {}
    for joint, name, parent, child in zip(ordered, BASE_NAMES, parents, targets):
        names[joint.get('name')] = name
        joint.set('name', name)
        joint.find('parent').set('link', parent)
        joint.find('child').set('link', child)
    return dict(source_coordinates=original, source_joint_to_interface_name=names,
                palm_link=current, runtime_rotation='Rz(base_yaw) Ry(base_pitch) Rx(base_roll)',
                mapping='source rotational angles = Rotation.from_euler(ZYX, public[yaw,pitch,roll]).as_euler(source intrinsic axis sequence)',
                scope='coordinate reparameterization of six geometry-free virtual joints; no added DOFs')


def install_simulation_modules(root, calibration):
    """Attach a declared shared simulation camera module to fixed URDF housings."""
    pool_name = calibration.get('simulation_camera_pool')
    if not pool_name:
        return []
    pool = json.loads((HERE / 'hand_calibrations' / pool_name).read_text())
    mounts = []
    for module in calibration['simulation_modules']:
        housing = module['housing_link']
        fixed = next(j for j in root.findall('joint') if j.find('child').get('link') == housing)
        if fixed.get('type') != 'fixed':
            raise ValueError('Simulation camera housing must be a fixed mount: ' + housing)
        for profile, row in pool['modules'][module['module_profile']].items():
            name = housing + '_' + profile
            optical = '__sim_optical_' + name
            ET.SubElement(root, 'link', name=optical)
            joint = ET.SubElement(root, 'joint', name=name, type='fixed')
            ET.SubElement(joint, 'parent', link=housing)
            ET.SubElement(joint, 'child', link=optical)
            ET.SubElement(joint, 'origin', xyz=' '.join(map(str, row['translation_m'])), rpy=' '.join(map(str, row['rpy_rad'])))
            mounts.append(dict(name=name, joint_name=name, camera_link=optical, parent_link=housing,
                translation=row['translation_m'], rpy=row['rpy_rad'], optics_profile=profile,
                optical_transform_source=pool['source'], configuration_kind='shared_simulation_configuration'))
    return mounts


def install_optical_frames(root, mounts, calibration):
    """Apply static mechanical-to-optical calibration to generated URDF only.

    A mechanical camera part is not necessarily an OpenGL optical frame.
    Virtual fixed links keep rendering, visibility and projection on the same
    FK path, including the calibration at the exposure's measured joint state.
    """
    from scipy.spatial.transform import Rotation
    offsets = calibration.get('camera_optical_frames', {})
    for mount in mounts:
        row = offsets.get(mount['joint_name'], offsets.get('*'))
        if row is None:
            continue
        translation = np.asarray(row['translation_m'], dtype=float)
        rotation = np.asarray(row['rpy_rad'], dtype=float)
        if translation.shape != (3,) or rotation.shape != (3,) or not np.isfinite(np.r_[translation, rotation]).all():
            raise ValueError('Invalid static camera optical transform')
        mechanical = mount['camera_link']
        optical = '__optical_' + mount['joint_name']
        if any(link.get('name') == optical for link in root.findall('link')):
            raise ValueError('Duplicate optical frame: ' + optical)
        ET.SubElement(root, 'link', name=optical)
        joint = ET.SubElement(root, 'joint', name=optical + '_fixed', type='fixed')
        ET.SubElement(joint, 'parent', link=mechanical)
        ET.SubElement(joint, 'child', link=optical)
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, translation)), rpy=' '.join(map(str, rotation)))
        parent_rotation = Rotation.from_euler('xyz', mount['rpy'])
        mount['translation'] = (np.asarray(mount['translation']) + parent_rotation.apply(translation)).tolist()
        mount['rpy'] = (parent_rotation * Rotation.from_euler('xyz', rotation)).as_euler('xyz').tolist()
        mount['mechanical_camera_link'] = mechanical
        mount['camera_link'] = optical
        mount['optical_transform_source'] = row['source']


def selected():
    path = os.environ.get('FINGEREYE_HAND_DESCRIPTION')
    return json.loads(Path(path).read_text()) if path else None


def discover_chains(hand):
    """Use the same topology implementation as policy FK and limits."""
    return hand.discover_finger_chains()


def materialize_collision_primitives(root, destination):
    """Expose all source collision shapes to the shared mesh path checker.

    Boxes are exact; curved primitives use enclosing meshes. Only the runtime
    derivative changes, and the same derivative goes to planning and physics.
    """
    import trimesh
    records = []
    for number, collision in enumerate(root.findall('.//collision')):
        geometry = collision.find('geometry')
        if geometry is None or geometry.find('mesh') is not None:
            continue
        shape = next(iter(geometry))
        parameters = dict(shape.attrib)
        if shape.tag == 'box':
            mesh = trimesh.creation.box(extents=np.fromstring(shape.get('size'), sep=' '))
        elif shape.tag == 'cylinder':
            mesh = trimesh.creation.cylinder(radius=float(shape.get('radius')) / np.cos(np.pi / 128),
                                             height=float(shape.get('length')), sections=128)
        elif shape.tag == 'sphere':
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(shape.get('radius')))
            plane_radius = np.einsum('ij,ij->i', mesh.face_normals, mesh.triangles[:, 0]).min()
            mesh.apply_scale(float(shape.get('radius')) / plane_radius)
        else:
            raise ValueError('Unsupported URDF collision primitive: ' + shape.tag)
        path = destination / ('collision_primitive_%04d.stl' % number)
        mesh.export(path)
        geometry.remove(shape)
        ET.SubElement(geometry, 'mesh', filename=str(path))
        records.append(dict(collision_index=number, kind=shape.tag, parameters=parameters,
                            path=str(path), approximation='exact box or enclosing curved mesh'))
    return records


def prepare(source, destination):
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    palm_mapping = normalize_virtual_palm(root)
    normalized = destination / 'normalized_source.urdf'
    tree.write(normalized, encoding='utf-8', xml_declaration=True)
    hand = HandDescription(normalized)
    # Historical absolute exports resolve in the selected checkout, never by
    # silently loading a different URDF with a coincident robot name.
    source_hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    for mesh in root.findall('.//mesh'):
        path = resolve_mesh(source, mesh.get('filename')).resolve()
        if not path.is_file():
            marker = '/assets/xhand3camFE_left/'
            original = mesh.get('filename')
            if marker in original:
                path = HERE.parents[1] / 'assets/xhand3camFE_left' / original.split(marker, 1)[1]
        if not path.is_file():
            raise FileNotFoundError('Source hand mesh is missing: ' + str(path))
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        mesh.set('filename', str(path))
    active = set(hand.joint_names)
    present = set(BASE_NAMES) & active
    if present and present != set(BASE_NAMES):
        raise ValueError('Incomplete world-to-palm coordinate chain')
    palm = hand.root
    if present:
        for _, child, _, _, _, index in hand.joints:
            if index is not None and hand.joint_names[index] == BASE_NAMES[-1]:
                palm = child
    else:
        world = '__runtime_world__'
        ET.SubElement(root, 'link', name=world)
        parent = world
        axes = ('1 0 0', '0 1 0', '0 0 1', '0 0 1', '0 1 0', '1 0 0')
        for index, (name, axis) in enumerate(zip(BASE_NAMES, axes)):
            child = hand.root if index == len(BASE_NAMES)-1 else '__runtime_' + name
            if child != hand.root:
                link = ET.SubElement(root, 'link', name=child)
                inertial = ET.SubElement(link, 'inertial')
                ET.SubElement(inertial, 'mass', value='0.001')
                ET.SubElement(inertial, 'inertia', ixx='1e-6', iyy='1e-6', izz='1e-6', ixy='0', ixz='0', iyz='0')
            joint = ET.SubElement(root, 'joint', name=name, type='prismatic' if index < 3 else 'revolute')
            ET.SubElement(joint, 'parent', link=parent); ET.SubElement(joint, 'child', link=child)
            ET.SubElement(joint, 'axis', xyz=axis)
            ET.SubElement(joint, 'limit', lower='-1' if index < 3 else '-3.141592653589793',
                          upper='1' if index < 3 else '3.141592653589793', effort='200', velocity='2')
            parent = child
    # USD identifiers cannot start with a digit. Record source-to-runtime names
    # explicitly instead of relying on undocumented importer renaming.
    source_to_runtime = {}
    for joint in root.findall('joint'):
        if joint.get('type') != 'fixed':
            name = joint.get('name')
            runtime_name = 'hand_joint_' + name if name[:1].isdigit() else name
            source_to_runtime[name] = runtime_name
            joint.set('name', runtime_name)
    order = [*BASE_NAMES, *(source_to_runtime[name] for name in hand.joint_names if name not in BASE_NAMES)]
    runtime_path = destination / 'hand.urdf'
    tree.write(runtime_path, encoding='utf-8', xml_declaration=True)
    runtime = HandDescription(runtime_path, joint_names=order, base_joint_names=BASE_NAMES)
    chains, paths = discover_chains(runtime)
    mounts = []
    for joint in root.findall('joint'):
        if joint.get('type') != 'fixed':
            continue
        child = joint.find('child').get('link')
        if 'cam' not in child.lower():
            continue
        # Restrict to leaf camera frames, not intermediate mechanical modules.
        if any(j.find('parent').get('link') == child for j in root.findall('joint')):
            continue
        chain = paths[child]
        origin = joint.find('origin')
        mounts.append(dict(name=joint.get('name'), camera_link=child,
                           parent_link=joint.find('parent').get('link'),
                           finger=next((row['finger'] for row in chains if child in row['links']), 'palm'), joint_name=joint.get('name'),
                           translation=np.fromstring(origin.get('xyz', '0 0 0'), sep=' ').tolist(),
                           rpy=np.fromstring(origin.get('rpy', '0 0 0'), sep=' ').tolist()))
    calibration_path=HERE/'hand_calibrations'/(source_hashes[str(source)]+'.json')
    calibration=json.loads(calibration_path.read_text()) if calibration_path.is_file() else {}
    for pair in calibration.get('joint_assembly_collision_exclusions', []):
        ET.SubElement(root, 'fingereye_collision_exclusion',
                      link_a=pair['link_a'], link_b=pair['link_b'])
    declared_mounts = install_simulation_modules(root, calibration)
    for mount in declared_mounts:
        mount['finger'] = next(row['finger'] for row in chains if mount['parent_link'] in row['links'])
    mounts.extend(declared_mounts)
    # Mechanical housings do not identify lens centers or optical axes. An
    # unknown URDF must never silently inherit another hand's optics.
    if not mounts or not calibration:
        readiness = dict(status='blocked', reason='missing_optical_calibration',
            source_urdf=str(source), source_sha256=source_hashes[str(source)],
            expected_sidecar=str(calibration_path), discovered_camera_mounts=mounts,
            finger_chains=chains, active_joint_count=len(hand.joint_names),
            requirement='lens link, lens-to-link transform and optics profile for each actual camera')
        (destination/'readiness.json').write_text(json.dumps(readiness, indent=2)+'\n')
        raise ValueError('missing_optical_calibration: selected URDF has no calibrated camera set; see '
                         + str(destination/'readiness.json'))
    if calibration.get('camera_optics_by_joint'):
        mounts=[m for m in mounts if m['joint_name'] in calibration['camera_optics_by_joint']]
    for mount in mounts:
        mount['optics_profile']=calibration.get('camera_optics_by_joint',{}).get(mount['joint_name'],mount.get('optics_profile','front1'))
    install_optical_frames(root, mounts, calibration)
    primitives = materialize_collision_primitives(root, destination)
    tree.write(runtime_path, encoding='utf-8', xml_declaration=True)
    runtime = HandDescription(runtime_path, joint_names=order, base_joint_names=BASE_NAMES)
    chains, paths = discover_chains(runtime)
    q = np.clip(np.zeros(len(order)), runtime.lower, runtime.upper)
    q[:3] = [-.18, 0., .30]
    result = dict(schema='fingereye.hand.v1', source_urdf_path=str(source),
                  source_sha256=source_hashes, urdf_path=str(runtime_path),
                  runtime_urdf_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
                  joint_names=order, joint_limits=np.stack([runtime.lower, runtime.upper], axis=1).tolist(),
                  base_joint_names=list(BASE_NAMES), palm_link=palm, root_link=runtime.root,
                  source_to_runtime_joint_names=source_to_runtime, finger_chains=chains,
                  virtual_palm_mapping=palm_mapping,
                  camera_mounts=mounts, initial_q=q.tolist(),
                  collision_primitive_derivatives=primitives,
                  runtime_derivative='source geometry unchanged; declared six-axis palm interface and USD-safe joint names')
    result['pads']=pad_catalog(runtime,chains,source_hashes[str(source)])
    result['task_frame']=calibration.get('task_frame',dict(forward_axis_in_palm=runtime.forward_axis_in_palm(palm,result['pads']).tolist(),source='neutral URDF finger roots to full fixed terminal assembly centroids; independent of selected contact regions'))
    path = destination / 'description.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    return result


def configure_env(cfg, sample_path, args):
    """Use the existing task/physics loop with a URDF-selected articulation."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    hand = selected()
    cfg.sample_pkl = str(sample_path)
    cfg.use_sample_mesh = True
    cfg.reset_to_sample_on_reset = True
    cfg.replay_mode = False
    cfg.force_usd_conversion = False
    cfg.converted_usd_dir = str(sample_path.parent / 'converted_objects')
    cfg.robot_urdf = hand['urdf_path']
    cfg.enable_cameras = True
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 10
    cfg.production_origin_preserving_batch = False
    cfg.async_batch_ground = False
    cfg.sim.gravity = (0., 0., 0.)
    cfg.sim.render_interval = 4
    cfg.synthe_robot_static_friction = cfg.synthe_robot_dynamic_friction = float(args.robot_friction)
    cfg.synthe_object_static_friction = cfg.synthe_object_dynamic_friction = float(args.object_friction)
    material = sim_utils.RigidBodyMaterialCfg(static_friction=float(args.robot_friction),
        dynamic_friction=float(args.robot_friction), friction_combine_mode='max')
    cfg.sim.physics_material = material
    cfg.robot_cfg = copy.deepcopy(cfg.robot_cfg)
    cfg.robot_cfg.prim_path = '/World/envs/env_.*/Robot'
    # Preserve the validated authored XHand decomposition. Other hands are
    # imported from the selected read-only URDF with the same contact offsets.
    if hand['source_to_runtime_joint_names'].get('left_hand_thumb_joint1'):
        usd_path = str(HERE / 'assets/round7b_xhand_allfine.usdc')
    else:
        conversion = UrdfConverter(UrdfConverterCfg(asset_path=hand['urdf_path'],
            usd_dir=str(sample_path.parent / 'converted_hand'), usd_file_name='hand.usd',
            force_usd_conversion=True, make_instanceable=False, fix_base=True,
            merge_fixed_joints=True, self_collision=True, collider_type='convex_decomposition',
            joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.))))
        usd_path = conversion.usd_path
        # Imported mesh subtrees can remain instanceable even when the robot
        # itself is not. Author millimetre contact offsets on the generated
        # derivative so the spawner cannot silently skip instance proxies.
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
        stage = Usd.Stage.Open(usd_path)
        while True:
            instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
            if not instances:
                break
            for prim in instances:
                prim.SetInstanceable(False)
        # STL importer puts collision APIs on a wrapper Xform. PhysX mesh
        # cooking parameters must be authored on the actual Mesh leaf.
        for wrapper in list(stage.Traverse()):
            if wrapper.HasAPI(UsdPhysics.CollisionAPI) and not wrapper.IsA(UsdGeom.Mesh):
                meshes=[leaf for leaf in Usd.PrimRange(wrapper) if leaf.IsA(UsdGeom.Mesh)]
                if not meshes:raise RuntimeError('Collision wrapper has no mesh: '+str(wrapper.GetPath()))
                for leaf in meshes:
                    UsdPhysics.CollisionAPI.Apply(leaf)
                    UsdPhysics.MeshCollisionAPI.Apply(leaf).CreateApproximationAttr('convexDecomposition')
                wrapper.RemoveAPI(UsdPhysics.CollisionAPI)
                wrapper.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        colliders = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr(.0002)
                api.CreateRestOffsetAttr(0.)
                cooking = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                cooking.CreateHullVertexLimitAttr(64)
                cooking.CreateMaxConvexHullsAttr(256)
                cooking.CreateMinThicknessAttr(.0002)
                cooking.CreateVoxelResolutionAttr(8000000)
                cooking.CreateErrorPercentageAttr(.1)
                cooking.CreateShrinkWrapAttr(True)
                colliders.append(str(prim.GetPath()))
        if not colliders:
            raise RuntimeError('Imported hand has no physical colliders')
        # URDF joint collisionEnabled alone does not install the articulation
        # pair filter. Use the exact same structural rule as the policy mesh
        # guard, applied to the generated fixed-joint-merged rigid bodies.
        model = HandDescription(hand['urdf_path'], joint_names=hand['joint_names'], base_joint_names=BASE_NAMES)
        bodies = {prim.GetName(): prim for prim in stage.Traverse()
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in model.links}
        exclusions = model.structural_collision_exclusions(list(bodies))
        for a, b in sorted(exclusions):
            for source, target in ((a, b), (b, a)):
                UsdPhysics.FilteredPairsAPI.Apply(bodies[source]).CreateFilteredPairsRel().AddTarget(bodies[target].GetPath())
        (sample_path.parent/'converted_hand'/'structural_collision_filters.json').write_text(
            json.dumps(dict(source='HandDescription.structural_collision_exclusions',
                            scope='fixed assemblies and direct movable neighbours only',
                            rigid_bodies=list(bodies), excluded_pairs=sorted(exclusions)), indent=2))
        stage.GetRootLayer().Save()
        (sample_path.parent/'converted_hand'/'collision_configuration.json').write_text(
            json.dumps(dict(contact_offset_m=.0002, rest_offset_m=0., cooking_source='validated round7b_xhand_allfine.json explicit_cooking_parameters', hullVertexLimit=64, maxConvexHulls=256, minThickness=.0002, voxelResolution=8000000, errorPercentage=.1, shrinkWrap=True, colliders=colliders), indent=2))
    cfg.robot_usd = usd_path
    cfg.robot_cfg.spawn = sim_utils.UsdFileCfg(usd_path=usd_path, activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0002, rest_offset=0.))
    cfg.robot_cfg.spawn.physics_material=material
    # Isaac validates its creation defaults before our task reset installs the
    # observation home pose. A source joint need not permit zero (e.g. a
    # mechanical abduction offset); keep the default inside its source limits.
    defaults = {n: float(np.clip(0., *limits))
                for n, limits in zip(hand['joint_names'], hand['joint_limits'])}
    cfg.robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.,0.,0.), joint_pos=defaults)
    tree=ET.parse(hand['urdf_path'])
    effort={j.get('name'):float(j.find('limit').get('effort')) for j in tree.findall('joint') if j.find('limit') is not None}
    fingers=[name for name in hand['joint_names'] if name not in BASE_NAMES]
    cfg.robot_cfg.actuators={
        'palm': ImplicitActuatorCfg(joint_names_expr=list(BASE_NAMES), effort_limit_sim=float(args.base_max_effort),
            stiffness=float(args.base_stiffness), damping=float(args.base_damping)),
        'fingers': ImplicitActuatorCfg(joint_names_expr=fingers, effort_limit_sim={n:effort[n] for n in fingers},
            stiffness=float(args.position_stiffness), damping=float(args.position_damping))}
    cfg.all_actuated_joint_names=list(hand['joint_names'])
    cfg.control_joint_names=fingers
    cfg.action_space=len(fingers)
    cfg.xarm_init_joint_values_degree=[0.]*len(BASE_NAMES)
    cfg.xhand_init_joint_values_radian=list(hand['initial_q'])
    cfg.leap_init_joint_values_radian=[0.]*len(fingers)
    cfg.enable_contact_audit=False  # Native diagnostic queries are installed for all physical bodies below.
    import types
    def cameras(self, finger=None, *, requested_fps=30., include_third_view=True, width=160, height=120):
        configure_cameras(self, requested_fps, width or 160, height or 120)
    cfg.configure_xhand_cameras=types.MethodType(cameras, cfg)
    configure_cameras(cfg, 30., 160, 120)
    from scipy.spatial.transform import Rotation
    cfg.cam_third_view.offset.pos=(0.,-.001,.5)
    forward=np.array([0.,.001,-.5]);forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    quat=Rotation.from_matrix(np.stack([right,up,-forward],axis=1)).as_quat()[[3,0,1,2]]
    cfg.cam_third_view.offset.rot=tuple(quat)


def configure_cameras(cfg, fps, width, height):
    from dataclasses import replace
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from fingereye.tasks.direct.fingereye.xhand_camera_config import XHandCameraSpec, _fisheye_spawn_cfg, _fisheye_metadata
    hand=selected(); configs={}; metadata={}
    for mount in hand['camera_mounts']:
        name=mount['name']
        spec=XHandCameraSpec(logical_name=name, finger=mount['finger'], mount_name=mount['optics_profile'],
            joint_name=mount['joint_name'],parent_link=mount['parent_link'],camera_link=mount['camera_link'],
            translation=tuple(mount['translation']),rpy=tuple(mount['rpy']),width=width,height=height,urdf_path=hand['urdf_path'])
        configs[name]=TiledCameraCfg(prim_path='/World/envs/env_.*/Robot/FingerEye_'+name,
            update_period=0.,spawn=_fisheye_spawn_cfg(sim_utils,spec),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.,0.,0.),rot=(1.,0.,0.,0.),convention='opengl'),
            data_types=['rgb'],width=width,height=height,update_latest_camera_pose=True)
        row=_fisheye_metadata(spec,1./fps)
        row['extrinsic_source']=mount.get('optical_transform_source',
            'source_URDF_camera_frame_used_as_simulation_optical_frame')
        row['calibration_status']='simulation_workspace_fisheye_not_hardware_optical_calibration'
        row['camera_prim_path_pattern']=configs[name].prim_path
        metadata[name]=row
    cfg.xhand_camera_configs=configs;cfg.xhand_camera_metadata=metadata
    cfg.recording_only_camera_names=list(configs)
    cfg.camera_name_list=['third_view',*configs]


def configure_physics(cfg):
    cfg.seed=0
    cfg.sim.physx.friction_offset_threshold=.0005
    cfg.sim.physx.friction_correlation_distance=.001
    cfg.sim.use_fabric=True


_model=None

def capture_cameras(env,cfg,names):
    global _model
    import torch
    from scipy.spatial.transform import Rotation
    from fingereye.tasks.direct.fingereye.xhand_camera_recorder import to_uint8_batch
    hand=selected()
    if _model is None:
        _model=HandDescription(hand['urdf_path'],joint_names=hand['joint_names'],base_joint_names=BASE_NAMES)
    actual=env.hand.data.joint_pos[0].cpu().numpy()
    q=np.array([actual[env.hand.joint_names.index(n)] for n in _model.joint_names])
    fk=_model.fk(q)
    for name,cam in env._active_cameras.items():
        if name=='third_view':continue
        T=fk[cfg.xhand_camera_metadata[name]['camera_link']]
        quat=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
        cam.set_world_poses(torch.tensor(T[None,:3,3],dtype=torch.float32,device=env.device),
            torch.tensor(quat[None],dtype=torch.float32,device=env.device),convention='opengl')
    env.sim.render()
    frames={}
    for name in names:
        cam=env._active_cameras[name];cam.update(0.,force_recompute=True)
        frames[name]=to_uint8_batch(cam.data.output['rgb'])
    return frames


def audit_runtime(env,cfg,out):
    """Prove the simulator's joint order/limits and loaded URDF, without policy truth."""
    hand=selected()
    actual=list(env.hand.joint_names)
    # Private initial collision representation: diagnose cooked convexes,
    # including solid containment that triangle surface intersection misses.
    if os.environ.get('FINGEREYE_AUDIT_COOKED')=='1':
        from pxr import UsdPhysics, UsdGeom, UsdUtils, PhysicsSchemaTools
        from omni.physx import get_physx_cooking_interface
        stage=env.sim.stage
        stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        cooked=[]
        for prim in stage.Traverse():
            if '/Robot/' not in str(prim.GetPath()) or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            transform=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
            path=str(prim.GetPath())
            def collect(result,convexes):
                cooked.append(dict(path=path,result=str(result),world_transform=transform.tolist(),
                    convex_vertices=[np.asarray(c.vertices).tolist() for c in convexes]))
            get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=stage_id,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),
                run_asynchronously=False,on_result=collect)
        (out/'cross_hand_initial_cooked_convexes.json').write_text(json.dumps(cooked))
    if set(actual)!=set(hand['joint_names']):
        raise RuntimeError('Imported actuator mismatch: '+repr((actual,hand['joint_names'])))
    shape = {key: getattr(env.hand.root_physx_view, method)().cpu().numpy().tolist()
             for key, method in [('contact_offsets', 'get_contact_offsets'), ('rest_offsets', 'get_rest_offsets')]}
    (out/'cross_hand_native_shape_properties.json').write_text(json.dumps(shape, indent=2))
    if not np.allclose(shape['contact_offsets'], .0002, atol=1e-9) or not np.allclose(shape['rest_offsets'], 0., atol=1e-9):
        raise RuntimeError('Selected hand contact offsets do not match the shared physics configuration')
    source={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hand['source_sha256']}
    if source!=hand['source_sha256']:
        raise RuntimeError('Source URDF or mesh changed during import')
    (out/'cross_hand_loaded.json').write_text(json.dumps(dict(hand, actual_simulator_joint_names=actual,
        loaded_usd=cfg.robot_usd,actual_body_names=list(env.hand.body_names)),indent=2,ensure_ascii=False)+'\n')


def _audit_camera_optics(camera, metadata, name, spawn_cfg):
    """Read renderer USD optics independently; native values stay private.

    Attribute names follow IsaacLab sim/spawners/sensors/sensors.py,
    CUSTOM_FISHEYE_CAMERA_ATTRIBUTES and the USD Camera schema.
    """
    prim = camera._sensor_prims[0].GetPrim()
    mapping = {
        'cameraProjectionType': 'projection_type',
        'fthetaWidth': 'fisheye_nominal_width',
        'fthetaHeight': 'fisheye_nominal_height',
        'fthetaCx': 'fisheye_optical_centre_x',
        'fthetaCy': 'fisheye_optical_centre_y',
        'fthetaMaxFov': 'fisheye_max_fov',
    }
    expected = {attribute: metadata.get(key) for attribute, key in mapping.items()}
    expected.update(focalLength=float(spawn_cfg.focal_length),
                    horizontalAperture=float(spawn_cfg.horizontal_aperture),
                    verticalAperture=float(spawn_cfg.vertical_aperture))
    polynomial = metadata.get('fisheye_polynomial', [])
    for index, suffix in enumerate('ABCDEF'):
        expected['fthetaPoly' + suffix] = polynomial[index] if len(polynomial) == 6 else None
    native = {key: prim.GetAttribute(key).Get() for key in expected}
    native['projection'] = prim.GetAttribute('projection').Get()
    checks = []
    for attribute, wanted in expected.items():
        actual = native[attribute]
        if isinstance(wanted, str):
            passed = actual == wanted
            absolute_error = None
        elif wanted is None or actual is None:
            passed = False
            absolute_error = None
        else:
            absolute_error = abs(float(actual) - float(wanted))
            passed = bool(np.isclose(float(actual), float(wanted), rtol=2e-6, atol=1e-12))
        checks.append(dict(attribute=attribute, expected=wanted, actual=actual,
                           absolute_error=absolute_error, passed=passed))
    native_k = camera.data.intrinsic_matrices[0].cpu().numpy()
    wanted_k = np.asarray(metadata['renderer_intrinsic_matrix'], dtype=float)
    k_error = float(np.max(np.abs(native_k - wanted_k)))
    checks.append(dict(attribute='renderer_intrinsic_matrix', expected=wanted_k.tolist(),
                       actual=native_k.tolist(), max_absolute_error_px=k_error,
                       passed=bool(np.allclose(native_k, wanted_k, rtol=0., atol=1e-5))))
    actual_shape = tuple(int(n) for n in camera.data.output['rgb'].shape[-3:-1])
    wanted_shape = (int(metadata['height']), int(metadata['width']))
    checks.append(dict(attribute='rendered_image_height_width', expected=list(wanted_shape),
                       actual=list(actual_shape), passed=actual_shape == wanted_shape))
    return dict(name=name, prim_path=str(prim.GetPath()), native_usd_attributes=native,
                source='actual renderer Camera prim; private diagnostic, never a policy observation',
                mapping_source='IsaacLab sim/spawners/sensors/sensors.py CUSTOM_FISHEYE_CAMERA_ATTRIBUTES',
                numeric_tolerance=dict(relative=2e-6, absolute=1e-12, intrinsic_absolute_px=1e-5),
                checks=checks, passed=all(row['passed'] for row in checks))


def audit_cameras(env,cfg,step,out):
    # Rendering uses the same model, FK exposure state and optical metadata as
    # the policy. Record native camera poses for independent offline checks.
    rows=[]; optics=[]
    for name,cam in env._active_cameras.items():
        if name=='third_view':continue
        if step==0:
            optics.append(_audit_camera_optics(cam,cfg.xhand_camera_metadata[name],name,cfg.xhand_camera_configs[name].spawn))
        rows.append(dict(name=name,position=cam.data.pos_w[0].cpu().numpy().tolist(),
            quaternion_opengl=cam.data.quat_w_opengl[0].cpu().numpy().tolist(),
            intrinsic_matrix=cam.data.intrinsic_matrices[0].cpu().numpy().tolist(),
            calibration=cfg.xhand_camera_metadata[name]))
    with (out/'cross_hand_camera_frames.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(step=step,simulator_joint_names=list(env.hand.joint_names),actual_q=env.hand.data.joint_pos[0].cpu().numpy().tolist(),native_body_names=list(env.hand.body_names),native_body_poses=env.hand.data.body_state_w[0,:,:7].cpu().numpy().tolist(),cameras=rows))+'\n')

    if step==0:
        report=dict(step=step,passed=all(row['passed'] for row in optics),cameras=optics)
        (out/'cross_hand_camera_optics.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['passed']:
            failures=[(row['name'],check['attribute']) for row in optics
                      for check in row['checks'] if not check['passed']]
            raise RuntimeError('Renderer optics disagree with projection metadata: '+repr(failures))


_PAD_CATALOG_CACHE={}


def pad_catalog(hand,chains,source_digest):
    # Cache immutable static surfaces, not current articulation or observations.
    # The kinematic/mesh signature also protects numerical URDF-frame audits.
    signature=(source_digest,repr(hand.joints),repr(hand.collision_meshes),repr(chains))
    calibration=HERE/'hand_calibrations'/(source_digest+'.json')
    key=signature+((calibration.read_bytes()if calibration.is_file()else b''),)
    if key not in _PAD_CATALOG_CACHE:
        if len(_PAD_CATALOG_CACHE)>=8:_PAD_CATALOG_CACHE.pop(next(iter(_PAD_CATALOG_CACHE)))
        _PAD_CATALOG_CACHE[key]=_build_pad_catalog(hand,chains,source_digest)
    return copy.deepcopy(_PAD_CATALOG_CACHE[key])


def _build_pad_catalog(hand, chains, source_digest):
    """Load static contact data by asset hash, or propose mesh-supported patches.

    Mesh proposals are metric geometric contact candidates, not measured pad
    compliance/friction. Native contacts remain private physical diagnostics.
    """
    import trimesh
    path=HERE/'hand_calibrations'/(source_digest+'.json')
    static=json.loads(path.read_text()) if path.is_file() else {}
    rows=copy.deepcopy(static.get('pads', []))
    groups={c['finger']:c for c in chains}
    for row in rows:
        if row['finger'] not in groups:
            raise ValueError('Static contact calibration names an unavailable finger: '+row['finger'])
        row['joint_names']=groups[row['finger']]['joint_names']
        row['geometry_class']='legacy_static_contact_surface'
        row['physical_pad_validation']='legacy contact location retained; surface identity and load require offline contact-point verification'
        for region in row.get('regions', []):
            region.setdefault('source', row.get('source', 'static calibration'))
    calibrated={row['finger'] for row in rows}
    zero_poses=hand.fk(np.clip(np.zeros(len(hand.joint_names)),hand.lower,hand.upper))
    for chain in chains:
        # Never replace an established calibration with a mesh proposal.
        # Other digits remain available even when a partial sidecar exists.
        link=chain['terminal_link']
        # Fixed descendants are one physical contact assembly, independently
        # of component names, inertial tags, or camera ownership.
        assembly={link}
        while True:
            before=len(assembly)
            for parent,child,_,_,kind,index in hand.joints:
                if parent in assembly and index is None:assembly.add(child)
            if len(assembly)==before:break
        pieces=[];face_links=[]
        for entry in hand.collision_meshes:
            if entry['link'] not in assembly:continue
            body=trimesh.load(entry['path'],force='mesh',process=True)
            body.apply_scale(entry['scale']);body.apply_transform(entry['origin'])
            if entry['link']!=link:
                body.apply_transform(np.linalg.inv(zero_poses[link])@zero_poses[entry['link']])
            pieces.append(body);face_links.extend([entry['link']]*len(body.faces))
        if not pieces:continue
        mesh=trimesh.util.concatenate(pieces)
        # Work in a physical frame tied to the motion chain, never in the
        # terminal mesh's arbitrary local XYZ chart. Compensated URDF frame
        # changes therefore rotate every sample with the unchanged geometry.
        rotation=zero_poses[link][:3,:3]
        joint_rows=[j for j in hand.joints if j[5] is not None and
                    hand.joint_names[j[5]] in chain['joint_names']]
        pivot=zero_poses[link][:3,3]
        center_world=rotation@mesh.center_mass+zero_poses[link][:3,3]
        upstream=[zero_poses[j[1]][:3,3] for j in joint_rows]
        anatomical=next((pivot-point for point in reversed(upstream)
                         if np.linalg.norm(pivot-point)>1e-6),center_world-pivot)
        if np.linalg.norm(anatomical)<1e-8:
            # Synthetic/zero-length chains: choose the principal *world*
            # geometry axis, with sign anchored to the unchanged root frame.
            world=np.asarray(mesh.vertices)@rotation.T
            _,_,axes=np.linalg.svd(world-world.mean(0),full_matrices=False)
            anatomical=axes[0]
            if anatomical[np.argmax(abs(anatomical))]<0:anatomical=-anatomical
        distal=rotation.T@(anatomical/np.linalg.norm(anatomical))
        if joint_rows:
            parent,child,origin,axis,kind,index=joint_rows[-1]
            lateral=rotation.T@(zero_poses[parent][:3,:3]@origin[:3,:3]@axis)
        else:lateral=rotation.T@np.eye(3)[np.argmin(abs(rotation@distal))]
        lateral-=distal*(lateral@distal)
        if np.linalg.norm(lateral)<1e-8:
            lateral=rotation.T@np.eye(3)[np.argmin(abs(rotation@distal))]
            lateral-=distal*(lateral@distal)
        lateral/=np.linalg.norm(lateral)
        basis=np.stack([distal,lateral,np.cross(distal,lateral)])
        terminal_basis=basis.copy();terminal_origin=mesh.center_mass.copy()
        canonical=np.eye(4);canonical[:3,:3]=basis;canonical[:3,3]=-basis@terminal_origin
        mesh.apply_transform(canonical)
        # 0.1nm canonical rounding removes arithmetic chart noise before the
        # float32 ray accelerator makes boundary/triangle decisions.
        mesh.vertices=np.round(np.asarray(mesh.vertices),10)
        basis=np.eye(3)
        projected=np.asarray(mesh.vertices)@basis.T
        bounds=np.array([projected.min(0),projected.max(0)]);extents=np.ptp(bounds,axis=0)
        regions=[]
        # Four percent of the narrowest assembly dimension is a geometric
        # support hypothesis. Curvature/edges can reduce it; it is not a pad
        # compliance or a minimum contact size requirement.
        radius=max(1e-5,float(extents.min())*.04)
        angles=np.arange(12)*np.pi/6
        def source_disc(point,normal,support_radius):
            u=np.eye(3)[np.argmin(abs(normal))];u-=normal*(u@normal);u/=np.linalg.norm(u);v=np.cross(normal,u)
            disc=np.vstack([point,point+support_radius*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)])
            reach=float(np.max(abs((np.asarray(mesh.vertices)-point)@normal)))+.02
            starts=disc+normal*reach
            hit,ray_ids,face_ids=mesh.ray.intersects_location(starts,np.tile(-normal,(13,1)),multiple_hits=True)
            support=[];normals=[];owners=[]
            for ray in range(13):
                ids=np.flatnonzero(ray_ids==ray)
                if not len(ids):return None
                j=ids[np.argmin(np.linalg.norm(hit[ids]-starts[ray],axis=1))]
                support.append(hit[j]);normals.append(mesh.face_normals[face_ids[j]]);owners.append(face_links[face_ids[j]])
            variation=float(np.max(np.arccos(np.clip(np.asarray(normals)@normal,-1,1))))
            if np.linalg.norm(support[0]-point)>1e-6 or variation>.25:return None
            if np.max(abs((np.asarray(support)-point)@normal))>support_radius*np.tan(.25)+1e-7:return None
            return dict(point=np.asarray(support[0]).tolist(),normal=normal.tolist(),radius_m=support_radius,
                normal_variation_rad=variation,surface_link=owners[0],surface_links=sorted(set(owners)),
                geometry_class='fixed_assembly_exterior_patch',exposure_check='13 first hits in actual surface-normal direction against complete rigid assembly',
                source='source surface-plane area and motion-chain coordinates; adaptive geometric support, not measured compliance')
        # Actual planar source features precede box scan proposals. Averaging
        # oblique scan-ray normals can miss the very plane that can engage.
        normals=np.asarray(mesh.face_normals);centers=np.asarray(mesh.triangles_center);areas=np.asarray(mesh.area_faces)
        keys=np.c_[np.round(normals,4),np.round(np.sum(normals*centers,axis=1),5)]
        unique,labels=np.unique(keys,axis=0,return_inverse=True)
        totals=np.bincount(labels,weights=areas)
        groups=sorted(range(len(unique)),key=lambda i:(-round(float(totals[i]),10),tuple(unique[i])))[:64]
        for group in groups:
            ids=np.flatnonzero(labels==group)
            if totals[group]<1e-9:continue
            normal=np.average(normals[ids],axis=0,weights=areas[ids]);normal/=np.linalg.norm(normal)
            # Surface bounds, not triangle-centroid quantiles, survive a
            # continuous plane being split into several fixed source links.
            u=np.eye(3)[np.argmin(abs(normal))];u-=normal*(u@normal);u/=np.linalg.norm(u);v=np.cross(normal,u)
            vertices=np.asarray(mesh.triangles)[ids].reshape(-1,3)
            uv=vertices@np.stack([u,v]).T;lower=uv.min(0);upper=uv.max(0)
            offset=float(np.average(centers[ids]@normal,weights=areas[ids]))
            proposals=[]
            for fraction in (.5,.2,.8):
                coordinate=(lower+upper)/2;axis=int(np.argmax(upper-lower));coordinate[axis]=lower[axis]+fraction*(upper[axis]-lower[axis])
                proposals.append(normal*offset+coordinate[0]*u+coordinate[1]*v)
            for point in proposals:
                for support_radius in (radius,radius/2,radius/4,radius/8):
                    region=source_disc(point,normal,support_radius)
                    if region is None:continue
                    if not any(np.linalg.norm(np.asarray(old['point'])-region['point'])<.0002 and np.dot(old['normal'],normal)>.99 for old in regions):regions.append(region)
                    break
        for normal_axis in (1,2,0):
          tangents=[axis for axis in range(3) if axis != normal_axis]
          for sign in (-1.,1.):
            fractions=(.95,.8,.65,.4,.2) if normal_axis else (.5,)
            for fraction in fractions:
              for cross_fraction in (0.,-.3,.3):
                center=bounds.mean(0)
                if normal_axis:
                    center[0]=bounds[0,0]+fraction*extents[0]
                    other=next(a for a in tangents if a!=0)
                    center[other]+=cross_fraction*extents[other]
                else:center[tangents[0]]+=cross_fraction*extents[tangents[0]]
                center[normal_axis]=bounds[0,normal_axis]-.02 if sign<0 else bounds[1,normal_axis]+.02
                direction=basis[normal_axis]*-sign
                for support_radius in (radius,radius/2,radius/4,radius/8):
                    offsets=np.zeros((13,3))
                    offsets[1:]=support_radius*(np.cos(angles)[:,None]*basis[tangents[0]]+np.sin(angles)[:,None]*basis[tangents[1]])
                    origins=center@basis+offsets
                    positions,rays,faces=mesh.ray.intersects_location(origins,np.tile(direction,(13,1)),multiple_hits=True)
                    points=[];normals=[];owners=[]
                    for i in range(13):
                        ids=np.flatnonzero(rays==i)
                        if not len(ids):break
                        j=ids[np.argmin(np.linalg.norm(positions[ids]-origins[i],axis=1))]
                        points.append(positions[j]);normals.append(mesh.face_normals[faces[j]]);owners.append(face_links[faces[j]])
                    if len(points)!=13:continue
                    normal=np.mean(normals,axis=0);normal/=np.linalg.norm(normal)
                    variation=float(np.max(np.arccos(np.clip(np.asarray(normals)@normal,-1,1))))
                    if variation>.25 or normal@(-direction)<.5:continue
                    if np.max(abs((np.asarray(points)-points[0])@normal))>support_radius*np.tan(.25)+1e-7:continue
                    # Verify exposure in the eventual contact-normal direction,
                    # rather than only in the proposal's anatomical scan axis.
                    # A side lip can otherwise hide a tilted face by centimetres.
                    u=basis[tangents[0]]-normal*(basis[tangents[0]]@normal)
                    if np.linalg.norm(u)<1e-8:continue
                    u/=np.linalg.norm(u);v=np.cross(normal,u)
                    disc=np.vstack([points[0],points[0]+support_radius*(np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v)])
                    reach=float(np.max(abs((np.asarray(mesh.vertices)-points[0])@normal)))+.02
                    starts=disc+normal*reach
                    hit,ray_ids,face_ids=mesh.ray.intersects_location(starts,np.tile(-normal,(13,1)),multiple_hits=True)
                    support=[];support_normals=[];support_owners=[]
                    for ray in range(13):
                        ids=np.flatnonzero(ray_ids==ray)
                        if not len(ids):break
                        j=ids[np.argmin(np.linalg.norm(hit[ids]-starts[ray],axis=1))]
                        support.append(hit[j]);support_normals.append(mesh.face_normals[face_ids[j]]);support_owners.append(face_links[face_ids[j]])
                    if len(support)!=13 or np.linalg.norm(support[0]-points[0])>1e-6:continue
                    if np.max(np.arccos(np.clip(np.asarray(support_normals)@normal,-1,1)))>.25:continue
                    if np.max(abs((np.asarray(support)-points[0])@normal))>support_radius*np.tan(.25)+1e-7:continue
                    points=support;owners=support_owners
                    # Link ownership is descriptive. A continuous exposed
                    # support is allowed to span several rigidly fixed parts.
                    if any(np.linalg.norm(np.asarray(old['point'])-points[0])<.0002 and
                           np.dot(old['normal'],normal)>.99 for old in regions):break
                    regions.append(dict(point=np.asarray(points[0]).tolist(),normal=normal.tolist(),radius_m=support_radius,
                        normal_variation_rad=variation,surface_link=owners[0],surface_links=sorted(set(owners)),
                        geometry_class='fixed_assembly_exterior_patch',exposure_check='13 nearest incoming rays against complete fixed assembly; continuous depth and normals across component seams',
                        source='motion-chain physical frame; scale/curvature adaptive exposed collision support, not a measured soft pad'))
                    break
        # Add geometry principal-axis scans. Their axes are computed in the
        # motion-chain chart, so a terminal-link coordinate change cannot alter
        # them. This covers curved distal skins as well as planar housings.
        # Integrate the uniform surface second moment exactly over triangles;
        # vertex PCA depends on how a fixed surface was tessellated/split.
        triangles=np.asarray(mesh.triangles);area=np.asarray(mesh.area_faces)
        total=float(area.sum());surface_mean=np.sum(mesh.triangles_center*area[:,None],axis=0)/total
        sums=triangles.sum(1)
        second=np.einsum('t,tvi,tvj->ij',area,triangles,triangles)+np.einsum('t,ti,tj->ij',area,sums,sums)
        covariance=second/(12*total)-np.outer(surface_mean,surface_mean)
        values,vectors=np.linalg.eigh(covariance);principal=vectors[:,np.argsort(values)[::-1]].T
        for axis in principal:
            if axis[np.argmax(abs(axis))]<0:axis*=-1
        projected=np.asarray(mesh.vertices)@principal.T
        pb=np.array([projected.min(0),projected.max(0)]);span=np.ptp(pb,axis=0)
        for normal_axis in (1,2,0):
            tangents=[i for i in range(3)if i!=normal_axis]
            for sign in (-1.,1.):
                for fraction in ((.95,.8,.65,.4,.2)if normal_axis else (.5,)):
                    coordinate=pb.mean(0)
                    if normal_axis:coordinate[0]=pb[0,0]+fraction*span[0]
                    coordinate[normal_axis]=pb[0,normal_axis]-.02 if sign<0 else pb[1,normal_axis]+.02
                    origin=coordinate@principal;direction=-sign*principal[normal_axis]
                    hits,ray_ids,face_ids=mesh.ray.intersects_location(origin[None],direction[None],multiple_hits=True)
                    if not len(hits):continue
                    j=np.argmin(np.linalg.norm(hits-origin,axis=1));point=hits[j];normal=mesh.face_normals[face_ids[j]]
                    for support_radius in (radius,radius/2,radius/4,radius/8):
                        region=source_disc(point,normal,support_radius)
                        if region is None:continue
                        if not any(np.linalg.norm(np.asarray(old['point'])-point)<.0002 and np.dot(old['normal'],normal)>.99 for old in regions):regions.append(region)
                        break
        # Bound combinatorics with farthest physical coverage, independent
        # of link/frame names and source triangle tessellation.
        if len(regions)>48:
            chosen=[regions[0]];remaining=regions[1:]
            scale=max(float(extents.max()),1e-6)
            while len(chosen)<48 and remaining:
                def separation(candidate):
                    return round(min(np.linalg.norm(np.asarray(candidate['point'])-old['point'])/scale+
                               .5*(1-np.dot(candidate['normal'],old['normal'])) for old in chosen),8)
                best=max(range(len(remaining)),key=lambda i:separation(remaining[i]))
                chosen.append(remaining.pop(best))
            regions=chosen
        for region in regions:
            region['point']=(np.asarray(region['point'])@terminal_basis+terminal_origin).tolist()
            region['normal']=(np.asarray(region['normal'])@terminal_basis).tolist()
        if not regions:continue
        if chain['finger'] in calibrated:
            row=next(row for row in rows if row['finger']==chain['finger'])
            # Each point is expressed in the same terminal link frame. Keep
            # calibrated locations first and preserve their selected defaults.
            existing=row.setdefault('regions',[])
            # Preserve a static surface description and do not multiply its
            # search budget with numerically equivalent generated patches.
            for region in regions:
                duplicate=any(old.get('surface_link',row.get('surface_link',link))==region['surface_link']
                    and np.linalg.norm(np.asarray(old['point'])-region['point'])<.0002
                    and np.dot(old['normal'],region['normal'])>.99 for old in existing)
                if not duplicate:existing.append(region)
            row['additional_region_source']='exposed collision surfaces of complete fixed terminal assembly'
        else:
            best=regions[0]
            rows.append(dict(finger=chain['finger'],link=link,surface_link=link,point_m=best['point'],
                outward_normal=best['normal'],joint_names=chain['joint_names'],support_radius_m=best['radius_m'],
                distal_direction=distal.tolist(),assembly_centroid_local_m=terminal_origin.tolist(),regions=regions,
                source='source URDF fixed terminal assembly exterior; rigid surface, no measured compliance'))
    for row in rows:
        radius=float(row.get('support_radius_m', .0012))
        row['contact_model']=dict(
            friction_coefficient=.3, normal_force_limit_n=1., support_radius_m=radius,
            compliance_m_per_n=None,
            source='explicit conservative simulation assumptions, not hardware measurements',
            force_limit_source='shared task-model cap; URDF effort and Jacobian additionally bound force',
            friction_source='shared planning assumption; environment material is separately configured',
            support_radius_source='static calibration' if row['finger'] in calibrated else '13-ray collision-mesh patch disc',
            compliance_source='unknown; online motor tracking is only a contact proxy')
        row.setdefault('geometry_class','terminal_body_mesh_patch')
        row.setdefault('physical_pad_validation','geometric solid-body support only; material/pad and sustained load unmeasured')
    if rows:
        poses=hand.fk(np.clip(np.zeros(len(hand.joint_names)),hand.lower,hand.upper))
        center=np.mean([poses[r['link']][:3,3] for r in rows],axis=0)
        for row in rows:
            if row['finger'] in calibrated:continue
            pose=poses[row['link']]
            def inward_score(region):
                point=pose[:3,:3]@region['point']+pose[:3,3]
                toward=center-point
                distal=pose[:3,:3]@row['distal_direction']
                toward-=distal*(toward@distal)
                return float((pose[:3,:3]@region['normal'])@toward/max(np.linalg.norm(toward),1e-9))
            world_points=np.asarray([r['point']for r in row['regions']])
            length=max(float(np.ptp(world_points@np.asarray(row['distal_direction']))),1e-6)
            best=max(row['regions'],key=lambda r:inward_score(r)+.2*float(np.asarray(r['point'])@row['distal_direction'])/length)
            row['point_m']=best['point'];row['outward_normal']=best['normal'];row['support_radius_m']=best['radius_m'];row['contact_model']['support_radius_m']=best['radius_m']
            row['selection_source']='static FK: mesh-supported patch facing the other finger chains'
    return rows
