"""Private Round7 physical asset configuration and runtime checks; never public inputs."""
import json,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent
ASSET=P/'assets/round7b_xhand_allfine.usdc'

def configure(cfg):
 cfg.seed=0
 cfg.robot_cfg.spawn.usd_path=str(ASSET)
 cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions=True
 # Tiny-object contact scales: avoid merging anchors 25 mm apart by default.
 cfg.sim.physx.friction_offset_threshold=.0005
 cfg.sim.physx.friction_correlation_distance=.001
 # Render cameras are driven from measured joint FK + static calibration.
 cfg.sim.use_fabric=True
 from isaaclab.sim import CollisionPropertiesCfg
 cfg.robot_cfg.spawn.collision_props=CollisionPropertiesCfg(contact_offset=.0002,rest_offset=0.)
 tree=ET.parse(cfg.robot_urdf).getroot()
 cfg.robot_cfg.actuators['xhand_fingers'].effort_limit_sim={j.get('name'):float(j.find('limit').get('effort')) for j in tree.findall('joint') if j.get('name').startswith('left_hand_') and j.find('limit') is not None}

 for name,cam in cfg.xhand_camera_configs.items():
  meta=cfg.xhand_camera_metadata[name]
  cam.prim_path=f"/World/envs/env_.*/Robot/FingerEye_{name}"
  meta['camera_prim_path_pattern']=cam.prim_path
  meta['extrinsic_source']='URDF_fixed_transform_FK_synchronized_renderer_round7'

def arr(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def audit_runtime(env,cfg,out):
 from pxr import Usd,UsdPhysics,UsdShade
 tree=ET.parse(cfg.robot_urdf).getroot();expected={}
 for l in tree.findall('link'):
  inert=l.find('inertial')
  if inert is None:continue
  i=inert.find('inertia');v={k:float(x) for k,x in i.attrib.items()}
  inertia=np.array([[v['ixx'],v['ixy'],v['ixz']],[v['ixy'],v['iyy'],v['iyz']],[v['ixz'],v['iyz'],v['izz']]])
  expected[l.get('name')]={'mass':float(inert.find('mass').get('value')),'inertia_eigenvalues':np.linalg.eigvalsh(inertia).tolist(),'com':np.fromstring(inert.find('origin').get('xyz','0 0 0'),sep=' ').tolist() if inert.find('origin') is not None else [0,0,0]}
 view=env.hand.root_physx_view;m=arr(view.get_masses())[0];ii=arr(view.get_inertias())[0];com=arr(view.get_coms())[0]
 assert len(m)==35 and abs(sum(m)-sum(e['mass'] for e in expected.values()))<1e-6
 rows=[]
 for index,name in enumerate(env.hand.body_names):
  e=expected.get(name);eigs=np.linalg.eigvalsh(ii[index].reshape(3,3));r={'body':name,'mass':float(m[index]),'inertia_matrix':ii[index].tolist(),'inertia_eigenvalues':eigs.tolist(),'com_pose':com[index].tolist(),'urdf':e}
  assert '_fingereye_camera_' not in name,'Optical frame still physical'
  assert e is not None and np.isclose(m[index],e['mass'],rtol=1e-5,atol=1e-10),(name,m[index],e)
  assert np.min(eigs)>0,(name,eigs)
  r['inertia_relative_error']=float(np.max(np.abs(eigs-np.array(e['inertia_eigenvalues']))/np.maximum(e['inertia_eigenvalues'],1e-15)))
  r['com_position_error_m']=float(np.max(np.abs(com[index,:3]-e['com'])))
  assert r['inertia_relative_error']<1e-5 and r['com_position_error_m']<1e-7,r
  rows.append(r)
 # Report rather than silently equating positive inertia to validated real hardware inertia.
 actuators={}
 for name in ['get_dof_stiffnesses','get_dof_dampings','get_dof_max_forces','get_dof_max_velocities','get_dof_limits','get_dof_friction_coefficients']:
  try:actuators[name]=arr(getattr(view,name)()).tolist()
  except Exception as exc:actuators[name]={'unavailable':str(exc)}
 shape={}
 for key,asset in [('hand',env.hand),('object',env.object)]:
  for method in ['get_material_properties','get_contact_offsets','get_rest_offsets']:
   try:shape[key+'_'+method]=arr(getattr(asset.root_physx_view,method)()).tolist()
   except Exception as exc:shape[key+'_'+method]={'unavailable':str(exc)}
 expected_efforts={j.get('name'):float(j.find('limit').get('effort')) for j in tree.findall('joint') if j.find('limit') is not None}
 for name,effort in zip(env.hand.joint_names,actuators['get_dof_max_forces'][0]):
  if name.startswith('left_hand_'):assert np.isclose(effort,expected_efforts[name],rtol=1e-6)
 assert np.allclose(shape['hand_get_contact_offsets'],.0002,atol=1e-9)
 assert np.allclose(shape['hand_get_rest_offsets'],0,atol=1e-9)
 cameras=[]
 for p in env.sim.stage.Traverse():
  if '_fingereye_camera_' in p.GetName() and p.GetTypeName()=='Xform' and p.GetName().rsplit('_',1)[-1] in ('front1','front2','back'):
   cameras.append({'path':str(p.GetPath()),'rigid':p.HasAPI(UsdPhysics.RigidBodyAPI),'mass_api':p.HasAPI(UsdPhysics.MassAPI)})
 dump(out/'camera_prim_audit.json',cameras)
 assert len(cameras)==12 and not any(c['rigid'] or c['mass_api'] for c in cameras),cameras
 data={'version':'R7B-allfine-C2-v1','asset_sha256':hashlib.sha256(ASSET.read_bytes()).hexdigest(),'body_names':env.hand.body_names,'body_count':len(rows),'total_mass_kg':float(sum(m)),'urdf_total_mass_kg':sum(e['mass'] for e in expected.values()),'bodies':rows,'joint_names':env.hand.joint_names,'actuators':actuators,'shape_properties':shape,'cameras':cameras,'gravity':list(cfg.sim.gravity),'hand_gravity_disabled':True,'object_world_gravity_force':'-mass*9.81 applied per step; audited separately','use_fabric':cfg.sim.use_fabric,'dt':cfg.sim.dt,'solver_type':cfg.sim.physx.solver_type,'enable_stabilization':cfg.sim.physx.enable_stabilization,'limitations':['URDF inertias are asset claims, not measured hardware validation','PhysX cooked convex hull exact surfaces not exported by this API; authored mesh signatures and actual contact links retained']}
 data['object_mass_kg']=arr(env.object.root_physx_view.get_masses()).tolist()
 data['seed']=cfg.seed
 data['base_pose_overlay_enabled']=False
 data['resolved_cfg_repr']=repr(cfg.to_dict())
 from round7_surface_probe import runtime_attributes
 runtime=runtime_attributes(env)
 scene_values=[]
 for prim in env.sim.stage.Traverse():
  if prim.IsA(UsdPhysics.Scene):
   attrs={name:prim.GetAttribute(name).Get() for name in ('physxScene:frictionOffsetThreshold','physxScene:frictionCorrelationDistance')}
   assert np.isclose(attrs['physxScene:frictionOffsetThreshold'],.0005,rtol=1e-6),attrs
   assert np.isclose(attrs['physxScene:frictionCorrelationDistance'],.001,rtol=1e-6),attrs
   scene_values.append({'path':str(prim.GetPath()),**attrs})
 assert scene_values
 data['tiny_contact_scene_attributes']=scene_values
 self_values=[a['attributes']['physxArticulation:enabledSelfCollisions']['value'] for a in runtime['actors'] if 'physxArticulation:enabledSelfCollisions' in a['attributes']]
 assert self_values and all(v is True for v in self_values),self_values
 data['enabled_self_collisions']=True
 data['self_collision_exclusion_scope']='same URDF fixed cluster or one directly adjacent movable joint; exact manifest retained'
 filtering=json.loads((P/'assets/round7b_xhand_selfcollisions.json').read_text())
 expected_pairs={tuple(sorted((p['body_a'],p['body_b']))) for p in filtering['excluded_pairs']}
 actual_pairs=set();directed=[]
 for prim in env.sim.stage.Traverse():
  if not str(prim.GetPath()).startswith('/World/envs/env_0/Robot/') or prim.GetName() not in env.hand.body_names:continue
  if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):continue
  for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
   target_prim=env.sim.stage.GetPrimAtPath(target)
   assert target_prim.IsValid() and str(target).startswith('/World/envs/env_0/Robot/'),str(target)
   actual_pairs.add(tuple(sorted((prim.GetName(),target_prim.GetName()))))
   directed.append([str(prim.GetPath()),str(target)])
 assert actual_pairs==expected_pairs,(actual_pairs-expected_pairs,expected_pairs-actual_pairs)
 data['runtime_filtered_pair_count']=len(actual_pairs)
 data['runtime_filtered_pairs']=directed
 dump(out/'runtime_asset_attributes.json',runtime)
 dump(out/'dynamics_audit.json',data)

_robot=None

def sync_cameras(env,cfg):
 """Renderer-only pose update from proprioception; no object/force/state access.
 CPU TiledCamera USD pose reads remain stale for nested moving non-rigid frames
 on this Isaac build. World cameras avoid that backend limitation while using
 the exact original URDF mount FK. Physical optical frames remain in the USD.
 """
 global _robot
 import torch
 from mainline.strategy import Robot,NAMES
 if _robot is None:_robot=Robot(cfg.robot_urdf)
 q=arr(env.hand.data.joint_pos)[0];q=np.array([q[env.hand.joint_names.index(n)] for n in NAMES]);fk=_robot.fk(q)
 for name,cam in env._active_cameras.items():
  if name=='third_view':continue
  T=fk[cfg.xhand_camera_metadata[name]['camera_link']];quat=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
  cam.set_world_poses(torch.tensor(T[None,:3,3],dtype=torch.float32,device=env.device),torch.tensor(quat[None],dtype=torch.float32,device=env.device),convention='opengl')

def capture_cameras(env,cfg,names):
 from fingereye.tasks.direct.fingereye.xhand_camera_recorder import to_uint8_batch
 sync_cameras(env,cfg);env.sim.render()
 frames={}
 for name in names:
  cam=env._active_cameras[name];cam.update(0.,force_recompute=True)
  frames[name]=to_uint8_batch(cam.data.output['rgb'])
 return frames

def audit_cameras(env,cfg,step,out):
 global _robot
 if step%120 and step!=0:return
 from mainline.strategy import Robot,NAMES
 if _robot is None:_robot=Robot(cfg.robot_urdf)
 q=arr(env.hand.data.joint_pos)[0];q=np.array([q[env.hand.joint_names.index(n)] for n in NAMES]);fk=_robot.fk(q)
 rows=[]
 for name,cam in env._active_cameras.items():
  if name=='third_view':continue
  meta=cfg.xhand_camera_metadata[name];T=fk[meta['camera_link']]
  actualpos=arr(cam.data.pos_w)[0];quat=arr(cam.data.quat_w_opengl)[0];R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
  rows.append({'name':name,'position_error_m':float(np.linalg.norm(T[:3,3]-actualpos)),'rotation_error_rad':float(Rotation.from_matrix(T[:3,:3].T@R).magnitude()),'fk_position':T[:3,3].tolist(),'camera_position':actualpos.tolist(),'parent_physics_position':arr(env.hand.data.body_pos_w)[0,env.hand.body_names.index(meta['parent_link'])].tolist()})
 with (out/'camera_fk_audit.jsonl').open('a') as f:f.write(json.dumps({'step':step,'cameras':rows})+'\n')
 assert max(r['position_error_m'] for r in rows)<.0001,rows
 assert max(r['rotation_error_rad'] for r in rows)<.002,rows
