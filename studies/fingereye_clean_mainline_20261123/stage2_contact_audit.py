"""Passive all-hand/object contact sidecar. Original evaluator contacts unchanged."""
import json
import numpy as np
from pathlib import Path

def enable_native_contact_processing(env):
 """Enable the same CPU report path as IsaacLab ContactSensor, without a policy sensor.

 SimulationContext disables contact processing regardless of spawn ContactReportAPI.
 Direct tensor views do not perform ContactSensor.__init__'s explicit re-enable.
 This must run after SimulationContext construction and before sampled physics.
 """
 import carb.settings
 from pxr import PhysxSchema, UsdPhysics
 settings=carb.settings.get_settings()
 key='/physics/disableContactProcessing'
 before=bool(settings.get_as_bool(key))
 settings.set_bool(key,False)
 if settings.get_as_bool(key):
  raise RuntimeError('PhysX CPU native contact processing could not be enabled')
 stage=env.sim.stage
 bodies=[]
 for name in env.hand.body_names:
  path='/World/envs/env_0/Robot/'+name
  prim=stage.GetPrimAtPath(path)
  if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
   raise RuntimeError('Contact diagnostic body is not a live rigid body: '+path)
  if not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
   raise RuntimeError('Contact diagnostic body lacks spawn ContactReportAPI: '+path)
  api=PhysxSchema.PhysxContactReportAPI.Get(stage,path)
  threshold=api.GetThresholdAttr().Get()
  if threshold is None or float(threshold)!=0.:
   raise RuntimeError('Contact report threshold must be zero: '+path)
  bodies.append(dict(path=path,rigid_body_api=True,contact_report_api=True,threshold_n=float(threshold)))
 return dict(setting=key,before=before,after=False,rigid_bodies=bodies,
             enable_source='same setting as IsaacLab ContactSensor.__init__; direct tensor views need explicit enable',
             physics_change='CPU contact report processing only; solver, geometry, forces and evaluator code untouched',
             capture_validation='independent horizontal and tilted fixture passed normal positions and aggregate vectors; raw friction anchors failed 0.5mm spatial gate; finite-pad tangential attribution unvalidated')

def install(out):
 import diagnostics
 original=diagnostics.Contacts
 class AuditedContacts(original):
  def __init__(self,env):
   native_setup=enable_native_contact_processing(env)
   (Path(out)/'native_contact_capture_setup.json').write_text(json.dumps(native_setup,indent=2))
   from cross_hand_runtime import selected, HandDescription
   hand=selected()
   if hand is None:
    super().__init__(env)
   else:
    model=HandDescription(hand['urdf_path'],joint_names=hand['joint_names'],base_joint_names=hand['base_joint_names'])
    self.names=sorted({entry['link'] for entry in model.collision_meshes}&set(env.hand.body_names))
    if not self.names:raise RuntimeError('Selected hand has no physical contact bodies')
    self.views={name:env.hand._physics_sim_view.create_rigid_contact_view(
     '/World/envs/env_0/Robot/'+name,['/World/envs/env_0/Object'],max_contact_data_count=256) for name in self.names}
   paths=['/World/envs/env_0/Robot/'+n for n in env.hand.body_names]
   self.extra=env.hand._physics_sim_view.create_rigid_contact_view(paths,[['/World/envs/env_0/Object']]*len(paths),max_contact_data_count=16384)
   self.extra_names=[Path(x).name for x in self.extra.sensor_paths]
   self.extra_stream=(Path(out)/'all_hand_object_contacts.jsonl').open('w',buffering=1)
   self.extra_step=0
   self.extra_cadence_steps=1
   self.friction_capability=callable(getattr(self.extra,'get_friction_data',None))
   self.contact_env=env
   # Event reports expose articulation-internal contacts that tensor contact
   # queries can omit. This is a private sidecar, never a policy observation.
   from omni.physx import get_physx_simulation_interface
   from pxr import PhysicsSchemaTools
   self.report_stream=(Path(out)/'native_contact_events.jsonl').open('w',buffering=1)
   def contact_report(headers,data):
    # Private native diagnostics cover approach, closure and lift as well as initialization.
    for header in headers:
     names={key:str(PhysicsSchemaTools.intToSdfPath(getattr(header,key))) for key in ('actor0','actor1','collider0','collider1')}
     count=int(header.num_contact_data);offset=int(header.contact_data_offset)
     contacts=[]
     for row in data[offset:offset+count]:
      values={}
      for key in ('separation','impulse','position','point','normal'):
       if not hasattr(row,key):continue
       value=getattr(row,key)
       if np.isscalar(value):values[key]=float(value)
       elif all(hasattr(value,k) for k in ('x','y','z')):values[key]=[float(value.x),float(value.y),float(value.z)]
       else:
        try:values[key]=np.asarray(value,dtype=float).tolist()
        except (TypeError,ValueError):values[key+'_unavailable']=True
      contacts.append(values)
     self.report_stream.write(json.dumps(dict(step=self.extra_step+1,event=int(header.type),count=count,contacts=contacts,**names))+'\n')
   self.report_subscription=get_physx_simulation_interface().subscribe_contact_report_events(contact_report)
   (Path(out)/'all_hand_object_contact_manifest.json').write_text(json.dumps(dict(names=self.extra_names,body_pose_names=list(env.hand.body_names),filters=self.extra.filter_paths,schema=2,cadence_steps=self.extra_cadence_steps,physics_dt_s=float(env.cfg.sim.dt),normal_force_source='PhysX tensor get_contact_data(dt)',friction_api_available=self.friction_capability,friction_source='PhysX tensor get_friction_data(dt), aggregate forces on sensor body; raw solver anchor positions are NOT spatially calibrated',event_impulse_units='N s; native contact event vector is normal-only; independent friction API required',body_pose_convention='xyz wxyz at post-physics scene update',private_diagnostic_only=True,original_evaluator_contacts_unchanged=True),indent=2))
  def read(self,dt):
   result=super().read(dt)
   if self.extra_step%self.extra_cadence_steps==0:
    force,point,normal,separation,count,start=[diagnostics.array(x) for x in self.extra.get_contact_data(dt)]
    rows=[]
    for i,name in enumerate(self.extra_names):
     n=int(count[i,0]);a=int(start[i,0])
     if n:rows.append(dict(link=name,count=n,normal_force_sum_n=float(force[a:a+n].sum()),minimum_separation_m=float(separation[a:a+n].min()),points_world_m=np.asarray(point[a:a+n]).tolist(),normals_world=np.asarray(normal[a:a+n]).tolist(),normal_forces_n=np.asarray(force[a:a+n]).reshape(-1).tolist(),separations_m=np.asarray(separation[a:a+n]).reshape(-1).tolist()))
    # Friction counts/start buffers alias the normal API buffers: normal rows
    # are fully copied above before querying this independent anchor stream.
    friction_rows=[];friction_status='api_unavailable'
    if self.friction_capability:
     try:
      tangential,anchors,friction_count,friction_start=[diagnostics.array(x) for x in self.extra.get_friction_data(dt)]
      friction_status='available'
      for i,name in enumerate(self.extra_names):
       n=int(friction_count[i,0]);a=int(friction_start[i,0])
       if n:friction_rows.append(dict(link=name,count=n,points_world_m=np.asarray(anchors[a:a+n]).tolist(),forces_world_n=np.asarray(tangential[a:a+n]).tolist()))
     except Exception as exc:
      friction_status='query_failed:'+type(exc).__name__+':'+str(exc)
    env=self.contact_env
    positions=diagnostics.array(env.hand.data.body_pos_w[0]);quaternions=diagnostics.array(env.hand.data.body_quat_w[0])
    poses={name:np.r_[positions[i],quaternions[i]].tolist() for i,name in enumerate(env.hand.body_names)}
    self.extra_stream.write(json.dumps(dict(step=self.extra_step+1,time_s=(self.extra_step+1)*float(dt),contacts=rows,friction_contacts=friction_rows,friction_status=friction_status,physics_dt_s=float(dt),normal_api_status='available',body_poses_world_wxyz=poses,sample_phase='post_physics_scene_update',private_diagnostic_only=True))+'\n')
   self.extra_step+=1
   return result
 diagnostics.Contacts=AuditedContacts
