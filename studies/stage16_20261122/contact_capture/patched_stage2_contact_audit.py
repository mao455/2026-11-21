"""Passive all-hand/object contact sidecar. Original evaluator contacts unchanged."""
import json
import numpy as np
from pathlib import Path

def install(out):
 import diagnostics
 original=diagnostics.Contacts
 class AuditedContacts(original):
  def __init__(self,env):
   super().__init__(env)
   self.extra_env=env
   paths=['/World/envs/env_0/Robot/'+n for n in env.hand.body_names]
   self.extra=env.hand._physics_sim_view.create_rigid_contact_view(paths,[['/World/envs/env_0/Object']]*len(paths),max_contact_data_count=16384)
   self.extra_names=[Path(x).name for x in self.extra.sensor_paths]
   self.extra_stream=(Path(out)/'all_hand_object_contacts.jsonl').open('w',buffering=1)
   self.extra_step=0
   (Path(out)/'all_hand_object_contact_manifest.json').write_text(json.dumps(dict(names=self.extra_names,filters=self.extra.filter_paths,cadence_steps=6,cadence_hz=20,point_frame='world',point_unit='m',force_unit='N',separation_unit='m',actual_native_contact_points=True,private_diagnostic_only=True,original_evaluator_contacts_unchanged=True),indent=2))
  def read(self,dt):
   result=super().read(dt)
   if self.extra_step%6==0:
    force,point,normal,separation,count,start=[diagnostics.array(x) for x in self.extra.get_contact_data(dt)]
    rows=[]
    for i,name in enumerate(self.extra_names):
     n=int(count[i,0]);a=int(start[i,0])
     if n:
      # Native query buffers may be reused by later queries. Copy every slice
      # immediately; these are evaluator-only engine points, never controller inputs.
      forces=np.array(force[a:a+n],copy=True).reshape(-1)
      points=np.array(point[a:a+n],copy=True)
      normals=np.array(normal[a:a+n],copy=True)
      separations=np.array(separation[a:a+n],copy=True).reshape(-1)
      rows.append(dict(link=name,count=n,normal_force_sum_n=float(forces.sum()),minimum_separation_m=float(separations.min()),points_world_m=points.tolist(),normals_world=normals.tolist(),normal_forces_n=forces.tolist(),separations_m=separations.tolist()))
    q=self.extra_env.hand.data.joint_pos[0].detach().cpu().numpy().copy().tolist()
    self.extra_stream.write(json.dumps(dict(step=self.extra_step,time_s=(self.extra_step+1)*float(dt),physics_dt_s=float(dt),joint_order='env.hand.joint_names',joint_names=list(self.extra_env.hand.joint_names),actual_joint_pos=q,contacts=rows))+'\n')
   self.extra_step+=1
   return result
 diagnostics.Contacts=AuditedContacts
