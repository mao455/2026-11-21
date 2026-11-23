"""Private readout and known-spin controls, excluded from scored episode."""
import json,inspect
import numpy as np
ROWS=[]
def sample(env,rec):
    view=env.object.root_physx_view
    p=view.get_transforms().clone().cpu().numpy()[0]
    v=view.get_velocities().clone().cpu().numpy()[0]
    ROWS.append(dict(step=rec['step'],raw_pose_xyzw=p.tolist(),raw_velocity=v.tolist(),timestamp=float(env.object.data._sim_timestamp),cache_pose_error=float(np.max(np.abs(p-np.r_[rec['object_pose'][:3],np.asarray(rec['object_pose'][3:])[[1,2,3,0]]]))),cache_velocity_error=float(np.max(np.abs(v-rec['object_velocity'])))))
def finish(env,out):
    import torch
    data=dict(data_source=inspect.getfile(type(env.object.data)),readout=ROWS,controls=[])
    # Executed only after the original evaluate and recorder close. No scored
    # records are added and no control observation sees these diagnostic states.
    env.object.reset()
    for rate in (0.,2.):
        env.object.reset()
        pose=torch.tensor([[0.,0.,1.,1.,0.,0.,0.]],device=env.device)
        vel=torch.tensor([[0.,0.,0.,0.,0.,rate]],device=env.device)
        env.object.write_root_pose_to_sim(pose)
        env.object.write_root_com_velocity_to_sim(vel)
        env.scene.write_data_to_sim();env.sim.forward();env.scene.update(dt=0.)
        rows=[]
        for k in range(24):
            env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(dt=env.physics_dt)
            p=env.object.root_physx_view.get_transforms().clone().cpu().numpy()[0]
            v=env.object.root_physx_view.get_velocities().clone().cpu().numpy()[0]
            rows.append(dict(step=k,pose_xyzw=p.tolist(),velocity=v.tolist()))
        data['controls'].append(dict(label='free_static' if rate==0 else 'free_known_spin',commanded_omega_world=[0,0,rate],dt=env.physics_dt,rows=rows))
    (out/'angular_probe.json').write_text(json.dumps(data)+'\n')
