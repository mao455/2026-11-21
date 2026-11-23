"""R4-V: image/joint-only observation boundary; private simulator audit.

The temporary scene.pkl is an IsaacLab scene-description transport produced here;
no historical dataset, grasp target, motion plan, or source PKL is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import traceback

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]


def clean(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-json', default='{"dx":0,"dy":0,"yaw":0}')
    args = parser.parse_args()
    # Native Isaac logs write fd 1; keep an independent protocol fd before redirecting.
    protocol = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    def respond(value):
        protocol.write(json.dumps(clean(value), allow_nan=False) + '\n')
        protocol.flush()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    public = out / 'public'
    public.mkdir(exist_ok=True)
    from run_round14_episode import validate_case
    case = validate_case(json.loads(args.case_json))
    import numpy as np
    parameters = {'mass_kg': .005, 'friction': .8, 'stiffness': 20.,
                  'depth_noise_m': 0., 'depth_bias_m': 0., 'depth_dropout': 0., 'noise_seed': 0}
    parameters.update({key: case[key] for key in parameters if key in case})
    parameters['rgb_noise_std'] = case.get('rgb_noise_std', 0.)
    sensor_rng = np.random.default_rng(parameters['noise_seed'])
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(LAB / 'scripts'))
    sys.path.insert(0, str(HERE.parent / 'rax_tinyobj_round1'))
    from xhand_grasp_target import HAND_ONLY_JOINT_NAMES
    from evaluator import evaluate
    q_start = np.zeros(18, dtype=np.float32)
    q_start[:3] = [-.18, 0., .30]
    quaternion = Rotation.from_euler('z', case['yaw']).as_quat()[[3, 0, 1, 2]]
    object_pose = np.r_[case['dx'], case['dy'], .0001, quaternion].astype(np.float32)
    mesh_path = LAB / 'assets/objects/manifest_native_obj/abc_90621_native.obj'
    if 'object_id' in case:
        from round14_objects import create
        mesh_path=create(case['object_id'])
    payload = {'object_mesh_path': str(mesh_path), 'object_pose_env': object_pose,
               'object_scale': np.ones(3), 'q_start': q_start}
    sys.argv = [str(HERE / 'backend.py'), '--motion_pkl', str(out / 'scene.pkl'),
                '--device', 'cpu', '--headless', '--save_video', '--video_width', '320',
                '--video_height', '240', '--camera-fps', '5',
                '--camera_eye', '0,-0.001,0.5', '--camera_target', '0,0,0']
    backend = env = writer = finger_writer = recorder = None
    records = []
    initial_object_z = float(object_pose[2])
    try:
        import rollout_xhand_motion_dual as backend
        import torch
        import trimesh
        import imageio.v2 as imageio
        from geometry import matrix, usd_collision_meshes
        from round8_convex_audit import Audit
        from diagnostics import Contacts
        from run_world_gravity import set_world_gravity_force, world_to_body
        sample = backend._motion_sample_for_dynamic_env(payload, q_start, parameters['mass_kg'],
                                                       parameters['friction'], parameters['friction'])
        sample_path = out / 'scene.pkl'
        sample_path.write_bytes(pickle.dumps({'format': 'fingereye.combined_motion_static_gravity_batch_samples.v1', 'batch_samples': [sample]}))
        cfg = backend.parse_env_cfg(backend.args_cli.task, device='cpu', num_envs=1)
        backend.args_cli.object_mass = parameters['mass_kg']
        backend.args_cli.robot_friction = parameters['friction']
        backend.args_cli.object_friction = parameters['friction']
        backend.args_cli.position_stiffness = parameters['stiffness']
        backend.args_cli.position_damping = 2.
        backend._configure_env(cfg, sample_path, backend.args_cli)
        cfg.configure_xhand_cameras('index', requested_fps=20, include_third_view=True, width=160, height=120)
        finger_names = [f'{f}_{m}' for f in ('index','thumb') for m in ('front1','front2','back')]
        cfg.camera_name_list = ['third_view', *finger_names]
        cfg.cam_third_view.data_types = ['rgb']
        cfg.cam_third_view.update_period = 0.
        for name in finger_names:
            cfg.xhand_camera_configs[name].update_period = 0.
        cfg.sim.physx.enable_stabilization = False
        cfg.cam_third_view.spawn.clipping_range = (.01, 3.)
        cfg.sim.dt = 1 / 120
        cfg.sim.render_interval = 6
        cfg.sim.enable_scene_query_support = True
        from round7c_dynamics import configure, audit_runtime, audit_cameras, capture_cameras
        configure(cfg)
        env = backend.gym.make(backend.args_cli.task, cfg=cfg).unwrapped
        from pxr import UsdGeom
        for prim in env.sim.stage.Traverse():
            path = str(prim.GetPath())
            if '/Robot/' in path and '/visuals' in path and 'fingereye_camera_' in path:
                UsdGeom.Imageable(prim).MakeInvisible()
        backend._remove_background_only()
        backend._ensure_ground_table()
        backend.args_cli.base_pose_overlay = False
        backend._configure_base_pose_overlay(env, backend.args_cli)
        env.reset()
        ids = torch.tensor([0], device=env.device)
        backend._set_q_state_and_target_batch(env, q_start[None], ids, snap_state=True)
        backend._write_object_pose_batch(env, object_pose[None], ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.)
        meshes, manifest = usd_collision_meshes(env.sim.stage, '/World/envs/env_0/Robot', env.hand.body_names)
        object_mesh = trimesh.load(mesh_path, force='mesh', process=True)
        audit = Audit(meshes, object_mesh)
        contacts = Contacts(env)
        from round7b_native_contacts import NativeSelfContacts
        self_contacts = NativeSelfContacts(env, out)
        audit_runtime(env, cfg, out)
        limits = env.hand.data.soft_joint_pos_limits[0].cpu().numpy()
        canonical_limits = np.stack([backend._env_to_canonical_q(env, limits[:, j]) for j in [0, 1]], axis=1)
        robot_description = {'urdf_path': backend.args_cli.robot_urdf,
                             'joint_names': list(HAND_ONLY_JOINT_NAMES), 'joint_limits': canonical_limits,
                             'dt': env.physics_dt, 'table_z': 0.,
                             'fingereye_calibration': {n: cfg.xhand_camera_metadata[n] for n in finger_names},
                             'observation_profile': 'initial_overhead_rgb_live_fingereye_rgb_joint_state_v1',
                             'action': '18 position targets: xyz, yaw pitch roll (Rz Ry Rx), 12 fingers; meters/radians'}
        save(out / 'scene.json', {'robot_description': robot_description, 'mesh': str(mesh_path),
             'mesh_sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
             'initial_object_pose': object_pose, 'collision_geometry': manifest,
             'case': case, 'realized_parameters': parameters, 'object_mass_kg': parameters['mass_kg'],
             'friction': parameters['friction'], 'object_gravity': [0, 0, -9.81],
             'finger_stiffness': parameters['stiffness'], 'finger_damping': 2.,
             'physics': backend._physics_configuration_snapshot(env, 1, independent_ground=False)})
        palm_index = env.hand.body_names.index('left_hand_palm_link')
        writer = imageio.get_writer(out / 'rollout.mp4', fps=5, macro_block_size=16)
        stream = (out / 'steps.jsonl').open('w', buffering=1)
        observation_index = 0
        sensor_cache = None
        overhead_initial = None
        finger_writer = imageio.get_writer(out / 'fingereye.mp4', fps=20, macro_block_size=16)
        frame_stream = (out/'image_manifest.jsonl').open('w',buffering=1)
        gravity_stream = (out/'gravity_force_audit.jsonl').open('w',buffering=1)
        save(out/'physics_version.json', {'version':'R7B-D','physical_dynamics':'R7B-allfine-C2-v1',
             'gravity_frame':'world','enable_stabilization':False,
             'new_sensor_profile':True,'evaluator_thresholds_unchanged':True})
        def sensor_observation():
            nonlocal observation_index, sensor_cache, overhead_initial
            if sensor_cache is not None and sensor_cache['step']==len(records):
                return sensor_cache
            frames = capture_cameras(env, cfg, ['third_view', *finger_names])
            audit_cameras(env, cfg, len(records), out)
            if set(frames)!=set(['third_view',*finger_names]):
                raise RuntimeError('Required RGB camera unavailable')
            top = env._active_cameras['third_view']
            rgb=frames['third_view'][0]
            # Overhead rendered for private evidence; only its INITIAL frame is public.
            if observation_index==0:
                path=public/'overhead_initial.npy';np.save(path,rgb)
                overhead_initial={'rgb_path':str(path),'intrinsics':top.data.intrinsic_matrices[0].cpu().numpy(),
                    'camera_pose_world':np.r_[top.data.pos_w[0].cpu().numpy(),top.data.quat_w_ros[0].cpu().numpy()],
                    'width':rgb.shape[1],'height':rgb.shape[0],'capture_step':0}
            if observation_index==0 or len(records)%24==0:
                writer.append_data(rgb)
            observed={}
            for name in finger_names:
                rgb=frames[name][0].copy()
                if parameters['rgb_noise_std']:
                    rgb=np.clip(rgb.astype(float)+sensor_rng.normal(0,parameters['rgb_noise_std'],rgb.shape),0,255).astype(np.uint8)
                path=public/f'{name}_{observation_index:05d}.npy';np.save(path,rgb)
                observed[name]={'rgb_path':str(path),'width':rgb.shape[1],'height':rgb.shape[0],
                                'capture_step':len(records),'frame_id':observation_index}
            composite=np.vstack([np.hstack([np.load(observed[n]['rgb_path']) for n in finger_names[i:i+3]]) for i in (0,3)])
            finger_writer.append_data(composite)
            frame_stream.write(json.dumps({'frame_id':observation_index,'step':len(records),
                'time_s':len(records)*env.physics_dt,'cameras':list(observed),
                'sha256':{n:hashlib.sha256(Path(d['rgb_path']).read_bytes()).hexdigest() for n,d in observed.items()}})+'\n')
            observation_index+=1
            sensor_cache={'time_s':len(records)*env.physics_dt,'step':len(records),
                    'actual_q':backend._env_to_canonical_q(env,env.hand.data.joint_pos[0].cpu().numpy()),
                    'actual_qvel':backend._env_to_canonical_q(env,env.hand.data.joint_vel[0].cpu().numpy()),
                    'initial_overhead':overhead_initial,'fingereye':observed}
            return sensor_cache
        from record_success_views import Recorder
        recorder = Recorder(env, cfg, out, case)
        recorder.capture(0, 'initial')
        respond({'ok': True, 'observation': sensor_observation(), 'robot_description': robot_description})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request.get('op')
                if op == 'finish':
                    summary = evaluate(records, initial_object_z)
                    save(out / 'summary.json', summary)
                    recorder.close()
                    recorder = None
                    respond({'ok': True, 'summary': summary})
                    break
                if op == 'step':
                    target = np.asarray(request['target_q'], dtype=np.float32)
                    count = request['steps']
                    if target.shape != (18,) or not np.isfinite(target).all() or not isinstance(count, int) or not 1 <= count <= 24:
                        raise ValueError('invalid action')
                    if len(records) + count > 24000:
                        raise ValueError('physical step budget exhausted')
                    if np.any(target < canonical_limits[:, 0] - 1e-5) or np.any(target > canonical_limits[:, 1] + 1e-5):
                        raise ValueError('joint target outside robot joint limits')
                    for _ in range(count):
                        backend._set_q_state_and_target_batch(env, target[None], ids, snap_state=False)
                        set_world_gravity_force(env, ids, parameters['mass_kg'], 1.)
                        quat=env.object.data.body_link_quat_w[ids].clone();quat[...,1:]*=-1
                        actual_force=world_to_body(env.object.permanent_wrench_composer.composed_force_as_torch[ids],quat)
                        expected_force=torch.zeros_like(actual_force);expected_force[...,2]=-9.81*parameters['mass_kg']
                        err=float(torch.max(torch.abs(actual_force-expected_force)))
                        gravity_stream.write(json.dumps({'step':len(records),'max_error_n':err})+'\n')
                        if err>1e-6:raise RuntimeError('World gravity audit failed')
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=env.physics_dt)
                        obj = backend._object_pose_batch(env, ids)[0]
                        bodies = np.c_[env.hand.data.body_pos_w[0].cpu().numpy(), env.hand.data.body_quat_w[0].cpu().numpy()]
                        check = audit.check(dict(zip(env.hand.body_names, bodies)), obj)
                        rec = clean({'step': len(records), 'time_s': (len(records) + 1) * env.physics_dt,
                           'phase': request.get('phase', ''), 'target_q': target,
                           'actual_q': backend._env_to_canonical_q(env, env.hand.data.joint_pos[0].cpu().numpy()),
                           'actual_qvel': backend._env_to_canonical_q(env, env.hand.data.joint_vel[0].cpu().numpy()),
                           'object_pose': obj, 'object_velocity': backend._object_velocity_batch(env, ids)[0],
                           'body_poses': bodies, 'penetration': check, 'contacts': contacts.read(env.physics_dt),
                           'hand_object_relative_matrix': np.linalg.inv(matrix(bodies[palm_index])) @ matrix(obj)})
                        records.append(rec)
                        if len(records) % 6 == 0:
                            recorder.capture(len(records), request.get('phase', ''))
                        self_contacts.record(env.physics_dt, len(records)-1, request.get('phase',''))
                        if len(records)==1:
                            from round7_surface_probe import save_probes
                            save_probes(env,meshes,out,'first_existing_physics_step',len(records))
                        stream.write(json.dumps(rec, allow_nan=False) + '\n')
                        if not check['passed'] and not (out / 'first_failure.json').exists():
                            save(out / 'first_failure.json', rec)
                elif op != 'observe':
                    raise ValueError('unsupported operation')
                respond({'ok': True, 'observation': sensor_observation()})
            except Exception as exc:
                traceback.print_exc()
                respond({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        stream.close()
        frame_stream.close()
        gravity_stream.close()
        if not (out / 'summary.json').exists():
            save(out / 'summary.json', evaluate(records, initial_object_z))
    except BaseException as exc:
        traceback.print_exc()
        save(out / 'backend_error.json', {'error': f'{type(exc).__name__}: {exc}'})
        respond({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
    finally:
        if recorder is not None: recorder.close()
        if writer is not None: writer.close()
        if finger_writer is not None: finger_writer.close()
        if env is not None: env.close()
        if backend is not None: backend.simulation_app.close()


if __name__ == '__main__':
    main()
