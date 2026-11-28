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
import time

HERE = Path(__file__).resolve().parent
LAB = HERE


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
    _request_start = None
    def timed_requests(stream):
        nonlocal _request_start
        while True:
            before=time.perf_counter();line=stream.readline();_request_start=time.perf_counter()
            _measure_idle(_request_start-before)
            if not line:return
            yield line
    def respond(value):
        protocol.write(json.dumps(clean(value), allow_nan=False) + '\n')
        protocol.flush()
        if _request_start is not None:_active_requests.append(dict(start_wall_s=_request_start,end_wall_s=time.perf_counter()))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    public = out / 'public'
    public.mkdir(exist_ok=True)
    from host import validate_case
    case = validate_case(json.loads(args.case_json))
    import numpy as np
    parameters = {'mass_kg': .005, 'friction': .8, 'stiffness': 20.,
                  'depth_noise_m': 0., 'depth_bias_m': 0., 'depth_dropout': 0., 'noise_seed': 0}
    parameters.update({key: case[key] for key in parameters if key in case})
    parameters['rgb_noise_std'] = case.get('rgb_noise_std', 0.)
    sensor_rng = np.random.default_rng(parameters['noise_seed'])
    from scipy.spatial.transform import Rotation
    from hand_runtime import selected
    HAND_ONLY_JOINT_NAMES = selected()['joint_names']
    from evaluator import evaluate
    q_start = np.zeros(len(HAND_ONLY_JOINT_NAMES), dtype=np.float32)
    q_start[:3] = [-.18, 0., .30]
    if os.environ.get('FINGEREYE_HAND_DESCRIPTION'):
        from hand_runtime import selected
        q_start = np.asarray(selected()['initial_q'], dtype=np.float32)
    quaternion = Rotation.from_euler('z', case['yaw']).as_quat()[[3, 0, 1, 2]]
    object_pose = np.r_[case['dx'], case['dy'], .0001, quaternion].astype(np.float32)
    mesh_path = HERE / 'examples/box.obj'
    if 'object_id' in case:
        from objects import create
        mesh_path=create(case['object_id'])
    payload = {'object_mesh_path': str(mesh_path), 'object_pose_env': object_pose,
               'object_scale': np.ones(3), 'q_start': q_start}
    sys.argv = [str(HERE / 'simulator.py'), '--motion_pkl', str(out / 'scene.pkl'),
                '--device', 'cpu', '--headless', '--save_video', '--video_width', '320',
                '--video_height', '240', '--camera-fps', '5',
                '--camera_eye', '0,-0.001,0.5', '--camera_target', '0,0,0']
    backend = env = writer = finger_writer = recorder = None
    records = []
    initial_object_z = float(object_pose[2])
    try:
        import simulator as backend
        import torch
        import trimesh
        import imageio.v2 as imageio
        from geometry import matrix, usd_collision_meshes
        from convex_geometry import Audit
        from diagnostics import Contacts
        from gravity import set_world_gravity_force, world_to_body
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
        cfg.configure_xhand_cameras('index', requested_fps=30, include_third_view=True, width=160, height=120)
        finger_names = list(cfg.xhand_camera_configs) if os.environ.get('FINGEREYE_HAND_DESCRIPTION') else [f'{f}_{m}' for f in ('index','thumb') for m in ('front1','front2','back')]
        cfg.camera_name_list = ['third_view', *finger_names]
        cfg.cam_third_view.data_types = ['rgb']
        cfg.cam_third_view.update_period = 0.
        for name in finger_names:
            cfg.xhand_camera_configs[name].update_period = 0.
        cfg.sim.physx.enable_stabilization = False
        cfg.cam_third_view.spawn.clipping_range = (.01, 3.)
        cfg.sim.dt = 1 / 120
        cfg.sim.render_interval = 4
        cfg.sim.enable_scene_query_support = True
        if os.environ.get('FINGEREYE_HAND_DESCRIPTION'):
            from hand_runtime import configure_physics as configure, audit_runtime, audit_cameras, capture_cameras
        configure(cfg)
        env = backend.gym.make(backend.args_cli.task, cfg=cfg).unwrapped
        from lighting import configure_lights
        configure_lights(env, out)
        parameters['light_multiplier'] = float(os.environ['FINGEREYE_LIGHT_SCALE'])
        from pxr import UsdGeom
        for prim in env.sim.stage.Traverse():
            path = str(prim.GetPath())
            if '/Robot/' in path and '/visuals' in path and 'fingereye_camera_' in path:
                UsdGeom.Imageable(prim).MakeInvisible()
        backend._remove_background_only()
        backend._ensure_ground_table()
        backend.args_cli.base_pose_overlay = False
        env.reset()
        ids = torch.tensor([0], device=env.device)
        backend._set_q_state_and_target_batch(env, q_start[None], ids, snap_state=True)
        backend._write_object_pose_batch(env, object_pose[None], ids)
        _measure('physics',env.scene.write_data_to_sim)
        env.sim.forward()
        env.scene.update(dt=0.)
        meshes, manifest = usd_collision_meshes(env.sim.stage, '/World/envs/env_0/Robot', env.hand.body_names)
        object_mesh = trimesh.load(mesh_path, force='mesh', process=True)
        audit = Audit(meshes, object_mesh)
        contacts = Contacts(env)
        from self_contacts import NativeSelfContacts
        self_contacts = NativeSelfContacts(env, out)
        audit_runtime(env, cfg, out)
        limits = env.hand.data.soft_joint_pos_limits[0].cpu().numpy()
        canonical_limits = np.stack([backend._env_to_canonical_q(env, limits[:, j]) for j in [0, 1]], axis=1)
        robot_description = {'urdf_path': backend.args_cli.robot_urdf,
                             'joint_names': list(HAND_ONLY_JOINT_NAMES), 'joint_limits': canonical_limits,
                             'dt': env.physics_dt, 'table_z': 0.,
                             'fingereye_calibration': {n: cfg.xhand_camera_metadata[n] for n in finger_names},
                             'observation_profile': 'initial_overhead_rgb_live_fingereye_rgb_joint_state_v1',
                             'action': 'declared joint-name order position targets; meters/radians'}
        if os.environ.get('FINGEREYE_HAND_DESCRIPTION'):
            hand = __import__('hand_runtime').selected()
            robot_description.update({key: hand[key] for key in ('source_urdf_path','base_joint_names','palm_link','root_link','finger_chains','source_to_runtime_joint_names','pads')})
            robot_description['contact_mode'] = hand.get('contact_mode','auto')
            robot_description['action'] = 'declared joint-name order position targets; six palm coordinates plus hand joints; meters/radians'
        save(out / 'scene.json', {'robot_description': robot_description, 'mesh': str(mesh_path),
             'mesh_sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
             'initial_object_pose': object_pose, 'collision_geometry': manifest,
             'case': case, 'realized_parameters': parameters, 'object_mass_kg': parameters['mass_kg'],
             'friction': parameters['friction'], 'object_gravity': [0, 0, -9.81],
             'finger_stiffness': parameters['stiffness'], 'finger_damping': 2.,
             'physics': backend._physics_configuration_snapshot(env, 1, independent_ground=False)})
        palm_index = env.hand.body_names.index(__import__('hand_runtime').selected()['palm_link'] if os.environ.get('FINGEREYE_HAND_DESCRIPTION') else 'left_hand_palm_link')
        writer = None
        stream = (out / 'steps.jsonl').open('w',buffering=1048576)
        _s16_finalized = False
        def _s16_finalize():
            nonlocal _s16_finalized
            if _s16_finalized:return
            _marks['private_finalize_start']=time.perf_counter()
            if _S16_MODE == 'deferred':
                details=_s16_finalize_records(records,audit,env.hand.body_names,stream,out,_measure)
            else:
                stream.flush();details=dict(states=len(records),elapsed_s=0.,private_evaluator_only=True)
            _marks['private_finalize_done']=time.perf_counter()
            _s16_finalized=True
            save(out/'stage16_evaluation_placement.json',dict(mode=_S16_MODE,**details))
        observation_index = 0
        sensor_cache = None
        overhead_initial = None
        finger_writer = None
        frame_stream = (out/'image_manifest.jsonl').open('w',buffering=1)
        gravity_stream = (out/'gravity_force_audit.jsonl').open('w',buffering=1)
        save(out/'physics_version.json', {'version':'R7B-D','physical_dynamics':'R7B-allfine-C2-v1',
             'gravity_frame':'world','enable_stabilization':False,
             'new_sensor_profile':True,'evaluator_thresholds_unchanged':True})
        capture_state = None
        initial_rgb_geometry = None
        def capture_sensor(phase='initial'):
            nonlocal observation_index, capture_state, overhead_initial, initial_rgb_geometry
            if capture_state is not None and capture_state['step']==len(records):return
            start=time.perf_counter()
            names=['third_view',*finger_names] if observation_index==0 or FULL else finger_names
            frames=_measure('policy_rgb_render_readback',capture_cameras,env,cfg,names)
            done=time.perf_counter()
            q=backend._env_to_canonical_q(env,env.hand.data.joint_pos[0].cpu().numpy())
            qv=backend._env_to_canonical_q(env,env.hand.data.joint_vel[0].cpu().numpy())
            if observation_index==0:
                top=env._active_cameras['third_view'];rgb=frames['third_view'][0]
                overhead_initial={'intrinsics':top.data.intrinsic_matrices[0].cpu().numpy(),
                    'camera_pose_world':np.r_[top.data.pos_w[0].cpu().numpy(),top.data.quat_w_ros[0].cpu().numpy()],
                    'width':rgb.shape[1],'height':rgb.shape[0],'capture_step':0}
                from initial_image import paired_initial_rgb
                rgb=paired_initial_rgb(case,rgb,overhead_initial,public)
                path=public/'overhead_initial.npy';_measure('rgb_disk',np.save,path,rgb)
                overhead_initial['rgb_path']=str(path)
                from initial_rgb_geometry import publish_initial_rgb_geometry
                initial_rgb_geometry=publish_initial_rgb_geometry(case,public,overhead_initial)
            observed={};hashes={}
            for name in finger_names:
                rgb=frames[name][0][...,:3].copy()
                if parameters['rgb_noise_std']:
                    rgb=np.clip(rgb.astype(float)+sensor_rng.normal(0,parameters['rgb_noise_std'],rgb.shape),0,255).astype(np.uint8)
                frames[name]=rgb[None]  # recorder uses the exact post-noise policy pixels
                path=public/f'{name}_{observation_index:05d}.npy';_measure('rgb_disk',np.save,path,rgb)
                hashes[name]=hashlib.sha256(rgb.tobytes()).hexdigest()
                observed[name]={'rgb_path':str(path),'width':rgb.shape[1],'height':rgb.shape[0],
                    'capture_step':len(records),'frame_id':observation_index}
            capture_state=dict(step=len(records),time_s=len(records)*env.physics_dt,
                actual_q=q,actual_qvel=qv,fingereye=observed,capture_start_wall_s=start,
                capture_done_wall_s=done,available_wall_s=time.perf_counter())
            _measure('frame_log_write',frame_stream.write,json.dumps(clean(dict(frame_id=observation_index,**capture_state,sha256=hashes)))+'\n')
            if FULL or observation_index==0 or os.environ.get('FINGEREYE_HAND_DESCRIPTION'):
                _measure('camera_diagnostic',audit_cameras,env,cfg,len(records),out)
            if FULL:
                _measure('extra_visualization',recorder.capture,len(records),phase,frames)
            observation_index+=1
        def sensor_observation():
            if capture_state is None:capture_sensor()
            return dict(time_s=len(records)*env.physics_dt,step=len(records),
                actual_q=backend._env_to_canonical_q(env,env.hand.data.joint_pos[0].cpu().numpy()),
                actual_qvel=backend._env_to_canonical_q(env,env.hand.data.joint_vel[0].cpu().numpy()),
                initial_overhead=overhead_initial,fingereye=capture_state['fingereye'],
                image_state={k:v for k,v in capture_state.items() if k!='fingereye'},
                **({'initial_rgb_geometry':initial_rgb_geometry} if initial_rgb_geometry is not None else {}))
        from recording import Recorder
        recorder = Recorder(env, cfg, out, case, enabled=FULL)
        capture_sensor()
        _marks['ready']=time.perf_counter(); respond({'ok': True, 'observation': sensor_observation(), 'robot_description': robot_description})
        for line in timed_requests(sys.stdin):
            try:
                request = json.loads(line)
                op = request.get('op')
                if op == 'finish':
                    _marks['finish_request']=time.perf_counter(); _s16_finalize(); summary = _measure('final_evaluator',evaluate,records,initial_object_z); _marks['evaluation_done']=time.perf_counter()
                    save(out / 'summary.json', summary)
                    _measure('video_finalize',recorder.close)
                    recorder = None
                    _dump()
                    respond({'ok': True, 'summary': summary})
                    break
                if op == 'step':
                    target = np.asarray(request['target_q'], dtype=np.float32)
                    count = request['steps']
                    if target.shape != (len(HAND_ONLY_JOINT_NAMES),) or not np.isfinite(target).all() or not isinstance(count, int) or not 1 <= count <= 24:
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
                        gravity_stream.write(json.dumps({'step':len(records),'max_error_n':err})+'\n') if FULL else None
                        if err>1e-6:raise RuntimeError('World gravity audit failed')
                        _measure('physics',env.scene.write_data_to_sim)
                        _measure('physics',env.sim.step,render=False)
                        _measure('physics',env.scene.update,dt=env.physics_dt)
                        obj = backend._object_pose_batch(env, ids)[0]
                        bodies = np.c_[env.hand.data.body_pos_w[0].cpu().numpy(), env.hand.data.body_quat_w[0].cpu().numpy()]
                        check = None  # private evaluation reconstructed after the last command
                        rec = clean({'step': len(records), 'time_s': (len(records) + 1) * env.physics_dt,
                           'phase': request.get('phase', ''), 'target_q': target,
                           'actual_q': backend._env_to_canonical_q(env, env.hand.data.joint_pos[0].cpu().numpy()),
                           'actual_qvel': backend._env_to_canonical_q(env, env.hand.data.joint_vel[0].cpu().numpy()),
                           'object_pose': obj, 'object_velocity': backend._object_velocity_batch(env, ids)[0],
                           'body_poses': bodies, 'penetration': check, 'contacts': _measure('strict_contacts',contacts.read,env.physics_dt),
                           'hand_object_relative_matrix': np.linalg.inv(matrix(bodies[palm_index])) @ matrix(obj)})
                        records.append(rec)
                        if len(records) % 4 == 0:
                            capture_sensor(request.get('phase',''))
                        _measure('contact_diagnostic',self_contacts.record,env.physics_dt,len(records)-1,request.get('phase','')) if FULL or len(records)<=12 else None
                        pass  # private JSON serialization deferred; actual state remains in records
                        if check is not None and not check['passed'] and not (out / 'first_failure.json').exists():
                            save(out / 'first_failure.json', rec)
                elif op != 'observe':
                    raise ValueError('unsupported operation')
                respond({'ok': True, 'observation': sensor_observation()})
            except Exception as exc:
                traceback.print_exc()
                respond({'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
        _s16_finalize()
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



from collections import defaultdict
TIMES = defaultdict(lambda: dict(exclusive_s=0., calls=0))
STACK=[]
def measured(category, fn, *args, **kwargs):
    frame = [time.perf_counter(), 0.]
    STACK.append(frame)
    try: return fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter()-frame[0]
        STACK.pop()
        TIMES[category]['exclusive_s'] += elapsed-frame[1]
        TIMES[category]['calls'] += 1
        if STACK: STACK[-1][1] += elapsed


def finalize_records(records, audit, body_names, stream, out, measure):
    """Exactly reconstruct private geometry and serialize immutable snapshots."""
    import numpy as np
    start = time.perf_counter()
    # finish can be retried by the backend's EOF/close path. Keep a failed
    # attempt from leaving a prefix that the retry would append a second time.
    # Successfully computed private geometry may stay cached in records.
    write_start = stream.tell()
    try:
        for rec in records:
            if rec['penetration'] is None:
                rec['penetration'] = measure('offline_strict_geometry', audit.check,
                    dict(zip(body_names, np.asarray(rec['body_poses']))), np.asarray(rec['object_pose']))
            measure('offline_log_serialize_write', stream.write,
                json.dumps(rec, allow_nan=False) + '\n')
            if not rec['penetration']['passed'] and not (out / 'first_failure.json').exists():
                pending = out / 'first_failure.json.partial'
                pending.write_text(json.dumps(rec, allow_nan=False, indent=2) + '\n')
                pending.replace(out / 'first_failure.json')
        stream.flush()
    except BaseException:
        stream.seek(write_start)
        stream.truncate()
        stream.flush()
        raise
    return dict(states=len(records), elapsed_s=time.perf_counter()-start,
        final_physical_step=len(records), private_evaluator_only=True)


_S16_MODE='deferred'
_s16_finalize_records=finalize_records
_measure=measured
_marks={};_active_requests=[];_idle=[0.]
FULL=False

def _measure_idle(seconds):_idle[0]+=seconds

def _dump():
    output=Path(sys.argv[sys.argv.index('--output')+1])
    save(output/'backend_timing.json',dict(marks=_marks,categories=dict(TIMES),stdin_idle_s=_idle[0]))

if __name__=='__main__':
    from resources import simulator_slot
    output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    from contact_diagnostics import install
    install(output)
    from geometry_acceleration import install as accelerate
    accelerate()
    _marks['start']=time.perf_counter()
    with simulator_slot(output):main()
