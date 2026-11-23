"""120 Hz original physics/scorer, independent 30 fps RGB acquisition.

Source substitutions are asserted and their executed source is saved. Private
per-step geometry/contact verdicts remain unchanged. Encoding runs in a worker.
"""
import hashlib,json,os,sys,time
from pathlib import Path
from stage7_diagnostic_backend import measured,TIMES
HERE=Path(__file__).resolve().parent

def main():
    from stage4_registry import install
    install()
    sys.path.insert(0,str(HERE.parent/'rax_tinyobj_round1'))
    from stage2_contact_audit import install as contacts
    out=Path(sys.argv[sys.argv.index('--output')+1]);contacts(out)
    source=(HERE/'round14_backend.py').read_text()
    def change(old,new):
        nonlocal source
        assert old in source,old
        source=source.replace(old,new)
    change('import traceback','import traceback\nimport time')
    change("    def respond(value):", "    _request_start = None\n    def timed_requests(stream):\n        nonlocal _request_start\n        while True:\n            before=time.perf_counter();line=stream.readline();_request_start=time.perf_counter()\n            _measure_idle(_request_start-before)\n            if not line:return\n            yield line\n    def respond(value):")
    change("        protocol.flush()", "        protocol.flush()\n        if _request_start is not None:_active_requests.append(dict(start_wall_s=_request_start,end_wall_s=time.perf_counter()))")
    change("for line in sys.stdin:","for line in timed_requests(sys.stdin):")
    change("requested_fps=20", "requested_fps=30")
    change('cfg.sim.render_interval = 6','cfg.sim.render_interval = 4')
    change("writer = imageio.get_writer(out / 'rollout.mp4', fps=5, macro_block_size=16)","writer = None")
    change("finger_writer = imageio.get_writer(out / 'fingereye.mp4', fps=20, macro_block_size=16)","finger_writer = None")
    a=source.index('        def sensor_observation():');b=source.index('        from record_success_views import Recorder',a)
    source=source[:a]+'''        capture_state = None
        def capture_sensor(phase='initial'):
            nonlocal observation_index, capture_state, overhead_initial
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
                from stage13_inputs import paired_initial_rgb
                rgb=paired_initial_rgb(case,rgb,overhead_initial,public)
                path=public/'overhead_initial.npy';_measure('rgb_disk',np.save,path,rgb)
                overhead_initial['rgb_path']=str(path)
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
            _measure('frame_log_write',frame_stream.write,json.dumps(clean(dict(frame_id=observation_index,**capture_state,sha256=hashes)))+'\\n')
            if FULL:
                _measure('camera_diagnostic',audit_cameras,env,cfg,len(records),out)
                _measure('extra_visualization',recorder.capture,len(records),phase,frames)
            observation_index+=1
        def sensor_observation():
            if capture_state is None:capture_sensor()
            return dict(time_s=len(records)*env.physics_dt,step=len(records),
                actual_q=backend._env_to_canonical_q(env,env.hand.data.joint_pos[0].cpu().numpy()),
                actual_qvel=backend._env_to_canonical_q(env,env.hand.data.joint_vel[0].cpu().numpy()),
                initial_overhead=overhead_initial,fingereye=capture_state['fingereye'],
                image_state={k:v for k,v in capture_state.items() if k!='fingereye'})
'''+source[b:]
    change('from record_success_views import Recorder','from stage9_recording import Recorder')
    change("recorder = Recorder(env, cfg, out, case)","recorder = Recorder(env, cfg, out, case, enabled=FULL)")
    change("recorder.capture(0, 'initial')","capture_sensor()")
    change('if len(records) % 6 == 0:\n                            recorder.capture(len(records), request.get(\'phase\', \'\'))',"if len(records) % 4 == 0:\n                            capture_sensor(request.get('phase',''))")
    change("self_contacts.record(env.physics_dt, len(records)-1, request.get('phase',''))", "_measure('contact_diagnostic',self_contacts.record,env.physics_dt,len(records)-1,request.get('phase','')) if FULL else None")
    change("if len(records)==1:","if FULL and len(records)==1:")
    change('env.sim.step(render=False)',"_measure('physics',env.sim.step,render=False)")
    change('env.scene.update(dt=env.physics_dt)',"_measure('physics',env.scene.update,dt=env.physics_dt)")
    change('env.scene.write_data_to_sim()',"_measure('physics',env.scene.write_data_to_sim)")
    change('check = audit.check(dict(zip(env.hand.body_names, bodies)), obj)',"check = _measure('strict_geometry',audit.check,dict(zip(env.hand.body_names,bodies)),obj)")
    change('contacts.read(env.physics_dt)',"_measure('strict_contacts',contacts.read,env.physics_dt)")
    change("stream = (out / 'steps.jsonl').open('w', buffering=1)","stream = (out / 'steps.jsonl').open('w',buffering=1048576)")
    change("stream.write(json.dumps(rec, allow_nan=False) + '\\n')", "_measure('strict_log_write',stream.write,json.dumps(rec,allow_nan=False)+'\\n')")
    change("gravity_stream.write(json.dumps({'step':len(records),'max_error_n':err})+'\\n')", "gravity_stream.write(json.dumps({'step':len(records),'max_error_n':err})+'\\n') if FULL else None")
    change("respond({'ok': True, 'observation': sensor_observation(), 'robot_description': robot_description})", "_marks['ready']=time.perf_counter(); respond({'ok': True, 'observation': sensor_observation(), 'robot_description': robot_description})")
    change("summary = evaluate(records, initial_object_z)","_marks['finish_request']=time.perf_counter(); summary = evaluate(records, initial_object_z)")
    change('recorder.close()\n                    recorder = None',"_measure('video_finalize',recorder.close)\n                    recorder = None\n                    _dump()")
    out.mkdir(parents=True,exist_ok=True)
    (out/'stage9_backend_executed.py').write_text(source)
    marks={'start':time.perf_counter()};active=[];idle=[0.]
    def measure_idle(seconds):idle[0]+=seconds
    def dump():
        (out/'wall_profile_backend.json').write_text(json.dumps(dict(marks=marks,total_s=time.perf_counter()-marks['start'],exclusive_categories=dict(TIMES),base_sha256=hashlib.sha256((HERE/'round14_backend.py').read_bytes()).hexdigest(),executed_sha256=hashlib.sha256(source.encode()).hexdigest(),full=FULL,active_request_intervals=active,backend_stdin_idle_s=idle[0]),indent=2))
    FULL=os.environ.get('S9_MODE','minimal')=='full'
    ns=dict(__name__='stage9_executed',__file__=str(HERE/'round14_backend.py'),_measure=measured,_marks=marks,_dump=dump,FULL=FULL,_measure_idle=measure_idle,_active_requests=active)
    exec(compile(source,str(out/'stage9_backend_executed.py'),'exec'),ns)
    try:ns['main']()
    finally:dump()
if __name__=='__main__':main()
