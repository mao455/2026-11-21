"""Passive exclusive wall instrumentation of the original Stage4 backend.

Original physics/evaluation code is compiled unmodified except wrapping named
calls with a timer. No additional sim advance/render, thresholds or inputs.
The source hash and exact substitutions are saved with every episode.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
TIMES = defaultdict(lambda: dict(exclusive_s=0., calls=0))
STACK = []

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

def main():
    output = Path(sys.argv[sys.argv.index('--output')+1])
    from stage4_registry import install
    install()
    sys.path.insert(0, str(HERE.parent/'rax_tinyobj_round1'))
    from stage2_contact_audit import install as install_contacts
    install_contacts(output)
    import imageio.v2 as imageio
    original_writer = imageio.get_writer
    class Writer:
        def __init__(self, writer): self.writer = writer
        def append_data(self, *a, **kw): return measured('disk_video', self.writer.append_data, *a, **kw)
        def close(self): return measured('disk_video', self.writer.close)
        def __getattr__(self, name): return getattr(self.writer, name)
    imageio.get_writer = lambda *a, **kw: Writer(original_writer(*a, **kw))
    # Both public acquisition and private recording import this same function.
    import round7c_dynamics
    original_capture = round7c_dynamics.capture_cameras
    round7c_dynamics.capture_cameras = lambda *a, **kw: measured('render_and_image', original_capture, *a, **kw)
    source_path = HERE/'round14_backend.py'
    source = source_path.read_text()
    from stage7_private_probe import sample as probe_sample, finish as probe_finish
    (output/'stage7_private_probe_snapshot.py').write_bytes((HERE/'stage7_private_probe.py').read_bytes())
    source = source.replace("self_contacts.record(env.physics_dt, len(records)-1, request.get('phase',''))", "_s7_sample(env,rec); self_contacts.record(env.physics_dt, len(records)-1, request.get('phase',''))")
    source = source.replace("respond({'ok': True, 'summary': summary})", "_s7_finish(env,out); respond({'ok': True, 'summary': summary})")
    substitutions = {
        'np.save(': "_s6_call('disk_rgb', np.save, ",
        'audit.check(': "_s6_call('private_geometry', audit.check, ",
        'audit_cameras(': "_s6_call('camera_diagnostic', audit_cameras, ",
        'stream.write(': "_s6_call('disk_json', stream.write, ",
        'frame_stream.write(': "_s6_call('disk_json', frame_stream.write, ",
        'gravity_stream.write(': "_s6_call('disk_json', gravity_stream.write, ",
        'env.sim.step(': "_s6_call('physics', env.sim.step, ",
        'env.scene.update(': "_s6_call('physics', env.scene.update, ",
        'env.scene.write_data_to_sim()': "_s6_call('physics', env.scene.write_data_to_sim)",
        'self_contacts.record(': "_s6_call('contact_diagnostic', self_contacts.record, ",
        'recorder.capture(': "_s6_call('recording_other', recorder.capture, ",
        'sensor_observation()': "_s6_call('sensor_other', sensor_observation)",
    }
    # Longer tokens first avoids stream.write matching frame_stream.write.
    import re
    pattern = re.compile('|'.join(re.escape(k) for k in sorted(substitutions, key=len, reverse=True)))
    counts = {k: len(re.findall(re.escape(k), source)) for k in substitutions}
    source = pattern.sub(lambda m: substitutions[m.group()], source)
    source = source.replace("respond({'ok': True, 'summary': summary})", "_s6_dump(); respond({'ok': True, 'summary': summary})")
    source = source.replace("def _s6_call('sensor_other', sensor_observation):", 'def sensor_observation():')
    def dump():
        (output/'wall_profile_backend.json').write_text(json.dumps(dict(total_s=time.perf_counter()-start, exclusive_categories=TIMES, original_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), substitutions=counts), indent=2)+'\n')
    namespace = dict(_s7_sample=probe_sample, _s7_finish=probe_finish, _s6_dump=dump, __name__='stage6_instrumented_backend', __file__=str(source_path), _s6_call=measured)
    exec(compile(source, str(source_path), 'exec'), namespace)
    start = time.perf_counter()
    try: namespace['main']()
    finally:
        (output/'wall_profile_backend.json').write_text(json.dumps(dict(
            total_s=time.perf_counter()-start, exclusive_categories=TIMES,
            original_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            substitutions=counts), indent=2)+'\n')

if __name__ == '__main__': main()
