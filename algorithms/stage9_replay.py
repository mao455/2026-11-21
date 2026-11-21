"""Replay ONLY public recorded inputs, zero physics; no success claim."""
import argparse,hashlib,importlib.util,io,json,sys,time,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--controller',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();e=a.episode
initial=json.loads((e/'public/initial.json').read_text());responses=[json.loads(x)['observation'] for x in (e/'public_responses.jsonl').read_text().splitlines()]
commands=[json.loads(x) for x in (e/'agent_trace.jsonl').read_text().splitlines() if json.loads(x).get('op')=='step']
controller=a.controller or e/'program/controller.py'
# Lossless archived RGB is still the recorded policy input. Restore individual
# arrays in memory so computation replay does not require duplicate disk data.
original_load=np.load
archive_path=e/'sensor_frames.zip'
archive=zipfile.ZipFile(archive_path) if archive_path.exists() else None
archive_index=json.loads((e/'sensor_archive.json').read_text())['sha256'] if archive else {}
def recorded_load(source,*args,**kwargs):
    if isinstance(source,(str,Path)) and not Path(source).exists() and archive is not None:
        name=Path(source).name
        if name in archive_index:
            data=archive.read(name)
            assert hashlib.sha256(data).hexdigest()==archive_index[name], 'recorded RGB SHA mismatch'
            return original_load(io.BytesIO(data),*args,**kwargs)
    return original_load(source,*args,**kwargs)
np.load=recorded_load
spec=importlib.util.spec_from_file_location('replay_controller',controller);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
# Nested CPU microprofiles: diagnostics of recorded-input computation only.
# Mask kernels exclude NumPy thresholding and are explicitly a subset of vision.
micro={}
def profile_call(label,fn):
    def call(*args,**kwargs):
        begin=time.perf_counter()
        try:return fn(*args,**kwargs)
        finally:micro.setdefault(label,[]).append(time.perf_counter()-begin)
    return call
for label,name in [('rgb_transfer_decode','_s2v_read_rgb'),('encoder_settling_predicate','stable_joint_response'),('interpolation_generation','smooth_segment')]:
    if hasattr(m,name):setattr(m,name,profile_call(label,getattr(m,name)))
for name in ('cvtColor','connectedComponentsWithStats'):
    setattr(m.cv2,name,profile_call('mask_kernel_'+name,getattr(m.cv2,name)))
m.Robot.fk=profile_call('robot_fk_nested',m.Robot.fk)
class Replay:
    def __init__(self):self.robot_description=initial['robot_description'];self.i=0;self.j=0;self.times=[];self.notes=[];self.last=time.perf_counter()
    def observe(self):
        o=responses[self.i];self.i+=1;return o
    def step(self,q,steps=1,phase=''):
        now=time.perf_counter();self.times.append(now-self.last)
        c=commands[self.j];self.j+=1
        assert c['steps']==steps and c['phase']==phase and np.max(np.abs(np.asarray(q)-c['target_q']))<1e-9
        o=responses[self.i];self.i+=1;self.last=time.perf_counter();return o
    def note(self,text):self.notes.append(json.loads(text))
r=Replay();start=time.perf_counter();m.main(r);elapsed=time.perf_counter()-start
assert r.j==len(commands) and r.i==len(responses)
t=np.array(r.times)*1000
result=dict(scope='public input computation replay only; strict outcomes exclusively from physics',commands_equal=True,commands=len(commands),total_wall_s=elapsed,mean_ms=float(t.mean()),p50_ms=float(np.percentile(t,50)),p95_ms=float(np.percentile(t,95)),p99_ms=float(np.percentile(t,99)),max_ms=float(t.max()),over50_ratio=float((t>50).mean()),cycle_compute_ms=t.tolist(),component_profiles=[x for x in r.notes if 's6_wall_profile' in x])
result['nested_microprofiles_ms']={k:dict(n=len(v),p50=float(np.percentile(v,50)*1000),p95=float(np.percentile(v,95)*1000),p99=float(np.percentile(v,99)*1000),max=float(max(v)*1000),total_s=float(sum(v))) for k,v in micro.items()}
result['microprofile_scope']='nested subsets; mask kernels exclude NumPy thresholding; recorded RGB decode includes archive read, SHA validation and NumPy decode; no camera exposure/USB/physics wait'
(a.output or e/'replay_timing.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('cycle_compute_ms','component_profiles')}))
