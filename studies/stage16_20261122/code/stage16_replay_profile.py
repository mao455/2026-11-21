"""Profile only public recorded controller input; no physical or hardware success claim."""
import argparse
import functools
import hashlib
import importlib.util
import io
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--controller',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ep=a.episode;controller=a.controller or ep/'program/controller.py'
    initial=json.loads((ep/'public/initial.json').read_text())
    responses=[json.loads(line)['observation'] for line in (ep/'public_responses.jsonl').read_text().splitlines()]
    commands=[r for r in map(json.loads,(ep/'agent_trace.jsonl').read_text().splitlines()) if r.get('op')=='step']
    old_load=np.load;archive=zipfile.ZipFile(ep/'sensor_frames.zip') if (ep/'sensor_frames.zip').exists() else None
    index=json.loads((ep/'sensor_archive.json').read_text())['sha256'] if archive else {}
    def recorded_load(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and not Path(path).exists() and archive and Path(path).name in index:
            name=Path(path).name;data=archive.read(name)
            assert hashlib.sha256(data).hexdigest()==index[name]
            return old_load(io.BytesIO(data),*args,**kwargs)
        return old_load(path,*args,**kwargs)
    np.load=recorded_load
    spec=importlib.util.spec_from_file_location('stage16_replayed_controller',controller);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    rows=[];stack=[];solver=[];instrumented=set();current={'command':0,'physical_step':0}
    def timed(name,function):
        instrumented.add(name)
        @functools.wraps(function)
        def call(*args,**kwargs):
            begin=time.perf_counter();frame=[begin,0.];stack.append(frame)
            try:return function(*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-begin;stack.pop()
                rows.append(dict(function=name,start_s=begin,inclusive_s=elapsed,exclusive_s=elapsed-frame[1],command=current['command'],physical_step=current['physical_step']))
                if stack:stack[-1][1]+=elapsed
        return call
    functions=['_S3_ESTIMATE','initial_hue','s5_plan','s5_local_candidates','s11_post_release_plan',
        '_s3_candidates_on_axes','_s14_surface_candidates','_s12_three_candidates','s11_contact_candidates',
        '_s12_endpoint_gate','_s12_path_gate','s12_task_quality','s12_order_candidates',
        'optimize','close_ik','solve_close_ik','fit_shape','stage3_live_geometry','_s2v_read_rgb',
        'capture_preload_snapshot','stage3_support_snapshot','stage3_support_pair','compare_preload',
        's13_preflight','s13_local_fit','s13_local_probe','s13_release_recover','s13_evaluate',
        'stable_joint_response','smooth_segment','_S6_BASE_CHECK']
    for name in functions:
        if hasattr(m,name):setattr(m,name,timed(name,getattr(m,name)))
    for cls,names in [('Robot',['fk','tips']),('SelfAwareGeometry',['check']),('Stage3LiveVolume',['carve','frames']),('RobotVisibility',['depth'])]:
        for name in names:
            if hasattr(getattr(m,cls,None),name):setattr(getattr(m,cls),name,timed(cls+'.'+name,getattr(getattr(m,cls),name)))
    for name in ['cvtColor','connectedComponentsWithStats','findContours']:
        if hasattr(m.cv2,name):setattr(m.cv2,name,timed('cv2.'+name,getattr(m.cv2,name)))
    # The controller imports FCL inside checker construction; instrument the
    # shared module rather than looking for a controller global that is absent.
    import fcl
    for name in ['collide','distance']:
        if hasattr(fcl,name):setattr(fcl,name,timed('fcl.'+name,getattr(fcl,name)))
    old_ls=m.least_squares
    def least_squares(fun,*args,**kwargs):
        residual_calls=[0]
        def residual(*args,**kwargs):
            residual_calls[0]+=1;return fun(*args,**kwargs)
        begin=time.perf_counter();fit=old_ls(timed('least_squares.residual',residual),*args,**kwargs)
        solver.append(dict(command=current['command'],physical_step=current['physical_step'],elapsed_s=time.perf_counter()-begin,
            residual_calls=residual_calls[0],nfev=int(fit.nfev),njev=None if fit.njev is None else int(fit.njev),
            max_nfev=kwargs.get('max_nfev'),status=int(fit.status),cost=float(fit.cost)))
        return fit
    m.least_squares=timed('least_squares',least_squares)
    import general_grasp_contacts as contacts
    if hasattr(contacts,'linprog'):contacts.linprog=timed('task_wrench.linprog',contacts.linprog)
    if hasattr(m,'linprog'):m.linprog=timed('perception.linprog',m.linprog)
    contacts.task_wrench_margin=timed('task_wrench_margin',contacts.task_wrench_margin)
    class Replay:
        def __init__(self):self.robot_description=initial['robot_description'];self.i=0;self.j=0;self.notes=[];self.cycles=[];self.last=time.perf_counter();self.max_error=0.
        def observe(self):
            obs=responses[self.i];self.i+=1;self.last=time.perf_counter();return obs
        def step(self,q,steps=1,phase=''):
            now=time.perf_counter();c=commands[self.j];error=float(np.max(np.abs(np.asarray(q)-c['target_q'])));self.max_error=max(self.max_error,error)
            assert c['steps']==steps and c['phase']==phase and error<1e-9,(self.j,phase,c['phase'],error)
            self.cycles.append(dict(command=self.j,phase=phase,compute_s=now-self.last,physical_step=current['physical_step']))
            self.j+=1;current['command']=self.j;current['physical_step']+=steps
            obs=responses[self.i];self.i+=1;self.last=time.perf_counter();return obs
        def note(self,text):self.notes.append(json.loads(text))
    replay=Replay();start=time.perf_counter();error=None
    try:m.main(replay)
    except Exception as exc:error=repr(exc)
    elapsed=time.perf_counter()-start
    functions={}
    for name in sorted({r['function'] for r in rows}):
        rr=[r for r in rows if r['function']==name]
        functions[name]=dict(calls=len(rr),inclusive_s=sum(r['inclusive_s'] for r in rr),exclusive_s=sum(r['exclusive_s'] for r in rr),median_inclusive_ms=statistics.median(r['inclusive_s'] for r in rr)*1000,
            before_first_command_s=sum(r['exclusive_s'] for r in rr if r['command']==0))
    payload=dict(episode=str(ep),controller=str(controller),controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
        scope='recorded simulation public inputs; CPU only; instrumented overhead included; no physics, camera capture, transport wait or actual hardware',
        commands_equal=error is None and replay.j==len(commands) and replay.i==len(responses),error=error,
        commands_replayed=replay.j,commands_expected=len(commands),max_command_error=replay.max_error,wall_s=elapsed,
        compute_over50ms=sum(c['compute_s']>.05 for c in replay.cycles),cycles=replay.cycles,
        functions=functions,instrumented_functions=sorted(instrumented),
        three_search_calls=sum(r['function']=='_s12_three_candidates' for r in rows),
        least_squares=solver,events=rows,notes=[n for n in replay.notes if 's12_search' in n],
        uninstrumented_s=elapsed-sum(r['exclusive_s'] for r in rows),
        interpretation='Functions inclusive times overlap; exclusive totals can be added. FK and FCL isolate portions of geometry; AABB and NumPy branches remain in check exclusive. Existing wall-clock search cutoffs may change under profiling; commands_equal validates this run.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps({k:payload[k] for k in ('commands_equal','error','wall_s','compute_over50ms','functions')},indent=2))
    if error:raise RuntimeError(error)

if __name__=='__main__':main()
