"""Compare archived commands plus semantic gates/perception under identical public input.

No physics or success-rate claim. Timing-only fields are omitted from comparison.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
TIMING = {'compute_s', 'elapsed_s', 'total_wall_s', 'component_samples_s', 'exclusive_s',
          'wall_s', 'wall_time_s', 'started_wall_s', 'host_s'}

def canonical(value):
    if isinstance(value, np.ndarray): return canonical(value.tolist())
    if isinstance(value, np.generic): return canonical(value.item())
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()
                if k not in TIMING and not k.endswith('_wall_s')}
    if isinstance(value, (list, tuple)): return [canonical(v) for v in value]
    return value

def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=True).encode()).hexdigest()

def run(ep, controller, output):
    ep = ep.resolve()
    initial = json.loads((ep/'public/initial.json').read_text())
    responses = [json.loads(x)['observation'] for x in (ep/'public_responses.jsonl').read_text().splitlines()]
    for obs in responses:
        for slot in [*obs['fingereye'].values(), obs['initial_overhead']]:
            slot['rgb_path'] = str(ep/'public'/Path(slot['rgb_path']).name)
    commands = [x for x in map(json.loads, (ep/'agent_trace.jsonl').read_text().splitlines()) if x['op']=='step']
    spec = importlib.util.spec_from_file_location('replayed', controller)
    m = importlib.util.module_from_spec(spec);sys.modules[spec.name] = m;spec.loader.exec_module(m)
    events = []
    def wrap(name, fn):
        def call(*args, **kwargs):
            value = fn(*args, **kwargs)
            # Record complete returned verdict, numerical witnesses and explicit query.
            query = None
            if name.endswith('.check') or name.endswith('.diagnose'):
                query = dict(q=np.asarray(args[1]).tolist(), object_check=kwargs.get('object_check', args[2] if len(args)>2 else True))
            events.append(dict(function=name, query=query, sha256=digest(value)))
            return value
        return call
    for name in ('color_foreground', 'color_reliability', '_S3_ESTIMATE', 'stage3_live_geometry',
                 '_s12_endpoint_gate', '_s12_path_gate', 's13_preflight', 's5_decide', 's13_preload_proposal'):
        setattr(m, name, wrap(name, getattr(m,name)))
    for name in ('check','diagnose'):
        setattr(m.SelfAwareGeometry,name,wrap('SelfAwareGeometry.'+name,getattr(m.SelfAwareGeometry,name)))
    class Replay:
        robot_description = initial['robot_description']
        def __init__(self): self.i=0;self.j=0;self.max_error=0.;self.notes=[]
        def observe(self):
            obs=responses[self.i];self.i+=1;return obs
        def step(self,q,steps=1,phase=''):
            c=commands[self.j];error=float(np.max(abs(np.asarray(q)-c['target_q'])))
            self.max_error=max(error,self.max_error)
            if c['steps']!=steps or c['phase']!=phase or error>1e-9:
                raise RuntimeError(str((self.j,phase,c['phase'],error)))
            self.j+=1;return self.observe()
        def note(self,text):
            value=json.loads(text)
            if 's9_io_audit' not in value:self.notes.append(digest(value))
    replay=Replay();start=time.perf_counter();error=None
    try:m.main(replay)
    except Exception as exc:
        import traceback
        traceback.print_exc();error=repr(exc)
    result=dict(controller=str(controller.resolve()),controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
                episode=str(ep),commands_equal=error is None and replay.j==len(commands) and replay.i==len(responses),
                error=error,commands=replay.j,expected_commands=len(commands),observations=replay.i,
                max_command_error=replay.max_error,semantic_events=events,notes=replay.notes,
                elapsed_s=time.perf_counter()-start)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('semantic_events','notes')},flush=True)
    if error:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True);p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.episode,a.controller,a.output)
