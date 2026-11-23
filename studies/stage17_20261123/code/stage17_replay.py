"""Exact same-input controller replay; no dynamics or success-rate claim."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ep=a.episode.resolve()
    initial=json.loads((ep/'public/initial.json').read_text())
    responses=[json.loads(x)['observation'] for x in (ep/'public_responses.jsonl').read_text().splitlines()]
    for obs in responses:
        for slot in [*obs['fingereye'].values(),obs['initial_overhead']]:
            slot['rgb_path']=str(ep/'public'/Path(slot['rgb_path']).name)
    commands=[x for x in map(json.loads,(ep/'agent_trace.jsonl').read_text().splitlines()) if x['op']=='step']
    spec=importlib.util.spec_from_file_location('replayed',a.controller);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    class Replay:
        robot_description=initial['robot_description']
        def __init__(self):self.i=0;self.j=0;self.error=0.;self.notes=[]
        def observe(self):
            obs=responses[self.i];self.i+=1;return obs
        def step(self,q,steps=1,phase=''):
            c=commands[self.j];e=float(np.max(abs(np.asarray(q)-c['target_q'])));self.error=max(e,self.error)
            if c['steps']!=steps or c['phase']!=phase or e>1e-9:raise RuntimeError(str((self.j,phase,c['phase'],e)))
            self.j+=1;return self.observe()
        def note(self,text):self.notes.append(json.loads(text))
    replay=Replay();start=time.perf_counter();error=None
    try:m.main(replay)
    except Exception as exc:error=repr(exc)
    result=dict(commands_equal=error is None and replay.j==len(commands) and replay.i==len(responses),error=error,commands=replay.j,expected=len(commands),max_error=replay.error,elapsed_s=time.perf_counter()-start)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(result)
    if not result['commands_equal']:raise SystemExit(1)
if __name__=='__main__':main()
