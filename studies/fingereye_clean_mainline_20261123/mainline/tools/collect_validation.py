"""Read-only collection of the two new physical episodes and historical comparison."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(HERE))
from stage15_readout import read


def trace(ep):
    rows=[json.loads(line) for line in (ep/'agent_trace.jsonl').open()]
    commands=[r for r in rows if r.get('op')=='step']
    notes=[json.loads(r['text']) for r in rows if r.get('op')=='note']
    phases=[];step=0
    for i,c in enumerate(commands):
        if not phases or phases[-1]['phase']!=c['phase']:
            phases.append(dict(phase=c['phase'],start_step=step,end_step=step,
                               start_s=step/120,end_s=step/120,commands=0))
        step+=c['steps'];phases[-1].update(end_step=step,end_s=step/120,commands=phases[-1]['commands']+1)
    selected=[]
    for n in notes:
        for key,v in n.items():
            if key=='s13_recovery' and v.get('event') in ('grasp_dispatched','failure_decision','considered','guard_refused','selected','executed','unknown_final_excitation'):
                selected.append(dict(kind=key,**v))
            if key.startswith('s5_') and key.endswith('_decision'): selected.append(dict(kind=key,**v))
    return commands,dict(phases=phases,events=selected)


def collect():
    out=HERE/'mainline/evidence';pairs=[]
    for case,name in [('A','first_A'),('B','recovery_B')]:
        old=HERE/f'../../outputs/stage17/lighting_v2/{case}_L1_hs_v2/episode'
        new=HERE/f'../../outputs/clean_mainline/{name}/episode'
        old,new=old.resolve(),new.resolve()
        before,after=read(old),read(new)
        a,ta=trace(old);b,tb=trace(new)
        initial=[np.load(p/'public/overhead_initial.npy') for p in [old,new]]
        first=None;maximum=0.;equal=0
        for i,(x,y) in enumerate(zip(a,b)):
            error=float(np.max(abs(np.asarray(x['target_q'])-y['target_q'])))
            maximum=max(maximum,error)
            same=x['steps']==y['steps'] and x['phase']==y['phase'] and error==0
            equal+=int(same)
            if first is None and not same:first=dict(command=i,old_phase=x['phase'],new_phase=y['phase'],max_q_error=error)
        fields=('success','legacy_strict_success','actual_grasps','physical_steps','physical_s','host_elapsed_s','first_grasp_success','recovered_after_first_failure')
        pair=dict(case=case,old_episode=str(old),new_episode=str(new),before={k:before[k] for k in fields},after={k:after[k] for k in fields},
                  initial_overhead_changed_channels=int(np.count_nonzero(initial[0]!=initial[1])),
                  initial_overhead_max_uint8_difference=int(abs(initial[0].astype(int)-initial[1]).astype(int).max()),
                  old_commands=len(a),new_commands=len(b),equal_commands_at_same_index=equal,
                  first_fresh_command_difference=first,max_q_difference_at_same_index=maximum,
                  phase_steps_before=dict(Counter({p:sum(c['steps'] for c in a if c['phase']==p) for p in {c['phase'] for c in a}})),
                  phase_steps_after=dict(Counter({p:sum(c['steps'] for c in b if c['phase']==p) for p in {c['phase'] for c in b}})),
                  interpretation='fresh rendering and physics are different input records; same-record extraction equivalence is established separately')
        pairs.append(pair)
        (out/f'trace_{case}.json').write_text(json.dumps(dict(episode=str(new),controller_sha256=after['controller_sha256'],**tb),indent=2)+'\n')
    (out/'fresh_comparison.json').write_text(json.dumps(dict(pairs=pairs),indent=2)+'\n')
    print(json.dumps(pairs,indent=2))

if __name__=='__main__':collect()
