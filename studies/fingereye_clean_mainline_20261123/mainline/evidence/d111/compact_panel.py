"""Retain small offline failure/control receipts before retiring raw episodes.

Truth and native contact records are read here only after the episode terminates;
this module is never imported by the online strategy.
"""
from pathlib import Path
import argparse, collections, hashlib, json
import numpy as np
from scipy.spatial import ConvexHull

H=Path(__file__).resolve().parent

def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)

def minimal_geometry(value):
    if isinstance(value,dict):
        out={k:minimal_geometry(v)for k,v in value.items()}
        cloud=value.get('collision_points')
        if isinstance(cloud,list)and len(cloud)>100:
            array=np.asarray(cloud,dtype=float)
            if array.ndim==2 and array.shape[1]==3:
                vertices=array[ConvexHull(array).vertices]
                out['collision_points']=vertices.tolist()
                out['collision_points_compaction']=dict(original_shape=list(array.shape),
                    original_float64_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                    retained='convex hull vertices, identical finite convex contact/collision geometry; interior sampling is regenerable')
        return out
    if isinstance(value,list):return [minimal_geometry(v)for v in value]
    return value

def compact(session):
    p=session/'episode'; notes=[]
    for e in records(p/'agent_trace.jsonl'):
        if e.get('op')=='note':notes.append(json.loads(e['text']))
    wanted=('s13_recovery','s5_plan','s12_candidate_command_binding',
        'planning_comparison_budget','s5_measured_stop','s5_guard',
        'selected_surface_update','local_rgb_repair','rgb_changed_command',
        'stage3_release','stage3_release_retreat_ticket','s7_release_clearance',
        'stage3_recovery_stop','stage3_recovery_reject','s5_rgb_prefix_blocked',
        's5_finger_control_decisions','s5_probe_budget_complete')
    retained=[n for n in notes if any(k in n for k in wanted) or any('feedback' in k or 'closure' in k or 'release' in k or 'recovery_phase' in k or 'close_probe' in k or 'preload' in k or 'support' in k for k in n)]
    # Plans already have exact candidate/initial joint vectors. Avoid repeating
    # thousands of geometric rejections that have compact numerical receipts.
    for n in retained:
        if 's5_plan' in n:
            n['s5_plan']['rejected']=[dict(first_rejection=r.get('first_rejection'),fingers=r.get('fingers'),error_m=r.get('error_m'),normal_error_rad=r.get('normal_error_rad'))for r in n['s5_plan'].get('rejected',[])]
    # Use exactly the evaluator's pre-simulation reference. The first recorded
    # physics step is already about 0.1 mm lower after normal settling.
    initial_z=json.loads((p/'scene.json').read_text())['initial_object_pose'][2]
    phases={}; phase_changes=[]; first_invalid=None; samples=[]; anchors=set()
    for n in retained:
        for k,v in n.items():
            if isinstance(v,dict):
                if 'step'in v:anchors.add(int(v['step']))
    last_phase=None
    for s in records(p/'steps.jsonl'):
        phase=s['phase'];d=phases.setdefault(phase,dict(first_step=s['step'],last_step=s['step'],samples=0,peak_lift_m=0.,all_geometry_valid=True,max_qvel=0.))
        d['last_step']=s['step'];d['samples']+=1;d['peak_lift_m']=max(d['peak_lift_m'],s['object_pose'][2]-initial_z)
        d['all_geometry_valid'] &= bool(s['penetration']['passed']);d['max_qvel']=max(d['max_qvel'],max(abs(v)for v in s['actual_qvel']))
        if not s['penetration']['passed']and first_invalid is None:first_invalid=dict(step=s['step'],phase=phase,geometry=s['penetration']);anchors.add(s['step'])
        if phase!=last_phase:phase_changes.append(dict(step=s['step'],phase=phase));anchors.add(s['step']);last_phase=phase
    # Native records use end-of-physics-step numbering. Preserve that number;
    # match to steps.jsonl's step-1 only for assigning a phase label.
    byphase={}; bylink={}; byattempt={}
    closure_starts=[v['s13_recovery'].get('started_step',v['s13_recovery']['step'])for v in notes if v.get('s13_recovery',{}).get('event')=='grasp_dispatched']
    transitions=iter(phase_changes);current=next(transitions,None);following=next(transitions,None)
    for s in records(p/'all_hand_object_contacts.jsonl'):
        step=int(s['step']);physical_step=step-1
        while following is not None and physical_step>=following['step']:current=following;following=next(transitions,None)
        phase='unknown'if current is None else current['phase']
        d=byphase.setdefault(phase,dict(samples=0,normal_force_by_link_n={}))
        d['samples']+=1
        attempt=sum(st<=physical_step for st in closure_starts)
        group=byattempt.setdefault(attempt,{})
        for c in s['contacts']:
            link=c['link'];force=float(c['normal_force_sum_n'])
            ag=group.setdefault(link,{})
            ap=ag.setdefault(phase,dict(first_above_10mN=None,last_above_10mN=None,max_normal_force_n=0.,positive_steps=0))
            ap['max_normal_force_n']=max(ap['max_normal_force_n'],force)
            if force>.01:
                if ap['first_above_10mN']is None:ap['first_above_10mN']=step
                ap['last_above_10mN']=step;ap['positive_steps']+=1
            a=bylink.setdefault(link,dict(first_above_10mN=None,last_above_10mN=None,max_normal_force_n=0.,peak_step=None))
            if force>.01:
                if a['first_above_10mN']is None:a['first_above_10mN']=step
                a['last_above_10mN']=step
            if force>a['max_normal_force_n']:a['max_normal_force_n']=force;a['peak_step']=step;a['peak_contact_record']=c
            b=d['normal_force_by_link_n'].setdefault(link,dict(sum_n=0.,max_n=0.,positive_samples=0));b['sum_n']+=force;b['max_n']=max(b['max_n'],force);b['positive_samples']+=int(force>.01)
    for v in bylink.values():
        for k in ('first_above_10mN','last_above_10mN','peak_step'):
            if v[k]is not None:anchors.add(v[k]-1)
    for s in records(p/'steps.jsonl'):
        if any(abs(s['step']-a)<=6 for a in anchors):samples.append({k:s[k]for k in ('step','phase','target_q','actual_q','actual_qvel','object_pose','penetration')})
    return dict(session=str(session),scope='post-run offline evidence only; native records retain their end-of-step numbering',
        initial_object_z_m=initial_z,lift_reference='scene.initial_object_pose[2], identical to evaluator',
        control_and_plan_notes=minimal_geometry(retained),phase_summary=phases,phase_changes=phase_changes,
        native_contact_by_link=bylink,native_contact_by_phase=byphase,native_contact_by_attempt=byattempt,actual_closure_starts=closure_starts,first_invalid=first_invalid,short_step_windows=samples)

def write_receipts(source_commit, rows):
    """One small index plus one file per case; preserve all retained samples."""
    index=[]
    for row in rows:
        name='PHYSICAL_'+row['hand']+'_'+row['case']+'.json'
        blob=(json.dumps(dict(source_commit=source_commit,**row),separators=(',',':'))+'\n').encode()
        (H/name).write_bytes(blob)
        index.append(dict(hand=row['hand'],case=row['case'],diagnosis_file=name,
            sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),
            actual_closure_starts=row['actual_closure_starts'],first_invalid=row['first_invalid'],
            initial_object_z_m=row['initial_object_z_m'],
            peak_lift_m=max(v['peak_lift_m']for v in row['phase_summary'].values())))
    (H/'FINAL_PHYSICAL_DIAGNOSIS.json').write_text(json.dumps(dict(source_commit=source_commit,
        scope='index; per-case files retain the exact compact evidence and raw control windows',rows=index),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path);a=parser.parse_args()
    state=json.loads((a.batch/'STATE.json').read_text());rows=[]
    for r in state['rows']:
        assert r['status']=='terminal'
        # Both remaining failures and one successful mechanical comparison.
        if not r.get('result',{}).get('success',False) or (r['hand']=='wuji'and r['case']=='pentagon_28'):
            rows.append(dict(hand=r['hand'],case=r['case'],**compact(Path(r['session']))))
    write_receipts(state['source_commit'],rows)
    print('retained compact rows',[(r['hand'],r['case'])for r in rows])
