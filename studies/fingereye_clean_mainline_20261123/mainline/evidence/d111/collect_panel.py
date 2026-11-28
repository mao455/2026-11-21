"""Compact final panel from immutable normal-entry receipts; truth offline only."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
E=Path(__file__).resolve().parents[3];H=Path(__file__).parent;O=E.parents[2]/'FingerEyeLab/outputs/d111'

def collect(batch):
 state=json.load((O/batch/'STATE.json').open());rows=[]
 for row in state['rows']:
  if row['status']!='terminal':continue
  session=Path(row['session']);P=session/'episode';result=json.load((P/'result.json').open())if(P/'result.json').exists()else None
  notes=[]
  for l in(P/'agent_trace.jsonl').open():
   x=json.loads(l)
   if x.get('op')=='note':notes.append(json.loads(x['text']))
  grasps=[n['s13_recovery']for n in notes if n.get('s13_recovery',{}).get('event')=='grasp_dispatched']
  decisions=[n['s13_recovery']for n in notes if n.get('s13_recovery',{}).get('event')=='failure_decision']
  plans=[n['s5_plan']for n in notes if 's5_plan'in n]
  selected=[n['s12_candidate_command_binding']for n in notes if 's12_candidate_command_binding'in n]
  receipt=json.load((session/'budget_receipt.json').open());assert receipt['source_unchanged']and not receipt['leftover_descendants']
  scene=json.load((P/'scene.json').open());initial_z=scene['initial_object_pose'][2]
  attempts=[dict(number=i+1,start_step=g.get('started_step',g['step']),fingers=[],candidate_id=g['candidate_id'])for i,g in enumerate(grasps)]
  for a in attempts:
   binding=next((s for s in reversed(selected)if s['step']<=a['start_step']),None)
   a.update(contact_count=None if binding is None else binding['contact_count'],peak_lift_m=0.,all_geometry_valid=True)
   plan=next((p for p in reversed(plans)if p['step']<=a['start_step']),None)
   if plan:
    candidate=next((c for c in plan['candidates']if c['meta'].get('planner_candidate_id')==a['candidate_id']),None)
    if candidate is None:candidate=next((c for c in plan['candidates']if c['meta'].get('candidate_id')==binding.get('candidate_id')),None)if binding else None
    if candidate:a['fingers']=candidate['meta']['contact_fingers']
  last=None;first=None;invalid=None
  for l in(P/'steps.jsonl').open():
   s=json.loads(l);last=s
   if first is None:first=s
   if not s['penetration']['passed']and invalid is None:invalid=dict(step=s['step'],phase=s['phase'],geometry=s['penetration'])
   eligible=[a for a in attempts if a['start_step']<=s['step']]
   if eligible:
    a=eligible[-1];a['peak_lift_m']=max(a['peak_lift_m'],s['object_pose'][2]-initial_z)
    a['all_geometry_valid'] &= bool(s['penetration']['passed'])
  success=bool(result and result['success']);closures=len(grasps)
  for a in attempts:
   a['two_second_hold_success']=bool(success and a['number']==closures)
   a['decision']=next((d['decision']for d in decisions if d['actual_grasps']==a['number']),None)
  timings=next((n['s9_io_audit']for n in reversed(notes)if 's9_io_audit'in n),{})
  components={k:dict(sum_s=float(sum(v)),samples=len(v))for k,v in timings.get('component_samples_s',{}).items()}
  blocks=[n['s5_rgb_prefix_blocked']for n in notes if 's5_rgb_prefix_blocked'in n]
  stops=[n.get('stage3_recovery_stop',n.get('stage3_recovery_reject',n.get('s13_recovery')))for n in notes if 'stage3_recovery_stop'in n or 'stage3_recovery_reject'in n or n.get('s13_recovery',{}).get('event')in ('stopped','guard_refused')]
  rows.append(dict(hand=row['hand'],case=row['case'],source_commit=state['source_commit'],session=str(session),
   controller_sha256=None if result is None else result['controller_sha256'],
   first_grasp_success=success and closures==1,within_three_success=success,zero_closure=closures==0,
   actual_closures=closures,rescued_on=closures if success and closures>1 else None,
   attempts=attempts,result=result,first_invalid=invalid,terminal_stops=stops[-3:],last_rgb_blocks=blocks[-2:],
   plans=[dict(attempt=p['attempt'],accepted=p['accepted'],first_rejections=[x.get('first_rejection')for x in p['rejected']])for p in plans],
   timing=dict(host_s=receipt['actual_host_elapsed_s'],policy_cpu_user_system_s=sum(p.get('cpu_user_s',0.)+p.get('cpu_system_s',0.)for p in receipt['processes']if p.get('is_policy_worker')),controller_s=None if result is None else result['elapsed_s'],components=components),
   argv=row['argv'],budget={k:receipt[k]for k in ['status','source_unchanged','leftover_descendants']},
   source_files=state['files'],artifact_sha256={name:hashlib.sha256((P/name).read_bytes()).hexdigest()for name in ['scene.json','result.json','agent_trace.jsonl']if(P/name).exists()}))
 return rows

if __name__=='__main__':
 batches=sys.argv[1:]or['final_c9'];rows=[]
 for b in batches:
  if(O/b/'STATE.json').exists():
   state=json.load((O/b/'STATE.json').open())
   assert state.get('completed') and all(r['status']=='terminal'for r in state['rows']), 'Final panel is not complete'
   rows+=collect(b)
 out=dict(evaluator_sha256=hashlib.sha256((E/'evaluator.py').read_bytes()).hexdigest(),rows=rows,
  by_hand={h:dict(cases=sum(r['hand']==h for r in rows),first_grasp=sum(r['first_grasp_success']for r in rows if r['hand']==h),within_three=sum(r['within_three_success']for r in rows if r['hand']==h),zero_closure=sum(r['zero_closure']for r in rows if r['hand']==h),second_rescue=sum(r['rescued_on']==2 for r in rows if r['hand']==h),third_rescue=sum(r['rescued_on']==3 for r in rows if r['hand']==h))for h in ['xhand','new','wuji']})
 (H/'FINAL_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['by_hand']))
