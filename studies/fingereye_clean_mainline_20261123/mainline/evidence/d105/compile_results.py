"""Read terminal runs and independently apply the original evaluator."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
E=Path(__file__).resolve().parents[3];sys.path.insert(0,str(E))
from evaluator import evaluate,reassess_episode

def read(session):
 e=session/'episode';host=json.loads((e/'result.json').read_text());notes=[];actions=[]
 for line in(e/'agent_trace.jsonl').open():
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  if r.get('op')=='note':
   try:notes.append(json.loads(r['text']))
   except json.JSONDecodeError:pass
  elif r.get('op')=='step':actions.append(r)
 events=[n['s13_recovery']for n in notes if 's13_recovery'in n]
 starts=[r for r in events if r.get('event')=='grasp_dispatched']
 if not starts:
  starts=[n['s11_close_begin']for n in notes if 's11_close_begin'in n]
 checked=reassess_episode(e);assert checked['success']==host['success']
 first=bool(host['success'])if len(starts)==1 else False
 first_physics=None
 if len(starts)>1:
  decision=next((r for r in events if r.get('event')=='failure_decision'and r.get('actual_grasps')==1),None)
  if decision:
   records=[json.loads(l)for l in(e/'steps.jsonl').open()]
   scene=json.loads((e/'scene.json').read_text())
   first_physics=evaluate([r for r in records if r['step']<decision['step']],scene['initial_object_pose'][2]);first=bool(first_physics['success'])
 selection=[n['s11_finger_selection']for n in notes if 's11_finger_selection'in n]
 failures=[r for r in events if r.get('event')in ['failure_decision','stopped','guard_refused','candidate_refused']]
 row=dict(session=str(session),success=checked['success'],first_success=first,actual_closures=len(starts),controller_completed=host['controller_completed'],error=host['error'],controller_sha256=host['controller_sha256'],evaluator_and_adapter_unchanged=host['evaluator_and_adapter_unchanged'],physics=checked['physics'],first_physics=first_physics,host_wall_s=host['elapsed_s'],physical_s=checked['physical_s'],closure_starts=starts,last_phase=actions[-1]['phase']if actions else None,last_failures=failures[-5:],selection_rounds=[dict(selected=r.get('selected'),planning_cpu_s=r.get('planning_cpu_s'),planning_wall_s=r.get('planning_wall_s'),rejected=len(r.get('rejected',[])))for r in selection])
 return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for root in a.runs:
  state=json.loads((root/'STATE.json').read_text())
  for item in state['rows']:
   assert item['status']=='terminal',item
   r=read(Path(item['session']));r.update(case=item['case'],hand=item['hand'],run_root=str(root));rows.append(r)
 totals={}
 for hand in sorted({r['hand']for r in rows}):
  group=[r for r in rows if r['hand']==hand];totals[hand]=dict(n=len(group),zero_closure=sum(r['actual_closures']==0 for r in group),first_success=sum(r['first_success']for r in group),max3_success=sum(r['success']for r in group),rescued_second=sum(r['success']and not r['first_success']and r['actual_closures']==2 for r in group),rescued_third=sum(r['success']and not r['first_success']and r['actual_closures']==3 for r in group),actual_closures=sum(r['actual_closures']for r in group))
 with a.output.open('x')as f:json.dump(dict(rows=rows,totals=totals),f,indent=2)
 print(json.dumps(totals))
if __name__=='__main__':main()
