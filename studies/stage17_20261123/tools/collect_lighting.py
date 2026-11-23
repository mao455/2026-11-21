"""Independent original-scorer recomputation, including excluded development runs."""
from pathlib import Path
import csv,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from stage15_readout import read
E=ROOT/'stage17_evidence';O=ROOT/'../../outputs/stage17';rows=[];details=[];launches={}
for path in [O/'lighting/launches.jsonl',O/'lighting_v2/launches.jsonl',O/'immutable_reruns.jsonl']:
 if path.exists():
  for line in path.read_text().splitlines():
   x=json.loads(line);launches[Path(x['rerun']).name if 'rerun' in x else x['name']]=x.get('wall_s',x.get('launch_wall_s'))
for cohort,plan in [('lighting','lighting_plan.json'),('lighting_v2','lighting_plan_v2.json')]:
 for item in json.loads((E/plan).read_text()):
  session=Path(item['argv'][item['argv'].index('--session')+1]);orig=session;rerun=session.with_name(session.name+'_immutable_rerun')
  chosen=rerun if (rerun/'episode/result.json').exists() else session
  for p in ([orig,chosen] if orig!=chosen else [chosen]):
   ep=p/'episode'
   if not (ep/'result.json').exists():continue
   # Full readout is cached only for unchanged result+raw-state hashes.
   key=hashlib.sha256((ep/'result.json').read_bytes()+(ep/'steps.jsonl').read_bytes()).hexdigest();cache=E/'readout_cache';cache.mkdir(exist_ok=True);f=cache/(key+'.json')
   d=json.loads(f.read_text()) if f.exists() else read(ep)
   if not f.exists():f.write_text(json.dumps(d,indent=2)+'\n')
   d.update(name=item['name'],cohort=cohort,light=item['light'],case_label=item['case'],selected=p==chosen);details.append(d)
   pp=d['physics'];attemptfail=next((a for a in d['attempts'] if not a['prefix_official_success']),None)
   first=('geometry at '+str(d['first_invalid_step']) if not d['geometry_passed'] else 'first proof failed; recovery' if attemptfail and d['actual_grasps']>1 else 'final lift below 28mm' if not pp['hold_physics_passed'] else 'none')
   rows.append(dict(name=item['name'],run=p.name,cohort=cohort,case=item['case'],light=item['light'],selected=p==chosen,execution_valid=d['execution_valid'],success=d['success'],strict=d['legacy_strict_success'],first_grasp=d['first_grasp_success'],grasps=d['actual_grasps'],physical_s=d['physical_s'],host_s=d['host_elapsed_s'],launch_s=launches.get(p.name),lift_mm=pp.get('hold_min_lift_m',0)*1000,clearance_mm=pp.get('hold_min_table_clearance_m',0)*1000,first_error=first,controller_sha256=d['controller_sha256']))
with (E/'lighting_episodes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(E/'lighting_readout.json').write_text(json.dumps(details,indent=2)+'\n')
selected=[r for r in rows if r['selected'] and r['execution_valid']];pairs=[]
for light in (.4,1.,1.6):
 for case in ('A','B'):
  a=next((r for r in selected if r['cohort']=='lighting' and r['name'].endswith('_legacy') and r['case']==case and r['light']==light),None);b=next((r for r in selected if r['cohort']=='lighting_v2' and r['case']==case and r['light']==light),None)
  if a and b:pairs.append(dict(case=case,light=light,baseline=a['success'],hs_v2=b['success'],transition=('S' if a['success'] else 'F')+('S' if b['success'] else 'F'),baseline_grasps=a['grasps'],hs_grasps=b['grasps'],baseline_s=a['physical_s'],hs_s=b['physical_s']))
if pairs:
 with (E/'lighting_pairs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
print(json.dumps(dict(total_completed=len(rows),selected_valid=len(selected),paired=len(pairs),rows=rows),indent=2))
