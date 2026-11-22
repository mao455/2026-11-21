"""Evidence-derived grasp, recovery, paired-state and timing readout (no policy)."""
import argparse,collections,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
from evaluator import evaluate
from scipy.spatial.transform import Rotation

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
 if p.exists():
  with p.open() as stream:
   for line in stream:
    if line.strip():yield json.loads(line)
def load(p):return json.loads(p.read_text()) if p.exists() else {}
GESTURE_ANGLE_RAD=math.pi/6
GESTURE_CONTACT_SHIFT_M=.005
def distribution(values):
 values=sorted(float(x) for x in values if x is not None)
 if not values:return dict(n=0,mean=None,median=None,p90=None)
 at=.9*(len(values)-1);lo=int(at);hi=min(lo+1,len(values)-1)
 fraction=at-lo
 return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),p90=values[lo]*(1-fraction)+values[hi]*fraction)
def object_relative_points(closure):
 pose=closure.get('preclose_object_pose')
 if not pose or len(pose)!=7:return None
 center=np.asarray(pose[:3],dtype=float)
 quaternion=np.asarray(pose[3:],dtype=float)
 rotation=Rotation.from_quat(quaternion[[1,2,3,0]])
 return rotation.inv().apply(np.asarray(closure.get('preclose_actual_pad_points',closure['points']),dtype=float)-center)
def gesture_delta(a,b):
 ma=a.get('preclose_hand_object_relative_matrix');mb=b.get('preclose_hand_object_relative_matrix')
 angle=None
 if ma is not None and mb is not None:
  angle=float((Rotation.from_matrix(np.asarray(ma)[:3,:3]).inv()*Rotation.from_matrix(np.asarray(mb)[:3,:3])).magnitude())
 pa=object_relative_points(a);pb=object_relative_points(b);shift=None
 if pa is not None and pb is not None:
  common=min(len(pa),len(pb));shift=float(np.linalg.norm(pa[:common]-pb[:common],axis=1).mean()) if common else None
 return dict(from_count=a['contact_count'],to_count=b['contact_count'],wrist_attitude_delta_rad=angle,common_finger_mean_landing_delta_m=shift,common_finger_count=min(a['contact_count'],b['contact_count']),large_gesture=bool((angle is not None and angle>=GESTURE_ANGLE_RAD) or (shift is not None and shift>=GESTURE_CONTACT_SHIFT_M)),hand_switch=a['contact_count']!=b['contact_count'],thresholds=dict(wrist_attitude_rad=GESTURE_ANGLE_RAD,common_finger_mean_landing_delta_m=GESTURE_CONTACT_SHIFT_M),evidence_complete=angle is not None and shift is not None)
def episode(path):
 path=Path(path);host=load(path/'result.json');scene=load(path/'scene.json');events=[];trace=list(rows(path/'agent_trace.jsonl'))
 initial_object_pose=scene.get('initial_object_pose')
 initial_height=initial_object_pose[2] if initial_object_pose and len(initial_object_pose)>=3 else 0.
 for row in trace:
  if row.get('op')=='note':
   try:
    for key,data in json.loads(row['text']).items():events.append(dict(key=key,data=data,elapsed_s=row['elapsed_s']))
   except (ValueError,TypeError):pass
 records=list(rows(path/'steps.jsonl'));phase_physics=collections.Counter(r.get('phase','unknown') for r in records)
 # Trace requests can be rejected. Actual 120Hz records establish execution.
 markers=[e for e in events if e['key']=='s11_close_begin']
 closures=[]
 for marker in markers:
  start=marker['data']['step'];next_begin=min([x['data']['step'] for x in markers if x['data']['step']>start],default=10**12)
  first=next((r for r in records if start<=r['step']<next_begin and r['phase']=='s5_close'),None)
  if first is None:continue
  binding=next((e['data'] for e in reversed(events) if e['key']=='s12_candidate_command_binding' and e['elapsed_s']<=marker['elapsed_s']),{})
  closures.append(dict(start_step=first['step'],contact_count=len(marker['data']['points']),points=marker['data']['points'],planned_q=binding.get('planned_q'),target_q=first['target_q'],actual_q=first['actual_q'],preclose_object_pose=first.get('object_pose'),preclose_hand_object_relative_matrix=first.get('hand_object_relative_matrix'),binding=binding,branch='initial' if not closures else 'legacy_or_recovery'))
 s13=[e for e in events if e['key']=='s13_recovery']
 # Bind acknowledged closures, then independently recover preload attempts
 # from actual contiguous physics. A runtime exception after a dispatch can
 # omit the policy acknowledgement even though the motor command took effect.
 for e in s13:
  d=e['data']
  if d.get('event')=='grasp_dispatched':
   matching=next((c for c in closures if c['start_step']==d['started_step']),None)
   if matching is not None:matching.update(branch=d['branch'],dispatch_event=d,dispatch_acknowledged=True)
 preload_starts=[(i,r) for i,r in enumerate(records) if r['phase']=='s13_preload_recovery' and (i==0 or records[i-1]['phase']!='s13_preload_recovery' or records[i-1]['step']+1!=r['step'])]
 for first_index,first in preload_starts:
  start=first['step']
  end_index=next((i for i in range(first_index+1,len(records)) if records[i]['phase']!='s13_preload_recovery' or records[i-1]['step']+1!=records[i]['step']),len(records))
  segment_end=records[end_index-1]['step']+1
  ack=next((e['data'] for e in s13 if e['data'].get('event')=='grasp_dispatched' and e['data'].get('branch')=='preload' and e['data'].get('started_step')==start),{})
  selected=next((e['data'] for e in reversed(s13) if e['data'].get('event')=='selected' and e['data'].get('branch')=='preload' and e['data'].get('step',10**12)<=start),{})
  guard=next((e['data'] for e in reversed(s13) if e['data'].get('event')=='guard' and e['data'].get('branch')=='preload' and e['data'].get('step',10**12)<=start and e['data'].get('recovery_id')==selected.get('recovery_id')),{})
  binding=next((e['data'] for e in reversed(events) if e['key']=='s12_candidate_command_binding' and e['data'].get('step',10**12)<=start),{})
  marker=next((e['data'] for e in reversed(markers) if e['data'].get('step',10**12)<=start),{})
  previous=max((c for c in closures if c['start_step']<start),key=lambda c:c['start_step'],default={})
  points=ack.get('points',marker.get('points',previous.get('points',[])))
  count=ack.get('contact_count',binding.get('contact_count',len(points)))
  attempt=dict(start_step=start,contact_count=count,points=points,planned_q=ack.get('planned_q',binding.get('planned_q',previous.get('planned_q'))),target_q=first['target_q'],actual_q=first['actual_q'],preclose_object_pose=first.get('object_pose') if points else None,preclose_hand_object_relative_matrix=first.get('hand_object_relative_matrix'),branch='preload',binding=binding,recovery_id=ack.get('recovery_id',selected.get('recovery_id')),selected_event=selected,guard_event=guard,dispatch_acknowledged=bool(ack),preload_segment_end_step=segment_end,attempt_evidence='first actual step of contiguous s13_preload_recovery phase')
  if ack:attempt['dispatch_event']=ack
  closures.append(attempt)
 closures.sort(key=lambda r:r['start_step'])
 # Match the physical instant immediately before first closure dispatch.
 # Nominal IK contact points are retained separately from actual FK pad poses.
 if closures and scene.get('robot_description',{}).get('urdf_path'):
  from visual_controller import Robot
  robot=Robot(scene['robot_description']['urdf_path'])
  for c in closures:
   i=int(c['start_step'])-1
   prior=records[i] if i>=0 else dict(actual_q=next(rows(path/'public_responses.jsonl'),{}).get('observation',{}).get('actual_q'),object_pose=scene.get('initial_object_pose'))
   robot.configure_contacts(c['contact_count'])
   c['preclose_actual_q']=prior['actual_q']
   c['preclose_actual_pad_points']=robot.tips(np.asarray(prior['actual_q']))[0].tolist()
   c['preclose_object_pose']=prior.get('object_pose')
   if prior.get('hand_object_relative_matrix') is not None:
    c['preclose_hand_object_relative_matrix']=np.linalg.inv(np.asarray(prior['hand_object_relative_matrix'])).tolist()
   c['preclose_stage']='actual state immediately before first closure dispatch'
 # First trial ends before the first physical recovery command, irrespective of
 # whether the policy believed it had failed. This never uses policy support as success.
 recovery_prefix=('s3_','s7_release','s11_local','s13_recovery','s13_local','s13_preload')
 def next_recovery(record,current):
  own_preload=current.get('branch')=='preload' and record['phase']=='s13_preload_recovery' and record['step']<current.get('preload_segment_end_step',current['start_step'])
  return record['phase'].startswith(recovery_prefix) and not own_preload
 first_start=closures[0]['start_step'] if closures else 10**12
 cutoff=next((r['step'] for r in records if r['step']>first_start and next_recovery(r,closures[0])),len(records))
 first_physics=evaluate(records[:cutoff],initial_height)
 for i,c in enumerate(closures):
  next_start=closures[i+1]['start_step'] if i+1<len(closures) else len(records)
  end=next((r['step'] for r in records if c['start_step']<r['step']<next_start and next_recovery(r,c)),next_start)
  c['trial_end_step']=end
  c['physics']=evaluate(records[:end],initial_height)
  c['success']=bool(initial_object_pose and c['physics']['success'] and host.get('controller_completed') and host.get('evaluator_and_adapter_unchanged') and host.get('error') is None)
 first_success=bool(initial_object_pose and first_physics['success'] and host.get('controller_completed') and host.get('evaluator_and_adapter_unchanged') and host.get('error') is None)
 branch_counts={b:dict(considered=0,selected=0,executed=0,rescued=0,guard_refused=0,conditional_fallback_considered=0) for b in ('preload','local_pose','release_resample')}
 last_recovery_branch=None
 for e in s13:
  d=e['data'];b=d.get('branch')
  if b in branch_counts and d.get('event') in ('selected','executed') and not d.get('conditional_fallback'):last_recovery_branch=b
  if b=='planning' and d.get('event')=='guard_refused' and last_recovery_branch in branch_counts:
   b=last_recovery_branch
   e['offline_branch_attribution']=b
  if b not in branch_counts:continue
  event=d.get('event')
  if event=='guard' and d.get('audit',{}).get('passed') is False:
   event='guard_refused';e['offline_guard_attribution']='failed_preflight_audit'
  if event=='selected' and d.get('conditional_fallback'):
   branch_counts[b]['conditional_fallback_considered']+=1
   branch_counts[b]['considered']+=1
   continue
  if event in ('considered','selected','executed','guard_refused'):branch_counts[b][event]+=1
 final_success=bool(host.get('success') and host.get('controller_completed') and host.get('evaluator_and_adapter_unchanged') and not host.get('error'));rescued=final_success and not first_success and len(closures)>1
 if rescued:
  b=closures[-1].get('branch')
  if b in branch_counts:branch_counts[b]['rescued']+=1
 transitions=[dict(attempt_from=i,attempt_to=i+1,**gesture_delta(a,b)) for i,(a,b) in enumerate(zip(closures,closures[1:]),1)]
 phases_wall=collections.defaultdict(lambda:dict(dispatch_rpc_s=0.,between_dispatch_compute_s=0.))
 audits=[e['data'] for e in events if e['key']=='s9_io_audit']
 for audit in audits:
  for cycle in audit.get('cycles',[]):
   p=phases_wall[cycle['phase']];p['dispatch_rpc_s']+=cycle['dispatch_rpc_s'];p['between_dispatch_compute_s']+=cycle['compute_between_dispatch_s']
 frames=list(rows(path/'image_manifest.jsonl'));video=load(path/'video_timing.json')
 initial_public=next(rows(path/'public_responses.jsonl'),{}).get('observation',{})
 initial_state=dict(object_pose=scene.get('initial_object_pose'),actual_q=initial_public.get('actual_q'),actual_qvel=initial_public.get('actual_qvel'),parameters=scene.get('realized_parameters'),mesh_sha256=scene.get('mesh_sha256'),object_mass_kg=scene.get('object_mass_kg'),friction=scene.get('friction'),object_gravity=scene.get('object_gravity'),finger_stiffness=scene.get('finger_stiffness'),finger_damping=scene.get('finger_damping'),physical_inertia=[{k:v for k,v in row.items() if 'physx_' in k} for row in scene.get('physics',[]) ])
 state_for_pair=dict(initial_state);params=dict(state_for_pair.get('parameters') or {});params.pop('rgb_noise_std',None);state_for_pair['parameters']=params
 policy_compute_s=sum(v['between_dispatch_compute_s'] for v in phases_wall.values())
 from stage13_report import policy_timing
 full_timing=policy_timing(dict(episode=str(path),policy_compute_s=policy_compute_s))
 legacy=host.get('physics',{}).get('legacy_strict',{})
 legacy_success=bool(legacy.get('success') and host.get('controller_completed') and host.get('evaluator_and_adapter_unchanged') and host.get('error') is None)
 return dict(name=path.parent.name,episode=str(path.resolve()),result_present=(path/'result.json').exists(),controller_completed=bool(host.get('controller_completed')),paired_initial_state_available=bool(initial_public.get('actual_q') and scene.get('initial_object_pose')),initial_overhead_sha256=sha(path/'public/overhead_initial.npy') if (path/'public/overhead_initial.npy').exists() else None,controller_sha256=host.get('controller_sha256'),success=final_success,legacy_strict_success=legacy_success,first_success=first_success,rescued=rescued,actual_grasps=len(closures),finger_counts=dict(collections.Counter(str(c['contact_count']) for c in closures)),closures=closures,first_successful_attempt=next((i for i,c in enumerate(closures,1) if c.get('success')),None),transitions=transitions,branch_counts=branch_counts,first_trial_end_step=cutoff,first_trial_verdict=first_physics,physical_s=len(records)/120,first_trial_physical_s=min(cutoff,len(records))/120,retry_physical_s=max(0,len(records)-cutoff)/120 if closures else 0.,policy_compute_s=policy_compute_s,policy_total_wall_s=full_timing.get('policy_nonrpc_wall_s'),policy_initial_wall_s=full_timing.get('initial_policy_nonrpc_s'),policy_timing_audit=full_timing,phase_physical_s={k:v/120 for k,v in phase_physics.items()},phase_wall_components=dict(phases_wall),wall_profile=load(path/'wall_profile_backend.json').get('exclusive_categories'),episode_elapsed_s=host.get('elapsed_s'),error=host.get('error'),integrity_valid=host.get('evaluator_and_adapter_unchanged'),frame_count=len(frames),frame_stride_4=all(b['step']-a['step']==4 for a,b in zip(frames,frames[1:])),video=video,initial_state=initial_state,paired_initial_state_sha256=hashlib.sha256(json.dumps(state_for_pair,sort_keys=True).encode()).hexdigest(),selection_events=[e for e in events if e['key']=='s11_finger_selection'],recovery_events=s13,source_sha256={name:sha(path/name) for name in ('result.json','scene.json','agent_trace.jsonl','steps.jsonl','image_manifest.jsonl') if (path/name).exists()})
def aggregate(root,output,source_commit=None):
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True);results=[]
 contract=load(root/'contract.json')
 matching_plans=[p for p in root.parent.glob('frozen*/plan.json') if sha(p)==contract.get('plan_sha256')]
 registered_plan=load(matching_plans[0]) if matching_plans else {};registered_runs={r['name']:r for r in registered_plan.get('runs',[])}
 for receipt_path in sorted(root.glob('*/batch_result.json')):
  path=receipt_path.parent/'episode';row=episode(path);receipt=load(receipt_path);launch=load(receipt_path.parent/'launch.json');row['error']=row.get('error') or receipt.get('error');row.update(source_commit=receipt.get('source_commit') or launch.get('source_commit') or source_commit,condition=receipt.get('condition'),object_id=receipt.get('object_id'),batch_wall_s=receipt.get('wall_s'),returncode=receipt.get('returncode'))
  # Incomplete physics does not censor an otherwise exact scheduler duration.
  # Censoring is a property of the receipt, independent of episode completion.
  case=receipt.get('case') or launch.get('case') or {};planned=registered_runs.get(row['name'],{})
  row.update(registered_case=case,noise_seed=case.get('noise_seed'),rgb_noise_std=case.get('rgb_noise_std'),initial_dx_m=case.get('dx'),initial_dy_m=case.get('dy'),initial_yaw_rad=case.get('yaw'),profile=planned.get('profile'),fingers_mode=planned.get('fingers'),max_actual_grasps=planned.get('max_attempts'),recording_mode=planned.get('mode'))
  row['wall_s_censored']=bool(receipt.get('wall_s_censored'))
  row['batch_wall_observed_exact']=bool(not row['wall_s_censored'] and row.get('batch_wall_s') is not None)
  row['timing_complete']=bool(row['result_present'] and row['controller_completed'] and not row.get('error'))
  # A recovered scheduler receipt can only bound its launch-to-exit wall span.
  # Its completed physics and backend monotonic clocks remain exact; do not
  # drop those observations merely because the scheduler receipt was recovered.
  row['batch_wall_timing_complete']=bool(row['timing_complete'] and row['batch_wall_observed_exact'])
  row['physical_time_scope']='completed_exact' if row['timing_complete'] else 'observed_lower_bound_incomplete'
  row['observed_physical_lower_bound_s']=row['physical_s'] if not row['timing_complete'] else None
  row['integrity_valid']=bool(row.get('integrity_valid'))
  row['controller_sha256']=row.get('controller_sha256') or receipt.get('controller_sha256')
  row['failure_class']=('success' if row['success'] else 'infrastructure_timeout' if row['returncode']==124 else 'infrastructure_error' if row['error'] else 'planning_rejection' if row['actual_grasps']==0 else 'physical_failure')
  (output/(row['name']+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False));results.append(row)
 pairs=[];grouped=collections.defaultdict(dict)
 for r in results:
  if r['object_id'] and r['condition']:grouped[r['object_id']][r['condition']]=r
 for obj,p in grouped.items():
  if set(p)=={'clean','noise'}:pairs.append(dict(object_id=obj,initial_state_equal=(p['clean']['paired_initial_state_sha256']==p['noise']['paired_initial_state_sha256']) if p['clean']['paired_initial_state_available'] and p['noise']['paired_initial_state_available'] else None,initial_overhead_equal=(p['clean']['initial_overhead_sha256']==p['noise']['initial_overhead_sha256']) if p['clean'].get('initial_overhead_sha256') and p['noise'].get('initial_overhead_sha256') else None,clean_success=p['clean']['success'],noise_success=p['noise']['success'],clean_first=p['clean']['first_success'],noise_first=p['noise']['first_success']))
 def group_stats(group):
  transitions=[t for r in group for t in r['transitions']]
  counts=collections.Counter(r['actual_grasps'] for r in group)
  failure=collections.Counter(r['failure_class'] for r in group)
  phases=sorted({name for r in group for name in r['phase_physical_s']})
  recovery={b:{k:sum(r['branch_counts'][b][k] for r in group) for k in ('considered','selected','executed','rescued','guard_refused','conditional_fallback_considered')} for b in ('preload','local_pose','release_resample')}
  timing={k:distribution(r.get(k) for r in group if r['batch_wall_timing_complete' if k=='batch_wall_s' else 'timing_complete']) for k in ('physical_s','first_trial_physical_s','retry_physical_s','policy_compute_s','policy_total_wall_s','policy_initial_wall_s','batch_wall_s')}
  return dict(recovery_counts=recovery,n=len(group),timing_complete_episodes=sum(r['timing_complete'] for r in group),batch_wall_timing_complete_episodes=sum(r['batch_wall_timing_complete'] for r in group),censored_timing_episodes=sum(r['wall_s_censored'] for r in group),incomplete_timing_episodes=sum(not r['timing_complete'] for r in group),observed_infrastructure_batch_wall_s=distribution(r['batch_wall_s'] for r in group if r['failure_class'].startswith('infrastructure_') and r['batch_wall_observed_exact']),observed_all_physical_s=distribution(r['physical_s'] for r in group),incomplete_observed_physical_lower_bound_s=distribution(r['physical_s'] for r in group if not r['timing_complete']),successes=sum(r['success'] for r in group),legacy_strict_successes=sum(r['legacy_strict_success'] for r in group),first_successes=sum(r['first_success'] for r in group),second_added=sum(r['success'] and r['first_successful_attempt']==2 for r in group),third_added=sum(r['success'] and r['first_successful_attempt']==3 for r in group),attempt_distribution={str(i):counts[i] for i in range(4)},failure_classes=dict(failure),actual_grasps=distribution(r['actual_grasps'] for r in group),successful_actual_grasps=distribution(r['actual_grasps'] for r in group if r['success']),first_actual_fingers={str(i):sum(bool(r['closures']) and r['closures'][0]['contact_count']==i for r in group) for i in (2,3)},attempt_fingers={str(i):sum(r['finger_counts'].get(str(i),0) for r in group) for i in (2,3)},final_success_fingers={str(i):sum(r['success'] and bool(r['closures']) and r['closures'][-1]['contact_count']==i for r in group) for i in (2,3)},transitions={f'{a}->{b}':sum(t['from_count']==a and t['to_count']==b for t in transitions) for a in (2,3) for b in (2,3)},large_gesture_changes=sum(t['large_gesture'] for t in transitions),hand_switches=sum(t['hand_switch'] for t in transitions),large_gesture_and_hand_switch=sum(t['large_gesture'] and t['hand_switch'] for t in transitions),incomplete_gesture_evidence=sum(not t['evidence_complete'] for t in transitions),timing_s=timing,phase_physical_s={name:distribution(r['phase_physical_s'].get(name,0.) for r in group if r['timing_complete']) for name in phases},observed_incomplete_phase_physical_s={name:distribution(r['phase_physical_s'].get(name,0.) for r in group if not r['timing_complete']) for name in phases},selection_scope=dict(collections.Counter(e['data'].get('selection_scope','unknown') for r in group for e in r['selection_events'])))
 conditions={c:group_stats([r for r in results if r['condition']==c]) for c in ('clean','noise')}
 summary=dict(episodes=len(results),successes=sum(r['success'] for r in results),first_successes=sum(r['first_success'] for r in results),rescues=sum(r['rescued'] for r in results),actual_grasps=sum(r['actual_grasps'] for r in results),overall=group_stats(results),branch_counts={b:{k:sum(r['branch_counts'][b][k] for r in results) for k in ('considered','selected','executed','rescued','guard_refused','conditional_fallback_considered')} for b in ('preload','local_pose','release_resample')},pairs=pairs,clean_success_noise_failure=[p['object_id'] for p in pairs if p['clean_success'] and not p['noise_success']],clean_failure_noise_success=[p['object_id'] for p in pairs if not p['clean_success'] and p['noise_success']],paired_state_mismatches=[p['object_id'] for p in pairs if p['initial_state_equal'] is False],paired_overhead_mismatches=[p['object_id'] for p in pairs if p['initial_overhead_equal'] is False],source_commits=sorted(set(str(r['source_commit']) for r in results)),conditions=conditions,results=[{k:v for k,v in r.items() if k not in ('selection_events','recovery_events','closures','initial_state')} for r in results])
 contract=load(root/'contract.json')
 summary['planned_episodes']=contract.get('planned')
 summary['paired_state_unavailable']=[p['object_id'] for p in pairs if p['initial_state_equal'] is None]
 summary['paired_overhead_unavailable']=[p['object_id'] for p in pairs if p['initial_overhead_equal'] is None]
 summary['pair_input_audit_complete']=bool(len(pairs)==40 and not summary['paired_state_unavailable'] and not summary['paired_overhead_unavailable'] and not summary['paired_state_mismatches'] and not summary['paired_overhead_mismatches'])
 summary['completion_scope']='all80 preregistered receipts returned; infrastructure failures retained; complete_physical_evidence separately requires all episodes completed and integrity valid'
 summary['fully_completed_episodes']=sum(r['controller_completed'] for r in results)
 summary['infrastructure_episodes']=sum(r['failure_class'].startswith('infrastructure_') for r in results)
 summary['integrity_valid_episodes']=sum(r['integrity_valid'] for r in results)
 summary['complete_physical_evidence']=len(results)==80 and all(r['controller_completed'] and r['integrity_valid'] for r in results)
 summary['timing_scope']=dict(completed_episode_metrics='completed episodes with a host result and no infrastructure error; censored scheduler receipts do not remove exact physical or policy timing',batch_wall_s='completed episodes with an exact non-censored scheduler duration',incomplete_observed_physical_lower_bound_s='recorded steps / 120 Hz for incomplete/error episodes, reported separately from completed-episode timing means',observed_infrastructure_batch_wall_s='exact observed scheduler duration of infrastructure failures; not a completed-episode duration')
 summary['complete']=bool(contract.get('planned')==80 and len(results)==80 and len(pairs)==40 and not summary['paired_state_mismatches'] and not summary['paired_overhead_mismatches'] and len(summary['source_commits'])==1 and all(r['condition'] in ('clean','noise') and r['object_id'] and r['actual_grasps']<=3 for r in results))
 (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
 keys=('name','object_id','condition','source_commit','controller_sha256','profile','fingers_mode','max_actual_grasps','recording_mode','initial_dx_m','initial_dy_m','initial_yaw_rad','noise_seed','rgb_noise_std','paired_initial_state_sha256','success','legacy_strict_success','first_success','first_successful_attempt','rescued','actual_grasps','failure_class','physical_s','first_trial_physical_s','retry_physical_s','policy_compute_s','policy_total_wall_s','policy_initial_wall_s','batch_wall_s','frame_count','frame_stride_4','controller_completed','integrity_valid','result_present','timing_complete','batch_wall_timing_complete','batch_wall_observed_exact','wall_s_censored','physical_time_scope','observed_physical_lower_bound_s','paired_initial_state_available','initial_overhead_sha256','error')
 with (output/'episodes.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r.get(k) for k in keys} for r in results)
 with (output/'transitions.csv').open('w') as f:
  keys=('object_id','condition','source_commit','attempt_from','attempt_to','from_count','to_count','wrist_attitude_delta_rad','common_finger_mean_landing_delta_m','common_finger_count','large_gesture','hand_switch','evidence_complete','rescued_on_next_attempt')
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
  for r in results:
   for t in r['transitions']:
    data=dict(object_id=r['object_id'],condition=r['condition'],source_commit=r['source_commit'],**t,rescued_on_next_attempt=bool(r['success'] and not r['first_success'] and t['attempt_to']==r['actual_grasps']))
    writer.writerow({k:data.get(k) for k in keys})
 print(json.dumps({k:v for k,v in summary.items() if k not in ('results','pairs')},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit');a=p.parse_args();aggregate(a.root,a.output,a.source_commit)
