"""Reconstruct recorded monotonic timing; never infer unavailable hardware timings."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def episode(ep, outer=None):
    result = load(ep/'result.json')
    profile = load(ep/'wall_profile_backend.json')
    rpc = load(ep/'rpc_timing.json')
    marks = profile['marks']
    trace = list(lines(ep/'agent_trace.jsonl'))
    commands = [r for r in trace if r.get('op') == 'step']
    steps = [r for r in rpc if r['op'] == 'step']
    assert len(commands) == len(steps)
    io = next((json.loads(r['text'])['s9_io_audit'] for r in trace
               if r.get('op') == 'note' and 's9_io_audit' in json.loads(r['text'])), {})
    first_frame = next(lines(ep/'image_manifest.jsonl'))
    start, ready, finish = (marks[k] for k in ('start', 'ready', 'finish_request'))
    first_command = steps[0]['start_wall_s']
    last_command_return = steps[-1]['start_wall_s'] + steps[-1]['inclusive_s']
    outer = result['elapsed_s'] if outer is None else outer
    categories = profile['exclusive_categories']
    cat_sum = sum(v['exclusive_s'] for v in categories.values())
    physics = result['physics']
    cycle = io.get('cycles', [])
    intervals = [b['start_wall_s']-a['start_wall_s'] for a,b in zip(steps,steps[1:])]
    last_hold_tick = physics.get('hold_end_step')
    cumulative = 0
    hold_rpc = None
    for command, call in zip(commands, steps):
        cumulative += command['steps']
        if last_hold_tick is not None and cumulative > last_hold_tick:
            hold_rpc = call['start_wall_s'] + call['inclusive_s']
            break
    phase_rows = {}
    for i,(command,call) in enumerate(zip(commands,steps)):
        p = phase_rows.setdefault(command['phase'], dict(commands=0,physical_s=0.,rpc_s=0.,between_command_wall_s=0.))
        p['commands'] += 1
        p['physical_s'] += command['steps']/120
        p['rpc_s'] += call['inclusive_s']
        if i+1 < len(steps):p['between_command_wall_s'] += steps[i+1]['start_wall_s']-call['start_wall_s']
    decomposition = {k:v['exclusive_s'] for k,v in categories.items()}
    decomposition['backend_uninstrumented_and_controller_wait_s'] = profile['total_s']-cat_sum
    decomposition['outside_backend_profile_s'] = outer-profile['total_s']
    lifecycle = load(ep/'host_backend_lifecycle.json')
    return dict(episode=str(ep),controller_sha256=result['controller_sha256'],success=result['success'],
        strict_success=result.get('legacy_strict_success'),physical_s=cumulative/120,outer_wall_s=outer,
        scope='simulation; task dispatch absolute monotonic timestamp and process-exit timestamp were not logged',
        marks_monotonic_s=dict(backend_start=start,first_exposure_start=first_frame['capture_start_wall_s'],
            first_exposure_done=first_frame['capture_done_wall_s'],first_frame_available=first_frame['available_wall_s'],
            backend_ready=ready,first_command=first_command,last_command_return=last_command_return,
            final_hold_observed_by_rpc_return=hold_rpc,finish_request=finish,
            task_dispatch=None,offline_evaluation_done=None,process_exit=None),
        ready_to_last_command_s=last_command_return-ready,
        first_frame_to_last_command_s=last_command_return-first_frame['available_wall_s'],
        backend_initialization_s=ready-start,initial_planning_s=first_command-ready,
        command_execution_and_replanning_s=last_command_return-first_command,
        post_last_command_to_finish_s=finish-last_command_return,
        backend_after_finish_s=profile['total_s']-(finish-start),
        lifecycle=lifecycle,outer_partition=decomposition,
        categories={k:dict(**v,mean_call_ms=1000*v['exclusive_s']/v['calls']) for k,v in categories.items()},
        policy_exclusive={k:dict(n=len(v),sum_s=sum(v),median_call_ms=statistics.median(v)*1000)
                          for k,v in io.get('component_samples_s',{}).items()},
        dispatch=dict(n=len(steps),interval_count=len(intervals),interval_over50ms=sum(t>.050 for t in intervals),
            interval_max_s=max(intervals,default=0),compute_cycles=len(cycle),
            compute_over50ms=sum(r['compute_between_dispatch_s']>.050 for r in cycle),
            compute_max_s=max((r['compute_between_dispatch_s'] for r in cycle),default=0),
            includes_initial_planning=True,simulation_not_realtime_scheduler=True),
        phases=phase_rows,
        command_sha256=digest([{k:r[k] for k in ('target_q','steps','phase')} for r in commands]),
        state_sha256=digest([{k:r.get(k) for k in ('actual_q','object_pose')} for r in lines(ep/'steps.jsonl')]),
        source_sha256={n:hashlib.sha256((ep/n).read_bytes()).hexdigest() for n in
            ('result.json','wall_profile_backend.json','rpc_timing.json','agent_trace.jsonl','image_manifest.jsonl')})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipts',type=Path,default=HERE/'stage14_evidence/registration/timing_ab_plan.receipts.jsonl')
    p.add_argument('--acceptance',type=Path,default=HERE/'stage15_evidence/timing/final_acceptance/timing.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for receipt in lines(a.receipts):
        if not receipt['name'].startswith('01_'):continue
        command=receipt['command'];ep=Path(command[command.index('--session')+1])/'episode'
        r=episode(ep,receipt['elapsed_wall_s']);r.update(name=receipt['name'],mode='reference' if 'reference' in receipt['name'] else 'optimized',command=command,env=receipt['env'])
        records.append(r)
    decomposition={}
    for category in records[0]['outer_partition']:
        values={m:statistics.median(r['outer_partition'][category] for r in records if r['mode']==m) for m in ('reference','optimized')}
        decomposition[category]={**values,'saved_s':values['reference']-values['optimized']}
    groups={m:dict(n=2,median_outer_s=statistics.median(r['outer_wall_s'] for r in records if r['mode']==m)) for m in ('reference','optimized')}
    saved=groups['reference']['median_outer_s']-groups['optimized']['median_outer_s']
    acceptance=[episode(Path(r['episode'])) for r in load(a.acceptance)['rows']]
    payload=dict(stage14_object01=records,groups=groups,saved_s=saved,savings_partition=decomposition,
        closure_error_s=saved-sum(x['saved_s'] for x in decomposition.values()),
        command_hashes_equal=len({r['command_sha256'] for r in records})==1,
        state_hashes_equal=len({r['state_sha256'] for r in records})==1,
        stage15_acceptance=acceptance,
        forty_seconds=dict(denominator=len(acceptance),official_success=sum(r['success'] for r in acceptance),
            simulation_ready_to_final_return_success_within40=sum(r['success'] and r['ready_to_last_command_s']<=40 for r in acceptance),
            physical_duration_success_within40=sum(r['success'] and r['physical_s']<=40 for r in acceptance),
            real_hardware_measured=0),
        limitations=['n=2 per mode; additive medians valid here because median of two equals arithmetic mean',
            'Categories are exclusive backend call timers. Policy functions and RPC are nested/overlapping and are NOT added.',
            'backend_uninstrumented_and_controller_wait_s combines startup, Python, controller and IPC idle; no unsupported causal attribution.',
            'No archival timestamps for task dispatch, completed offline evaluator, or exact process exit. Receipt elapsed gives launch-to-exit only.',
            'Physical-duration <=40 is a kinematic necessary condition, not hardware end-to-end acceptance.'])
    (a.output/'timing_audit.json').write_text(json.dumps(payload,indent=2))
    for name,rows in [('savings', [dict(category=k,**v) for k,v in decomposition.items()]),
        ('episodes',[{k:r[k] for k in ('episode','success','physical_s','outer_wall_s','ready_to_last_command_s','initial_planning_s')} for r in records+acceptance])]:
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(groups=groups,saved_s=saved,partition=decomposition,forty_seconds=payload['forty_seconds']),indent=2))

if __name__=='__main__':main()
