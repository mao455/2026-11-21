"""Offline verification of immutable normal-entry episodes; never controller input."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import sys

E = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(E))
from evaluator import reassess_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(row):
    session = Path(row['session'])
    episode = session / 'episode'
    verdict = reassess_episode(episode)
    receipt = json.loads((session / 'budget_receipt.json').read_text())
    closures, recovery, selected, refusals, stops = [], [], [], Counter(), []
    phases = Counter()
    closure_actions, closure_stops, support_decisions = [], [], []
    trace_elapsed_s = 0.
    planning_cpu_max_s = 0.
    for line in (episode / 'agent_trace.jsonl').open():
        event = json.loads(line)
        trace_elapsed_s = max(trace_elapsed_s, float(event.get('elapsed_s', 0.)))
        if event.get('op') == 'step':
            phases[event.get('phase', 'unspecified')] += int(event.get('steps', 0))
        if event.get('op') != 'note':
            continue
        note = json.loads(event['text'])
        planning_cpu_max_s = max(planning_cpu_max_s, float(note.get('planning_progress', {}).get('process_cpu_s', 0.)))
        if 's5_closure_action' in note:
            item=note['s5_closure_action']
            closure_actions.append({k:item[k] for k in ('step','round','mode','reason','squeeze_m','contact_proxy_at_m','settled_blocked_streak','unsettled_tracking') if k in item})
        if 's5_closure_geometry_stop' in note:
            closure_stops.append(note['s5_closure_geometry_stop'])
        for phase in ('proof','middle','final'):
            key='s5_'+phase+'_decision'
            if key in note:
                support_decisions.append(dict(note[key], phase=phase))
        if 's13_recovery' in note:
            item = note['s13_recovery']
            recovery.append({k: item[k] for k in ('event', 'step', 'actual_grasps', 'decision', 'reason') if k in item})
            if item.get('event') == 'grasp_dispatched':
                closures.append(item)
        for key in ('s5_measured_stop', 's5_stop', 's5_plan_failure'):
            if key in note:
                stops.append({key: note[key]})
        plan = note.get('s5_plan', {})
        for rejected in plan.get('rejected', []):
            refusals[str(rejected.get('first_rejection', rejected.get('reason', 'unspecified')))] += 1
        if plan.get('accepted'):
            selected.append({k: plan[k] for k in ('attempt', 'step', 'accepted', 'candidate_id', 'selected', 'candidates') if k in plan})
    # Native contact and object state remain offline evidence only.
    hold = deque(maxlen=240)
    valid, continuous, previous, steps = True, True, None, 0
    by_attempt = {}
    initial_z = json.loads((episode / 'scene.json').read_text())['initial_object_pose'][2]
    starts = sorted((int(c.get('started_step', c['step'])), int(c['actual_grasps'])) for c in closures)
    for line in (episode / 'steps.jsonl').open():
        record = json.loads(line)
        valid &= bool(record['penetration']['passed'])
        continuous &= record['step'] == steps and abs(record['time_s'] - (steps + 1) / 120) < 1e-9
        steps += 1
        hold.append(record)
        attempt = max((a for start, a in starts if record['step'] >= start), default=0)
        if not attempt:
            continue
        data = by_attempt.setdefault(attempt, dict(steps=0, contact_steps=0, max_lift_m=-1., contact_links=Counter()))
        data['steps'] += 1
        data['max_lift_m'] = max(data['max_lift_m'], record['object_pose'][2] - initial_z)
        touching = [link for link, contact in record['contacts'].items() if contact['normal_force_sum_n'] > 0]
        data['contact_steps'] += bool(touching)
        data['contact_links'].update(touching)
    hold_links = Counter()
    for record in hold:
        hold_links.update(link for link, contact in record['contacts'].items() if contact['normal_force_sum_n'] > 0)
    actual = max((int(item.get('actual_grasps', 0)) for item in recovery), default=0)
    assert actual == len(closures), (session, actual, len(closures))
    assert verdict['success'] == verdict['recorded_success']
    assert receipt['source_unchanged'] and not receipt['leftover_descendants']
    assert actual <= 3
    return dict(hand=row['hand'], case=row['case'], session=str(session),
                source_unchanged=receipt['source_unchanged'], budget_receipt_sha256=sha(session/'budget_receipt.json'),
                actual_closures=actual, first_grasp_success=bool(verdict['success'] and actual == 1),
                within_three_success=verdict['success'], success_on_closure=actual if verdict['success'] else None,
                physical_steps=steps, geometry_valid_every_step=valid, continuous_120hz=continuous,
                physics=verdict['physics'], source_sha256=verdict['source_sha256'],
                closure_dispatches=[{k:c[k]for k in ('actual_grasps','started_step','step','candidate_id','branch','points','command_plan_sha256','recovery_id')if k in c}for c in closures],
                native_offline_by_attempt=by_attempt, hold_contact_steps_by_link=hold_links,
                endpoint_refusals=dict(refusals), recovery=recovery, stops=stops,
                trace_elapsed_s=trace_elapsed_s, planning_cpu_max_s=planning_cpu_max_s,
                requested_simulation_s_by_phase={k:v/120 for k,v in phases.items()},
                closure_actions=closure_actions, closure_geometry_stops=closure_stops,
                support_decisions=support_decisions, selected_plans=selected,
                controller_completed=json.loads((episode/'result.json').read_text()).get('controller_completed'),
                controller_error=json.loads((episode/'result.json').read_text()).get('error'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    if not args.partial:
        assert all(r['status'] == 'terminal' for r in state['rows'])
    rows = []
    for row in state['rows']:
        if row['status'] == 'terminal':
            rows.append(audit(row))
            print(json.dumps({k:rows[-1][k]for k in ('hand','case','actual_closures','first_grasp_success','within_three_success')}), flush=True)
    report = dict(source_commit=state['source_commit'], state_sha256=sha(args.state),
                  original_evaluator_sha256=sha(E/'evaluator.py'), offline_only=True, rows=rows)
    report['by_hand'] = {hand:dict(cases=sum(r['hand']==hand for r in rows),
        zero_closures=sum(r['hand']==hand and r['actual_closures']==0 for r in rows),
        first_successes=sum(r['hand']==hand and r['first_grasp_success'] for r in rows),
        within_three_successes=sum(r['hand']==hand and r['within_three_success'] for r in rows),
        second_grasp_rescues=sum(r['hand']==hand and r['success_on_closure']==2 for r in rows),
        third_grasp_rescues=sum(r['hand']==hand and r['success_on_closure']==3 for r in rows))for hand in sorted({r['hand']for r in rows})}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
