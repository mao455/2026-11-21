"""Independent raw-trajectory readout; preserves whole-episode and attempt verdicts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from evaluator import evaluate


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center-radius, center+radius]


def read(ep):
    host = json.loads((ep/'result.json').read_text())
    scene = json.loads((ep/'scene.json').read_text())
    rows = [json.loads(line) for line in (ep/'steps.jsonl').open() if line.strip()]
    score = evaluate(rows, scene['initial_object_pose'][2])
    events = []
    for line in (ep/'agent_trace.jsonl').open():
        item = json.loads(line)
        if item.get('op') == 'note':
            value = json.loads(item['text']).get('s13_recovery')
            if value:
                events.append(value)
    dispatched = [e for e in events if e['event'] == 'grasp_dispatched']
    decisions = [e for e in events if e['event'] == 'failure_decision']
    attempts = []
    previous = 0
    for decision in decisions:
        end = min(int(decision['step']), len(rows))
        prefix = evaluate(rows[:end], scene['initial_object_pose'][2])
        attempts.append(dict(actual_grasps=decision.get('actual_grasps'),
            decision_step=end, decision=decision.get('decision'),
            terminal_window_lift_passed=prefix.get('hold_physics_passed', False),
            prefix_official_success=prefix['success'],
            prefix_geometry_passed=prefix.get('geometry_passed'),
            geometry_since_previous_decision=all(r['penetration']['passed'] for r in rows[previous:end]),
            lift_m=prefix.get('hold_min_lift_m'), clearance_m=prefix.get('hold_min_table_clearance_m'),
            hold_duration_s=prefix.get('hold_duration_s'),
            note='Terminal 2 s at decision time, with original initial height; whole-prefix geometry retained.'))
        previous = end
    valid = bool(host.get('controller_completed') and host.get('error') is None
        and host.get('evaluator_and_adapter_unchanged') and host['physics'] == score
        and sha(ep/'program/controller.py') == host.get('controller_sha256'))
    first_success = bool(attempts and attempts[0]['prefix_official_success'])
    actual = len(dispatched)
    considered = [e for e in events if e['event'] == 'considered']
    executed = [e for e in events if e['event'] == 'executed']
    failure = host.get('error') or score.get('failure') or ('success' if score['success'] else 'physical_failure')
    raw = {name: dict(path=str(ep/name), sha256=sha(ep/name)) for name in
        ('result.json','scene.json','steps.jsonl','agent_trace.jsonl','public_responses.jsonl',
         'image_manifest.jsonl','all_hand_object_contacts.jsonl') if (ep/name).is_file()}
    videos = [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(ep.rglob('*.mp4'))]
    return dict(episode=str(ep.resolve()), execution_valid=valid,
        controller_sha256=host.get('controller_sha256'), case=scene.get('case'),
        success=bool(valid and score['success']), legacy_strict_success=bool(valid and score['legacy_strict']['success']),
        first_grasp_success=bool(valid and first_success), actual_grasps=actual,
        first_failed_entered_recovery=bool(not first_success and considered),
        recovery_actions_dispatched=len(executed), subsequent_grasp_dispatched=actual > 1,
        recovered_after_first_failure=bool(valid and not first_success and actual > 1 and score['success']),
        geometry_passed=score.get('geometry_passed'), first_invalid_step=score.get('first_invalid_step'),
        final_single_lift_passed=score.get('hold_physics_passed',False), failure=failure,
        physical_steps=len(rows), physical_s=len(rows)/120,
        host_elapsed_s=host.get('elapsed_s'), attempts=attempts,
        compact_events=[{k:e[k] for k in ('event','step','frame_id','actual_grasps','branch',
            'recovery_id','reason','candidate_id') if k in e} for e in events
            if e['event'] in ('grasp_dispatched','considered','guard_refused','executed','stopped','preload_reproof')],
        physics=score, raw_evidence=raw, videos=videos)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', action='append', type=Path, default=[])
    p.add_argument('--batch', type=Path)
    p.add_argument('--runs-root', action='append', type=Path, default=[])
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    episodes=list(a.episode)
    if a.batch:
        episodes.extend(Path(r['session'])/'episode' for r in json.loads((a.batch/'summary.json').read_text())['rows']
                        if r['completed'])
    for root in a.runs_root:
        episodes.extend(path.parent for path in sorted(root.rglob('result.json'))
                        if path.parent.name == 'episode')
    episodes=list(dict.fromkeys(ep.resolve() for ep in episodes))
    rows=[read(ep) for ep in episodes]
    counters=('success','legacy_strict_success','first_grasp_success','first_failed_entered_recovery',
        'subsequent_grasp_dispatched','recovered_after_first_failure','geometry_passed','final_single_lift_passed')
    totals={name:sum(bool(row[name]) for row in rows) for name in counters}
    result=dict(episodes=len(rows), valid=sum(r['execution_valid'] for r in rows), counts=totals,
        success_wilson95=wilson(totals['success'],len(rows)),
        uncertainty_note='Descriptive small-sample interval only. Paired noise conditions and repeated objects are dependent; no population generalization claim.',
        total_actual_grasps=sum(r['actual_grasps'] for r in rows),
        evaluator_sha256=sha(Path(__file__).with_name('evaluator.py')), rows=rows)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'readout.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=('episode','execution_valid',*counters,'actual_grasps','recovery_actions_dispatched',
            'first_invalid_step','failure','physical_steps','physical_s','host_elapsed_s')
    with (a.output/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        writer.writerows({key:r[key] for key in fields} for r in rows)
    (a.output/'video_index.json').write_text(json.dumps([dict(episode=r['episode'],videos=r['videos']) for r in rows],indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='rows'}))


if __name__=='__main__':
    main()
