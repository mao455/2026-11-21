"""Read-only timeout evidence audit; writes only ignored audit artifacts."""
import collections
import hashlib
import importlib.util
import json
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[3]
ROOT = LAB / 'outputs/three_regrasp_20261122_01'
RUN = ROOT / 'formal_runs/02_abc_76395_clean'
EP = RUN / 'episode'
AUDIT = ROOT / 'audit'

def load(path):
    return json.loads(path.read_text()) if path.exists() else {}

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def scalar_fields(value):
    return {k: v for k, v in value.items() if v is None or isinstance(v, (int, float, str, bool))}

receipt = load(RUN / 'batch_result.json')
# Bind the immutable batch-end snapshot: final delivery legitimately enriches the live compact with cohort metadata.
COMPACT_SNAPSHOT = AUDIT / 'timeout_02_abc76395_batch_end_compact_v4.json'
compact = load(COMPACT_SNAPSHOT)
steps = rows(EP / 'steps.jsonl')
trace = rows(EP / 'agent_trace.jsonl')
frames = rows(EP / 'image_manifest.jsonl')
responses = rows(EP / 'public_responses.jsonl')
commands = []
position = 0
for row in trace:
    if row.get('op') == 'step':
        commands.append(dict(row, start_step=position, end_step=position + row['steps']))
        position += row['steps']
events = compact['recovery_events']
checks = {}
checks['timeout_is_retained_failure'] = receipt.get('returncode') == 124 and receipt.get('error') == 'wall_timeout' and receipt.get('success') is False
checks['no_completed_host'] = not (EP / 'result.json').exists() and not receipt['controller_completed'] and not compact['controller_completed']
checks['no_invented_success_or_rescue'] = all(compact[k] is False for k in ('success', 'first_success', 'rescued')) and compact['first_successful_attempt'] is None
checks['source_prefix_hashes_match_final_files'] = all(sha(EP / f) == h for f, h in compact['source_sha256'].items())
checks['persisted_steps_are_contiguous'] = [r['step'] for r in steps] == list(range(len(steps)))
checks['all_persisted_physics_timestamps_match_120hz'] = all(abs(r['time_s'] - (i + 1) / 120) < 1e-9 for i, r in enumerate(steps))

chain = []
for attempt, closure in enumerate(compact['closures'], 1):
    start = closure['start_step']
    ack = closure['dispatch_event']
    command = next(c for c in commands if c['start_step'] == start)
    public = next((r['observation'] for r in responses if r.get('observation', {}).get('step') == command['end_step']), None)
    physical = [r for r in steps if start <= r['step'] < command['end_step']]
    passed = len(physical) == 6 and all(r['phase'] == 's5_close' for r in physical) and ack['started_step'] == start and ack['step'] == command['end_step'] and command['target_q'] == ack['target_q'] and public is not None
    checks[f'attempt_{attempt}_command_physics_and_ack'] = passed
    chain.append(dict(attempt=attempt, start_step=start, ack_step=ack['step'], branch=closure['branch'], finger_count=closure['contact_count'], candidate_id=ack['candidate_id'], command_plan_sha256=ack['command_plan_sha256'], planned_q=ack['planned_q'], command_target_q=command['target_q'], first_persisted_motor_target_q=physical[0]['target_q'], first_persisted_actual_q=physical[0]['actual_q'], last_ack_actual_q=public['actual_q'], command_elapsed_s=command['elapsed_s'], physical_phase_steps=len(physical), matched_public_ack=True, evidence_passed=passed))

checks['exactly_three_physical_closures_retained'] = len(chain) == compact['actual_grasps'] == receipt['actual_grasps'] == 3
checks['all_actual_attempts_are_twofinger'] = all(c['finger_count'] == 2 for c in chain) and receipt['actual_finger_sequence'] == [2, 2, 2]
checks['two_real_transitions_without_hand_switch'] = len(compact['transitions']) == 2 and all(t['from_count'] == t['to_count'] == 2 and not t['hand_switch'] for t in compact['transitions'])

last_public = [r['observation'] for r in responses if r.get('observation')][-1]
last_capture = frames[-1]
sys.path.insert(0, str(ROOT / 'reproduction'))
from timing_readout import episode as timing_episode
timing = timing_episode(RUN)
checks['full_timing_is_explicitly_unavailable'] = timing.get('valid') is False and 'missing completed' in timing.get('error', '')
checks['no_final_rpc_or_backend_profile'] = not (EP / 'rpc_timing.json').exists() and not (EP / 'wall_profile_backend.json').exists()

partial = load(AUDIT / 'final80_partial_first5_v4.json')
timeout_validation = next(r for r in partial['episodes'] if r['name'] == RUN.name)
checks['first5_partial_validator_passed'] = partial['episodes_audited'] == 5 and partial['all_available_checks_passed'] and not partial['failures']
checks['partial_never_claims_final_complete'] = partial['final_complete'] is False and partial['all80_physical_input_pairing_verified'] is False
checks['timeout_validator_retains_attempts_and_failure'] = timeout_validation['infrastructure'] is True and timeout_validation['success'] is False and timeout_validation['actual_grasps'] == 3

compact_events = []
for entry in events:
    d = entry['data']
    event = dict(scalar_fields(d), elapsed_s=entry['elapsed_s'])
    for key in ('decision', 'audit', 'close_audit'):
        if key in d:
            event[key] = scalar_fields(d[key])
    if 'new_RGB' in d:
        event['new_RGB'] = {k: d['new_RGB'].get(k) for k in ('method', 'step', 'valid', 'usable_views', 'occupied_voxels', 'compute_s')}
    compact_events.append(event)

phase_steps = collections.Counter(r['phase'] for r in steps)
cutoff = compact['first_trial_end_step']
phase_bands = {'first_grasp': {}, 'release_retreat': {}, 'regrasp': {}}
for row in steps:
    stage = 'first_grasp' if row['step'] < cutoff else 'release_retreat' if row['phase'].startswith(('s3_', 's7_release')) or 'release' in row['phase'] or 'retreat' in row['phase'] else 'regrasp'
    phase_bands[stage][row['phase']] = phase_bands[stage].get(row['phase'], 0) + 1

latency_path = AUDIT / 'latency_diagnostic_02_abc76395_v4.json'
latency = load(latency_path)
latency_bounds = {k: latency[k] for k in ('observed_unix', 'sources', 'same_rpc_capture_gap_top10', 'recent20_camera_capture_function_duration_s')}
out = dict(schema=1, scope='Final read-only audit of terminated formal V4 episode. Never edits simulator, policy, receipt, compact or verdict.', name=RUN.name, source_commit=receipt['source_commit'], controller_sha256=receipt['controller_sha256'], success=False, failure_class='infrastructure_timeout', returncode=124, integrity_valid=False, completed=False, actual_grasps=3, actual_finger_sequence=[2, 2, 2], physical_closure_chain=chain, recovery_counts=compact['branch_counts'], recovery_chain=compact_events, transitions=compact['transitions'], timing=dict(scheduler_observed_exact_wall_s=receipt['wall_s'], scheduler_scope='Observed launch-to-timeout-cleanup wall, not a completed episode latency.', persisted_contiguous_physics_steps=len(steps), persisted_contiguous_physics_lower_bound_s=len(steps)/120, last_public_ack_step=last_public['step'], last_public_ack_physics_time_s=last_public['time_s'], trace_step_requests=len(commands), trace_requested_steps=position, requested_nominal_physical_s=position/120, receipt_physical_s=receipt['physical_s'], requested_time_scope='Sum of submitted step requests; final request has no public acknowledgement. Not completed physical time.', last_capture_step=last_capture['step'], last_capture_frame=last_capture['frame_id'], persisted_phase_lower_bound_s={k: v/120 for k, v in phase_steps.items()}, persisted_stage_lower_bound_s={stage: sum(phases.values())/120 for stage, phases in phase_bands.items()}, full_stage_backend_wall_s=None, full_stage_policy_nonrpc_wall_s=None, policy_compute_s=None, unavailable_reason='No completed host result, rpc_timing.json, wall_profile_backend.json or final s9_io_audit. Compact policy_compute_s=0 is an empty accumulator, not measured zero computation.', completed_timing_statistics_eligible=False, observed_infrastructure_wall_statistics_eligible=True, persistent_evidence_limit='Different streams end at different steps. The cause of the difference is not established; no causal claim of buffer loss.', timing_reader_result=timing), latency_interim_bounds=latency_bounds, partial_validator=dict(path=str((AUDIT/'final80_partial_first5_v4.json').relative_to(ROOT)), sha256=sha(AUDIT/'final80_partial_first5_v4.json'), episodes_audited=partial['episodes_audited'], checks_passed=partial['all_available_checks_passed'], final_complete=partial['final_complete'], frames_checked=partial['frame_files_checked'], timeout_episode=timeout_validation), checks=checks, all_checks_passed=all(checks.values()), source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [RUN/'batch_result.json', COMPACT_SNAPSHOT, EP/'scene.json', EP/'steps.jsonl', EP/'agent_trace.jsonl', EP/'public_responses.jsonl', EP/'image_manifest.jsonl', latency_path, pathlib.Path(__file__)]})
(AUDIT/'timeout_chain_02_abc76395_v4.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
assert out['all_checks_passed'], checks
print(json.dumps(dict(output=str(AUDIT/'timeout_chain_02_abc76395_v4.json'), checks=len(checks), all_checks_passed=out['all_checks_passed'], actual_grasps=3, persisted_physics_lower_bound_s=len(steps)/120, last_public_ack_physics_s=last_public['time_s'], requested_nominal_physical_s=position/120, full_timing_available=False)))
