"""Measured 30 mm no-drop verdict, with the historical strict verdict retained.

Private rollout evidence only; no controller phase labels affect scoring. Run this
module on recorded episode directories to reassess without changing the originals.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

DT = 1 / 120
HOLD_STEPS = 240
PROTOCOL_ID = 'lift30_no_drop_v1'
TARGET_LIFT_M = .030
LIFT_TOLERANCE_M = .002  # Evaluation convention, not a hardware calibration.
MIN_COMMANDED_LIFT_M = .0295
MIN_CLEARANCE_M = .005


def evaluate_strict(records, initial_object_z):
    """Original strict scorer; thresholds and returned fields are unchanged."""
    base = {'success': False, 'steps': len(records), 'required_hold_steps': HOLD_STEPS,
            'geometry_scope': 'authored USD collision meshes; not internal cooked convex pieces',
            'protocol': 'final continuous 2 s, actual lift >=10mm, clearance >=5mm, stable relative pose, contact; geometry valid throughout'}
    if not records:
        return {**base, 'failure': 'no_physical_steps'}
    invalid = next((r for r in records if not r['penetration']['passed']), None)
    base.update(geometry_passed=invalid is None,
                first_invalid_step=None if invalid is None else invalid['step'])
    if len(records) < HOLD_STEPS:
        return {**base, 'failure': 'insufficient_hold_duration'}
    hold = records[-HOLD_STEPS:]
    relative = np.asarray([r['hand_object_relative_matrix'] for r in hold])
    drift = np.linalg.norm(relative[:, :3, 3] - relative[0, :3, 3], axis=1)
    rotations = relative[0, :3, :3].T @ relative[:, :3, :3]
    rotation = Rotation.from_matrix(rotations).magnitude()
    clear = min(r['penetration']['object_min_z'] for r in hold)
    lift = min(r['object_pose'][2] - initial_object_z for r in hold)
    linear = max(np.linalg.norm(r['object_velocity'][:3]) for r in hold)
    angular = max(np.linalg.norm(r['object_velocity'][3:]) for r in hold)
    contact = min(sum(max(0., c['normal_force_sum_n']) for c in r['contacts'].values()) for r in hold)
    # Commanded lift is checked against preceding command heights; actual lift remains mandatory.
    before = records[:-HOLD_STEPS]
    commanded = min(r['target_q'][2] for r in hold) - min((r['target_q'][2] for r in before), default=hold[0]['target_q'][2])
    passed = bool(clear >= .005 and lift >= .01 and drift.max() <= .005 and rotation.max() <= .25
                  and linear <= .05 and angular <= 1. and contact >= .01 and commanded >= .0295)
    return {**base, 'hold_physics_passed': passed, 'hold_min_table_clearance_m': float(clear),
            'hold_min_lift_m': float(lift), 'hold_relative_translation_drift_m': float(drift.max()),
            'hold_relative_rotation_drift_rad': float(rotation.max()), 'hold_max_linear_speed_m_s': float(linear),
            'hold_max_angular_speed_rad_s': float(angular), 'hold_min_contact_force_n': float(contact),
            'commanded_lift_m': float(commanded), 'success': bool(passed and invalid is None)}


def evaluate(records, initial_object_z):
    strict = evaluate_strict(records, initial_object_z)
    # Preserve the historical metrics at the top level for existing diagnostics.
    result = {**strict, 'success': False, 'hold_physics_passed': False,
              'protocol_id': PROTOCOL_ID,
              'protocol': ('final continuous 2 s at commanded 30mm lift, actual '
                           'object lift >=28mm (2mm tolerance), clearance >=5mm; '
                           'geometry valid throughout; speed/rotation/contact diagnostic only'),
              'target_lift_m': TARGET_LIFT_M,
              'lift_tolerance_m': LIFT_TOLERANCE_M,
              'minimum_actual_lift_m': TARGET_LIFT_M - LIFT_TOLERANCE_M,
              'minimum_commanded_lift_m': MIN_COMMANDED_LIFT_M,
              'legacy_strict': strict}
    if len(records) < HOLD_STEPS:
        return result
    # A row count cannot establish a continuous physical-time hold if samples
    # were omitted or duplicated. The backend emits one row per 120 Hz step.
    steps = np.asarray([r.get('step', np.nan) for r in records], dtype=float)
    times = np.asarray([r.get('time_s', np.nan) for r in records], dtype=float)
    continuous = bool(np.array_equal(steps, np.arange(len(records)))
                      and np.isfinite(times).all()
                      and np.allclose(times, (steps + 1) * DT, rtol=0, atol=1e-9))
    result['continuous_physics_evidence'] = continuous
    if not continuous:
        return {**result, 'failure': 'noncontinuous_physics_evidence'}
    hold = records[-HOLD_STEPS:]
    lifts = np.asarray([r['object_pose'][2] - initial_object_z for r in hold])
    clearances = np.asarray([r['penetration']['object_min_z'] for r in hold])
    commands = np.asarray([r['target_q'][2] for r in records])
    finite = bool(np.isfinite(initial_object_z) and np.isfinite(lifts).all()
                  and np.isfinite(clearances).all() and np.isfinite(commands).all())
    if not finite:
        return {**result, 'failure': 'nonfinite_lift_evidence'}
    commanded = float(commands[-HOLD_STEPS:].min()
                      - commands[:-HOLD_STEPS].min()) if len(records) > HOLD_STEPS else 0.
    checks = dict(actual_lift=bool(lifts.min() + 1e-12 >= TARGET_LIFT_M - LIFT_TOLERANCE_M),
                  table_clearance=bool(clearances.min() >= MIN_CLEARANCE_M),
                  completed_commanded_lift=bool(commanded >= MIN_COMMANDED_LIFT_M))
    held = all(checks.values())
    result.update(hold_physics_passed=held, hold_checks=checks,
                  commanded_lift_m=commanded, hold_min_lift_m=float(lifts.min()),
                  hold_min_table_clearance_m=float(clearances.min()),
                  hold_start_step=int(steps[-HOLD_STEPS]), hold_end_step=int(steps[-1]),
                  hold_start_time_s=float(times[-HOLD_STEPS] - DT),
                  hold_end_time_s=float(times[-1]), hold_duration_s=HOLD_STEPS * DT,
                  success=bool(held and result['geometry_passed']))
    if not result['geometry_passed']:
        result['failure'] = 'invalid_episode_geometry'
    elif not held:
        result['failure'] = next(name for name, passed in checks.items() if not passed)
    return result


def reassess_episode(episode):
    """Read original physical evidence and host gates; never rewrite it."""
    import hashlib
    import json
    from pathlib import Path
    episode = Path(episode)
    paths = {name: episode / name for name in ('result.json', 'scene.json', 'steps.jsonl')}
    host = json.loads(paths['result.json'].read_text())
    scene = json.loads(paths['scene.json'].read_text())
    records = [json.loads(line) for line in paths['steps.jsonl'].read_text().splitlines() if line.strip()]
    physics = evaluate(records, scene['initial_object_pose'][2])
    execution = bool(host['controller_completed'] and host['error'] is None
                     and host['evaluator_and_adapter_unchanged'])
    historical = host['physics'].get('legacy_strict', host['physics'])
    return dict(name=episode.parent.name, episode=str(episode.resolve()),
                source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in paths.items()},
                physical_s=len(records) * DT, execution_valid=execution,
                recorded_success=host['success'],
                legacy_strict_success=bool(execution and physics['legacy_strict']['success']),
                legacy_strict_matches_recorded=physics['legacy_strict'] == historical,
                success=bool(execution and physics['success']), physics=physics)


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refuse to overwrite an existing reassessment')
    report = dict(protocol_id=PROTOCOL_ID, reassessment_only=True,
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  episodes=[reassess_episode(path) for path in args.episodes])
    report.update(cases=len(report['episodes']),
                  successes=sum(r['success'] for r in report['episodes']),
                  legacy_strict_successes=sum(r['legacy_strict_success'] for r in report['episodes']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}))


if __name__ == '__main__':
    main()
