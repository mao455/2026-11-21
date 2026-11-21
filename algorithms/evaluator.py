"""Frozen measured-state verdict. No controller phase labels affect scoring."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

DT = 1 / 120
HOLD_STEPS = 240


def evaluate(records, initial_object_z):
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
