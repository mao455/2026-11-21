"""Command-interval encoder evidence, without native contacts or force claims."""
import numpy as np


def interval_feedback(before, after, command_delta):
    q0=np.asarray(before['actual_q'],float);q1=np.asarray(after['actual_q'],float)
    velocity=np.asarray(after['actual_qvel'],float);measured=q1-q0
    result=dict(measured_delta=measured.tolist(),commanded_delta=None,tracking_residual=None,
                fingers={},force_observable=False,interval_steps=[before.get('step'),after.get('step')])
    if command_delta is None:
        result['uncertainty']='command increment unavailable'
        return result
    command=np.asarray(command_delta,float)
    if command.shape!=measured.shape or not np.isfinite(command).all():
        raise ValueError('command delta must match finite measured joint vector')
    result.update(commanded_delta=command.tolist(),tracking_residual=(command-measured).tolist())
    prismatic=set(before.get('prismatic_indices',[]))
    for finger,indices in before['groups'].items():
        units={};active=[]
        for unit,linear in [('rad',False),('m',True)]:
            ids=[int(j)for j in indices if (j in prismatic)==linear]
            if not ids:continue
            epsilon=1e-7 if linear else 1e-6
            driven=[j for j in ids if abs(command[j])>epsilon];active+=driven
            speed=float(np.max(abs(velocity[ids])));settled=speed<(.002 if linear else .06)
            ratio=None if not driven else float(measured[driven]@command[driven]/(command[driven]@command[driven]))
            units[unit]=dict(indices=ids,active_indices=driven,command_increment=command[ids].tolist(),
                measured_increment=measured[ids].tolist(),qvel=velocity[ids].tolist(),
                max_abs_qvel=speed,settled=settled,tracking_ratio=ratio,
                drive_deficit=None if ratio is None else 1.-ratio)
        valid=[r for r in units.values()if r['drive_deficit']is not None]
        settled=all(r['settled']for r in units.values())
        deficit=None if not valid else min(r['drive_deficit']for r in valid)
        state=('information_insufficient_zero_command' if not valid else
               'settled_resistance_proxy' if settled and deficit>=.45 else
               'suspected_lag' if deficit>=.45 else 'following_command')
        result['fingers'][finger]=dict(joint_indices=list(indices),active_joint_indices=active,
            tracking_ratio=None if deficit is None else 1.-deficit,drive_deficit=deficit,
            settled=settled,max_abs_qvel=float(np.max(abs(velocity[list(indices)]))),
            unit_groups=units,evidence_state=state,
            confidence=.6 if state=='settled_resistance_proxy'else .25 if state=='suspected_lag'else 0.,
            contact_confirmed=False)
    result['uncertainty']='same-interval target increment and measured increment; static tracking error is separate; resistance proxy is not contact force'
    return result


def saturated_pd(q_target,q,qvel,kp,kd,effort_limit,target_velocity=None):
    """Sampled force-PD proxy; implicit integration can differ from this sample."""
    q=np.asarray(q,float);v=np.zeros_like(q)if target_velocity is None else np.asarray(target_velocity,float)
    raw=np.asarray(kp)*(np.asarray(q_target)-q)+np.asarray(kd)*(v-np.asarray(qvel))
    limit=np.asarray(effort_limit)
    return np.clip(raw,-limit,limit),abs(raw)>=limit


def matched_extra_effort(loaded,empty,configuration):
    """Subtract only comparable, same-command-direction unloaded responses."""
    q=np.asarray(loaded['q']);eq=np.asarray(empty['q'])
    if (np.max(abs(q-eq))>configuration['pose_tolerance'] or
        np.max(abs(np.asarray(loaded['qvel'])-empty['qvel']))>configuration['speed_tolerance'] or
        not np.array_equal(np.sign(loaded['delta']),np.sign(empty['delta']))):
        return dict(available=False,reason='unmatched_unloaded_response',force_observable=False)
    values=[];saturations=[]
    for row in (loaded,empty):
        torque,saturated=saturated_pd(row['target'],row['q'],row['qvel'],configuration['kp'],configuration['kd'],configuration['effort_limit'])
        values.append(torque);saturations.append(saturated)
    return dict(available=True,extra_joint_effort=(values[0]-values[1]).tolist(),
        saturation=[s.tolist()for s in saturations],force_observable=False,
        uncertainty='sampled implicit force-PD proxy; no fingertip force without a matched contact Jacobian and identifiable normal')
