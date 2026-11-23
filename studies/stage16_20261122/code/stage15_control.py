"""Bounded preload recovery using fresh RGB association and absolute encoder error.

Concatenated after Stage14 recovery. A stationary hold cannot identify drive
compliance: its near-zero requested displacement made the old ratio ill posed.
No online force, object identity, pose or object geometry is used here.
"""
_S15_COMPARE_PRELOAD = compare_preload
_S15_PRELOAD = s13_preload
S15_PRELOAD = False


def compare_preload(before, after, commanded_delta=None):
    result = _S15_COMPARE_PRELOAD(before, after, commanded_delta)
    result['exposure_steps'] = [before.get('step'), after.get('step')]
    result['bounded_probe_association'] = {}
    for name, row in result['views'].items():
        first, last = before['views'].get(name), after['views'].get(name)
        if not first or not last or first['status'] != 'ok' or last['status'] != 'ok':
            continue
        if not first['visible'] or not last['visible'] or first['size'] != last['size']:
            continue
        camera_translation = float(np.linalg.norm(last['camera'][:3, 3] - first['camera'][:3, 3]))
        camera_rotation = _s3s_rotation(first['camera'][:3, :3], last['camera'][:3, :3])
        intersection = int(np.logical_and(first['mask'], last['mask']).sum())
        union = int(np.logical_or(first['mask'], last['mask']).sum())
        iou = intersection / max(1, union)
        shift = float(np.linalg.norm(np.asarray(last['centroid_px']) - first['centroid_px']))
        # This verifies association/visibility during a fixed-camera observation,
        # never stationary contact or support. The subsequent 0.25mm move is a probe.
        associated = bool(min(first['area_px'], last['area_px']) >= 24 and iou >= .8
                          and shift <= 1. and camera_translation <= .0001 and camera_rotation <= .002)
        row.update(association_iou=iou, centroid_displacement_px=shift,
                   exposure_camera_translation_m=camera_translation,
                   exposure_camera_rotation_rad=camera_rotation,
                   bounded_probe_association=associated)
        if associated:
            result['bounded_probe_association'].setdefault(row['finger'], []).append(name)
    result['probe_association_thresholds'] = dict(min_area_px=24, min_mask_iou=.8,
        max_centroid_displacement_px=1., max_camera_translation_m=.0001,
        max_camera_rotation_rad=.002, evidence_scope='same-color silhouette association; not contact')
    return result


def s13_preload_proposal(s, evidence):
    names = s['robot'].finger_names
    q = np.asarray(s['obs']['actual_q'], dtype=float)
    qvel = np.asarray(s['obs']['actual_qvel'], dtype=float)
    error = np.asarray(s['last']) - q
    groups = s5_static(s)['finger_joint_indices']
    errors = {name: float(np.max(abs(error[groups[name]]))) for name in names}
    speeds = {name: float(np.max(abs(qvel[groups[name]]))) for name in names}
    fresh = bool(evidence.get('step_after') == s.get('closure_evidence_step')
        and 0 <= int(s['obs']['step']) - int(evidence.get('step_after', -1000)) <= 12
        and evidence.get('wrist_fixed')
        and not evidence.get('visual_evidence_reused_for_state_only'))
    associated = all(evidence.get('bounded_probe_association', {}).get(name) for name in names)
    settled = all(speeds[name] < .06 for name in names)
    # Absolute tracking error is observable during a hold. The displacement
    # ratio is retained in diagnostics only and never used to infer contact.
    tracking = all(errors[name] <= .03 for name in names)
    grounded = bool(s.get('s8_previous_decision', {}).get('grounded_evidence'))
    squeeze = np.asarray(s['squeeze']); cap = float(s.get('last_closure_cap', S5_CAP))
    reason = dict(eligible=False, fresh=fresh, bilateral_association=associated,
        settled=settled, absolute_tracking_valid=tracking, grounded_evidence=grounded,
        max_abs_tracking_error_rad=errors, max_abs_qvel_rad_s=speeds,
        squeeze_m=squeeze.tolist(), cap_m=cap,
        thresholds=dict(max_q_error_rad=.03, max_qvel_rad_s=.06, probe_increment_m=.00025,
                        max_evidence_age_steps=12), force_observable=False)
    if not globals().get('S15_PRELOAD', True):
        return None, dict(reason, reason='preload_disabled_for_paired_control')
    if (fresh and associated and settled and tracking and grounded
        and not evidence.get('moving_fingers') and np.all(squeeze >= .0005)
        and np.all(squeeze + .00025 <= cap)):
        return squeeze + .00025, dict(reason, eligible=True, direction='increase',
            reason='fresh_bilateral_association_grounded_bounded_probe_with_cap_reserve')
    return None, dict(reason, reason='no_fresh_safe_bounded_probe_evidence_or_cap_reserve')


def s13_preload(s):
    before=capture_preload_snapshot(s['obs'],s['robot'],s5_static(s))
    s5_move(s,s['last'],24,'s13_recovery_observe')
    after=capture_preload_snapshot(s['obs'],s['robot'],s5_static(s))
    evidence=compare_preload(before,after,commanded_delta=s['last']-before['actual_q'])
    s['closure_evidence_step']=evidence.get('step_after')
    proposal,reason=s13_preload_proposal(s,evidence)
    s13_log(s,'fresh_preload_evidence',evidence=evidence)
    s13_log(s,'considered',branch='preload',**reason)
    if proposal is None:return False
    actual=np.asarray(s['obs']['actual_q'],dtype=float)
    points=s['robot'].tips(actual)[0]
    increment=proposal-np.asarray(s['squeeze'])
    # Keep the current measured wrist. Historical recovery used c.q, which
    # returned the wrist to its old pre-proof height instead of adjusting in situ.
    goals=points+s['c']['inward']*increment[:,None]
    target=close_ik(s['robot'],actual,goals)
    inactive=np.setdiff1d(np.arange(len(actual)),s['robot'].active_finger_indices)
    projection=float(np.max(abs(np.clip(actual,s['robot'].lo,s['robot'].hi)-actual)))
    if projection>1e-5:
        s13_log(s,'guard_refused',branch='preload',reason='measured_joint_limit',projection=projection)
        return False
    # Unused encoders can settle a few microradians below a hard stop. Keep
    # their already legal position command instead of commanding that noise.
    target[inactive]=s['last'][inactive]
    predicted=s['robot'].tips(target)[0]-points
    along=np.sum(predicted*s['c']['inward'],axis=1)
    lateral=np.linalg.norm(predicted-s['c']['inward']*along[:,None],axis=1)
    residual=np.linalg.norm(s['robot'].tips(target)[0]-goals,axis=1)
    if (np.any(along<.00005) or np.any(along>.0004)
            or np.any(lateral>.00015) or np.any(residual>.00015)):
        s13_log(s,'guard_refused',branch='preload',reason='bounded_tip_endpoint',
            predicted_tip_increment_m=predicted.tolist(),along_inward_m=along.tolist(),
            lateral_m=lateral.tolist(),goal_residual_m=residual.tolist(),
            thresholds=dict(min_inward_m=.00005,max_inward_m=.0004,max_lateral_m=.00015,max_residual_m=.00015))
        return False
    n=_s3_steps(target-s['last'],48,.06,.8)
    path=smooth_segment(s['last'].copy(),target,n,'s13_preload_recovery')
    audit=s13_preflight(s,path)
    rid=s.setdefault('recovery_serial',0)+1;s['recovery_serial']=rid
    s13_log(s,'selected',branch='preload',recovery_id=rid,**reason)
    s13_log(s,'guard',branch='preload',recovery_id=rid,audit=audit,
        proposed_target_q=target.tolist(),measured_seed_q=actual.tolist(),
        proposed_tip_increment_m=increment.tolist(),goal_points_m=goals.tolist(),
        wrist_command_delta=(target[:6]-s['last'][:6]).tolist(),
        predicted_tip_increment_m=predicted.tolist(),goal_residual_m=residual.tolist())
    if not audit['passed']:return False
    old_squeeze=np.asarray(s['squeeze']).copy();start_step=int(s['obs']['step'])
    s['s13_pending_grasp']=dict(branch='preload',recovery_id=rid)
    for q,count,label in path:s5_step(s,q,count,label)
    s['squeeze']=proposal;s5_move(s,s['last'],48,'s5_preload_hold')
    s13_log(s,'executed',branch='preload',recovery_id=rid,
        old_squeeze_m=old_squeeze.tolist(),new_squeeze_m=proposal.tolist(),force_known=False)
    s13_log(s,'preload_encoder_response',before_step=start_step,
        target_q=target.tolist(),before_q=actual.tolist(),actual_q=s['obs']['actual_q'],
        actual_qvel=s['obs']['actual_qvel'],
        measured_delta=(np.asarray(s['obs']['actual_q'])-actual).tolist(),
        measured_tip_increment_m=(s['robot'].tips(np.asarray(s['obs']['actual_q']))[0]-points).tolist(),
        next_action='normal_proof_middle_final_lift_and_hold',force_known=False)
    s['lift_reports']=[];s['hold_reports']=[];s['s15_pending_reproof']=True;s['s15_preload_height_floor']=float(s['obs']['actual_q'][2]-s['c']['q'][2])
    return True


_S15_LIFT = s5_lift
def s5_lift(s,height=.03,hold=360,_full=False):
    if height <= .0031 and s.pop('s15_pending_reproof',False):
        height=max(height,float(s['obs']['actual_q'][2]-s['c']['q'][2])+.003)
        s['s15_preload_height_floor']=height
        s13_log(s,'preload_reproof',target_height_from_original_m=height,
                additional_wrist_lift_m=.003,force_known=False)
    if 's15_preload_height_floor' in s:
        height=max(height,s['s15_preload_height_floor'])
    return _S15_LIFT(s,height,hold,_full)


_S15_LOCAL_PROBE = s13_local_probe
def s13_local_probe(mover,state):
    # One actual local retry is enough to test a small placement correction.
    # If it also fails, reserve the last closure for full release/resampling.
    if state.get('s15_local_retry_executed', False):
        s13_log(state,'guard_refused',branch='local_pose',
            reason='failed_local_retry_requires_full_release_resample')
        return False
    accepted = _S15_LOCAL_PROBE(mover,state)
    if accepted:
        state['s15_local_retry_executed'] = True
    return accepted


_S15_RELEASE_RECOVER = s13_release_recover
def s13_release_recover(s):
    recovered = _S15_RELEASE_RECOVER(s)
    if recovered and s.get('next_branch') == 'local_pose':
        s['s15_local_retry_executed'] = True
    return recovered


def s13_evaluate(s):
    for phase,height,hold in (('proof',.003,48),('middle',.0165,0),('final',.03,360)):
        s5_lift(s,height,hold)
        decision=s5_decide(s,phase)
        if decision['action']=='reobserve':
            s5_reobserve(s)
            decision=s5_decide(s,phase,reobservations=1)
        if phase=='final' and decision['action']=='stop_unknown':
            # Unknown RGB can mean an intact but marginally lifted grasp. A
            # single guarded 2mm excitation preserves it and supplies a new
            # motion/hold observation. Object height is never read online.
            target=max(.03,float(s['obs']['actual_q'][2]-s['c']['q'][2]))+.002
            if target<=.035 and not s.get('s15_final_excitation_used',False):
                s['s15_final_excitation_used']=True
                s13_log(s,'unknown_final_excitation',
                    hypothesis='possible_carried_grasp_with_unresolved_RGB',
                    trigger=decision,additional_commanded_lift_m=.002,
                    target_height_from_grasp_m=target,
                    outcome='requires_independent_trajectory_verification')
                s5_lift(s,target,360)
                decision=s5_decide(s,phase,reobservations=1)
        if decision['action']!='continue_lift':
            return decision
    return decision


def policy_main(ctx):
    if not globals().get('S11_CONFIG', {}).get('evidence_recovery'):
        return _S13_LEGACY_POLICY(ctx)
    s=s5_init(ctx);s['actual_grasps']=0;s['next_branch']='initial'
    # A preload is dispatched inside recovery. Even the last allowed closure
    # must receive its normal proof/lift/hold evaluation before the loop exits.
    while s['actual_grasps'] < S5_ATTEMPTS or s.get('preload_ready', False):
        if not s.pop('preload_ready', False):
            local=s.pop('local_retry_ready', False)
            if not local:
                axis=S5_AXIS if not s['actual_grasps'] else 'previous_orthogonal'
                if not s5_plan(s, axis):
                    s13_log(s,'stopped',reason='no_feasible_next_grasp');return
                s5_approach(s)
                if not s.get('refined'):s5_refine(s)
            s.setdefault('tried_contacts', []).append(dict(points=s['c']['points'].tolist()))
            s['s13_pending_grasp']=dict(branch=s['next_branch'],recovery_id=s.get('s13_recovery_id'))
            s5_close(s, S5_CLOSURE, S5_CAP)
        decision=s13_evaluate(s)
        s13_log(s,'failure_decision',decision=decision)
        if decision['action'] != 'retry' or s['actual_grasps'] >= S5_ATTEMPTS:
            return
        if not s.get('s15_preload_used', False) and s13_preload(s):
            s['s15_preload_used']=True;s['preload_ready']=True;continue
        if s.get('s15_preload_used', False):
            s13_log(s,'guard_refused',branch='preload',reason='one_bounded_probe_per_episode')
        s.pop('s15_preload_height_floor',None)
        if not s13_release_recover(s):return
