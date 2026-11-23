"""Public-only numerical diagnostics; optional single-factor ablations.

Append to a generated controller. Flags default to historical decisions:
S16_COMMON_NORMAL_WEIGHT=None, S16_MARGIN_TIE='historical'.
Only final fixed-target optimize and candidate ordering are wrapped. The region
search residual and all per-finger/full-path hard gates remain unchanged.
"""

def s16_ik_terms(robot, q, prior, targets, inward, normal_weight):
    q=np.asarray(q);prior=np.asarray(prior);targets=np.asarray(targets);inward=np.asarray(inward)
    p,n,_,poses=robot.tips(q);vertices=robot.vertices(poses)
    count=len(targets);scale=float(np.sqrt(2/count))
    errors=np.linalg.norm(p-targets,axis=1)
    angles=np.arccos(np.clip(np.sum(n*inward,axis=1),-1,1))
    position=(p-targets)*1500*scale;normal=(n-inward)*normal_weight*scale
    table=np.minimum(vertices[:,2]-.001,0)*5000
    wrist=(q[3:6]-prior[3:6])*.25
    fingers=(q[robot.active_finger_indices]-prior[robot.active_finger_indices])*.08
    components=dict(position=float(np.sum(position**2)),normal=float(np.sum(normal**2)),
        table=float(np.sum(table**2)),wrist_regularizer=float(np.sum(wrist**2)),
        finger_regularizer=float(np.sum(fingers**2)))
    joint_margin=np.minimum(q-robot.lo,robot.hi-q)
    return dict(contact_count=count,units=dict(position='m',angle='rad',wrist_translation='m',
        wrist_rotation_and_finger_joints='rad',residual='dimensionless weighted mixed objective'),
        position_weight_per_m=1500.,normal_chord_weight=float(normal_weight),
        contact_count_normalization=scale,per_finger_position_error_m=errors.tolist(),
        max_position_error_m=float(errors.max()),rms_position_error_m=float(np.sqrt(np.mean(errors**2))),
        per_finger_normal_error_rad=angles.tolist(),max_normal_error_rad=float(angles.max()),
        rms_normal_error_rad=float(np.sqrt(np.mean(angles**2))),
        per_finger_position_squared_residual=np.sum(position**2,axis=1).tolist(),
        per_finger_normal_squared_residual=np.sum(normal**2,axis=1).tolist(),
        squared_residual_contributions=components,reconstructed_scipy_cost=.5*sum(components.values()),
        per_coordinate_limit_margin=joint_margin.tolist(),minimum_translation_limit_margin_m=float(joint_margin[:3].min()),
        minimum_angular_limit_margin_rad=float(joint_margin[3:].min()),
        table_clearance_m=float(vertices[:,2].min()),q=q.tolist(),prior_q=prior.tolist(),
        targets_world_m=targets.tolist(),desired_inward_world=inward.tolist(),
        actual_pad_points_world_m=p.tolist(),actual_pad_normals_world=n.tolist(),
        hard_gate_scope='each finger position and normal plus full endpoint/path checker; RMS never replaces these gates')

_S16_ORIGINAL_OPTIMIZE=optimize

def optimize(robot,prior,targets,inward,normal_weight=4.,seed_index=0):
    requested=float(normal_weight)
    common=globals().get('S16_COMMON_NORMAL_WEIGHT',None)
    used=requested if common is None else float(common)
    q,meta=_S16_ORIGINAL_OPTIMIZE(robot,prior,targets,inward,normal_weight=used,seed_index=seed_index)
    meta['s16_numerics']=s16_ik_terms(robot,q,prior,targets,inward,used)
    meta['s16_numerics'].update(requested_normal_weight=requested,used_normal_weight=used,
        nfev=meta['nfev'],max_nfev=120,solver_success=meta['solver_success'],
        termination_detail_available=False,search_scope='final fixed-target IK only; region IK is a distinct preceding solver')
    return q,meta

_S16_HISTORICAL_ORDER=s12_order_candidates

def s16_order_candidates(candidates,tolerance=.1):
    """Use a declared relative alpha tolerance, never a nominal/robust gap.

    Alpha is the dimensionless common-task load multiplier. A nominal-robust
    difference is a sensitivity to a model scenario, not a confidence interval.
    """
    if tolerance<0 or not np.isfinite(tolerance):raise ValueError('finite nonnegative margin tolerance required')
    remaining=list(candidates);ordered=[]
    while remaining:
        supported=[c for c in remaining if c['meta']['task_quality']['nominal']['gravity_feasible']]
        pool=supported or remaining
        best=max(pool,key=lambda c:c['meta']['task_quality']['robust']['minimum_task_margin'])
        br=best['meta']['task_quality']['robust']['minimum_task_margin']
        tied=[];comparisons=[]
        for c in pool:
            r=c['meta']['task_quality']['robust']['minimum_task_margin']
            threshold=tolerance*max(1.,abs(br),abs(r))
            comparisons.append(dict(candidate_id=c['meta']['candidate_id'],robust_margin=r,
                best_minus_candidate=br-r,tie_threshold=threshold,within_tie=bool(br-r<=threshold)))
            if br-r<=threshold:tied.append(c)
        def motion(c):
            t=c['meta']['quality_terms']
            return (t['planned_motion_s'],-t['min_table_clearance_m'],t['uncertainty_m'],c['meta']['quality_score'])
        chosen=min(tied,key=motion)
        chosen['meta']['rank_reason']=dict(best_robust_margin=br,relative_tolerance=tolerance,
            tied_candidate_ids=[c['meta']['candidate_id'] for c in tied],comparisons=comparisons,
            rule='bounded relative task-alpha tie; motion, clearance, observation, normalized error',
            nominal_robust_gap_is_tie_band=False,
            load_gate='nominal_gravity_supported' if supported else 'uncertified_empirical_fallback')
        ordered.append(chosen);remaining=[c for c in remaining if c is not chosen]
    return ordered


def s12_order_candidates(candidates,tolerance=.1):
    mode=globals().get('S16_MARGIN_TIE','historical')
    if mode=='historical':return _S16_HISTORICAL_ORDER(candidates,tolerance)
    if mode=='bounded_relative':return s16_order_candidates(candidates,tolerance)
    raise ValueError('unknown S16_MARGIN_TIE')
