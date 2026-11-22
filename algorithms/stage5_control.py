"""Shared FSM and control; built by the normal entry."""
def s5_note(ctx, key, value):
    ctx.note(json.dumps({key: value}, allow_nan=False))

def s5_init(ctx):
    obs = ctx.observe()
    robot = Robot(ctx.robot_description.get('urdf_path', URDF))
    robot.contact_points = np.array(PRIMITIVE['contact_points'])
    robot.contact_normals = np.array(PRIMITIVE['contact_normals'])
    mode = globals().get('S11_CONFIG', {}).get('fingers', 'two')
    robot.configure_contacts(3 if mode == 'three' else 2)
    info = _S3_ESTIMATE(obs)
    volume = _S3_VOLUME(robot, ctx.robot_description['fingereye_calibration'], info)
    rgb = np.load(obs['initial_overhead']['rgb_path'])
    hue = initial_hue(rgb)
    return dict(hand=xhand_from_runtime(ctx.robot_description, PRIMITIVE, len(robot.contact_points)), ctx=ctx, obs=obs, robot=robot, info=info, volume=volume, checker=SelfAwareGeometry(robot, info), last=np.asarray(obs['actual_q']).copy(), hue=hue, attempt=1, squeeze=np.full(len(robot.contact_points), 0.0005), lift_reports=[], hold_reports=[], started=time.monotonic())

def s5_static(s):
    return dict(s['ctx'].robot_description, volume=s['volume'], info=s['info'], hue=s['hue'], calibration=s['volume'].calibration, color=s['info']['color'], finger_joint_indices={name:list(range(6,9)) if name=='thumb' else list(range(9,12)) if name=='index' else [12,13] for name in s['robot'].finger_names})

def s5_step(s, q, n, phase, object_check=False):
    if not 1 <= n <= 24:
        raise ValueError('each dispatch must contain 1..24 physical steps')
    if s['obs']['step'] + n > 24000 or time.monotonic() - s['started'] > 2300:
        raise RuntimeError('finite episode budget')
    q = np.asarray(q)
    robot = s['robot']
    last = s['last']
    check = s['checker'].check(q, object_check=object_check)
    rate = np.abs(q - last) / (n / 120)
    if not check['passed'] or np.any(q < robot.lo) or np.any(q > robot.hi) or (rate[:3].max() > 0.14) or (rate[3:].max() > 1.4):
        s5_note(s['ctx'], 's5_guard', dict(phase=phase, check=check, rate=rate.tolist(), step=s['obs']['step']))
        raise RuntimeError('original geometry/joint/rate guard: ' + phase)
    while n:
        count = min(24, n)
        s['obs'] = s['ctx'].step(q, steps=count, phase=phase)
        n -= count
        if globals().get('S11_CONFIG', {}).get('fast_close',False) and phase in ('s5_close','s5_preload_adjust','s5_preload_hold','s13_preload_recovery'):
            tracking=q-np.asarray(s['obs']['actual_q']);speed=np.asarray(s['obs']['actual_qvel'])
            s.setdefault('servo_samples',[]).append(dict(step=s['obs']['step'],phase=phase,frame_id=s9_frame(s['obs']),target_q=q.tolist(),actual_q=s['obs']['actual_q'],actual_qvel=s['obs']['actual_qvel'],tracking_error=tracking.tolist()))
            if abs(speed[:3]).max()>.15 or abs(speed[3:]).max()>1.5:
                s5_note(s['ctx'],'s11_servo_stop',dict(reason='measured_native_robot_speed',step=s['obs']['step']));raise RuntimeError('measured closure velocity guard')
    s['last'] = q.copy()
    return s['obs']

def s5_move(s, target, n, phase, object_check=False):
    target = np.asarray(target)
    if phase == 's5_close':
        delta = abs(target - s['last'])
        duration = 1.875 * max(float(delta[:3].max()) / 0.06, float(delta[3:].max()) / 0.8)
        n = max(12, int(np.ceil(duration * 120 / 6)) * 6)
    elif phase == 's5_preload_adjust':
        n = _s3_steps(target - s['last'], minimum=24, translation=0.06, joint=0.8)
        s5_note(s['ctx'], 's8_preload_motion', dict(step=s['obs']['step'], old_minimum_steps=60, new_steps=n))
    elif phase in ('s5_close_observe', 's5_preload_hold'):
        total = 0
        stable_count = 0
        maximum = 120 if phase == 's5_close_observe' else 60
        while total < maximum:
            count = 6 if phase == 's5_preload_hold' or S9_EARLY_SETTLE else 24
            s5_move_segment(s, target, count, phase, object_check)
            total += count
            settled, velocity = stable_joint_response(s['obs']['actual_qvel'], s['hand'])
            stable_count = stable_count + 1 if settled else 0
            if total >= 24 and settled and (phase == 's5_close_observe' or stable_count >= 2):
                break
        return
    else:
        n = max(n, _s3_steps(target - s['last'], minimum=n, translation=0.06, joint=0.8))
    robot=s['robot']
    path = smooth_segment(s['last'].copy(), target, n, phase)
    if phase == 's5_close' and globals().get('S11_CONFIG', {}).get('fast_close', False):
        start_check=time.perf_counter();previous=s['last'].copy()
        for index,(node,count,label) in enumerate(path):
            node=np.asarray(node);rate=abs(node-previous)/(count/120);check=s['checker'].check(node,object_check=object_check)
            if not check['passed'] or np.any(node<robot.lo) or np.any(node>robot.hi) or rate[:3].max()>.14 or rate[3:].max()>1.4:
                s5_note(s['ctx'],'s11_close_refused',dict(node=index,check=check));raise RuntimeError('closure preflight original guard')
            previous=node
        s5_note(s['ctx'],'s11_close_preflight',dict(nodes=len(path),compute_s=time.perf_counter()-start_check))
    for q, count, label in path:
        s5_step(s, q, count, label, object_check)

def s11_contact_quality(robot, candidate, initial):
    meta = candidate['meta']
    errors = np.asarray(meta.get('per_finger_error_m', [meta['error_m']]*len(candidate['points'])))
    normals = np.asarray(meta['normal_error_rad'])
    points = candidate['points']
    axis = candidate['inward'][0]
    tangent = np.cross([0., 0., 1.], axis)
    span = float(np.ptp(points @ tangent))
    time_s = sum(n for _,n,_ in candidate['prefix'])/120
    clearance = float(meta['endpoint']['table_clearance_m'])
    terms = dict(mean_position= float(np.mean(errors/.0008)), mean_normal=float(np.mean(normals/.5)),
                 uncertainty_m=.0015, tangent_span_m=span, min_table_clearance_m=clearance,
                 planned_motion_s=time_s, per_finger_error_m=errors.tolist(), all_fingers_passed=bool(np.all(errors<.0008) and np.all(normals<.5)),
                 support_scope='geometric spread only; friction and forces not measured')
    score = terms['mean_position'] + terms['mean_normal'] + .02*time_s + .001/max(clearance,.00065) - min(span/.01,1.)*.2
    return score, terms

def s12_task_quality(robot, candidate, info):
    """Same world-frame tasks/COM assumption for both groups, no private inputs."""
    from general_grasp_contacts import task_wrench_margin
    q = candidate['q']; count = len(candidate['points'])
    indices = robot.active_finger_indices
    jacobian = np.zeros((3*count,len(indices)))
    for column,index in enumerate(indices):
        plus=q.copy();minus=q.copy();plus[index]+=1e-6;minus[index]-=1e-6
        jacobian[:,column]=((robot.tips(plus)[0]-robot.tips(minus)[0])/(2e-6)).ravel()
    limits={joint.get('name'):float(joint.find('limit').get('effort','inf'))
            for joint in ET.parse(URDF).getroot().findall('joint') if joint.find('limit') is not None}
    effort=np.array([limits.get(NAMES[i],np.inf) for i in indices])
    # Geometric targets are the public surface hypotheses. FK residual is kept
    # separately; no hypothetical LP force is interpreted as measured preload.
    points=np.asarray(candidate['meta']['targets'])
    parameters=dict(globals().get('S11_CONFIG',{}).get('task_wrench',{}))
    sigma=parameters.pop('position_sigma_m',.00025)
    parameters.pop('margin_tolerance',None)
    common=dict(reference=np.asarray(info['center']),torque_map=jacobian.T,effort_limits=effort,**parameters)
    nominal=task_wrench_margin(points,-np.asarray(candidate['inward']),**common)
    robust=task_wrench_margin(points,-np.asarray(candidate['inward']),
        covariances=np.tile(np.eye(3)*(sigma**2),(count,1,1)),**common)
    return dict(nominal=nominal,robust=robust,position_sigma_m=sigma,
        classification='conditional_load_model_not_physical_certificate',
        measured_force_available=False)

def s12_select_contacts(s, candidate):
    count=len(candidate['points'])
    s['robot'].configure_contacts(count)
    s['hand']=xhand_from_runtime(s['ctx'].robot_description,PRIMITIVE,count)
    s['squeeze']=np.full(count,.0005)
    s5_note(s['ctx'],'s12_candidate_command_binding',dict(attempt=s['attempt'],
        step=s['obs']['step'],frame_id=s9_frame(s['obs']),contact_count=count,
        candidate_id=candidate['meta'].get('candidate_id'),
        actual_q=s['obs']['actual_q'],planned_q=candidate['q'].tolist(),
        task_quality=candidate['meta'].get('task_quality'),
        approach_first_command=candidate['prefix'][0][0] if candidate['prefix'] else None))

def s12_order_candidates(candidates, tolerance=.1):
    """Compare conditional margins; overlapping error intervals use motion.

    Nominal versus sampled-uncertainty margins are not confidence intervals.
    Relative tolerance prevents insignificant alpha differences determining
    finger count. Unsupported grasps remain an explicit empirical fallback.
    """
    remaining=list(candidates);ordered=[]
    while remaining:
        supported=[c for c in remaining if c['meta']['task_quality']['nominal']['gravity_feasible']]
        pool=supported or remaining
        best=max(pool,key=lambda c:c['meta']['task_quality']['robust']['minimum_task_margin'])
        bt=best['meta']['task_quality'];br=bt['robust']['minimum_task_margin'];bn=bt['nominal']['minimum_task_margin']
        tied=[]
        for c in pool:
            t=c['meta']['task_quality'];r=t['robust']['minimum_task_margin'];n=t['nominal']['minimum_task_margin']
            threshold=max(abs(n-r),abs(bn-br),tolerance*max(1.,abs(br),abs(r)))
            if br-r<=threshold:
                tied.append(c)
        def motion(c):
            terms=c['meta']['quality_terms']
            return (terms['planned_motion_s'],-terms['min_table_clearance_m'],terms['uncertainty_m'],c['meta']['quality_score'])
        chosen=min(tied,key=motion)
        chosen['meta']['rank_reason']=dict(best_robust_margin=br,relative_tolerance=tolerance,
            tied_candidate_ids=[c['meta']['candidate_id'] for c in tied],
            rule='uncertainty-overlap or relative margin tie: shortest motion, clearance, observation, normalized error',
            load_gate='nominal_gravity_supported' if supported else 'uncertified_empirical_fallback')
        ordered.append(chosen)
        remaining=[c for c in remaining if c is not chosen]
    return ordered

def s11_contact_candidates(s, robot, info, prior, checker, initial, axes, heights, shift, geometry_source, **kwargs):
    settings=globals().get('S11_CONFIG',{})
    mode=settings.get('fingers','two')
    if mode=='two' and not settings.get('task_ranking',False):
        robot.configure_contacts(2)
        return _s3_candidates_on_axes(robot,info,prior,checker,initial,axes,heights,shift,geometry_source,**kwargs)
    modes=(2,3) if mode=='auto' else (3,) if mode=='three' else (2,)
    if kwargs.get('refinement') and s.get('c') is not None:
        modes=(len(s['c']['points']),)
    accepted=[];rejected=[];qualities=[]
    for count in modes:
        robot.configure_contacts(count)
        choices,failures=_s3_candidates_on_axes(robot,info,prior,checker,initial,axes,heights,shift,geometry_source,**kwargs)
        for failure in failures:failure['contact_count']=count
        rejected+=failures
        for index,candidate in enumerate(choices):
            repeated=any(len(old['points'])==len(candidate['points']) and
                np.mean(np.linalg.norm(np.asarray(old['points'])-candidate['points'],axis=1))<.001
                for old in s.get('tried_contacts',[]))
            if repeated:
                rejected.append(dict(contact_count=count,reason='already_physically_tried_placement',
                    targets=candidate['points'].tolist()))
                continue
            score,terms=s11_contact_quality(robot,candidate,initial)
            task=s12_task_quality(robot,candidate,info)
            candidate['meta'].update(contact_count=count,quality_score=score,quality_terms=terms,
                task_quality=task,candidate_id=str(s['attempt'])+'-'+str(count)+'-'+str(index))
            # Unknown force/COM/patch friction stays explicit. Geometry-safe
            # empirical attempts are retained even without a load certificate.
            accepted.append(candidate)
            qualities.append(dict(candidate_id=candidate['meta']['candidate_id'],contact_count=count,
                score=score,terms=terms,task_quality=task))
    accepted=s12_order_candidates(accepted,settings.get('task_wrench',{}).get('margin_tolerance',.1))
    robot.configure_contacts(len(accepted[0]['points']) if accepted else modes[0])
    s5_note(s['ctx'],'s11_finger_selection',dict(step=s['obs']['step'],frame_id=s9_frame(s['obs']),
        mode=mode,selected=len(accepted[0]['points']) if accepted else None,
        selection_scope='both_groups_geometry_and_path_feasible' if {len(c['points']) for c in accepted}=={2,3} else 'only_one_group_feasible' if accepted else 'no_feasible_group',
        ordered_candidate_ids=[c['meta']['candidate_id'] for c in accepted],
        candidates=qualities,rejected=rejected,actual_attempts=s['attempt']-1,
        candidate_rejections=len(rejected),
        selection_reason='nominal gravity; robust task reserve; overlapping uncertainty or relative tolerance uses motion/clearance',
        rank_reasons=[dict(candidate_id=c['meta']['candidate_id'],**c['meta']['rank_reason']) for c in accepted],
        middle_calibration=MIDDLE_CONTACT))
    return accepted,rejected

def s5_plan(s, axis='short', height=None, shift=0.0):
    if s['attempt'] >= 3 and globals().get('S11_CONFIG',{}).get('fingers','two')=='two':
        previous = np.asarray(s['c']['inward'][0])
        choices = []
        for angle in (np.pi / 4, -np.pi / 4):
            axis = Rotation.from_euler('z', angle).apply(previous)
            width = 2 * float(np.sum(abs(np.asarray(s['info']['axes']).T @ axis[:2]) * np.asarray(s['info']['half'])[:2]))
            choices.append((width, axis))
        width, axis = min(choices, key=lambda item: item[0])
        s5_note(s['ctx'], 's8_retry_geometry', dict(step=int(s['obs']['step']), attempt=s['attempt'], previous_failure=s.get('s8_previous_decision', {}), chosen_axis_world=axis.tolist(), projected_envelope_width_m=width, rule='observed_failure_then_new_diagonal_minimum_envelope_width', uncertainty='RGB/encoder evidence does not certify contact force'))
        axis = axis.tolist()
    info = s['info']
    robot = s['robot']
    side = int(np.argmin(info['half'][:2]))
    direction = np.r_[info['axes'][:, side], 0.0]
    if max(info['half'][:2]) / min(info['half'][:2]) < 1.15:
        direction = np.array([1.0, 0.0, 0.0])
    if axis == 'auto':
        ratio = float(max(info['half'][:2]) / min(info['half'][:2]))
        axis = 'orthogonal' if ratio >= 1.45 else 'short'
        s5_note(s['ctx'], 's5_direction_choice', dict(step=s['obs']['step'], aspect_ratio=ratio, threshold=1.45, chosen=axis, scope='two_development_shapes_not_universal'))
    if axis == 'orthogonal':
        direction = np.cross([0.0, 0.0, 1.0], direction)
    if axis == 'previous_orthogonal':
        direction = np.cross([0.0, 0.0, 1.0], s['c']['inward'][0])
    if isinstance(axis, (list, tuple)):
        direction = np.asarray(axis)
    heights = (0.4, 0.6) if height is None else (float(height),)
    initial = s13_legal_start(s) if globals().get('S11_CONFIG',{}).get('evidence_recovery') else np.asarray(s['obs']['actual_q']) if globals().get('S11_CONFIG',{}).get('task_ranking') else s['last'] if s['attempt'] > 1 else np.asarray(s['obs']['actual_q'])
    if initial is None:return False
    if s['attempt'] > 1:
        ticket = STAGE3_RELEASE_TICKET
        if not ticket or not ticket.get('ready') or ticket.get('source_attempt') != s['attempt'] - 1:
            s5_note(s['ctx'], 's5_orientation_refusal', dict(reason='no_completed_release_retreat_ticket'))
            return False
    if s['attempt'] > 1 and S5_LOCAL_RECOVERY:
        candidates, rejected = s5_local_candidates(s, direction, heights, shift, initial)
        for scan in range(2):
            if candidates:
                break
            s5_note(s['ctx'], 's5_local_refusal_scan', dict(step=s['obs']['step'], scan=scan, rejected=rejected))
            if not s5_scan_release(s):
                return False
            info = s['info']
            candidates, failures = s5_local_candidates(s, direction, heights, shift, s['last'])
            rejected += failures
    else:
        candidates, rejected = s11_contact_candidates(s, robot, info, np.array(PRIMITIVE['grasp_q']), s['checker'], initial, [direction], heights, shift, 's5_current_rgb' if s['attempt'] > 1 else 's5_initial_rgb', retime=True)
    s5_note(s['ctx'], 's5_plan', dict(attempt=s['attempt'], step=s['obs']['step'], axis_world=direction.tolist(), accepted=len(candidates), rejected=rejected, info={k: np.asarray(info[k]).tolist() for k in ('center', 'half', 'axes')}, candidates=[dict(meta=c['meta'], q=c['q'].tolist(), points=c['points'].tolist(), inward=c['inward'].tolist()) for c in candidates]))
    if not candidates:
        return False
    s['c'] = candidates[0]
    s12_select_contacts(s,s['c'])
    s['axis'] = direction
    s['lift_reports'] = []
    s['hold_reports'] = []
    s['refined'] = False
    return True

def s5_local_candidates(s, direction, heights, shift, initial):
    pts = s['volume'].points
    axis = np.asarray(direction, dtype=float).copy()
    axis /= np.linalg.norm(axis)
    tangent = np.cross([0.0, 0.0, 1.0], axis)
    along = pts @ tangent
    center = float(np.median(along)) + shift
    results, rejected = ([], [])
    for h in heights:
        z = s['info']['table_z'] + max(0.004, h * 2 * s['info']['half'][2])
        band = (abs(pts[:, 2] - z) <= 0.0016) & (abs(along - center) <= 0.0016)
        selected = pts[band]
        if len(selected) < 25:
            rejected.append(dict(reason='insufficient_local_multiview_surface', height_m=z, voxels=len(selected)))
            continue
        across = selected @ axis
        lo, hi = (float(across.min()), float(across.max()))
        if hi - lo < 0.003:
            rejected.append(dict(reason='no_resolved_local_chord', height_m=z, width_m=hi - lo))
            continue
        ends = np.array([axis * lo + tangent * center, axis * hi + tangent * center])
        ends[:, 2] = z
        local = dict(s['info'], collision_points=ends)
        choices, failures = s11_contact_candidates(s, s['robot'], local, np.array(PRIMITIVE['grasp_q']), s['checker'], initial, [axis], (h,), 0.0, 'current_rgb_local_height_chord', retime=True)
        for c in choices:
            c['meta']['local_surface_selection'] = dict(height_m=z, along_world_m=center, band_radius_m=0.0016, voxels=len(selected), chord_m=[lo, hi], full_collision_map_retained=True, surface_normal_measured=False)
        results += choices
        rejected += failures
    return (results, rejected)

def s5_scan_release(s):
    if not STAGE3_RELEASE_TICKET or not STAGE3_RELEASE_TICKET.get('ready'):
        return False
    target = s['last'].copy()
    target[2] += 0.006
    s5_move(s, target, 144, 's5_release_view_retreat', True)
    s5_move(s, target, 48, 's5_release_view_settle', True)
    current, report = stage3_live_geometry(s['volume'], s['obs'], s['info'])
    qvel = np.abs(s['obs']['actual_qvel'])
    settled = bool(qvel[:3].max() < 0.005 and qvel[3:].max() < 0.05)
    s5_note(s['ctx'], 's5_release_view_map', dict(step=s['obs']['step'], settled=settled, report=report))
    if current is None or not settled:
        STAGE3_RELEASE_TICKET['ready'] = False
        return False
    s['info'] = current
    s['checker'] = SelfAwareGeometry(s['robot'], current)
    STAGE3_RELEASE_TICKET.update(fresh_map_step=int(s['obs']['step']), retreat_completed_step=int(s['obs']['step']))
    return True

def s5_approach(s):
    prefix = s['c']['prefix']
    total = sum((n for _, n, phase in prefix if phase == 'descend'))
    descended = 0
    next_fraction = 0.9
    checked = 0
    for q, n, phase in prefix:
        if phase in ('close', 'probe_hold'):
            break
        s5_step(s, q, n, 's5_' + phase, object_check=True)
        if phase == 'descend':
            descended += n
            if checked < 1 and descended >= next_fraction * total:
                before = int(s['obs']['step'])
                checked += 1
                next_fraction = 0.9
                changed = s5_refine(s)
                s5_note(s['ctx'], 's5_approach_observation', dict(step=s['obs']['step'], observation_step=before, descent_steps_completed=descended, descent_steps_planned=total, remaining_prefix_replaced=changed))
                if changed:
                    break
    s5_note(s['ctx'], 's5_arrived', dict(step=s['obs']['step'], actual_q=s['obs']['actual_q'], actual_pad_m=s['robot'].tips(np.asarray(s['obs']['actual_q']))[0].tolist(), planned_points_m=s['c']['points'].tolist()))

def s5_refine(s):
    report = s['volume'].carve(s['obs'])
    s5_note(s['ctx'], 's5_near_map', report)
    if not report.get('valid'):
        return False
    info = s['info']
    points = s['volume'].points
    local = points[:, :2] @ info['axes']
    lo, hi = (local.min(0), local.max(0))
    refined = dict(info, center=np.r_[info['axes'] @ ((lo + hi) / 2), points[:, 2].max() / 2], half=np.r_[(hi - lo) / 2, points[:, 2].max() / 2], collision_points=np.concatenate([points + np.array([a, b, c]) for a in [-0.0005, 0.0005] for b in [-0.0005, 0.0005] for c in [-0.0005, 0.0005]]))
    change = max(float(np.linalg.norm(refined['center'] - info['center'])), float(np.max(abs(refined['half'] - info['half']))))
    s5_note(s['ctx'], 's6_target_change', dict(step=s['obs']['step'], change_m=change, threshold_m=S6_UPDATE_MM, material=change >= S6_UPDATE_MM))
    if change < S6_UPDATE_MM:
        return False
    checker = SelfAwareGeometry(s['robot'], refined)
    actual = np.asarray(s['obs']['actual_q'])
    legal_actual = np.clip(actual, s['robot'].lo + 1e-07, s['robot'].hi - 1e-07)
    projection = float(np.max(abs(legal_actual - actual)))
    if projection > 1e-05:
        s5_note(s['ctx'], 's6_refinement_refused', dict(reason='measured_joint_limit_violation', projection=projection))
        return False
    choices, rejected = s11_contact_candidates(s, s['robot'], refined, np.array(PRIMITIVE['grasp_q']), checker, legal_actual, [s['c']['inward'][0]], (s['c']['meta']['height_fraction'],), 0.0, 's6_current_rgb_actual_joints', retime=True, refinement=True)
    s5_note(s['ctx'], 's5_refinement_gate', dict(step=s['obs']['step'], accepted=len(choices), rejected=rejected, planning_from='projected_actual_joints', limit_projection=projection, actual_minus_command=(actual - s['last']).tolist()))
    if not choices:
        return False
    c = choices[0]
    opened = close_ik(s['robot'], c['q'], c['points'] - c['inward'] * 0.006)
    if not checker.check(legal_actual)['passed']:
        return False
    n = _s3_steps(opened - s['last'], 240, 0.045, 0.7)
    path = smooth_segment(s['last'], opened, n, 's5_rgb_realign')
    failed = [i for i, (q, _, _) in enumerate(path) if not checker.check(np.asarray(q))['passed']]
    if failed:
        s5_note(s['ctx'], 's5_refinement_connection_refused', dict(step=s['obs']['step'], failed_nodes=failed))
        return False
    s['checker'] = checker
    s5_move(s, opened, n, 's5_rgb_realign', True)
    s['c'] = c
    s12_select_contacts(s,c)
    s['info'] = refined
    s['refined'] = True
    return True

def s11_post_release_plan(s):
    previous=s['c'];actual=np.asarray(s['obs']['actual_q'])
    legal=np.clip(actual,s['robot'].lo+1e-7,s['robot'].hi-1e-7)
    if np.max(abs(legal-actual))>1e-5:
        s5_note(s['ctx'],'s11_replan_stop',dict(reason='measured_joint_limit'));return False
    rejected=[];started=time.perf_counter()
    for h,shift in ((.25,-.0015),(.25,.0015),(.55,-.0015),(.55,.0015),(.75,-.0015),(.75,.0015)):
        candidates,failures=_s3_candidates_on_axes(s['robot'],s['info'],np.array(PRIMITIVE['grasp_q']),
            s['checker'],legal,[previous['inward'][0]],(h,),shift,'s11_released_fresh_RGB',retime=True)
        rejected.extend(failures)
        for c in candidates:
            novelty=float(np.mean(np.linalg.norm(c['points']-previous['points'],axis=1)))
            if novelty<.001:
                rejected.append(dict(reason='no_material_novelty',novelty_m=novelty));continue
            if any(np.mean(np.linalg.norm(c['points']-x,axis=1))<.001 for x in s.get('tried_points',[])):
                rejected.append(dict(reason='already_tried_placement'));continue
            s.update(c=c,refined=True,lift_reports=[],hold_reports=[])
            s5_note(s['ctx'],'s11_released_placement',dict(step=s['obs']['step'],frame_id=s9_frame(s['obs']),
                actual_q=s['obs']['actual_q'],points=c['points'].tolist(),planned_q=c['q'].tolist(),
                height_fraction=h,shift_m=shift,novelty_m=novelty,rejected=rejected,
                unknown_failure_cause=True,search_s=time.perf_counter()-started))
            return True
        if time.perf_counter()-started>12.:break
    s5_note(s['ctx'],'s11_released_placement_refused',dict(step=s['obs']['step'],rejected=rejected,
        fallback='original released high pregrasp reorientation',search_s=time.perf_counter()-started))
    return False

def s11_motor_snapshot(s):
    hand=s['hand']
    groups={p.finger:[hand.index[n] for n in p.joint_names] for p in hand.pads}
    return dict(actual_q=np.asarray(s['obs']['actual_q']), actual_qvel=np.asarray(s['obs']['actual_qvel']), groups=groups)

def s5_close(s, mode='fixed', cap=0.0025):
    c = s['c']
    robot = s['robot']
    ctx = s['ctx']
    fast = globals().get('S11_CONFIG', {}).get('fast_close', False)
    displacements = [0.0005] if fast else np.linspace(-0.006, 0.0005, 30)
    s5_note(ctx, 's11_close_begin', dict(step=s['obs']['step'], fast=fast, frame_id=s9_frame(s['obs']), capture_step=s['obs']['image_state']['step'], actual_q=s['obs']['actual_q'], last_command=s['last'].tolist(), points=c['points'].tolist()))
    for displacement in displacements:
        target = close_ik(robot, c['q'], c['points'] + c['inward'] * displacement)
        s5_move(s, target, 24, 's5_close')
    s5_move(s, s['last'], 120, 's5_close_observe')
    squeeze = np.full(len(c['points']), 0.0005)
    evidence = None
    contact_at = np.full(len(c['points']), np.nan)
    last_increment = np.full(len(c['points']), 0.00025)
    for i in range(15 if mode == 'servo' else 6):
        # Fresh RGB is mandatory at entry, on anomaly and every second probe.
        # Intermediate rounds use raw encoder deltas, never duplicate image evidence.
        previous_visual = evidence
        visual_round = not fast or i % 2 == 0 or evidence is None or bool(evidence.get('moving_fingers')) or not all(m.get('settled') for m in evidence.get('motor', {}).get('fingers', {}).values())
        before = capture_preload_snapshot(s['obs'], robot, s5_static(s)) if visual_round else s11_motor_snapshot(s)
        delta = np.full(len(c['points']), 0.00025 if i == 0 else 0.00035)
        reason = 'fixed_schedule'
        if mode == 'feedback' and evidence is not None:
            delta, reason = stage2_response_action(evidence, i, 'preload')
        if mode in ('adaptive', 'servo') and evidence is not None:
            motors = evidence['motor']['fingers']
            visible = set(evidence.get('visible_fingers', []))
            delta = np.full(len(c['points']), 0.00035)
            reason = 'bounded_motor_probe_with_bilateral_visibility'
            for k, finger in enumerate(robot.finger_names):
                motor = motors.get(finger, {})
                if finger not in visible or not motor.get('settled', False):
                    delta[k] = 0.0
                    reason = 'visibility_or_motor_settle_hold'
                deficit = motor.get('drive_deficit')
                if last_increment[k] > 0.0001 and deficit is not None and (deficit >= 0.45) and np.isnan(contact_at[k]):
                    contact_at[k] = squeeze[k]
                allowance = 0.00105 if mode == 'servo' else 0.0007
                if not np.isnan(contact_at[k]):
                    delta[k] = min(delta[k], max(0.0, contact_at[k] + allowance - squeeze[k]))
            if evidence.get('moving_fingers') and evidence.get('confidence', 0.0) >= 0.5:
                delta[:] = 0.0
                reason = 'registered_object_motion_hold'
        proposed = np.clip(squeeze + delta, 0.0005, cap)
        target = close_ik(robot, c['q'], c['points'] + c['inward'] * proposed[:, None])
        s5_note(ctx, 's5_closure_action', dict(step=s['obs']['step'], round=i, mode=mode, reason=reason, response_used=evidence if mode in ('feedback', 'adaptive', 'servo') else None, contact_proxy_at_m=[float(x) if np.isfinite(x) else None for x in contact_at], squeeze_m=proposed.tolist(), target_q=target.tolist()))
        s5_move(s, target, 60, 's5_preload_adjust')
        s5_move(s, target, 60, 's5_preload_hold')
        after = capture_preload_snapshot(s['obs'], robot, s5_static(s)) if visual_round else s11_motor_snapshot(s)
        if visual_round:
            evidence = compare_preload(before, after, commanded_delta=target - before['actual_q'])
        else:
            evidence = dict(previous_visual, motor=_s2v_motor(before, after, target-before['actual_q']))
            evidence.update(confidence=0., moving_fingers=[], visual_evidence_reused_for_state_only=True)
        s5_note(ctx, 's11_probe_tracking', dict(step=s['obs']['step'], frame_id=s9_frame(s['obs']), visual_processed=visual_round, target_q=target.tolist(), actual_q=s['obs']['actual_q'], actual_qvel=s['obs']['actual_qvel'], tracking_error=(target-np.asarray(s['obs']['actual_q'])).tolist(), measured_tip_m=robot.tips(np.asarray(s['obs']['actual_q']))[0].tolist(), cap_m=cap, force_known=False))
        s5_note(ctx, 's5_closure_response', dict(step=s['obs']['step'], round=i, features=evidence))
        last_increment = proposed - squeeze
        squeeze = proposed
        if mode == 'servo' and np.all(last_increment < 1e-07):
            break
    s['squeeze'] = squeeze
    s['closure_evidence']=evidence or {}
    s['closure_evidence_step']=(evidence or {}).get('step_after')
    s['last_closure_cap']=cap
    if fast:
        s5_note(ctx,'s11_servo_audit',dict(attempt=s['attempt'],samples=s.pop('servo_samples',[]),rules=SENSOR_RULES,position_preload_cap_m=cap,force_estimate_available=False))

def s5_lift(s, height=0.03, hold=360, _full=False):
    if S9_EARLY_SETTLE and height <= 0.0031 and (not _full):
        return early_proof(s, height, hold)
    ctx = s['ctx']
    before = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    target = s['last'].copy()
    target[2] = s['c']['q'][2] + height
    n = 240 if height <= 0.0031 else 300
    recent = []
    previous = before
    s['lift_veto'] = False
    for q, count, label in smooth_segment(s['last'].copy(), target, n, 's5_proof' if height <= 0.0031 else 's5_lift'):
        s5_step(s, q, count, label)
        if abs(float(s['obs']['actual_q'][2]) - float(previous['actual_q'][2])) < 0.0022:
            continue
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        response = stage3_support_pair(previous, snap, 'lift')
        previous = snap
        recent.append(response)
        s5_note(ctx, 's5_lift_in_progress', response)
        if len(recent) >= 2 and all((x['state'] == 'grounded_evidence' for x in recent[-2:])):
            s['lift_veto'] = True
            s5_note(ctx, 's5_lift_veto', dict(step=s['obs']['step'], reason='two_current_RGB_intervals_grounded', commanded_height_m=height))
            break
    after = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    r = stage3_support_pair(before, after, 'lift')
    s['lift_reports'].append(r)
    s5_note(ctx, 's5_lift_response', dict(step=s['obs']['step'], height_m=height, report=r))
    for _ in range(0 if s['lift_veto'] else hold // 24):
        s5_step(s, s['last'], 24, 's5_hold')
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        r = stage3_support_pair(after, snap, 'hold')
        s['hold_reports'].append(r)
        after = snap
    if hold and (not s['lift_veto']):
        s5_note(ctx, 's5_hold_response', dict(step=s['obs']['step'], reports=s['hold_reports'][-hold // 24:]))
    return r

def s5_decide(s, phase, reobservations=0):
    result = stage3_support_decision(s['lift_reports'], s['hold_reports'], attempt=s['attempt'], max_attempts=S5_ATTEMPTS, phase=phase, reobservations=reobservations, remaining_steps=24000 - int(s['obs']['step']), remaining_wall_s=2300 - (time.monotonic() - s['started']))
    if s.get('lift_veto'):
        result.update(action='retry' if s['attempt'] < S5_ATTEMPTS else 'stop_limit', reason='in_progress_grounded_RGB_veto', support_state='failure_evidence')
    s5_note(s['ctx'], 's5_' + phase + '_decision', result)
    s['s8_previous_decision'] = dict(result)
    return result

def s5_reobserve(s):
    previous = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    for _ in range(10):
        s5_step(s, s['last'], 24, 's5_reobserve')
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        s['hold_reports'].append(stage3_support_pair(previous, snap, 'hold'))
        previous = snap
    s5_note(s['ctx'], 's5_reobservation_completed', dict(step=s['obs']['step'], reports=s['hold_reports'][-10:]))

# These rules select computation, never grant new policy inputs.
SENSOR_RULES = {
    'coarse': 'initial ordinary overhead RGB',
    'approach': 'new exposure-paired RGB at descent checkpoint or changed geometry',
    'close': 'encoder feedback every command; new RGB at entry and anomaly',
    'preload': 'new RGB on even probes/anomaly; encoders on every probe',
    'lift': 'new exposure-paired RGB for every independent support interval',
    'recover': 'new RGB after each bounded opening/retreat before local planning',
}

def s11_local_fit(robot, actual, goals, inward):
    """Bounded local translation and fingers; measured wrist attitude is fixed."""
    initial=np.asarray(actual,dtype=float).copy()
    indices=np.r_[np.arange(3),np.arange(6,12)]
    travel=float(globals().get('S11_CONFIG',{}).get('local_translation_m',.008))
    lower=np.maximum(robot.lo[indices]+1e-7,np.r_[initial[:3]-[.002,.002,travel],robot.lo[6:12]+1e-7])
    upper=np.minimum(robot.hi[indices]-1e-7,np.r_[initial[:3]+[.002,.002,.002],robot.hi[6:12]-1e-7])
    def residual(x):
        q=initial.copy();q[indices]=x
        points,normals,_,_=robot.tips(q)
        return np.r_[(points-goals).ravel()*1500,(normals-inward).ravel()*4,
                     (q[:3]-initial[:3])*2,(q[6:12]-initial[6:12])*.03]
    fit=least_squares(residual,np.clip(initial[indices],lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=45)
    q=initial.copy();q[indices]=fit.x
    return q

def s11_local_probe(mover, state):
    # Called only AFTER an original, physically executed opening/short retreat.
    # No collision exemption or synthetic completed-release ticket is created.
    mover.sync()
    state.update(obs=mover.obs, last=mover.last.copy())
    calls = state.get('local_probe_calls', 0)
    if calls >= 8:
        return False
    state['local_probe_calls'] = calls + 1
    ctx, robot = mover.ctx, mover.robot
    obs = mover.obs
    def log(reason, **extra):
        s5_note(ctx, 's11_local_gate', dict(step=obs['step'], frame_id=s9_frame(obs),
            capture_step=obs['image_state']['step'], actual_q=obs['actual_q'],
            last_command=mover.last.tolist(), opening_already_executed=True,
            search_call=calls+1, reason=reason, **extra))
    volume = Stage3LiveVolume(robot, state['volume'].calibration, state['info'])
    info, report = stage3_live_geometry(volume, obs, state['info'])
    if info is None:
        log('fresh_map', report=report)
        return False
    checker = SelfAwareGeometry(robot, info)
    actual = np.asarray(obs['actual_q'])
    actual_check, command_check = checker.check(actual), checker.check(mover.last)
    if not actual_check['passed'] or not command_check['passed']:
        log('start_clearance', measured=actual_check, commanded=command_check)
        return False
    legal = np.clip(actual, robot.lo+1e-7, robot.hi-1e-7)
    if np.max(abs(legal-actual)) > 1e-5:
        log('measured_joint_limits')
        return False
    previous = state['c']
    axis = np.asarray(previous['inward'][0]); tangent = np.cross([0.,0.,1.], axis)
    pts = volume.points
    center = float(np.mean(previous['points'] @ tangent))
    old_z = float(np.mean(previous['points'][:,2]))
    rejected=[];search_start=time.perf_counter()
    for dz, shift in ((.002,0.),(-.002,0.),(0.,.0015),(0.,-.0015),(.002,.0015),(.002,-.0015)):
        z = old_z + dz
        if z < info['table_z']+.004 or z > info['top_z']-.001:
            rejected.append(dict(reason='height_outside_RGB', dz_m=dz,shift_m=shift));continue
        band = pts[(abs(pts[:,2]-z)<=.0016) & (abs(pts@tangent-center-shift)<=.002)]
        if len(band)<12:
            rejected.append(dict(reason='unresolved_local_RGB_chord',voxels=len(band)));continue
        lo,hi = float((band@axis).min()),float((band@axis).max())
        goals=np.array([axis*(lo-.0003),axis*(hi+.0003)])+tangent*(center+shift);goals[:,2]=z
        if time.perf_counter()-search_start>10.:
            rejected.append(dict(reason='local_search_wall_budget'));break
        q=s11_local_fit(robot,legal,goals,np.asarray(previous['inward']))
        q[3:6]=mover.last[3:6]
        points,normals,_,_=robot.tips(q)
        error=np.linalg.norm(points-goals,axis=1)
        normal_error=np.arccos(np.clip(np.sum(normals*np.asarray(previous['inward']),axis=1),-1,1))
        novelty=float(np.mean(np.linalg.norm(points-previous['points'],axis=1)))
        if error.max()>.0008 or normal_error.max()>.5 or novelty<.001:
            rejected.append(dict(reason='candidate_IK_or_novelty',error_m=error.tolist(),normal_rad=normal_error.tolist(),novelty_m=novelty));continue
        opened=close_ik(robot,q,goals-np.asarray(previous['inward'])*.002)
        opened[3:6]=mover.last[3:6]
        n=_s3_steps(opened-mover.last,48,.045,.7)
        path=smooth_segment(mover.last,opened,n,'s11_local_reposition')
        last=mover.last.copy();failure=None
        for index,(node,count,label) in enumerate(path):
            node=np.asarray(node);rate=abs(node-last)/(count/120);check=checker.check(node)
            if not check['passed'] or np.any(node<robot.lo) or np.any(node>robot.hi) or rate[:3].max()>.14 or rate[3:].max()>1.4:
                failure=dict(reason='connection_path',node=index,check=check);break
            last=node
        if failure is not None:
            rejected.append(failure);continue
        # Preflight close under the SAME object-check convention as original s5_close.
        closed=close_ik(robot,q,points+np.asarray(previous['inward'])*.0005)
        close_n=_s3_steps(closed-opened,24,.06,.8)
        if any(not checker.check(np.asarray(node),object_check=False)['passed'] for node,_,_ in smooth_segment(opened,closed,close_n,'s5_close')):
            rejected.append(dict(reason='close_path'));continue
        log('accepted',points=points.tolist(),planned_q=q.tolist(),novelty_m=novelty,
            dz_m=dz,shift_m=shift,rejected=rejected,path_nodes=len(path),
            measured_start_q=actual.tolist(),base_translation_m=(opened[:3]-actual[:3]).tolist(),
            failure_cause='unknown; new RGB chord and measured joints select distinct candidate')
        mover.checker=checker
        # Wrist attitude remains fixed before a completed release ticket.
        # Translation is bounded at IK and checked over the complete open path.
        opened[3:6]=mover.last[3:6]
        if not mover.move(opened,n,'s11_local_reposition',True):
            log('execution_guard');return False
        state.update(info=info,newinfo=info,volume=volume,checker=checker,
            c=dict(previous,q=q,points=points,meta=dict(previous['meta'],local_novelty_m=novelty,
                local_dz_m=dz,local_shift_m=shift)),local_retry_ready=True,refined=True)
        mover.sync()
        return True
    log('bounded_candidates_exhausted',rejected=rejected)
    return False

def s5_recover(s):
    global STAGE3_ATTEMPT
    STAGE3_ATTEMPT = s['attempt']
    state = dict(s, info=s['info'], newinfo=s['info'], refined=False)
    state.pop('ctx', None)
    local = globals().get('S11_CONFIG', {}).get('fingers','two')=='two' and globals().get('S11_CONFIG', {}).get('local_retry', False) and s.get('s8_previous_decision', {}).get('grounded_evidence', False)
    recovered = stage3_recover(s['ctx'], state, s11_local_probe if local else None)
    if recovered is None:
        for key in ('obs', 'last'):
            s[key] = state[key]
        return False
    for key in ('obs', 'last', 'info', 'volume', 'checker', 'c', 'refined'):
        s[key] = recovered[key]
    s['local_retry_ready'] = recovered.get('local_retry_ready', False)
    s['attempt'] += 1
    s['squeeze'] = np.full(len(s['c']['points']), 0.0005)
    return True

def early_proof(s, height=0.03, hold=360):
    if not S9_EARLY_SETTLE or height > 0.0031:
        return _S9_LIFT(s, height, hold)
    before = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    start = int(s['obs']['step'])
    target = s['last'].copy()
    target[2] = s['c']['q'][2] + height
    for q, count, label in smooth_segment(s['last'].copy(), target, 240, 's5_proof'):
        s5_step(s, q, count, label)
        c = s['obs']['image_state']
        if float(c['actual_q'][2]) - float(before['actual_q'][2]) < 0.0022:
            continue
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        r = stage3_support_pair(before, snap, 'lift')
        if r['state'] == 'grounded_evidence':
            decision = stage3_support_decision(s['lift_reports'] + [r], s['hold_reports'], attempt=s['attempt'], max_attempts=S5_ATTEMPTS, phase='proof')
            if decision['action'] in ('retry', 'stop_limit'):
                s['lift_reports'].append(r)
                s['lift_veto'] = False
                event = dict(stage='s5_proof', attempt=s['attempt'], trigger_frame=s9_frame(s['obs']), trigger_step=s['obs']['step'], capture_step=c['step'], legacy_decision_step=start + 240 + hold, saved_s=(start + 240 + hold - int(s['obs']['step'])) / 120, predicate='existing V7 grounded evidence and proof decision', report=r, decision=decision)
                _S9_IO['early'].append(event)
                s5_note(s['ctx'], 's9_early_proof', event)
                return r
    after = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
    r = stage3_support_pair(before, after, 'lift')
    s['lift_reports'].append(r)
    s['lift_veto'] = False
    s5_note(s['ctx'], 's5_lift_response', dict(step=s['obs']['step'], height_m=height, report=r))
    for _ in range(hold // 24):
        s5_step(s, s['last'], 24, 's5_hold')
        snap = stage3_support_snapshot(s['obs'], s['robot'], s5_static(s))
        s['hold_reports'].append(stage3_support_pair(after, snap, 'hold'))
        after = snap
    return r

def s5_move_segment(s, target, n, phase, object_check=False):
    n = max(n, _s3_steps(target - s['last'], minimum=n, translation=0.06, joint=0.8))
    for q, count, label in smooth_segment(s['last'].copy(), target, n, phase):
        s5_step(s, q, count, label, object_check)

def policy_main(ctx):
    s = s5_init(ctx)
    for attempt in range(S5_ATTEMPTS):
        if not s.pop('local_retry_ready', False):
            placed = attempt > 0 and globals().get('S11_CONFIG', {}).get('fingers','two')=='two' and globals().get('S11_CONFIG', {}).get('local_retry', False) and s11_post_release_plan(s)
            next_axis=S5_AXIS if attempt==0 else 'previous_orthogonal'
            if globals().get('S11_CONFIG',{}).get('fingers')=='auto': next_axis=S5_AXIS
            if not placed and not s5_plan(s,next_axis):
                return
            s5_approach(s)
            if not s.get('refined'):
                s5_refine(s)
        s.setdefault('tried_points', []).append(s['c']['points'].copy())
        s.setdefault('tried_contacts',[]).append(dict(fingers=list(s['robot'].finger_names),points=s['c']['points'].tolist(),candidate_id=s['c']['meta'].get('candidate_id')))
        s5_close(s, S5_CLOSURE, S5_CAP)
        decision = None
        for phase, height, hold in (('proof', 0.003, 48), ('middle', 0.0165, 0), ('final', 0.03, 360)):
            s5_lift(s, height, hold)
            decision = s5_decide(s, phase)
            if decision['action'] == 'reobserve':
                s5_reobserve(s)
                decision = s5_decide(s, phase, reobservations=1)
            if decision['action'] != 'continue_lift':
                break
        if decision['action'] != 'retry':
            return
        if not s5_recover(s):
            return
