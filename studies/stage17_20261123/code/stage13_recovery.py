"""Public-evidence recovery for fast+auto; original guards remain authoritative."""
_S13_LEGACY_POLICY=policy_main
_S13_ORIGINAL_STEP=s5_step

def s13_log(s,event,**fields):
    s5_note(s['ctx'],'s13_recovery',dict(event=event,step=int(s['obs']['step']),
        frame_id=s9_frame(s['obs']),actual_grasps=s.get('actual_grasps',0),
        candidate_id=s.get('c',{}).get('meta',{}).get('planner_candidate_id'),**fields))

def s13_legal_start(s,checker=None):
    actual=np.asarray(s['obs']['actual_q'],dtype=float);robot=s['robot']
    legal=np.clip(actual,robot.lo+1e-7,robot.hi-1e-7)
    projection=float(np.max(abs(legal-actual)))
    if not np.isfinite(actual).all() or projection>1e-5:
        s13_log(s,'guard_refused',branch='planning',reason='measured_joint_limit',projection=projection);return None
    checker=checker or s['checker']
    for fraction in (0.,.25,.5,.75,1.):
        check=checker.check(actual+(legal-actual)*fraction)
        if not check['passed']:
            s13_log(s,'guard_refused',branch='planning',reason='actual_to_legal_connection',fraction=fraction,check=check);return None
    s13_log(s,'legal_command_seed',actual_q=actual.tolist(),legal_q=legal.tolist(),projection=projection)
    return legal

def s5_step(s,q,n,phase,object_check=False):
    pending=s.get('s13_pending_grasp')
    closure=phase in ('s5_close','s13_preload_recovery')
    if pending and closure and s.get('actual_grasps',0)>=S5_ATTEMPTS:
        s13_log(s,'guard_refused',branch=pending['branch'],reason='actual_grasp_limit');raise RuntimeError('actual grasp budget')
    before=int(s['obs']['step'])
    value=_S13_ORIGINAL_STEP(s,q,n,phase,object_check)
    if pending and closure and int(s['obs']['step'])>before:
        s['actual_grasps']=s.get('actual_grasps',0)+1;s['attempt']=s['actual_grasps']
        s13_log(s,'grasp_dispatched',branch=pending['branch'],started_step=before,
            target_q=np.asarray(q).tolist(),actual_q=s['obs']['actual_q'],
            contact_count=len(s['c']['points']),points=s['c']['points'].tolist(),
            command_plan_sha256=s['c']['meta'].get('command_plan_sha256'),
            planned_q=s['c']['q'].tolist(),recovery_id=pending.get('recovery_id'))
        s.pop('s13_pending_grasp',None)
    return value

def s13_preload_proposal(s,evidence):
    """Servo deficit is a proxy; never report a measured normal force."""
    squeeze=np.asarray(s['squeeze']);cap=float(s.get('last_closure_cap',S5_CAP))
    visible=set(evidence.get('visible_fingers',[]));motors=evidence.get('motor',{}).get('fingers',{})
    names=s['robot'].finger_names
    fresh=bool(evidence.get('step_after')==s.get('closure_evidence_step') and 0<=int(s['obs']['step'])-int(evidence.get('step_after',-1000))<=12 and evidence.get('wrist_fixed') and not evidence.get('visual_evidence_reused_for_state_only'))
    settled=all(motors.get(f,{}).get('settled',False) for f in names)
    deficits=[motors.get(f,{}).get('drive_deficit') for f in names]
    confidence=float(evidence.get('confidence',0.))
    observed=bool(fresh and confidence>=.5 and set(names)<=visible and settled and all(d is not None for d in deficits))
    motion=bool(evidence.get('moving_fingers'))
    if observed and (motion or max(deficits)>.85) and np.all(squeeze>.0005+.0001):
        return np.maximum(.0005,squeeze-.00025),dict(eligible=True,direction='decrease',reason='fresh_registered_motion_or_high_drive_deficit',confidence=confidence,deficits=deficits)
    if observed and s.get('s8_previous_decision',{}).get('grounded_evidence') and max(deficits)<.45 and np.all(squeeze+.00025<=cap):
        return squeeze+.00025,dict(eligible=True,direction='increase',reason='fresh_grounded_bilateral_visibility_low_deficit_with_cap_reserve',confidence=confidence,deficits=deficits)
    return None,dict(eligible=False,reason='no_credible_fresh_bilateral_response_or_preload_margin',fresh=fresh,confidence=confidence,settled=settled,deficits=deficits,squeeze_m=squeeze.tolist(),cap_m=cap)

def s13_preflight(s,path,checker=None,object_check=False):
    checker=checker or s['checker'];previous=s['last'].copy();robot=s['robot']
    for i,(node,count,label) in enumerate(path):
        node=np.asarray(node);check=checker.check(node,object_check=object_check);rate=abs(node-previous)/(count/120)
        if not check['passed'] or np.any(node<robot.lo) or np.any(node>robot.hi) or rate[:3].max()>.14 or rate[3:].max()>1.4:
            return dict(passed=False,node=i,phase=label,check=check,rate=rate.tolist())
        previous=node
    return dict(passed=True,nodes=len(path))

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
    rid=s.setdefault('recovery_serial',0)+1;s['recovery_serial']=rid
    s13_log(s,'selected',branch='preload',recovery_id=rid,**reason)
    target=close_ik(s['robot'],s['c']['q'],s['c']['points']+s['c']['inward']*proposal[:,None])
    n=_s3_steps(target-s['last'],48,.06,.8);path=smooth_segment(s['last'].copy(),target,n,'s13_preload_recovery')
    audit=s13_preflight(s,path)
    s13_log(s,'guard',branch='preload',recovery_id=rid,audit=audit)
    if not audit['passed']:return False
    before=s['squeeze'].copy();s['s13_pending_grasp']=dict(branch='preload',recovery_id=rid)
    for q,count,label in path:s5_step(s,q,count,label)
    s['squeeze']=proposal;s5_move(s,s['last'],48,'s5_preload_hold')
    s13_log(s,'executed',branch='preload',recovery_id=rid,old_squeeze_m=before.tolist(),new_squeeze_m=proposal.tolist(),force_known=False)
    s['lift_reports']=[];s['hold_reports']=[];return True

def s13_local_fit(robot,actual,goals,inward):
    indices=np.r_[np.arange(3),robot.active_finger_indices];start=np.asarray(actual).copy()
    lower=robot.lo[indices]+1e-7;upper=robot.hi[indices]-1e-7
    lower[:3]=np.maximum(lower[:3],start[:3]-[.002,.002,.008]);upper[:3]=np.minimum(upper[:3],start[:3]+[.002,.002,.002])
    def residual(x):
        q=start.copy();q[indices]=x;points,normals,_,_=robot.tips(q)
        return np.r_[(points-goals).ravel()*1500,(normals-inward).ravel()*4,(q[:3]-start[:3])*2,(q[robot.active_finger_indices]-start[robot.active_finger_indices])*.03]
    fit=least_squares(residual,np.clip(start[indices],lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=45)
    q=start.copy();q[indices]=fit.x;return q

def s13_local_probe(mover,state):
    """Called after actual opening; new map and all active pads required."""
    mover.sync();state.update(obs=mover.obs,last=mover.last.copy(),ctx=mover.ctx)
    if state.get('s13_local_calls',0)>=2:return False
    state['s13_local_calls']=state.get('s13_local_calls',0)+1
    rid=state.setdefault('recovery_serial',0)+1;state['recovery_serial']=rid
    s13_log(state,'selected',branch='local_pose',recovery_id=rid,opening_already_executed=True)
    volume=Stage3LiveVolume(mover.robot,state['volume'].calibration,state['info'])
    info,report=stage3_live_geometry(volume,mover.obs,state['info'])
    if info is None:
        s13_log(state,'guard_refused',branch='local_pose',recovery_id=rid,reason='fresh_RGB_map',report=report);return False
    checker=SelfAwareGeometry(mover.robot,info);legal=s13_legal_start(state,checker)
    if legal is None:return False
    if not checker.check(mover.last)['passed']:
        s13_log(state,'guard_refused',branch='local_pose',reason='commanded_start_clearance');return False
    previous=state['c'];old=np.asarray(previous['points']);inward=np.asarray(previous['inward']);pts=volume.points
    tangent=np.cross([0.,0.,1.],inward[0]);rejected=[]
    for dz,shift in ((.002,0.),(-.002,0.),(0.,.0015),(0.,-.0015)):
        guessed=old+np.array([0.,0.,dz])+tangent*shift;goals=[]
        for goal,normal in zip(guessed,inward):
            near=pts[(abs(pts[:,2]-goal[2])<.0016)&(np.linalg.norm((pts-goal)@np.array([tangent,[0.,0.,1.]]).T,axis=1)<.0025)]
            if len(near)<12:break
            # First visible surface along this inward ray, not interior voxels.
            projection=near@normal;surface=near[projection<=projection.min()+.0005]
            goals.append(surface[np.argmin(np.linalg.norm(surface-goal,axis=1))])
        if len(goals)!=len(old):rejected.append(dict(reason='unresolved_all_pad_RGB_patch',dz=dz,shift=shift));continue
        goals=np.asarray(goals)
        if np.any(goals[:,2]<info['table_z']+.003) or np.any(goals[:,2]>info['top_z']-.001):continue
        q=s13_local_fit(mover.robot,legal,goals,inward);q[3:6]=mover.last[3:6]
        points,normals,_,_=mover.robot.tips(q);error=np.linalg.norm(points-goals,axis=1);angle=np.arccos(np.clip(np.sum(normals*inward,axis=1),-1,1));novelty=float(np.mean(np.linalg.norm(points-old,axis=1)))
        if error.max()>.0008 or angle.max()>.5 or not .001<=novelty<=.004:
            rejected.append(dict(reason='IK_or_bounded_novelty',error_m=error.tolist(),normal_rad=angle.tolist(),novelty_m=novelty));continue
        opened=close_ik(mover.robot,q,goals-inward*.006);opened[3:6]=mover.last[3:6]
        path=smooth_segment(mover.last,opened,_s3_steps(opened-mover.last,48,.045,.7),'s13_local_reposition')
        audit=s13_preflight(state,path,checker,True)
        closed=close_ik(mover.robot,q,points+inward*.0005)
        close_path=smooth_segment(opened,closed,_s3_steps(closed-opened,24,.06,.8),'s5_close')
        close_audit=s13_preflight(dict(state,last=opened),close_path,checker,False)
        if not audit['passed'] or not close_audit['passed']:
            rejected.append(dict(reason='local_or_close_path',path=audit,close=close_audit));continue
        s13_log(state,'guard',branch='local_pose',recovery_id=rid,audit=audit,close_audit=close_audit,new_RGB=report)
        mover.checker=checker
        if not mover.move(opened,sum(n for _,n,_ in path),'s13_local_reposition',True):
            s13_log(state,'guard_refused',branch='local_pose',reason='runtime_guard');return False
        meta=dict(previous['meta'],targets=goals.tolist(),error_m=float(error.max()),per_finger_error_m=error.tolist(),normal_error_rad=angle.tolist(),local_novelty_m=novelty,local_dz_m=dz,local_shift_m=shift,local_close_path_nodes=len(close_path))
        candidate=dict(previous,q=q,points=points,prefix=path,meta=meta)
        meta['task_quality']=s12_task_quality(mover.robot,candidate,info)
        _s12_candidate_identity(candidate)
        state.update(info=info,newinfo=info,volume=volume,checker=checker,c=candidate,local_retry_ready=True,refined=True,s13_recovery_branch='local_pose',s13_recovery_id=rid)
        mover.sync();s13_log(state,'executed',branch='local_pose',recovery_id=rid,novelty_m=novelty,dz_m=dz,shift_m=shift)
        return True
    s13_log(state,'guard_refused',branch='local_pose',recovery_id=rid,reason='bounded_candidates_exhausted',rejected=rejected)
    return False

def s13_release_recover(s):
    global STAGE3_ATTEMPT
    STAGE3_ATTEMPT=s['actual_grasps'];state=dict(s,newinfo=s['info'],refined=False)
    s13_log(s,'considered',branch='local_pose',reason='fresh_RGB_and_local_path_required',contact_count=len(s['c']['points']))
    # Local callback may return after a guarded partial opening. Otherwise the
    # SAME physical release FSM completes full retreat and a fresh map ticket.
    s13_log(s,'selected',branch='release_resample',recovery_id=s.get('recovery_serial',0)+1,conditional_fallback=True)
    recovered=stage3_recover(s['ctx'],state,s13_local_probe)
    s.update({k:state[k] for k in ('obs','last','recovery_serial','s13_local_calls') if k in state})
    if recovered is None:
        s13_log(s,'guard_refused',branch='release_resample',reason='original_release_retreat_or_fresh_map_gate');return False
    for key in ('obs','last','info','volume','checker','c','refined','local_retry_ready','s13_recovery_branch','s13_recovery_id'):
        if key in recovered:s[key]=recovered[key]
    s['attempt']=s['actual_grasps']+1;s['squeeze']=np.full(len(s['c']['points']),.0005)
    if recovered.get('local_retry_ready'):
        s['next_branch']='local_pose';return True
    s['next_branch']='release_resample';s13_log(s,'executed',branch='release_resample',ticket=STAGE3_RELEASE_TICKET)
    return True

def s13_evaluate(s):
    for phase,height,hold in (('proof',.003,48),('middle',.0165,0),('final',.03,360)):
        s5_lift(s,height,hold);decision=s5_decide(s,phase)
        if decision['action']=='reobserve':s5_reobserve(s);decision=s5_decide(s,phase,reobservations=1)
        if decision['action']!='continue_lift':return decision
    return decision

def policy_main(ctx):
    if not globals().get('S11_CONFIG',{}).get('evidence_recovery'):return _S13_LEGACY_POLICY(ctx)
    s=s5_init(ctx);s['actual_grasps']=0;s['next_branch']='initial'
    while s['actual_grasps']<S5_ATTEMPTS:
        if not s.pop('preload_ready',False):
            local=s.pop('local_retry_ready',False)
            if not local:
                axis=S5_AXIS if not s['actual_grasps'] else 'previous_orthogonal'
                if not s5_plan(s,axis):s13_log(s,'stopped',reason='no_feasible_next_grasp');return
                s5_approach(s)
                if not s.get('refined'):s5_refine(s)
            s.setdefault('tried_contacts',[]).append(dict(points=s['c']['points'].tolist()))
            s['s13_pending_grasp']=dict(branch=s['next_branch'],recovery_id=s.get('s13_recovery_id'))
            s5_close(s,S5_CLOSURE,S5_CAP)
        decision=s13_evaluate(s)
        s13_log(s,'failure_decision',decision=decision)
        if decision['action']!='retry' or s['actual_grasps']>=S5_ATTEMPTS:return
        if s13_preload(s):s['preload_ready']=True;continue
        if not s13_release_recover(s):return
