"""Offline fixed-input IK and ranking audit; never runs the simulator."""
import argparse,copy,csv,gzip,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent

def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x

def load_policy(program):
    spec=importlib.util.spec_from_file_location('s16numericalpolicy',program)
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    exec((HERE/'stage16_numerics_overlay.py').read_text(),p.__dict__)
    return p

def audit_case(p,search_path):
    search=json.load(gzip.open(search_path,'rt') if search_path.suffix=='.gz' else search_path.open())
    episode=Path(search['episode']);obs=json.loads(next((episode/'public_responses.jsonl').open()))['observation']
    description=json.loads((episode/'scene.json').read_text())['robot_description']
    robot=p.Robot(description['urdf_path']);robot.configure_contacts(2)
    info=p._S3_ESTIMATE(obs);initial=np.asarray(obs['actual_q']);prior=np.asarray(p.PRIMITIVE['grasp_q'])
    checker=p.SelfAwareGeometry(robot,info)
    base=np.r_[info['axes'][:,int(np.argmin(info['half'][:2]))],0.]
    axes=[Rotation.from_euler('z',angle).apply(base) for angle in (0.,np.pi/2,np.pi/4,-np.pi/4)]
    two,rejected=p._s3_candidates_on_axes(robot,info,prior,checker,initial,axes,(.4,.6),0.,'fixed_public_RGB_stage16',retime=True)
    rows=[];candidates=[]
    for i,c in enumerate(two):
        score,terms=p.s11_contact_quality(robot,c,initial);task=p.s12_task_quality(robot,c,info)
        c['meta'].update(candidate_id='two-'+str(i),quality_score=score,quality_terms=terms,task_quality=task)
        rows.append(dict(group='two',variant='historical_weight4',candidate_index=i,meta=c['meta'],q=c['q'],path=c['prefix']))
        candidates.append(c)
    # Recreate one finite-surface IK input from each of the first two distinct
    # layouts with position+normal-valid historical final IK. Identical region
    # prior/targets are then passed independently to final weights7 and4.
    eligible=[(i,r) for i,r in enumerate(search['trials']) if r['meta']['error_m']<.0008 and max(r['meta']['normal_error_rad'])<.5]
    chosen=[]
    if '20_' in episode.parent.name:
        eligible=sorted(eligible,key=lambda value: (0 if value[0]==1 else 1,value[0]))
    for i,row in eligible:
        if not chosen or row['meta']['layout']!=chosen[0][1]['meta']['layout']:
            chosen.append((i,row))
        if len(chosen)>=2:break
    for index,trial in chosen:
        meta=trial['meta'];robot.configure_contacts(3)
        robot.contact_points[2]=np.asarray(meta['anchor_middle_link2_m'])
        axis=axes[meta['axis_index']].copy()
        if axis[0]<0:axis=-axis
        rot=Rotation.from_euler('z',np.arctan2(axis[1],axis[0])).as_matrix()
        seed=prior.copy();seed[:3]=rot@seed[:3]+np.r_[info['center'][:2],0.]
        seed[3:6]=Rotation.from_matrix(rot@Rotation.from_euler('ZYX',seed[3:6]).as_matrix()).as_euler('ZYX')
        fractions,angles=p._s12_three_layouts((.4,.6))[meta['layout']]
        regions=p._s12_rgb_contact_regions(info,axis,fractions,angles)
        targets=np.asarray([r['point_world_m'] for r in regions]);inward=-np.asarray([r['outward_normal_world'] for r in regions])
        stance=0 if '20_' in episode.parent.name else 2
        start=p._s12_three_start(robot,seed,targets,inward,stance)
        regionq,targets,refined,regionmeta=p._s12_region_ik(robot,start,regions,info)
        for weight in (7.,4.):
            q,fit=p.optimize(robot,regionq,targets,inward,normal_weight=weight)
            fit.update(regionmeta,contact_regions=refined,geometry_source='fixed_public_RGB_stage16',candidate_id='three-'+str(index)+'-w'+str(weight))
            passed=p._s12_endpoint_gate(robot,q,fit,checker,initial,False)
            candidate=p._s12_complete_candidate(robot,checker,initial,q,inward,fit,True) if passed else None
            pseudo=candidate or dict(q=q,points=robot.tips(q)[0],inward=inward,prefix=[],meta=fit)
            score,terms=p.s11_contact_quality(robot,pseudo,initial);task=p.s12_task_quality(robot,pseudo,info)
            fit.update(quality_score=score,quality_terms=terms,task_quality=task)
            rows.append(dict(group='three',variant='final_normal_weight'+str(weight),candidate_index=index,
                endpoint_passed=passed,full_path_passed=candidate is not None,meta=fit,q=q,path=pseudo['prefix'],
                path_time_available=candidate is not None,initial_region_input_q=start,historical_stance=stance,
                historical_final_q_max_abs_difference=float(np.max(np.abs(q-np.asarray(trial['q'])))) if weight==7. else None))
            if candidate:candidates.append(candidate)
    # The live feasibility-selected pool is kept separate from a declared
    # synthetic ranking witness. Rejected endpoints never get ranked as valid.
    orders={}
    if candidates:
        orders['historical']=[c['meta']['candidate_id'] for c in p._S16_HISTORICAL_ORDER(copy.deepcopy(candidates))]
        orders['bounded_relative']=[c['meta']['candidate_id'] for c in p.s16_order_candidates(copy.deepcopy(candidates))]
    frame=Path(obs['initial_overhead']['rgb_path'])
    return clean(dict(episode=str(episode),source_search=str(search_path),initial_rgb_sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),
        initial_measured_q=initial,common_task_assumptions=p.S11_CONFIG['task_wrench'],rows=rows,
        two_search_accepted=len(two),two_search_rejected=rejected,
        fixed_three_region_inputs=len(chosen),feasible_pool_orders=orders,
        warning='zero path duration on a rejected proposal means unavailable, not zero-cost motion; LP on rejected proposals is diagnostic only'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--program',type=Path,default=HERE/'stage15_evidence/release/controller.py')
    ap.add_argument('--search',type=Path,action='append');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();p=load_policy(a.program)
    searches=a.search or [HERE/'stage15_evidence/three'/f'{i}_surface_patch_search.json.gz' for i in (20,34)]
    results=[];a.output.mkdir(parents=True,exist_ok=True)
    for search in searches:
        print('starting',str(search),flush=True);results.append(audit_case(p,search))
        (a.output/'fixed_candidate_audit.json').write_text(json.dumps(dict(scope='public-only offline fixed-input numerical audit; not physical acceptance',program=str(a.program),program_sha256=hashlib.sha256(a.program.read_bytes()).hexdigest(),cases=results),indent=2))
        print('completed',str(search),flush=True)
    columns=['case','group','variant','candidate_index','position_max_m','position_rms_m','normal_max_rad','normal_rms_rad','normal_squared_loss','position_squared_loss','solver_cost','recomputed_cost','nfev','endpoint','path','nominal_alpha','robust_alpha']
    with (a.output/'candidate_table.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for case in results:
            for row in case['rows']:
                m=row['meta'];n=m['s16_numerics'];t=m['task_quality'];cost=n['squared_residual_contributions']
                writer.writerow(dict(case=Path(case['episode']).parent.name,group=row['group'],variant=row['variant'],candidate_index=row['candidate_index'],position_max_m=n['max_position_error_m'],position_rms_m=n['rms_position_error_m'],normal_max_rad=n['max_normal_error_rad'],normal_rms_rad=n['rms_normal_error_rad'],normal_squared_loss=cost['normal'],position_squared_loss=cost['position'],solver_cost=m['cost'],recomputed_cost=n['reconstructed_scipy_cost'],nfev=m['nfev'],endpoint=row.get('endpoint_passed',True),path=row.get('full_path_passed',True),nominal_alpha=t['nominal']['minimum_task_margin'],robust_alpha=t['robust']['minimum_task_margin']))
if __name__=='__main__':main()
