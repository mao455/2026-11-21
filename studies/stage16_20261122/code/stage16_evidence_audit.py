"""Re-read archived physical episodes; truth stays in this offline audit only."""
import argparse, csv, gzip, hashlib, json, statistics
from pathlib import Path
from evaluator import evaluate
from stage15_readout import read, sha

HERE=Path(__file__).resolve().parent

def load(p):
    return json.loads(Path(p).read_text())

def csvout(path, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def same_exposure_stats(rows):
    """Use the recorded schema and fail visibly if a required metric disappears."""
    result=[]
    for sigma in (0,3,6,12,20.4):
        selected=[r for r in rows if float(r['sigma_dn'])==sigma]
        out=dict(sigma_dn=sigma,rows=len(selected))
        for key in ('mask_iou','mask_boundary_iou','centroid_shift_px'):
            if selected:
                values=[float(r[key]) for r in selected if r[key] not in ('',None)]
                out[key+'_measured_rows']=len(values)
                out[key+'_missing_rows']=len(selected)-len(values)
                out[key+'_median']=statistics.median(values) if values else None
        result.append(out)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'stage16_evidence/evidence');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    prior=load(HERE/'stage15_evidence/acceptance_final/readout.json')
    accepted=[]; timelines=[]; failures=[]
    for archived in prior['rows']:
        ep=Path(archived['episode']); fresh=read(ep)
        fresh['archived_metrics_match']=all(fresh[k]==archived[k] for k in ('execution_valid','success','legacy_strict_success','actual_grasps','first_invalid_step','physical_steps','physics'))
        fresh['case_name']=ep.parent.parent.name
        accepted.append(fresh)
        tick=0
        for line in (ep/'agent_trace.jsonl').open():
            r=json.loads(line)
            if r.get('op')=='step':
                if any(x in r.get('phase','') for x in ('release','retreat','retry','recovery','reproof','observe','resample')):
                    timelines.append(dict(case=fresh['case_name'],step=tick,physical_s=tick/120,host_elapsed_s=r['elapsed_s'],kind='command',event=r['phase'],duration_steps=r['steps'],target_q=json.dumps(r['target_q'])))
                tick+=r['steps']
            elif r.get('op')=='note':
                d=json.loads(r['text'])
                for k,v in d.items():
                    if k=='s13_recovery' or ('map' in k and tick>3000):
                        timelines.append(dict(case=fresh['case_name'],step=v.get('step',tick) if isinstance(v,dict) else tick,physical_s=tick/120,host_elapsed_s=r['elapsed_s'],kind='note',event=k,data=json.dumps(v,sort_keys=True)))
        first=fresh['first_invalid_step']
        if first is not None:
            state=[]
            for line in (ep/'steps.jsonl').open():
                r=json.loads(line)
                if first-2<=r['step']<=first+2:state.append(r)
            failures.append(dict(case=fresh['case_name'],first_invalid_step=first,window=state,physics=fresh['physics'],episode=str(ep)))
    historical_root=Path('<LOCAL_DOCS>/fingereye_stage14/FingerEyeLab/outputs/stage14_20261122')
    historical=load(historical_root/'noise_closed_loop_readout/summary.json');noise=[]
    for cell in historical['selected']:
        ep=historical_root/'noise_closed_loop'/cell['name']/'episode'; scene=load(ep/'scene.json');host=load(ep/'result.json')
        raw=[json.loads(l) for l in (ep/'steps.jsonl').open() if l.strip()];score=evaluate(raw,scene['initial_object_pose'][2]);del raw
        noise.append(dict(**cell,raw_score_success=score['success'],raw_score_equal=score==host['physics'],summary_equal=score['success']==cell['success'],geometry_passed=score['geometry_passed'],hold_physics_passed=score['hold_physics_passed'],physical_s=score['steps']/120,host_elapsed_s=host['elapsed_s'],controller_sha256=host['controller_sha256'],steps_sha256=sha(ep/'steps.jsonl'),episode=str(ep),case=scene.get('case')))
    cells={r['name']:r for r in noise};pairs=[]
    for p in historical['pairs']:
        base=cells[p['baseline_episode']];noisy=cells[p['noisy_episode']]
        pairs.append(dict(**p,raw_verified=base['raw_score_equal'] and noisy['raw_score_equal'],case_equal_except_noise={k:v for k,v in base['case'].items() if k!='rgb_noise_std'}=={k:v for k,v in noisy['case'].items() if k!='rgb_noise_std'},outcome=('SS' if base['raw_score_success'] and noisy['raw_score_success'] else 'SF' if base['raw_score_success'] else 'FS' if noisy['raw_score_success'] else 'FF')))
    counts=[]
    for dose in (0,3,6,12,20.4):
        selected=[r for r in noise if r['sigma_dn']==dose]; pp=[r for r in pairs if r['sigma_dn']==dose]
        invalid=[r for r in historical['all_launches_rows'] if not r['execution_valid'] and ('_dn'+str(dose)+'_') in r['name']]
        counts.append(dict(sigma_dn=dose,valid_cells=len(selected),successes=sum(r['raw_score_success'] for r in selected),complete_pairs=len(pp),SS=sum(r['outcome']=='SS' for r in pp),SF=sum(r['outcome']=='SF' for r in pp),FS=sum(r['outcome']=='FS' for r in pp),FF=sum(r['outcome']=='FF' for r in pp),invalid_launches=len(invalid),scope='closed loop' if selected else 'same-exposure replay only; no archived closed-loop sample'))
    dose=list(csv.DictReader((HERE/'stage15_evidence/noise/same_exposure_dose.csv').open()))
    dose_stats=same_exposure_stats(dose)
    result=dict(scope='Offline raw re-read, no new physical episodes',acceptance_count=len(accepted),acceptance_valid=sum(r['execution_valid'] for r in accepted),acceptance_success=sum(r['success'] for r in accepted),acceptance_strict=sum(r['legacy_strict_success'] for r in accepted),all_acceptance_metrics_match=all(r['archived_metrics_match'] for r in accepted),noise_all_raw_scores_equal=all(r['raw_score_equal'] and r['summary_equal'] for r in noise),noise_counts=counts,noise_pairs=pairs,noise_cells=noise,acceptance=accepted,invalid_noise_launches=[r for r in historical['all_launches_rows'] if not r['execution_valid']],same_exposure=dose_stats,first_observed_degradation=dict(dose_dn=20.4,object_id='abc_311740',seed=141,closed_loop_seeds_at_this_dose=2,losses_on_this_object=1,missing_closed_loop_doses=[6,12],claim='First observed among sampled 0/3/20.4; not a physical threshold or isolated noise-only cause'),evaluator_sha256=sha(HERE/'evaluator.py'))
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with gzip.open(a.output/'failure_windows.json.gz','wt') as f:json.dump(failures,f)
    csvout(a.output/'noise_cells.csv',[{k:v for k,v in r.items() if k!='case'} for r in noise]);csvout(a.output/'noise_pairs.csv',pairs);csvout(a.output/'noise_counts.csv',counts);csvout(a.output/'recovery_timeline.csv',timelines)
    csvout(a.output/'acceptance.csv',[{k:r[k] for k in ('case_name','episode','success','legacy_strict_success','actual_grasps','physical_s','host_elapsed_s','first_invalid_step','failure','archived_metrics_match')} for r in accepted])
    print(json.dumps({k:v for k,v in result.items() if k not in ('acceptance','noise_cells','noise_pairs')}))
if __name__=='__main__':main()
