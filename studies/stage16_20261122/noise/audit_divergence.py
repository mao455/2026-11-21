"""Read two critical 0/20.4-DN pairs without starting simulation or publishing frames."""
import argparse, hashlib, json
from pathlib import Path


def records(p):
    with p.open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage14-root',type=Path,required=True)
    p.add_argument('--stage16-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    pairs=[('historical_stage14',a.stage14_root/'noise_closed_loop/01_abc_311740_clean_dn0_seed141_v2/episode',a.stage14_root/'noise_closed_loop/01_abc_311740_clean_dn20.4_seed141_v3/episode'),
           ('stage16_development',a.stage16_root/'noise/noise01_seed141_dn0/episode',a.stage16_root/'noise/noise01_seed141_dn20.4/episode')]
    out=[]
    for pop,b,n in pairs:
        traces=[]; maps=[];closures=[]
        for ep in (b,n):
            rr=list(records(ep/'agent_trace.jsonl'))
            traces.append([r for r in rr if r.get('op')=='step'])
            notes=[(k,v) for r in rr if r.get('op')=='note' for k,v in json.loads(r['text']).items()]
            maps.append(next(v for k,v in notes if k=='s5_near_map'))
            closures.append([{k:v.get(k) for k in ('step','round','mode','reason','squeeze_m')} for name,v in notes if name=='s5_closure_action'])
        step=0;exact=None;material=None
        for index,(x,y) in enumerate(zip(*traces)):
            diff=[abs(u-v) for u,v in zip(x['target_q'],y['target_q'])]
            changed=x['steps']!=y['steps'] or x['phase']!=y['phase']
            row=dict(command_index=index,start_tick=step,phase=x['phase'],linear_target_delta_max_m=max(diff[:3]),angular_target_delta_max_rad=max(diff[3:]))
            if exact is None and (changed or max(diff)>0):exact=row
            if material is None and (changed or max(diff[:3])>1e-6 or max(diff[3:])>1e-5):material=row
            step+=x['steps']
        first_state=None;first_bad=None
        for x,y in zip(records(b/'steps.jsonl'),records(n/'steps.jsonl')):
            if first_state is None and any(x[k]!=y[k] for k in ('actual_q','actual_qvel','object_pose','target_q')):
                first_state=dict(tick=x['step'],time_s=x['time_s'],phase=x['phase'],
                                 joint_coordinate_delta_max=max(abs(u-v) for u,v in zip(x['actual_q'],y['actual_q'])),
                                 object_pose_component_delta_max=max(abs(u-v) for u,v in zip(x['object_pose'],y['object_pose'])))
            if not y['penetration']['passed']:
                first_bad=dict(tick=y['step'],time_s=y['time_s'],phase=y['phase'],**y['penetration']); break
        exposure=[]
        for ep in (b,n):
            r=next(r for r in records(ep/'image_manifest.jsonl') if r['step']==1308)
            exposure.append(dict(capture_tick=r['step'],frame_id=r['frame_id'],capture_time_s=r['time_s'],actual_q=r['actual_q'],actual_qvel=r['actual_qvel'],
                                 cameras={k:dict(frame_id=v.get('frame_id',r['frame_id']),pixel_sha256=r['sha256'][k]) for k,v in r['fingereye'].items()}))
        keep=('method','step','usable_views','voxel_m','occupied_voxels','center_world_m','bounds_world_m','valid')
        out.append(dict(population=pop,initial_rgb_file_equal=sha(b/'public/overhead_initial.npy')==sha(n/'public/overhead_initial.npy'),
                        baseline_first_map={k:v for k,v in maps[0].items() if k in keep},noisy_first_map={k:v for k,v in maps[1].items() if k in keep},
                        first_exact_command_difference=exact,first_difference_above_reporting_resolution=material,
                        reporting_resolution=dict(linear_m=1e-6,angular_rad=1e-5,meaning='reporting only; not safety or success threshold'),
                        first_state_difference=first_state,first_noisy_geometry_failure=first_bad,closure_schedule_equal=closures[0]==closures[1],
                        critical_exposures=exposure,command_counts=[len(t) for t in traces]))
    (a.output/'first_divergence.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps([dict(population=r['population'],first_command=r['first_exact_command_difference'],first_bad=r['first_noisy_geometry_failure']) for r in out]))

if __name__=='__main__':main()
