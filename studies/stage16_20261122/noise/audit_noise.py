"""Offline raw noise audit; paths/assets stay local and only scalar evidence is written.

Needs the private episode archive, plus the evaluator in --here. Does not run
simulation or change inputs. Output paths are symbolic $STAGE14_ROOT/$STAGE16_ROOT.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, statistics, sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def jl(path):
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def csvout(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys, lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('here', 'stage14-root', 'stage16-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.here.resolve()))
    from evaluator import evaluate
    archive = load(a.here / 'stage16_evidence/final/readout.json')
    prior = load(a.here / 'stage16_evidence/evidence/audit.json')
    historical = load(a.stage14_root / 'noise_closed_loop_readout/summary.json')
    cells = []; provenance = []; errors = []
    fresh = {}
    def read(group, id, ep, expected):
        host = load(ep / 'result.json'); scene = load(ep / 'scene.json')
        states = jl(ep / 'steps.jsonl')
        score = evaluate(states, scene['initial_object_pose'][2])
        execution = bool(host['controller_completed'] and host['error'] is None
                         and host['evaluator_and_adapter_unchanged'])
        match = score == host['physics']
        success = bool(execution and score['success'])
        if not match or success != expected['success']: errors.append(id)
        case = scene['case']; first = next((r for r in states if not r['penetration']['passed']), None)
        root = '$STAGE14_ROOT' if group == 'historical_stage14' else '$STAGE16_ROOT'
        base = a.stage14_root if group == 'historical_stage14' else a.stage16_root
        raw = root + '/' + ep.relative_to(base).as_posix()
        row = dict(population=group, id=id, object_id=case['object_id'], seed=case['noise_seed'],
                   sigma_dn=case.get('rgb_noise_std', 0), success=success,
                   status='passed' if success else 'failed', execution_valid=execution,
                   score_matches_raw_record=match, actual_grasps=expected.get('actual_grasps', expected.get('attempts')),
                   physical_steps=len(states), physical_s=len(states)/120,
                   hold_lift_mm=1000*score['hold_min_lift_m'],
                   hold_clearance_mm=1000*score['hold_min_table_clearance_m'],
                   hold_duration_s=score['hold_duration_s'], geometry_passed=score['geometry_passed'],
                   single_hold_passed=score['hold_physics_passed'], failure=score.get('failure', 'formal_success'),
                   first_bad_tick=first['step'] if first else None,
                   first_bad_time_s=first['time_s'] if first else None,
                   first_bad_links=';'.join(first['penetration'].get('hand_object_intrusion_links', [])) if first else '',
                   controller_sha256=host['controller_sha256'], raw_episode=raw,
                   failure_time_boundary='first authored geometry bad state' if first else 'terminal hold window assessed at end of physical stream' if not success else '',
                   terminal_hold_start_tick=len(states)-240,terminal_hold_end_tick=len(states)-1,
                   required_actual_lift_mm=28,required_clearance_mm=5)
        cells.append(row); fresh[(group,id)] = row
        for name in ('result.json','scene.json','steps.jsonl','agent_trace.jsonl','image_manifest.jsonl',
                     'public_responses.jsonl','all_hand_object_contacts.jsonl','program/controller.py','stage9_backend_executed.py'):
            path = ep / name
            if path.is_file():
                provenance.append(dict(population=group,id=id,source=raw+'/'+name,sha256=sha(path),bytes=path.stat().st_size))
        del states
    for cell in historical['selected']:
        read('historical_stage14', cell['name'], a.stage14_root/'noise_closed_loop'/cell['name']/'episode', cell)
    for row in archive['rows']:
        if row['id'].startswith('noise/'):
            ep=a.stage16_root/row['id']/'episode'
            read('stage16_development', row['id'], ep, row)
    pairs=[]
    for row in prior['noise_pairs']:
        b=fresh[('historical_stage14',row['baseline_episode'])]; n=fresh[('historical_stage14',row['noisy_episode'])]
        pairs.append(dict(population='historical_stage14', sigma_dn=row['sigma_dn'], object_id=row['object_id'],seed=row['seed'],
                          baseline_id=b['id'],noisy_id=n['id'],baseline_success=b['success'],noisy_success=n['success'],
                          outcome=('S' if b['success'] else 'F')+('S' if n['success'] else 'F'),
                          case_equal_except_noise=row['case_equal_except_noise'],shared_initial_rgb='verified_for_critical_seed141;other_pairs_not_claimed'))
    for row in archive['paired_noise']:
        b=fresh[('stage16_development',row['baseline_id'])]; n=fresh[('stage16_development',row['noisy_id'])]
        pairs.append(dict(population='stage16_development',sigma_dn=row['sigma_dn'],object_id=n['object_id'],seed=row['seed'],
                          baseline_id=b['id'],noisy_id=n['id'],baseline_success=b['success'],noisy_success=n['success'],
                          outcome=('S' if b['success'] else 'F')+('S' if n['success'] else 'F'),case_equal_except_noise=True,
                          shared_initial_rgb=row['shared_initial_rgb_verified']))
    counts=[]
    for pop in ('historical_stage14','stage16_development'):
        for dose in (0,3,6,12,20.4):
            cc=[r for r in cells if r['population']==pop and r['sigma_dn']==dose]
            pp=[r for r in pairs if r['population']==pop and r['sigma_dn']==dose]
            invalid=[r for r in historical['all_launches_rows'] if pop=='historical_stage14' and not r['execution_valid'] and ('_dn'+str(dose)+'_') in r['name']]
            counts.append(dict(population=pop,sigma_dn=dose,valid_cells=len(cc),successes=sum(r['success'] for r in cc),
                               complete_pairs=len(pp),SS=sum(r['outcome']=='SS' for r in pp),SF=sum(r['outcome']=='SF' for r in pp),
                               FS=sum(r['outcome']=='FS' for r in pp),FF=sum(r['outcome']=='FF' for r in pp),invalid_launches=len(invalid)))
    invalid=[dict(population='historical_stage14',**r) for r in historical['all_launches_rows'] if not r['execution_valid']]
    dose=list(csv.DictReader((a.here/'stage15_evidence/noise/same_exposure_dose.csv').open()))
    dose_stats=[]
    for sigma in (0,3,6,12,20.4):
        rr=[r for r in dose if float(r['sigma_dn'])==sigma]
        out=dict(sigma_dn=sigma,view_exposure_seed_rows=len(rr),visibility_flips=sum(r['visible_flip']=='True' for r in rr),
                 ambiguity_flips=sum(r['ambiguity_flip']=='True' for r in rr))
        for key in ('mask_iou','mask_boundary_iou','centroid_shift_px','residual_std_dn','low_clipping_fraction','high_clipping_fraction'):
            values=[float(r[key]) for r in rr if r[key] != '']
            out[key+'_valid_rows']=len(values)
            out[key+'_median']=statistics.median(values) if values else None
        dose_stats.append(out)
    csvout(a.output/'episodes.csv',cells);csvout(a.output/'pairs.csv',pairs);csvout(a.output/'dose_counts.csv',counts)
    csvout(a.output/'infrastructure_failures.csv',invalid);csvout(a.output/'raw_provenance.csv',provenance)
    csvout(a.output/'same_exposure_dose.csv',dose);csvout(a.output/'same_exposure_summary.csv',dose_stats)
    for name in ('evaluator.py','stage15_noise_evidence.py','stage15_camera_noise.py',
                 'stage16_evidence/final/readout.json','stage16_evidence/evidence/noise_first_divergence.json',
                 'stage15_evidence/noise/same_exposure_dose.csv'):
        path=a.here/name;provenance.append(dict(population='audit_source',id='',source='$HERE/'+name,sha256=sha(path),bytes=path.stat().st_size))
    csvout(a.output/'raw_provenance.csv',provenance)
    report=dict(scope='Fresh independent raw re-read; no new physical simulation',source_evidence_commit='d584af11938b382a2dd7852c53a7b7802239812d',
                evaluator_sha256=sha(a.here/'evaluator.py'),physical_cells=len(cells),score_mismatches=errors,
                historical_valid=14,historical_invalid=3,stage16_valid=6,stage16_successes=sum(r['success'] for r in cells if r['population']=='stage16_development'),
                raw_file_hashes=len(provenance),same_exposure_rows=len(dose),real_camera_status='blocked: no authorized driver/static RGB sequence with exposure/gain/ISP/lighting provenance')
    (a.output/'audit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if errors:raise SystemExit('Raw evidence mismatch: '+repr(errors))
    print(json.dumps(report))

if __name__=='__main__':main()
