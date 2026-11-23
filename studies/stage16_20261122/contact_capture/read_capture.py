"""Read actual new native all-link points from the evaluator-only sidecar."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--experiment-here', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    sys.path.insert(0, str(args.experiment_here))
    from stage15_batch import result_record
    ep = args.session / 'episode'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    result = result_record(args.session)
    assert result['completed'] and result['raw_recording_verified'], result
    data = [json.loads(line) for line in (ep / 'steps.jsonl').open()]
    sidecar = [json.loads(line) for line in (ep / 'all_hand_object_contacts.jsonl').open()]
    hold_start = len(data) - 240
    window = [row for row in sidecar if row['step'] >= hold_start]
    assert len(window) == 40
    links = [f'left_hand_{finger}_camera_component{component}_link'
             for finger in ('thumb', 'index', 'middle') for component in (1, 2, 3)]
    summary = []
    points = []
    for link in links:
        forces = []
        positions = []
        separations = []
        loaded_positions = []
        loaded_separations = []
        counts = []
        for sample in window:
            contact = next((row for row in sample['contacts'] if row['link'] == link), None)
            forces.append(contact['normal_force_sum_n'] if contact else 0.)
            counts.append(contact['count'] if contact else 0)
            if contact is None:
                continue
            assert len(contact['points_world_m']) == contact['count']
            for i, (point, normal, force, separation) in enumerate(zip(
                    contact['points_world_m'], contact['normals_world'],
                    contact['normal_forces_n'], contact['separations_m'])):
                positions.append(point)
                separations.append(separation)
                if force > .001:
                    loaded_positions.append(point)
                    loaded_separations.append(separation)
                points.append(dict(step=sample['step'], time_s=sample['time_s'],
                    phase=data[sample['step']]['phase'], link=link, point_index=i,
                    x_world_m=point[0], y_world_m=point[1], z_world_m=point[2],
                    nx=normal[0], ny=normal[1], nz=normal[2],
                    normal_force_n=force, separation_mm=separation*1000))
        summary.append(dict(link=link, samples=len(window),
            positive_load_samples_over_1mN=sum(force > .001 for force in forces),
            min_force_n=min(forces), max_force_n=max(forces),
            native_point_count=sum(counts), contact_counts_per_sample=counts,
            actual_points_bounds_world_m=[[min(p[d] for p in positions) for d in range(3)],
                                          [max(p[d] for p in positions) for d in range(3)]] if positions else None,
            min_separation_mm=min(separations)*1000 if separations else None,
            max_separation_mm=max(separations)*1000 if separations else None,
            loaded_point_threshold_n=.001, loaded_native_points=len(loaded_positions),
            loaded_points_bounds_world_m=[[min(p[d] for p in loaded_positions) for d in range(3)],
                                          [max(p[d] for p in loaded_positions) for d in range(3)]] if loaded_positions else None,
            min_loaded_separation_mm=min(loaded_separations)*1000 if loaded_separations else None,
            max_loaded_separation_mm=max(loaded_separations)*1000 if loaded_separations else None))
    pads = [row for row in summary if 'component3' in row['link']]
    housing = [row for row in summary if 'component3' not in row['link'] and row['positive_load_samples_over_1mN']]
    host = json.loads((ep / 'result.json').read_text())
    commands = [json.loads(line) for line in (ep / 'agent_trace.jsonl').open()]
    closes = [row for row in commands if row.get('op') == 'note' and 's11_close_begin' in row.get('text', '')]
    output = dict(scope='new isolated physical diagnostic with private all-link native points capture',
        source_commit='60b5a7b007f', original_controller_sha256=host['controller_sha256'],
        evaluator_unchanged=True, public_controller_unchanged=True,
        private_sidecar_only_patch=True, independent_raw_result_verified=result['raw_recording_verified'],
        success=host['success'], legacy_strict=host['physics']['legacy_strict']['success'],
        physics=host['physics'], result_elapsed_s=host['elapsed_s'],
        physical_s=len(data)/120, actual_closures=len(closes),
        all_three_pads_accepted=bool(host['success'] and all(row['positive_load_samples_over_1mN'] == 40 for row in pads) and not housing),
        classification='housing_assisted_three_diagnostic' if housing else 'three_pad' if all(row['positive_load_samples_over_1mN'] == 40 for row in pads) else 'other',
        cadence_hz=20, native_physics_hz=120, hold_window_steps=[hold_start,len(data)-1],
        sampled_steps=[row['step'] for row in window],
        coordinates='actual engine contact points and unit normals in world frame; points m, force N, separation mm',
        contacts=summary, raw_hashes={name:sha(ep/name) for name in [
            'result.json','steps.jsonl','all_hand_object_contacts.jsonl',
            'all_hand_object_contact_manifest.json','agent_trace.jsonl']},
        sidecar_source_sha256=sha(args.experiment_here/'stage2_contact_audit.py'),
        initial_input_audit=json.loads((ep/'initial_overhead_input_audit.json').read_text()))
    original=args.experiment_here.parents[3]/'fingereye_stage16/FingerEyeLab/outputs/stage16/diagnostic/20_safe/episode/steps.jsonl'
    output['identical_steps_sha256_to_original_diagnostic']=original.exists() and sha(ep/'steps.jsonl')==sha(original)
    def portable(value):
        if isinstance(value,dict):return {key:portable(item) for key,item in value.items()}
        if isinstance(value,list):return [portable(item) for item in value]
        if isinstance(value,str):return value.replace(str(args.experiment_here.parents[3]),'<LOCAL_DOCS>').replace(str(args.experiment_here.parents[4]),'<LOCAL_HOME>')
        return value
    output=portable(output)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'actual_contacts_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    with (args.output/'hold_native_contacts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
    print(json.dumps(dict(success=output['success'],classification=output['classification'],
        all_three_pads_accepted=output['all_three_pads_accepted'],contacts=summary),indent=2))


if __name__ == '__main__':
    main()
