"""Build portable numerical evidence from existing logs; never run physics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-here', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    here = args.source_here
    if here is None:
        for ancestor in Path(__file__).resolve().parents:
            candidates = [ancestor, ancestor / 'FingerEyeLab/experiments/rax_dexbin_tinyobj']
            here = next((path for path in candidates
                         if (path / 'stage16_evidence/numerics').is_dir()), None)
            if here is not None:
                break
    if here is None:
        parser.error('--source-here must identify the experiment directory with local raw evidence')
    here = here.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = here / 'stage16_evidence/numerics'
    sys.path.insert(0, str(here))
    from stage15_batch import result_record

    # Public tables retain local evidence identities without publishing a user's
    # machine path. No pixel arrays, videos, or per-physics-step logs are copied.
    prefix = str(here.parents[3])
    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            return value.replace(prefix, '<LOCAL_DOCS>').replace(str(here.parents[4]), '<LOCAL_HOME>')
        return value

    source_files = [
        'candidate_table.csv', 'fixed_candidate_audit.json',
        'normal_and_tie_witnesses.json', 'jacobian_and_worst_loads.json',
        'scenario_registration.json', 'paired_scenarios.json',
        'physical/physical_readout.json', 'physical/physical_table.csv',
        'physical/actual_best_candidate_details.json',
        'physical/actual_best_candidate_table.csv',
        'physical/fixed_feasible_ranking_ab.json', 'physical/image_divergence.json',
        'physical/final_decision_divergence.json',
        'physical/native_contact_points_summary.json', 'physical/commands.json',
    ]
    sources = []
    for name in source_files:
        path = evidence / name
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            target.write_text(json.dumps(portable(json.loads(path.read_text())),
                                         indent=2, allow_nan=False) + '\n')
        else:
            target.write_text(portable(path.read_text()))
        sources.append(dict(source='stage16_evidence/numerics/' + name,
                            original_sha256=sha(path), public=name,
                            public_sha256=sha(target)))

    readout = json.loads((evidence / 'physical/physical_readout.json').read_text())
    verified = []
    paired = []
    for entry in readout['rows']:
        session = Path(entry['session'])
        actual = result_record(session, entry['controller_sha256'])
        raw_hashes = {name: sha(session / 'episode' / name)
                      for name in entry['raw_hashes']}
        assert raw_hashes == entry['raw_hashes'], entry['name']
        assert actual['completed'] and actual['raw_recording_verified'], entry['name']
        assert actual['success'] == entry['success'], entry['name']
        assert actual['physical_steps'] == entry['steps'], entry['name']
        verified.append(dict(name=entry['name'], **portable(actual)))
        selected = entry['selections'][0] if entry['selections'] else {}
        rejected = selected.get('rejection_counts', {})
        groups = selected.get('accepted_groups', {})
        paired.append(dict(
            name=entry['name'], kind=entry['kind'], success=entry['success'],
            legacy_strict=entry['legacy_strict'], actual_closures=entry['actual_closures'],
            physical_steps=entry['steps'], physical_s=entry['physical_s'],
            result_elapsed_s=entry['host_elapsed_s'],
            launch_wall_including_admission_s=entry['launch_wall_s'],
            feasible_two=groups.get('2', 0), feasible_three=groups.get('3', 0),
            rejected_normal=rejected.get('per_finger_normal', 0),
            rejected_position=rejected.get('per_finger_position', 0),
            rejected_endpoint=rejected.get('endpoint_clearance_collision', 0),
            search_budget_rejections=len(selected.get('budget', [])),
            three_pad_hold_accepted=entry['three_pad_hold_accepted'],
            controller_sha256=entry['controller_sha256'],
            steps_sha256=entry['raw_hashes']['steps.jsonl']))
    write_csv(output / 'paired_modes.csv', paired)

    details = json.loads((evidence / 'physical/actual_best_candidate_details.json').read_text())
    per_finger = []
    residuals = []
    maximum_cost_error = 0.
    for row in details:
        meta = row['meta']
        numeric = meta['s16_numerics']
        count = numeric['contact_count']
        positions = numeric['per_finger_position_error_m']
        angles = numeric['per_finger_normal_error_rad']
        position_squared = [2/count * (1500 * error)**2 for error in positions]
        normal_squared = [2/count * numeric['used_normal_weight']**2 *
                          2*(1-math.cos(angle)) for angle in angles]
        assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)
                   for a, b in zip(position_squared, numeric['per_finger_position_squared_residual']))
        assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-12)
                   for a, b in zip(normal_squared, numeric['per_finger_normal_squared_residual']))
        maximum_cost_error = max(maximum_cost_error,
            abs(meta['cost'] - .5 * sum(numeric['squared_residual_contributions'].values())))
        for index, (position, angle) in enumerate(zip(positions, angles)):
            per_finger.append(dict(run=row['run'], contact_count=count,
                selection=row['selection'], finger=meta['fingers'][index],
                position_error_m=position, normal_error_rad=angle,
                contact_normalization=math.sqrt(2/count),
                normal_chord_weight=numeric['used_normal_weight'],
                position_squared_residual=position_squared[index],
                normal_squared_residual=normal_squared[index]))
        for term, contribution in numeric['squared_residual_contributions'].items():
            residuals.append(dict(run=row['run'], contact_count=count,
                term=term, squared_residual_sum=contribution,
                scipy_cost_contribution=.5*contribution,
                total_scipy_cost=meta['cost'],
                position_weight_per_m=numeric['position_weight_per_m'],
                normal_chord_weight=numeric['used_normal_weight']))
    assert maximum_cost_error < 1e-10
    write_csv(output / 'per_finger_errors.csv', per_finger)
    write_csv(output / 'ik_residual_contributions.csv', residuals)

    normal = [row for row in paired if row['kind'] == 'normal']
    original = [row for row in normal if re.fullmatch(r'(20|34)_(two|three|auto)', row['name'])]
    report = dict(
        scope='existing development cases; independently re-read logs; no new physics runs',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=here, text=True).strip(),
        original_modes=dict(successes=sum(x['success'] for x in original), denominator=len(original)),
        with_single_factor_ablations=dict(successes=sum(x['success'] for x in normal), denominator=len(normal)),
        all_logged_episodes_verified=len(verified),
        normal_three_successes=sum(x['success'] for x in normal if '_three' in x['name']),
        normal_three_closures=sum(x['actual_closures'] for x in normal if '_three' in x['name']),
        both_groups_feasible=sum(bool(x['feasible_two'] and x['feasible_three']) for x in normal),
        accepted_three_pad_holds=sum(x['three_pad_hold_accepted'] for x in paired),
        maximum_recomputed_scipy_cost_error=maximum_cost_error,
        normal_equal_error_cost_ratio=49/16,
        hardware_trials=0,
        uncertainty='n=1 per condition in two deliberately chosen development scenarios; no population success estimate',
        real_time_limit='result elapsed and launch wall are simulation timings; neither establishes hardware task-to-grasp latency',
        trajectory_caveat='20 auto_tie has identical commands/state over 3294 common steps but later support decision differs; not sorting speedup',
        contact_caveat='120Hz actual thumb/index points retained locally; middle pad/housing actual point positions absent; 20Hz link forces show middle pad 0/40 and housing 40/40',
        raw_verification=verified, sources=sources)
    (output / 'AUDIT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for path in output.rglob('*'):
        if path.is_file() and path.name != 'build_public.py':
            assert '/home/' not in path.read_text(), str(path)
    public_files = [dict(path=str(path.relative_to(output)), bytes=path.stat().st_size,
                         sha256=sha(path)) for path in sorted(output.rglob('*'))
                    if path.is_file() and path.name != 'MANIFEST.json']
    (output / 'MANIFEST.json').write_text(json.dumps(dict(
        scope='structured numerical evidence; no pixels, video or raw per-physics-step logs',
        files=public_files), indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('all_logged_episodes_verified',
        'original_modes', 'with_single_factor_ablations', 'maximum_recomputed_scipy_cost_error')}))


if __name__ == '__main__':
    main()
