"""Portable, bounded batch runner around the normal FingerEye episode entry.

Manifest is a JSON list (or JSONL) of {id, case:{object_id,dx,dy,yaw,...}}.
An explicit case_file is also accepted. Relative reference paths are resolved
against the manifest, then hash checked. No policy input is manufactured here.
Each launch has a new immutable directory; execution-valid physical failures
are completed results and are skipped on resume just like successes.
"""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write_json(path, value):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_jobs(path, seeds=None):
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()]
    jobs = []
    for i, row in enumerate(rows):
        if 'case_file' in row:
            source = Path(row['case_file'])
            if not source.is_absolute():
                source = path.parent / source
            case = json.loads(source.read_text())
        else:
            case = dict(row.get('case', row))
            if 'case' not in row:
                for key in ('id', 'seeds', 'seed'):
                    case.pop(key, None)
        case.pop('name', None)
        if 'object_id' not in case:
            raise ValueError('manifest row needs object_id: ' + str(i))
        # Relative public-input locations are portable; recorded SHA stays exact.
        if case.get('initial_overhead_reference'):
            from stage13_inputs import validate_reference
            ref = dict(case['initial_overhead_reference'])
            for key in ('rgb_path', 'metadata_path'):
                p = Path(ref[key])
                ref[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
            validate_reference(ref)
            for prefix in ('rgb', 'metadata'):
                if sha(ref[prefix + '_path']) != ref[prefix + '_sha256']:
                    raise ValueError('changed public reference: ' + ref[prefix + '_path'])
            case['initial_overhead_reference'] = ref
        for seed in row.get('seeds', seeds or [row.get('seed', case.get('noise_seed', 0))]):
            current = dict(case, noise_seed=int(seed))
            identifier = row.get('id', str(i).zfill(4) + '_' + current['object_id'])
            identifier += '_seed' + str(seed)
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', identifier) or identifier in ('.', '..'):
                raise ValueError('unsafe manifest id: ' + identifier)
            jobs.append(dict(id=identifier, case=current))
    if len({x['id'] for x in jobs}) != len(jobs):
        raise ValueError('duplicate expanded manifest IDs')
    return jobs


def result_record(session, expected_controller=None):
    path = session / 'episode/result.json'
    if not path.exists():
        return dict(completed=False, success=False, reason='no_result')
    try:
        result = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        return dict(completed=False, success=False, reason='unreadable_result: ' + str(exc))
    completed = bool(result.get('controller_completed') and result.get('error') is None
                     and result.get('evaluator_and_adapter_unchanged'))
    physics = result.get('physics', {})
    verification = {}
    if completed:
        try:
            from evaluator import evaluate
            scene_path = session / 'episode/scene.json'
            steps_path = session / 'episode/steps.jsonl'
            scene = json.loads(scene_path.read_text())
            with steps_path.open() as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            score = evaluate(records, scene['initial_object_pose'][2])
            requested = result.get('requested_control_steps', physics['steps'])
            recording_valid = bool(len(records) == physics['steps'] == requested and
                all(r['step'] == i for i, r in enumerate(records)) and
                all(abs(r['time_s'] - (i + 1) / 120) <= 1e-9 for i, r in enumerate(records)) and
                bool(result.get('success')) == bool(score['success']) and
                physics == score and
                sha(session/'episode/program/controller.py') == result.get('controller_sha256') and
                (expected_controller is None or result.get('controller_sha256') == expected_controller))
            verification = dict(raw_recording_verified=recording_valid,
                independent_physical_success=score['success'],
                scene_sha256=sha(scene_path), steps_sha256=sha(steps_path),
                evaluator_sha256=sha(HERE/'evaluator.py'))
            if not recording_valid:
                completed = False
                verification['recording_verification_error'] = 'record count, timing, scorer, or controller mismatch'
        except (ValueError, OSError, KeyError, TypeError) as exc:
            completed = False
            verification = dict(raw_recording_verified=False, recording_verification_error=str(exc))
    return dict(completed=completed, success=bool(completed and result.get('success')),
                reason=result.get('error') or verification.get('recording_verification_error') or physics.get('failure') or ('success' if result.get('success') else 'physical_failure'),
                physical_steps=physics.get('steps', result.get('requested_control_steps')),
                physical_s=(physics.get('steps', result.get('requested_control_steps')) or 0) / 120,
                controller_sha256=result.get('controller_sha256'),
                integrity_valid=result.get('evaluator_and_adapter_unchanged'),
                result_sha256=sha(path), result_path=str(path), **verification)


def attempt_count(session):
    path = session / 'episode/steps.jsonl'
    count = 0
    in_close = False
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:
                    phase = json.loads(line)['phase']
                except (ValueError, KeyError):
                    # A killed startup may leave one truncated final line. Keep
                    # it in raw evidence and still emit the timeout receipt.
                    continue
                if phase in ('s5_close', 's13_preload_recovery') and not in_close:
                    count += 1
                    in_close = True
                elif phase not in ('s5_close', 's13_preload_recovery', 's5_close_observe', 's5_preload_adjust', 's5_preload_hold'):
                    in_close = False
    return count


def terminate_group(process):
    # Kill only this launch's new process group, never all Kit/Python processes.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def run_batch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.batch.lock').open('a') as batch_lock:
        try:
            fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another batch owns this output directory')
        runner = args.runner.resolve()
        if not runner.is_file():
            raise ValueError('normal entry not found: ' + str(runner))
        jobs = load_jobs(args.manifest.resolve(), args.seeds)
        version = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(HERE), 'status', '--porcelain'], text=True)
        descriptor = dict(schema='fingereye_batch_v1', jobs=jobs, runner=str(runner), runner_sha256=sha(runner),
                          controller_sha256=sha(args.controller) if args.controller else None,
                          asset_root=str(args.asset_root.resolve()) if args.asset_root else None,
                          registry_manifest_sha256=sha(args.registry_manifest) if args.registry_manifest else None,
                          mode=args.mode, phase=args.phase, profile=args.profile, fingers=args.fingers,
                          max_attempts=args.max_attempts, protocol='lift30_no_drop_v1',
                          physics_hz=120, control_hz=20, rgb_fps=30, version=version,
                          entry_source_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
                          dirty_diff_sha256=hashlib.sha256(subprocess.check_output(['git', '-C', str(HERE), 'diff', 'HEAD'])).hexdigest())
        digest = hashlib.sha256(canonical(descriptor).encode()).hexdigest()
        plan_path = args.output / 'batch_plan.json'
        if plan_path.exists():
            old = json.loads(plan_path.read_text())
            if old['plan_sha256'] != digest:
                raise ValueError('configuration/version changed; use a new output directory')
            if not args.resume:
                raise ValueError('batch exists; use --resume or a new output directory')
        else:
            write_json(plan_path, dict(plan_sha256=digest, configuration=descriptor, git_status=dirty))
        if args.dry_run:
            print(json.dumps(dict(jobs=len(jobs), plan_sha256=digest, dry_run=True)))
            return 0
        lock = threading.Lock()
        receipts = args.output / 'launches.jsonl'

        def append(row):
            with lock, receipts.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

        def launch(job):
            jobdir = args.output / 'episodes' / job['id']
            jobdir.mkdir(parents=True, exist_ok=True)
            existing = sorted(jobdir.glob('launch_*'))
            for session in existing:
                outcome = result_record(session, descriptor['controller_sha256'])
                if outcome['completed']:
                    row = dict(id=job['id'], session=str(session), skipped_completed=True, **outcome,
                               actual_attempts=attempt_count(session))
                    append(row)
                    return row
            # Preserve all interrupted/invalid launches. New launch cannot overwrite.
            session = jobdir / ('launch_' + str(len(existing) + 1).zfill(3))
            session.mkdir()
            case_path = session / 'case.json'
            write_json(case_path, job['case'])
            command = [sys.executable, str(runner), '--session', str(session), '--case-file', str(case_path), '--mode', args.mode, '--phase', args.phase]
            if args.controller:
                command += ['--controller', str(args.controller.resolve())]
            else:
                command += ['--profile', args.profile, '--fingers', args.fingers, '--max-attempts', str(args.max_attempts)]
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            if args.asset_root:
                env['FINGEREYE_ASSET_ROOT'] = str(args.asset_root.resolve())
            if args.registry_manifest:
                env['FINGEREYE_OBJECT_MANIFEST'] = str(args.registry_manifest.resolve())
            started = time.time()
            monotonic = time.perf_counter()
            write_json(session / 'launch.json', dict(id=job['id'], command=command, started_unix_s=started,
                       version=version, plan_sha256=digest, asset_root=descriptor['asset_root'],
                       registry_manifest=str(args.registry_manifest.resolve()) if args.registry_manifest else None,
                       timeout_s=args.timeout, concurrency=args.concurrency))
            status = 'exited'
            code = None
            with (session / 'launcher.log').open('w') as log:
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True)
                    try:
                        code = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        status = 'timeout'
                        terminate_group(process)
                        code = process.returncode
                    except BaseException:
                        terminate_group(process)
                        raise
                except OSError as exc:
                    status = 'startup_error'
                    log.write(str(exc) + '\n')
            outcome = result_record(session, descriptor['controller_sha256'])
            row = dict(id=job['id'], session=str(session), launch_status=status, returncode=code,
                       started_unix_s=started, elapsed_wall_s=time.perf_counter() - monotonic,
                       actual_attempts=attempt_count(session), **outcome)
            write_json(session / 'receipt.json', row)
            append(row)
            print(json.dumps(row), flush=True)
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(launch, jobs))
        write_json(args.output / 'summary.json', dict(plan_sha256=digest, total=len(rows),
                   completed=sum(x['completed'] for x in rows), successes=sum(x['success'] for x in rows),
                   incomplete=sum(not x['completed'] for x in rows), rows=rows))
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (args.output / 'episodes.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows(rows)
        return int(any(not x['completed'] for x in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--asset-root', type=Path, help='FingerEyeLab root containing assets/scripts/source; FINGEREYE_ASSET_ROOT')
    p.add_argument('--registry-manifest', type=Path, help='private object registry; FINGEREYE_OBJECT_MANIFEST')
    p.add_argument('--runner', type=Path, default=HERE / 'run_stage15.py')
    p.add_argument('--controller', type=Path)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--concurrency', type=int, choices=(1, 2), default=1)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--mode', choices=('minimal', 'full'), default='minimal')
    p.add_argument('--phase', choices=('explore', 'heldout'), default='explore')
    p.add_argument('--profile', choices=('v9', 'fast', 'retry'), default='fast')
    p.add_argument('--fingers', choices=('two', 'three', 'auto'), default='two')
    p.add_argument('--max-attempts', type=int, choices=(1, 2, 3), default=3)
    a = p.parse_args()
    if a.timeout <= 0:
        p.error('timeout must be positive')
    if a.asset_root and not a.asset_root.is_dir():
        p.error('asset root does not exist')
    return run_batch(a)


if __name__ == '__main__':
    raise SystemExit(main())
