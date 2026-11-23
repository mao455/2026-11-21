"""R4-V image/joint-only controller sandbox with private simulator diagnostics.

No model calls happen here: the coding agent edits controller.py between episodes.
The submitted program runs offline in bubblewrap, with sensors and robot assets only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
PYTHON = Path(sys.executable).absolute()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fingerprint(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Peer:
    def __init__(self, command, log, env):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=log, text=True, bufsize=1, env=env,
                                        start_new_session=True)
        self.lines = queue.Queue()
        def reader():
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=reader, daemon=True).start()

    def receive(self, timeout=180):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError('process produced no protocol response') from exc
        if line is None:
            raise RuntimeError(f'process exited without response: {self.process.poll()}')
        return json.loads(line)

    def send(self, message):
        self.process.stdin.write(json.dumps(message, allow_nan=False) + '\n')
        self.process.stdin.flush()

    def call(self, message, timeout=180):
        self.send(message)
        return self.receive(timeout)

    def close(self):
        import signal
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)


def sandbox_command(out):
    # Same mounts/guard as published runtime; portable interpreter location.
    from run_published import sandbox_command as portable_sandbox
    return portable_sandbox(out)


def validate_step(request):
    q = request.get('target_q')
    if not isinstance(q, list) or len(q) != 18 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in q):
        raise ValueError('target_q must contain 18 finite numbers')
    count = request.get('steps')
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 24:
        raise ValueError('step count must be 1..24')
    phase = request.get('phase')
    if not isinstance(phase, str) or len(phase) > 40:
        raise ValueError('invalid phase label')
    return {'op': 'step', 'target_q': q, 'steps': count, 'phase': phase}



# Parameters are private host-side conditions; the controller receives sensors only.
CASE_PARAMETER_RANGES = {
    'mass_kg': (.002, .015), 'friction': (.3, 1.2), 'stiffness': (10., 40.),
    'depth_noise_m': (0., .001), 'depth_bias_m': (-.001, .001),
    'depth_dropout': (0., .05), 'rgb_noise_std': (0., 20.),
}


def validate_case(case):
    """Validate without importing Isaac; retain omitted defaults for pilot compatibility."""
    required = {'dx', 'dy', 'yaw'}
    allowed = required | set(CASE_PARAMETER_RANGES) | {'noise_seed', 'collision_audit', 'object_id', 'initial_overhead_reference'}
    if not isinstance(case, dict) or not required <= set(case) or set(case) - allowed:
        raise ValueError('case requires dx,dy,yaw and only documented optional parameters')
    if 'object_id' in case:
        from round14_objects import OBJECTS
        if case['object_id'] not in OBJECTS:raise ValueError('Unknown private object fixture')
    for key in required | (set(case) & set(CASE_PARAMETER_RANGES)):
        value = case[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
    if abs(case['dx']) > .04 or abs(case['dy']) > .04 or abs(case['yaw']) > math.pi:
        raise ValueError('case outside table fixture')
    for key, (low, high) in CASE_PARAMETER_RANGES.items():
        if key in case and not low <= case[key] <= high:
            raise ValueError(f'{key} must lie within [{low}, {high}]')
    if 'noise_seed' in case:
        seed = case['noise_seed']
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('noise_seed must be a nonnegative integer')
    if 'collision_audit' in case and not isinstance(case['collision_audit'], bool):
        raise ValueError('collision_audit must be boolean')
    if case.get('collision_audit', False):
        raise ValueError('live cooked collision export is not implemented; use collision_audit=false')
    if 'initial_overhead_reference' in case:
        from stage13_inputs import validate_reference
        validate_reference(case['initial_overhead_reference'])
    return dict(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--controller', type=Path, default=HERE / 'mainline/strategy.py')
    parser.add_argument('--case-json', default='{"dx":0,"dy":0,"yaw":0}')
    parser.add_argument('--phase', choices=['explore', 'rehearsal', 'heldout'], default='explore')
    parser.add_argument('--max-episodes', type=int, default=8)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {'.', '..'}:
        raise ValueError('episode name must be a single path component')
    case = validate_case(json.loads(args.case_json))
    if any(k.startswith('depth_') for k in case):raise ValueError('R4-V has RGB only; depth perturbations are not applicable')
    if not 1 <= args.max_episodes <= 100:
        raise ValueError('episode budget must be 1..100')
    submitted = args.controller.read_bytes()
    compile(submitted, str(args.controller), 'exec')
    controller_hash = hashlib.sha256(submitted).hexdigest()
    if shutil.which('bwrap') is None:
        raise RuntimeError('bubblewrap required; no unsandboxed fallback')
    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=True)
    immutable = fingerprint([HERE/'run_published.py', HERE/'record_success_views.py', HERE/'round13_objects.py', HERE/'round12_objects.py', HERE/'round10_objects.py', *sorted((HERE/'assets/round13_objects').glob('*.obj')), HERE/'round8_convex_audit.py', HERE/'round14_objects.py', HERE/'round11_objects.py', *sorted((HERE/'assets/round14_objects').glob('*.obj')), HERE/'round8_objects.py', *sorted((HERE/'assets/round8_objects').glob('*.obj')), *sorted((HERE/'assets/round11_objects').glob('*.obj')), HERE / 'round14_backend.py', HERE / 'evaluator.py', HERE / 'worker.py',
                             HERE / 'run_round14_episode.py', HERE/'round9_objects.py', HERE/'round7c_dynamics.py', HERE/'assets/round7b_xhand_allfine.usdc', HERE/'assets/round7b_xhand_allfine.json', LAB / 'scripts/rollout_xhand_motion_dual.py', LAB/'scripts/xhand_rollout_cli.py', LAB/'source/fingereye/fingereye/tasks/direct/fingereye/fingereye_env.py', LAB/'source/fingereye/fingereye/tasks/direct/fingereye/fingereye_env_cfg.py', LAB/'source/fingereye/fingereye/tasks/direct/fingereye/fingereye_synthe_env_cfg.py', HERE/'mainline/strategy.py', HERE.parent / 'rax_tinyobj_round1/geometry.py',
                             HERE.parent / 'rax_tinyobj_round1/diagnostics.py', HERE/'run_world_gravity.py', HERE/'visual_observation.py', HERE/'round7_surface_probe.py', HERE/'round7b_native_contacts.py', HERE/'assets/round7b_xhand_selfcollisions.json', HERE/'assets/round7b_xhand_allfine.usdc', HERE/'assets/round7b_xhand_allfine.json',
                             LAB/'source/fingereye/fingereye/tasks/direct/fingereye/xhand_camera_config.py',
                             LAB/'source/fingereye/fingereye/tasks/direct/fingereye/xhand_camera_recorder.py',
                             LAB/'source/fingereye/fingereye/tasks/direct/fingereye/fingereye_synthe_env.py',
                             LAB/'source/fingereye/fingereye/tasks/direct/fingereye/fingereye_xhand_env_cfg.py',
                             LAB.parent/'fingereye_camera_cfg.py',
                             LAB/'assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf'])
    contract = {'max_episodes': args.max_episodes, 'max_control_steps': 24000,
                'evaluator_and_adapter_hashes': immutable,
                'observation_profile': 'initial_overhead_rgb_live_fingereye_rgb_joint_state_v1',
                'sensor_parameters': 'RGB noise only; no depth, live overhead, contacts or object state',
                'method': 'RAX-inspired code exploration; not official RAX harness'}
    with (session / 'ledger.jsonl').open('a+') as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        manifest = session / 'contract.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != contract:
                raise RuntimeError('session contract changed; audit before starting a new session')
        else:
            write_json(manifest, contract)
        ledger.seek(0)
        prior = [json.loads(x) for x in ledger if x.strip()]
        if args.phase == 'heldout':
            frozen = session / 'frozen_controller.json'
            if not frozen.exists() or json.loads(frozen.read_text())['sha256'] != controller_hash:
                raise RuntimeError('heldout requires explicitly frozen unchanged controller')
        if len(prior) >= args.max_episodes:
            raise RuntimeError('episode budget exhausted')
        out = session / args.name
        out.mkdir(exist_ok=False)
        entry = {'episode': len(prior) + 1, 'name': args.name, 'phase': args.phase,
                 'case': case, 'reserved_unix': time.time(),
                 'controller_sha256': controller_hash}
        ledger.write(json.dumps(entry) + '\n')
        ledger.flush()
        os.fsync(ledger.fileno())
    (out / 'program').mkdir()
    (out / 'public').mkdir()
    (out / 'program/controller.py').write_bytes(submitted)
    (out / 'harness_snapshot').mkdir()
    for source in immutable:
        p = Path(source)
        shutil.copy2(p, out / 'harness_snapshot' / p.name)
    shutil.copy2(HERE / 'worker.py', out / 'program/worker.py')
    write_json(out / 'episode.json', entry)
    from visual_observation import ObservationGuard
    observation_guard=ObservationGuard(out/'public')
    backend = worker = None
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    steps, complete, error = 0, False, None
    start = time.monotonic()
    with (out / 'backend.log').open('w') as backend_log, (out / 'controller.log').open('w') as worker_log, (out / 'agent_trace.jsonl').open('w', buffering=1) as trace:
        try:
            backend = Peer(['nice', '-n', '10', str(PYTHON), str(HERE / 'round14_backend.py'),
                            '--output', str(out), '--case-json', json.dumps(case)], backend_log, env)
            ready = backend.receive(timeout=300)
            if not ready.get('ok', True):
                raise RuntimeError(ready.get('error', 'backend startup failure'))
            observation_guard.validate(ready, initial=True)
            write_json(out / 'public/initial.json', ready)
            worker = Peer(sandbox_command(out), worker_log, env)
            worker.send(ready)
            while time.monotonic() - start < 2400:
                message = worker.receive(timeout=240)
                op = message.get('op')
                trace.write(json.dumps({'elapsed_s': time.monotonic() - start, **message}) + '\n')
                if op == 'complete':
                    worker.send({'ok': True})
                    complete = True
                    break
                if op == 'controller_error':
                    raise RuntimeError(message.get('error', 'controller error'))
                if op == 'note':
                    worker.send({'ok': True})
                    print('NOTE', message.get('text', '')[:1500], flush=True)
                    continue
                if op == 'step':
                    message = validate_step(message)
                    steps += message['steps']
                    if steps > 24000:
                        raise RuntimeError('physical step budget exhausted')
                elif op != 'observe':
                    raise ValueError('unsupported controller operation')
                response = backend.call(message)
                if response.get('ok',True):observation_guard.validate(response)
                worker.send(response)
                if not response.get('ok', True):
                    raise RuntimeError(response.get('error', 'backend rejected action'))
            else:
                raise TimeoutError('episode wall time exhausted')
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            print(error, file=sys.stderr, flush=True)
        finally:
            if worker is not None:
                worker.close()
            if backend is not None:
                try:
                    result = backend.call({'op': 'finish'}, timeout=60)
                    write_json(out / 'backend_final.json', result)
                except Exception as exc:
                    write_json(out / 'backend_finish_error.json', {'error': str(exc)})
                backend.close()
            unchanged = fingerprint([Path(p) for p in immutable]) == immutable
            host = {'controller_completed': complete, 'error': error,
                    'requested_control_steps': steps, 'elapsed_s': time.monotonic() - start,
                    'evaluator_and_adapter_unchanged': unchanged,
                    'controller_sha256': entry['controller_sha256']}
            summary_path = out / 'summary.json'
            physics = json.loads(summary_path.read_text()) if summary_path.exists() else {'success': False, 'failure': 'missing physics summary'}
            host['physics'] = physics
            host['success'] = bool(complete and error is None and unchanged and physics.get('success'))
            host['success_protocol'] = physics.get('protocol_id', 'legacy_strict')
            host['legacy_strict_success'] = bool(complete and error is None and unchanged
                                                and physics.get('legacy_strict', physics).get('success'))
            write_json(out / 'result.json', host)
            if args.phase == 'rehearsal' and host['success']:
                write_json(session / 'frozen_controller.json', {'sha256': controller_hash, 'rehearsal': args.name})
            print('RESULT', json.dumps(host), flush=True)
    return 0 if error is None and unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
