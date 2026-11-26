"""Linux-only outer wall-clock budget for both unchanged round1 and round2 runners.

The policy worker independently enforces its existing 600-second RLIMIT_CPU.
This supervisor starts its clock immediately before Popen, including preparation.
It becomes a child subreaper, so double-forked/setsid workers remain owned even
when an intermediate parent exits between process scans. Signals target verified
PID identities, never a shared process group. Cleanup time is reported separately.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid

import psutil


def fingerprint(paths):
    result = {}
    for name in paths:
        path = Path(name).resolve()
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def atomic_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def supervise(args):
    if sys.platform != 'linux':
        raise RuntimeError('Linux PR_SET_CHILD_SUBREAPER is required')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
    receipt = Path(args.receipt).resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists():
        raise FileExistsError(f'Preserving existing trial receipt: {receipt}')
    journal = receipt.with_suffix(receipt.suffix + '.jsonl')
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not argv:
        raise ValueError('A command after -- is required')
    cwd = Path(args.cwd).resolve()
    files = [str(Path(__file__).resolve()), *args.fingerprint_file]
    files.extend(str((cwd / x).resolve()) for x in argv if (cwd / x).is_file())
    if (cwd / 'worker.py').is_file():
        files.append(str(cwd / 'worker.py'))
    before = fingerprint(files)
    run_id = uuid.uuid4().hex
    data = dict(schema='fingereye.round2.budget.v1', run_id=run_id, argv=argv,
                cwd=str(cwd), planned_host_budget_s=args.host_timeout_s,
                cleanup_grace_s=args.cleanup_grace_s,
                deadline_start_semantics='monotonic immediately before Popen; includes runner preparation',
                started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                environment={k: os.environ.get(k) for k in (
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE')},
                supervisor_sha256=before[str(Path(__file__).resolve())],
                source_fingerprint_before=before, timeout_triggered=False,
                interrupted_signal=None, return_code=None, status='starting',
                cpu_budget_note='Worker existing per-process RLIMIT_CPU; not aggregate simulator/tree CPU',
                ownership='Linux child subreaper, ancestry, PID plus creation time; no process-group signals')

    def event(kind, **fields):
        with journal.open('a') as f:
            f.write(json.dumps(dict(event=kind, run_id=run_id, unix_time=time.time(), **fields)) + '\n')
            f.flush()
            os.fsync(f.fileno())

    atomic_json(receipt, data)
    event('start', receipt=data)
    owned = {}
    myself = psutil.Process()
    interrupted = []
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda number, frame: interrupted.append(number))
    process = None
    start = time.monotonic()
    stop = start

    def collect():
        # Subreaper-adopted orphans become direct children and are included here.
        for proc in myself.children(recursive=True):
            try:
                key = (proc.pid, proc.create_time())
                row = owned.setdefault(key, dict(pid=proc.pid, create_time=key[1],
                    command=proc.cmdline(), exit_code=None, exit_signal=None))
                cpu = proc.cpu_times()
                row.update(cpu_user_s=cpu.user, cpu_system_s=cpu.system)
                row['rlimit_cpu_soft_hard'] = list(resource.prlimit(proc.pid, resource.RLIMIT_CPU))
                row['is_policy_worker'] = any(Path(x).name == 'worker.py' for x in row['command'])
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, PermissionError):
                continue

    def living():
        out = []
        for (pid, created), row in owned.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == created and proc.status() != psutil.STATUS_ZOMBIE:
                    out.append(proc)
            except psutil.NoSuchProcess:
                pass
        return out

    def send(proc, sig):
        try:
            # psutil.send_signal checks PID reuse using the Process identity.
            proc.send_signal(sig)
        except psutil.NoSuchProcess:
            pass

    def reap():
        # Popen owns leader wait status. Reap only adopted descendants here.
        if process is not None:
            process.poll()
        for (pid, created), row in owned.items():
            if process is not None and pid == process.pid:
                if process.returncode is not None:
                    row['exit_code'] = process.returncode
                    row['exit_signal'] = -process.returncode if process.returncode < 0 else None
                continue
            try:
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited:
                    code = os.waitstatus_to_exitcode(status)
                    row.update(exit_code=code, exit_signal=-code if code < 0 else None)
            except ChildProcessError:
                pass

    try:
        process = subprocess.Popen(argv, cwd=cwd, start_new_session=True)
        data.update(status='running', leader_pid=process.pid)
        atomic_json(receipt, data)
        while True:
            collect()
            now = time.monotonic()
            if now - start >= args.host_timeout_s:
                data['timeout_triggered'] = True
                break
            if interrupted:
                data['interrupted_signal'] = interrupted[0]
                break
            if process.poll() is not None:
                break
            time.sleep(min(args.poll_s, max(0.001, args.host_timeout_s - (now - start))))
        stop = time.monotonic()
    except BaseException as exc:
        data['supervisor_error'] = f'{type(exc).__name__}: {exc}'
        stop = time.monotonic()
    finally:
        # Freeze action-producing processes before giving the leader bounded
        # opportunity to seal its episode. New/adopted descendants are rescanned.
        collect()
        active = living()
        if active:
            event('cleanup_start', timed_out=data['timeout_triggered'], owned_pids=[p.pid for p in active])
            for proc in active:
                send(proc, signal.SIGSTOP)
            if process is not None and process.poll() is None:
                leader = next((p for p in active if p.pid == process.pid), None)
                if leader is not None:
                    send(leader, signal.SIGINT)
                    send(leader, signal.SIGCONT)
            grace_end = time.monotonic() + args.cleanup_grace_s
            while time.monotonic() < grace_end and living():
                collect()
                for proc in living():
                    if process is None or proc.pid != process.pid:
                        send(proc, signal.SIGSTOP)
                reap()
                time.sleep(args.poll_s)
            # Multiple passes catch orphan adoption and children spawned just
            # before SIGSTOP. Only this invocation's descendants are signalled.
            kill_end = time.monotonic() + 5.0
            while True:
                collect()
                for proc in living():
                    send(proc, signal.SIGKILL)
                reap()
                if not living() or time.monotonic() >= kill_end:
                    break
                time.sleep(args.poll_s)
        reap()
        data.update(return_code=process.returncode if process is not None else None,
                    actual_host_elapsed_s=stop-start, cleanup_elapsed_s=time.monotonic()-stop,
                    ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    processes=list(owned.values()), leftover_descendants=[p.pid for p in living()])
        try:
            data['source_fingerprint_after'] = fingerprint(files)
            data['source_unchanged'] = data['source_fingerprint_after'] == before
        except OSError as exc:
            data['source_fingerprint_error'] = str(exc)
            data['source_unchanged'] = False
        data['status'] = ('timeout' if data['timeout_triggered'] else
                          'interrupted' if data['interrupted_signal'] else
                          'supervisor_error' if data.get('supervisor_error') else
                          'completed' if data['return_code'] == 0 else 'command_failed')
        atomic_json(receipt, data)
        event('finish', receipt=data)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if data['leftover_descendants'] or data.get('supervisor_error'):
        return 125
    if data['timeout_triggered']:
        return 124
    if data['interrupted_signal']:
        return 128 + data['interrupted_signal']
    code = data['return_code']
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-timeout-s', type=float, default=1800.0)
    parser.add_argument('--cleanup-grace-s', type=float, default=5.0)
    parser.add_argument('--poll-s', type=float, default=0.05)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--cwd', default=os.getcwd())
    parser.add_argument('--fingerprint-file', action='append', default=[])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.host_timeout_s <= 0 or not 0 < args.poll_s <= 0.1 or args.cleanup_grace_s < 0:
        parser.error('positive timeout, 0 < poll <= 0.1, and nonnegative grace required')
    return supervise(args)


if __name__ == '__main__':
    sys.exit(main())
