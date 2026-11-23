"""Independent phase accounting of archived physics and newly executed paced replays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(v) for v in p.read_text().splitlines()]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--replay-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    segments, summaries, replays, replay_segments = [], [], [], []
    jobs = [('A', 'finalA_dn0_seed16111'), ('B', 'finalB_dn3_seed16112'), ('D', 'finalD_dn20.4_seed16113')]
    for label, job in jobs:
        ep = a.raw_root/'episodes'/job/'launch_001/episode'
        commands = [r for r in lines(ep/'agent_trace.jsonl') if r.get('op') == 'step']
        rpc = [r for r in read(ep/'rpc_timing.json') if r['op'] == 'step']
        assert len(commands) == len(rpc)
        profile = read(ep/'wall_profile_backend.json')
        ready = profile['marks']['ready']
        tick, groups = 0, []
        recovery = False
        for i, (c, t) in enumerate(zip(commands, rpc)):
            if c['phase'].startswith(('s13_recovery', 's3_lower')):
                recovery = True
            name = ('recovery' if recovery else 'first_attempt', c['phase'])
            before = t['start_wall_s'] - (rpc[i-1]['start_wall_s']+rpc[i-1]['inclusive_s'] if i else ready)
            if not groups or name != groups[-1]['name']:
                groups.append(dict(name=name, begin_command=i, end_command=i, begin_tick=tick,
                                   end_tick=tick, rpc_s=0., preceding_compute_and_io_s=0., commands=0))
            g = groups[-1]
            g['end_command'] = i
            g['end_tick'] += c['steps']
            g['rpc_s'] += t['inclusive_s']
            g['preceding_compute_and_io_s'] += before
            g['commands'] += 1
            tick += c['steps']
        for g in groups:
            name = g.pop('name')
            segments.append(dict(case=label, scope='actual closed-loop simulation', attempt=name[0], phase=name[1],
                                 **g, physical_s=(g['end_tick']-g['begin_tick'])/120.))
        finish = rpc[-1]['start_wall_s']+rpc[-1]['inclusive_s']
        measured = sum(g['rpc_s']+g['preceding_compute_and_io_s'] for g in groups)
        assert abs(measured-(finish-ready)) < 1e-6
        summaries.append(dict(case=label, commands=len(commands), physical_s=tick/120.,
                              backend_ready_to_terminal_rpc_s=finish-ready,
                              rpc_inclusive_s=sum(g['rpc_s'] for g in groups),
                              between_rpc_compute_and_io_s=sum(g['preceding_compute_and_io_s'] for g in groups),
                              partition_error_s=measured-(finish-ready),
                              private_finalize_s=profile['marks']['private_finalize_done']-profile['marks']['private_finalize_start'],
                              source_steps_sha256=hashlib.sha256((ep/'steps.jsonl').read_bytes()).hexdigest()))
        rp = a.replay_root/('paced_final_'+label)
        if not (rp/'timing_summary.json').exists():
            continue
        s = read(rp/'timing_summary.json')
        ev = lines(rp/'monotonic.jsonl')
        ds = [r for r in ev if r['event'] == 'command_dispatch']
        rs = [r for r in ev if r['event'] == 'command_return']
        assert len(ds) == len(rs) == len(commands)
        assert s['evaluation']['all_commands_equal']
        first = s['marks']['first_command']-s['marks']['task_dispatch']
        wait = sum(r['transport_and_driver_s'] for r in rs)
        compute = sum(ds[i+1]['monotonic_s']-rs[i]['monotonic_s'] for i in range(len(ds)-1))
        replays.append(dict(case=label, source_controller_sha256=hashlib.sha256((ep/'program/controller.py').read_bytes()).hexdigest(),
                            condition='paced recorded final simulation inputs; ideal capture/transport zero; no physics/hardware',
                            commands=s['command_count'], all_commands_equal=s['evaluation']['all_commands_equal'],
                            task_to_first_command_s=first, synchronous_calls_s=wait, between_command_compute_and_io_s=compute,
                            task_to_archived_terminal_hold_s=s['task_to_stable_s'],
                            within40=s['official_success_within40s'], late_over_1ms=s['overrun_count'],
                            missed_50ms_periods=s['missed_periods'], max_lateness_s=s['max_lateness_s'],
                            hardware_n=0, new_physics_n=0))
        for g in groups:
            lo, hi = g['begin_command'], g['end_command']
            replay_segments.append(dict(case=label, phase=commands[lo]['phase'], begin_command=lo, end_command=hi,
                synchronous_calls_s=sum(rs[i]['transport_and_driver_s'] for i in range(lo, hi+1)),
                preceding_compute_and_io_s=sum(ds[i]['monotonic_s']-(rs[i-1]['monotonic_s'] if i else s['marks']['task_dispatch']) for i in range(lo, hi+1))))
    write_csv(a.output/'simulation_phase_timing.csv', segments)
    write_csv(a.output/'simulation_timing.csv', summaries)
    if replays:
        write_csv(a.output/'frozen_replay_timing.csv', replays)
        write_csv(a.output/'frozen_replay_phases.csv', replay_segments)
    print(json.dumps(dict(simulation=summaries, replay=replays), indent=2))


if __name__ == '__main__':
    main()
