"""Keep private trajectory scoring outside the online command critical path.

Both modes retain every actual 120 Hz state/contact, all RGB exposures and all
public controller guards. Deferred mode evaluates the same authored USD meshes
on the recorded actual body/object poses before returning the final score.
"""
import inspect
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def finalize_records(records, audit, body_names, stream, out, measure):
    """Exactly reconstruct private geometry and serialize immutable snapshots."""
    import numpy as np
    start = time.perf_counter()
    # finish can be retried by the backend's EOF/close path. Keep a failed
    # attempt from leaving a prefix that the retry would append a second time.
    # Successfully computed private geometry may stay cached in records.
    write_start = stream.tell()
    try:
        for rec in records:
            if rec['penetration'] is None:
                rec['penetration'] = measure('offline_strict_geometry', audit.check,
                    dict(zip(body_names, np.asarray(rec['body_poses']))), np.asarray(rec['object_pose']))
            measure('offline_log_serialize_write', stream.write,
                json.dumps(rec, allow_nan=False) + '\n')
            if not rec['penetration']['passed'] and not (out / 'first_failure.json').exists():
                pending = out / 'first_failure.json.partial'
                pending.write_text(json.dumps(rec, allow_nan=False, indent=2) + '\n')
                pending.replace(out / 'first_failure.json')
        stream.flush()
    except BaseException:
        stream.seek(write_start)
        stream.truncate()
        stream.flush()
        raise
    return dict(states=len(records), elapsed_s=time.perf_counter()-start,
        final_physical_step=len(records), private_evaluator_only=True)


def transform(source, mode):
    if mode not in ('inline', 'deferred'):
        raise ValueError('unknown evaluation mode')
    def change(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError('Stage16 source anchor changed: ' + old)
        source = source.replace(old, new)
    change("_marks['finish_request']=time.perf_counter(); summary = evaluate(records, initial_object_z)",
        "_marks['finish_request']=time.perf_counter(); _s16_finalize(); "
        "summary = _measure('final_evaluator',evaluate,records,initial_object_z); "
        "_marks['evaluation_done']=time.perf_counter()")
    change("        stream.close()", "        _s16_finalize()\n        stream.close()")
    helper = """        _s16_finalized = False
        def _s16_finalize():
            nonlocal _s16_finalized
            if _s16_finalized:return
            _marks['private_finalize_start']=time.perf_counter()
            if _S16_MODE == 'deferred':
                details=_s16_finalize_records(records,audit,env.hand.body_names,stream,out,_measure)
            else:
                stream.flush();details=dict(states=len(records),elapsed_s=0.,private_evaluator_only=True)
            _marks['private_finalize_done']=time.perf_counter()
            _s16_finalized=True
            save(out/'stage16_evaluation_placement.json',dict(mode=_S16_MODE,**details))
"""
    change("        observation_index = 0", helper + "        observation_index = 0")
    if mode == 'deferred':
        change("check = _measure('strict_geometry',audit.check,dict(zip(env.hand.body_names,bodies)),obj)",
            "check = None  # private evaluation reconstructed after the last command")
        change("_measure('strict_log_write',stream.write,json.dumps(rec,allow_nan=False)+'\\n')",
            "pass  # private JSON serialization deferred; actual state remains in records")
        change("if not check['passed'] and not (out / 'first_failure.json').exists():",
            "if check is not None and not check['passed'] and not (out / 'first_failure.json').exists():")
    if 'FINGEREYE_LIGHT_SCALE' in os.environ:
        change("        env = backend.gym.make(backend.args_cli.task, cfg=cfg).unwrapped",
               "        env = backend.gym.make(backend.args_cli.task, cfg=cfg).unwrapped\n"
               "        from stage17_lighting import configure_lights\n"
               "        configure_lights(env, out)\n"
               "        parameters['light_multiplier'] = float(os.environ['FINGEREYE_LIGHT_SCALE'])")
    return source


def install(stage9_backend, mode):
    source = inspect.getsource(stage9_backend.main)
    from stage15_runtime import portable_backend_source
    loader="source=(HERE/'round14_backend.py').read_text()"
    if source.count(loader)!=1:raise RuntimeError('unexpected Stage9 loader')
    source=source.replace(loader,"source=portable_backend_source((HERE/'round14_backend.py').read_text())")
    anchor = '    out.mkdir(parents=True,exist_ok=True)'
    if source.count(anchor) != 1:raise RuntimeError('unexpected Stage9 main')
    source = source.replace(anchor, '    source=_s16_transform(source,_S16_MODE)\n'+anchor)
    old="_active_requests=active)"
    if source.count(old) != 1:raise RuntimeError('unexpected Stage9 execution namespace')
    source=source.replace(old, "_active_requests=active,_S16_MODE=_S16_MODE,_s16_finalize_records=_s16_finalize_records)")
    stage9_backend._s16_transform=transform
    stage9_backend._S16_MODE=mode
    stage9_backend._s16_finalize_records=finalize_records
    stage9_backend.portable_backend_source=portable_backend_source
    exec(compile(source,str(HERE/'stage16_backend.py'),'exec'),stage9_backend.__dict__)
    stage9_backend._stage15_portable_source_installed=True


def main():
    mode=os.environ.get('S16_EVALUATION_MODE','deferred')
    if os.environ.get('S14_DIAGNOSTIC_STOP')=='1' and mode!='inline':
        raise ValueError('diagnostic stop requires inline geometry; it is a distinct private diagnostic')
    import stage15_runtime
    stage15_runtime.install()
    import stage9_backend
    install(stage9_backend,mode)
    import stage15_backend
    return stage15_backend.main()


if __name__=='__main__':
    from stage14_resources import simulator_slot
    with simulator_slot(Path(sys.argv[sys.argv.index('--output')+1])):
        main()
