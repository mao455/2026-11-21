"""Monotonic 20 Hz adapter harness. Device driver is supplied explicitly; no hardware auto-discovery.

Driver factory returns an initialized adapter with robot_description, observe(),
step(q, steps=6, phase=''), note(text), evaluate(), close(). Existing device safety
checks remain the driver's responsibility. evaluate() runs after control returns.
This harness only accepts actual stable time from evaluate(), never estimates it.
"""
import argparse
import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path


class Timeline:
    def __init__(self, path, clock=time.perf_counter):
        self.clock=clock;self.rows=[];self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('w',buffering=1)
    def emit(self,event,**fields):
        row=dict(event=event,monotonic_s=self.clock(),**fields);self.rows.append(row)
        self.stream.write(json.dumps(row,allow_nan=False)+'\n');return row
    def close(self):self.stream.close()


class TimingContext:
    """Synchronous transport pacing: late blocking calls never create catch-up bursts."""
    def __init__(self,adapter,timeline,period=.05,clock=time.perf_counter,sleep=time.sleep):
        if period<=0:raise ValueError('positive command period required')
        self.adapter=adapter;self.robot_description=adapter.robot_description
        self.timeline=timeline;self.clock=clock;self.sleep=sleep;self.period=period
        self.due=None;self.calls=0;self.overruns=0;self.missed_periods=0;self.first_frame=False
    def _observation(self,obs):
        image=obs.get('image_state',{});frameids=sorted({x.get('frame_id') for x in obs.get('fingereye',{}).values() if x.get('frame_id') is not None})
        self.timeline.emit('observation_available',frame_ids=frameids,capture_step=image.get('step'),
            exposure_monotonic_s=image.get('capture_start_wall_s'),readback_done_monotonic_s=image.get('capture_done_wall_s'),
            actual_joint_timestamp_s=obs.get('joint_monotonic_s'),exposure_joint_q=image.get('actual_q'),actual_q=obs.get('actual_q'))
        has_frame = bool(frameids or obs.get('initial_overhead') or image.get('frame_id') is not None or image.get('step') is not None)
        if has_frame and not self.first_frame:
            self.first_frame=True;self.timeline.emit('first_frame_available')
        return obs
    def observe(self):
        self.timeline.emit('observe_requested');return self._observation(self.adapter.observe())
    def step(self,q,steps=6,phase=''):
        if steps != 6:raise ValueError('hardware timing adapter requires one 20 Hz command, steps=6')
        if len(q)!=18 or not all(math.isfinite(float(x)) for x in q):raise ValueError('18 finite position targets required')
        now=self.clock()
        if self.due is None:self.due=now
        while now<self.due:
            self.sleep(self.due-now);now=self.clock()
        late=max(0.,now-self.due)
        # 1 ms tolerance separates OS wakeup jitter from material blocked cycles.
        overrun=late>.001;self.overruns+=int(overrun)
        missed=max(0,int(math.floor((late+1e-9)/self.period)));self.missed_periods+=missed
        if self.calls==0:self.timeline.emit('first_command')
        self.timeline.emit('command_dispatch',index=self.calls,phase=phase,target_q=list(map(float,q)),
            scheduled_monotonic_s=self.due,lateness_s=late,overrun=overrun,missed_periods=missed)
        sent=self.clock();self.due=sent+self.period
        result=self.adapter.step(q,steps=steps,phase=phase)
        self.timeline.emit('command_return',index=self.calls,transport_and_driver_s=self.clock()-sent)
        self.calls+=1;return self._observation(result)
    def note(self,text):
        self.timeline.emit('policy_note',text=text)
        return self.adapter.note(text)


def summarize(rows,evaluation):
    marks={r['event']:r['monotonic_s'] for r in rows if r['event'] not in ('command_dispatch','command_return','observation_available','policy_note')}
    dispatch=[r for r in rows if r['event']=='command_dispatch']
    task=marks.get('task_dispatch');stable=evaluation.get('stable_hold_monotonic_s')
    if stable is not None:
        if evaluation.get('clock')!='perf_counter':raise ValueError('stable time requires the same perf_counter clock')
        if not marks.get('first_command',float('inf'))<=stable<=marks.get('evaluation_done',-float('inf')):raise ValueError('stable time outside execution/evaluation timeline')
    duration=None if stable is None or task is None else stable-task
    return dict(clock='time.perf_counter',marks=marks,stable_hold_monotonic_s=stable,
        task_to_stable_s=duration,official_success=evaluation.get('official_success'),
        official_success_within40s=bool(evaluation.get('official_success') and duration is not None and duration<=40),
        timing_evidence_complete=duration is not None,command_count=len(dispatch),
        overrun_count=sum(r['overrun'] for r in dispatch),missed_periods=sum(r['missed_periods'] for r in dispatch),
        max_lateness_s=max((r['lateness_s'] for r in dispatch),default=0),evaluation=evaluation)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',required=True,help='Explicit installed module:factory for already authorized safe device adapter')
    parser.add_argument('--controller',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    timeline=Timeline(args.output/'monotonic.jsonl');timeline.emit('process_start')
    adapter=None;evaluation={};error=None
    try:
        module,factory=args.adapter.split(':',1);timeline.emit('initialization_start')
        adapter=getattr(importlib.import_module(module),factory)();timeline.emit('resident_ready')
        spec=importlib.util.spec_from_file_location('stage16_hardware_controller',args.controller)
        controller=importlib.util.module_from_spec(spec);sys.modules[spec.name]=controller;spec.loader.exec_module(controller)
        context=TimingContext(adapter,timeline);timeline.emit('task_dispatch');controller.main(context);timeline.emit('control_complete')
        timeline.emit('evaluation_start');evaluation=adapter.evaluate();timeline.emit('evaluation_done')
    except Exception as exc:
        error=repr(exc);timeline.emit('error',error=error)
    finally:
        if adapter is not None:
            timeline.emit('close_start')
            try:adapter.close()
            except Exception as exc:error=error or repr(exc);timeline.emit('close_error',error=repr(exc))
            timeline.emit('close_done')
        timeline.emit('process_exit_boundary')
        try:summary=summarize(timeline.rows,evaluation)
        except Exception as exc:summary=dict(error=repr(exc),official_success_within40s=False,timing_evidence_complete=False);error=error or repr(exc)
        summary.update(adapter=args.adapter,error=error,condition='real driver only if supplied adapter is real; no device was discovered or selected by this harness')
        (args.output/'timing_summary.json').write_text(json.dumps(summary,indent=2));timeline.close()
    if error:raise RuntimeError(error)

if __name__=='__main__':main()
