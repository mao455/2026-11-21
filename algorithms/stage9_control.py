"""Exposure/current joint separation and control/camera scheduling adapter.
The V7 policy prefix retains candidate generation, speed/path guards and FSM.
"""
_S9_IO={'cycles':[],'consumers':[],'early':[],'duplicate_cache_hits':0,'invalid_frames':0}
_S9_CACHE={};_S9_RECEIVED={};_S9_COMPUTE_START=time.perf_counter()

def s9_frame(obs):
    c=obs['image_state'];frames=obs['fingereye'];ids={f['frame_id'] for f in frames.values()}
    if len(ids)!=1 or c['step']>obs['step'] or obs['time_s']-c['time_s']>S9_MAX_FRAME_AGE:
        _S9_IO['invalid_frames']+=1;raise RuntimeError('no fresh valid RGB; safe stop')
    return next(iter(ids))

def s9_exposure(obs):
    s9_frame(obs);c=obs['image_state']
    return dict(obs,step=c['step'],time_s=c['time_s'],actual_q=c['actual_q'],actual_qvel=c['actual_qvel'])

def s9_visual(name,fn):
    def wrapped(obs,*a,**kw):
        fid=s9_frame(obs);key=(name,fid,id(a[0]) if a else None)
        if key in _S9_CACHE:
            _S9_IO['duplicate_cache_hits']+=1;return _S9_CACHE[key]
        start=time.perf_counter();c=obs['image_state']
        result=fn(s9_exposure(obs),*a,**kw)
        _S9_IO['consumers'].append(dict(function=name,control_step=obs['step'],frame_id=fid,
            capture_step=c['step'],capture_q=c['actual_q'],current_q=obs['actual_q'],consume_wall_s=start,
            capture_to_use_wall_s=start-c['capture_done_wall_s'],frame_age_s=obs['time_s']-c['time_s'],compute_s=time.perf_counter()-start))
        _S9_CACHE[key]=result
        # Keep a bounded state-estimation cache; never count a cache hit as evidence.
        if len(_S9_CACHE)>30:
            for k in list(_S9_CACHE):
                if k[1]<fid-2:del _S9_CACHE[k]
        return result
    return wrapped
capture_preload_snapshot=s9_visual('capture_preload_snapshot',capture_preload_snapshot)
stage3_support_snapshot=s9_visual('stage3_support_snapshot',stage3_support_snapshot)
_s9_policy_frames=Stage3LiveVolume.frames
def _s9_volume_frames(self,obs):return _s9_policy_frames(self,s9_exposure(obs))
Stage3LiveVolume.frames=_s9_volume_frames
def stage3_live_geometry(volume,obs,previous_info):
    start=time.perf_counter();result=_s9_geometry(volume,s9_exposure(obs),previous_info)
    _S9_IO['consumers'].append(dict(function='stage3_live_geometry',control_step=obs['step'],frame_id=s9_frame(obs),capture_step=obs['image_state']['step'],capture_q=obs['image_state']['actual_q'],current_q=obs['actual_q'],consume_wall_s=start,capture_to_use_wall_s=start-obs['image_state']['capture_done_wall_s'],frame_age_s=obs['time_s']-obs['image_state']['time_s'],compute_s=time.perf_counter()-start))
    return result

class _S9Context:
    def __init__(self,ctx):self.ctx=ctx;self.robot_description=ctx.robot_description;self.last_return=time.perf_counter();self.inbox=S9FrameInbox(S9_MAX_FRAME_AGE);_S9_IO['inbox']=self.inbox.counts
    def step(self,q,steps=1,phase=''):
        if steps%6:raise ValueError('20Hz commands require multiples of6 physical steps')
        for _ in range(steps//6):
            start=time.perf_counter();compute=start-self.last_return
            obs=self.inbox.select(self.ctx.step(q,steps=6,phase=phase));done=time.perf_counter();s9_frame(obs)
            _S9_IO['cycles'].append(dict(end_step=obs['step'],phase=phase,compute_between_dispatch_s=compute,components_s=dict(_S9_WORK),dispatch_rpc_s=done-start,frame_id=s9_frame(obs),capture_step=obs['image_state']['step'],frame_age_s=obs['time_s']-obs['image_state']['time_s']))
            _S9_WORK.clear();self.last_return=time.perf_counter()
        return obs
    def observe(self):
        obs=self.inbox.select(self.ctx.observe());s9_frame(obs);self.last_return=time.perf_counter();return obs
    def note(self,text):return self.ctx.note(text)

_S9_WORK={};_S9_STACK=[];_S9_SAMPLES={}
def s9_timed(category,fn):
    def call(*a,**kw):
        start=time.perf_counter();frame=[start,0.];_S9_STACK.append(frame)
        try:return fn(*a,**kw)
        finally:
            elapsed=time.perf_counter()-start;_S9_STACK.pop();exclusive=elapsed-frame[1]
            _S9_WORK[category]=_S9_WORK.get(category,0.)+exclusive
            _S9_SAMPLES.setdefault(category,[]).append(exclusive)
            if _S9_STACK:_S9_STACK[-1][1]+=elapsed
    return call
for _s9_n,_s9_c in (('optimize','ik'),('close_ik','ik'),('capture_preload_snapshot','vision'),('stage3_support_snapshot','vision'),('stage3_support_pair','vision'),('compare_preload','vision'),('stage3_live_geometry','vision'),('_S3_ESTIMATE','vision'),('s5_decide','fsm')):
    globals()[_s9_n]=s9_timed(_s9_c,globals()[_s9_n])
SelfAwareGeometry.check=s9_timed('path_checks',SelfAwareGeometry.check)


def main(ctx):
    start=time.perf_counter()
    try:return policy_main(_S9Context(ctx))
    finally:
        _S9_IO['component_samples_s']=_S9_SAMPLES
        _S9_IO.update(total_wall_s=time.perf_counter()-start,early_settle=S9_EARLY_SETTLE,physics_hz=120,control_hz=20,camera_fps=30)
        ctx.note(json.dumps({'s9_io_audit':_S9_IO},allow_nan=False))
