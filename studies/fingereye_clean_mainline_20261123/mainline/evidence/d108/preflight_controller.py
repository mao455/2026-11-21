"""Motion/RGB admission probe away from the object; no grasp candidate execution."""
import json
import numpy as np
S17_COLOR_MODE='hs'
S5_ATTEMPTS=3
S15_PRELOAD=False
S11_CONFIG={'fingers':'auto'}
def main(ctx):
 obs=ctx.observe();q=np.asarray(obs['actual_q'],float);home=q.copy();states=[]
 for index,change in [(0,.015),(1,.015),(2,.015),(3,.12),(4,.10),(5,.10),(6,.18),(10,.18),(14,.18)]:
  target=home.copy();target[index]+=change
  for i in range(1,11):obs=ctx.step(q+(target-q)*i/10,6,'admission_motion')
  for _ in range(10):obs=ctx.step(target,6,'admission_settle')
  states.append(dict(index=index,target=target.tolist(),actual_q=obs['actual_q'],step=obs['step']))
  q=np.asarray(obs['actual_q'])
  for i in range(1,11):obs=ctx.step(q+(home-q)*i/10,6,'admission_return')
  q=np.asarray(obs['actual_q'])
 views={}
 for name,frame in obs['fingereye'].items():
  rgb=np.load(frame['rgb_path']);views[name]=dict(shape=list(rgb.shape),std=float(rgb.std()),min=int(rgb.min()),max=int(rgb.max()))
 ctx.note(json.dumps(dict(admission_motion=states,rgb=views,actual_closures=0)))
