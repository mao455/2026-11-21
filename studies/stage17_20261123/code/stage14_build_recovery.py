"""Freeze experimental public-image registration recovery for physical review."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OVERLAY='''
# S14 public shape-prior recovery experiment. Current pose is freshly fitted.
_S14_INIT=s5_init
_S14_SHAPE_PRIOR=None
def s5_init(ctx):
    global _S14_SHAPE_PRIOR
    s=_S14_INIT(ctx)
    _S14_SHAPE_PRIOR={k:np.array(v,copy=True) if isinstance(v,np.ndarray) else v for k,v in s['info'].items()}
    return s
def stage3_live_geometry(volume,obs,previous_info):
    info,report=fit_shape(volume,s9_exposure(obs),_S14_SHAPE_PRIOR)
    return info,report
def _s14_registered_carve(self,obs,fuse=False):
    info,report=fit_shape(self,s9_exposure(obs),_S14_SHAPE_PRIOR)
    if info is None:self.points=None;self.boundary=None;self.tree=None
    return report
Stage3LiveVolume.carve=_s14_registered_carve
'''
FIRST_MISS='''
# Explicit first-miss diagnostic; subsequent recovery remains autonomous.
_S14_ORIGINAL_CLOSE=s5_close
def s5_close(s,mode='fixed',cap=.0045):
    if s.get('actual_grasps',0):return _S14_ORIGINAL_CLOSE(s,mode,cap)
    c=s['c'];gap=-.004
    target=close_ik(s['robot'],c['q'],c['points']+c['inward']*gap)
    s5_note(s['ctx'],'s14_first_miss_diagnostic',dict(step=s['obs']['step'],opening_offset_m=gap,subsequent_action_specified=False,no_reset=True))
    s5_move(s,target,48,'s5_close')
    s5_move(s,s['last'],48,'s5_close_observe')
    s['squeeze']=np.full(len(c['points']),gap)
    s['last_closure_cap']=cap
'''
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--surface-pair',action='store_true');p.add_argument('--fit-module',type=Path,default=HERE/'stage14_shape_fit.py');p.add_argument('--first-miss-diagnostic',action='store_true');a=p.parse_args()
    if a.output.exists():p.error('immutable controller already exists')
    module=a.fit_module;source=a.base.read_text()+'\n'+module.read_text()+OVERLAY+(FIRST_MISS if a.first_miss_diagnostic else '')
    if a.surface_pair:source+='\n'+(HERE/'stage14_surface_pair.py').read_text()
    compile(source,str(a.output),'exec');a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(source)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    a.output.with_suffix('.provenance.json').write_text(json.dumps(dict(base=str(a.base.resolve()),base_sha256=sha(a.base.read_bytes()),module_sha256=sha(module.read_bytes()),controller_sha256=sha(source.encode()),scope='experimental current RGB registration; not validated production policy',forced_first_miss=a.first_miss_diagnostic,private_policy_inputs=False,surface_pair=a.surface_pair,surface_module_sha256=sha((HERE/'stage14_surface_pair.py').read_bytes()) if a.surface_pair else None),indent=2))
if __name__=='__main__':main()
