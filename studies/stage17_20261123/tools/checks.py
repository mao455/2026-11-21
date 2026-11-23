"""Meaningful invariants and fixed-input equivalence for the maintained policy."""
import importlib.util,json,time,pathlib,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
def load(n):
 p=ROOT/'stage17_evidence/controllers'/n;s=importlib.util.spec_from_file_location(n.replace('.py',''),p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
old=load('stage16_frozen.py');new=load('hs_v2.py');tests=[]
def check(name,passed):
 tests.append(dict(name=name,passed=bool(passed)));assert passed,name
new.S17_COLOR_MODE='hs'
for rgb,valid in [([255,200,0],True),([255,255,0],True),([255,255,255],False),([8,4,0],False),([9,5,0],True)]:check('photometric_'+str(rgb),new.color_reliability(np.array([[rgb]],dtype=np.uint8))[0,0]==valid)
check('unknown_projection_no_vote',not new.color_projection_known(dict(reliable=np.zeros((5,5),bool)),np.array([[2.,2.]]))[0])
check('known_projection_votes',new.color_projection_known(dict(reliable=np.ones((5,5),bool)),np.array([[2.,2.]]))[0])
ep=ROOT/'../../outputs/stage17/lighting/A_L1_legacy/episode';obs=json.loads(next((ep/'public_responses.jsonl').open()))['observation'];info=old._S3_ESTIMATE(obs);r=old.Robot();a=old.SelfAwareGeometry(r,info);b=new.SelfAwareGeometry(r,info);states=[json.loads(s) for s in (ep/'steps.jsonl').read_text().splitlines()];qs=[np.array(states[i]['actual_q']) for i in np.linspace(0,len(states)-1,60,dtype=int)];result={}
for inspect_object in (False,True):
 start=time.perf_counter();aa=[old._S6_BASE_CHECK(a,q,inspect_object) for q in qs];ta=time.perf_counter()-start;start=time.perf_counter();bb=[new._R13_CHECKER.check(b,q,inspect_object) for q in qs];tb=time.perf_counter()-start;check('exact_geometry_'+str(inspect_object),aa==bb);result[str(inspect_object)]=dict(inputs=len(qs),same=aa==bb,old_s=ta,new_s=tb)
 # Actual cached checker equivalence, not approximate shortcuts.
 check('cached_geometry_'+str(inspect_object),[a.check(q,inspect_object) for q in qs]==[b.check(q,inspect_object) for q in qs])
(ROOT/'stage17_evidence/checks.json').write_text(json.dumps(dict(tests=tests,geometry=result,claim='one geometry implementation; timing variation is not a demonstrated speedup'),indent=2)+'\n');print(len(tests),'passed',result)
