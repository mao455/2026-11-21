"""Immutable, sequential physical trials; original evaluator and 120/20/30 Hz."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REL=Path('FingerEyeLab/experiments/rax_dexbin_tinyobj')
D98=HERE.parents[1]/'fixtures/d106'
X=HERE.parents[4]/'assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf'
NEW=Path('/home/CNS2025331827/docs/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM/urdf/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM.urdf')
PY=Path('/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='+',default=['box_center','cylinder_offset','wedge_offset']);p.add_argument('--hands',nargs='+',default=['new','xhand']);p.add_argument('--controller',type=Path);p.add_argument('--baseline',action='store_true');a=p.parse_args()
 e=a.root/REL;controller=a.controller or e/'mainline/strategy.py';a.out.mkdir(parents=True,exist_ok=False)
 files=[*e.glob('*.py'),* (e/'mainline').glob('*.py'),e/'stage9_config.json'];pin={str(x):sha(x) for x in files};rows=[]
 for case in a.cases:
  for hand in a.hands:
   session=a.out/(case+'_'+hand);casefile=D98/case/'case.json'
   if a.baseline:raise ValueError('Use the recoverable D105 commit for historical baseline RGB trials')
   cmd=[str(PY),str(e/'run_stage17.py'),'--session',str(session),'--case-file',str(casefile),'--robot-urdf',str(X if hand=='xhand' else NEW),'--controller',str(controller),'--evaluation-mode','inline','--mode','minimal','--fingers','two' if a.baseline else 'auto','--max-attempts','3','--no-preload','--phase','explore','--profile','fast','--color-mode','hs','--light-multiplier','1.0']
   rows.append(dict(case=case,hand=hand,session=str(session),argv=cmd,status='planned'))
 state=dict(root=str(a.root),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.root,text=True).strip(),controller_sha256=sha(controller),files=pin,rows=rows)
 def save():(a.out/'STATE.json').write_text(json.dumps(state,indent=2))
 save();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for row in rows:
  assert all(sha(Path(f))==h for f,h in pin.items());row['status']='running';save();session=Path(row['session'])
  cmd=[str(PY),str(HERE.parents[2]/'round2_evidence/budget_run.py'),'--host-timeout-s','1800','--receipt',str(session/'budget_receipt.json'),'--cwd',str(e)]
  for f in [controller,e/'mainline/hand_description.py',e/'mainline/contact_regions.py',e/'evaluator.py']:
   if f.exists():cmd+=['--fingerprint-file',str(f)]
  with (a.out/(session.name+'.log')).open('x') as f:rc=subprocess.call(cmd+['--']+row['argv'],cwd=e,env=env,stdout=f,stderr=subprocess.STDOUT)
  receipt=json.loads((session/'budget_receipt.json').read_text())
  assert receipt['source_unchanged'] and not receipt.get('leftover_descendants')
  row.update(status='terminal',return_code=rc,budget_status=receipt['status'],receipt_sha256=sha(session/'budget_receipt.json'));result=session/'episode/result.json'
  if result.exists():row['result']=json.loads(result.read_text())
  assert all(sha(Path(f))==h for f,h in pin.items());save();print(json.dumps({k:row[k] for k in ['case','hand','status','return_code']}),flush=True)
if __name__=='__main__':main()
