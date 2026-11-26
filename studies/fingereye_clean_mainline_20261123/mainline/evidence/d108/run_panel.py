"""Frozen normal-entry Wuji panel, with the original four D107 object/pose cases."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
E=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--controller',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 registration=json.loads((Path(__file__).parent/'registration.json').read_text());urdf=Path(registration['hand_urdf']);assert sha(urdf)==registration['hand_sha256'];controller=a.controller or E/'mainline/strategy.py'
 files=[*E.glob('*.py'),*(E/'mainline').glob('*.py'),*(E/'hand_calibrations').glob('*.json'),*(E/'mainline/fixtures/wuji_first').glob('*.json'),Path(__file__),Path(__file__).parent/'registration.json',controller];pin={str(f):sha(f) for f in files};rows=[]
 for relative in registration['cases']:
  case=E/relative;session=a.out/case.stem;command=[sys.executable,str(E/'run_stage17.py'),'--session',str(session),'--case-file',str(case),'--registry-manifest',str(E/'mainline/fixtures/wuji_first/objects.json'),'--robot-urdf',str(urdf),'--controller',str(controller),'--evaluation-mode','inline','--mode','minimal','--fingers','auto','--max-attempts','3','--no-preload','--phase','explore','--profile','fast','--color-mode','hs','--light-multiplier','1.0'];rows.append(dict(case=case.stem,hand='wuji',session=str(session),argv=command,status='planned'))
 state=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=E,text=True).strip(),controller_sha256=sha(controller),registration=registration,files=pin,rows=rows)
 def save():(a.out/'STATE.json').write_text(json.dumps(state,indent=2))
 save();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 for row in rows:
  assert all(sha(Path(f))==h for f,h in pin.items());row['status']='running';save();session=Path(row['session']);command=[sys.executable,str(E/'round2_evidence/budget_run.py'),'--host-timeout-s','1800','--receipt',str(session/'budget_receipt.json'),'--cwd',str(E)]
  for f in [controller,E/'cross_hand_runtime.py',E/'mainline/hand_description.py',E/'evaluator.py']:command+=['--fingerprint-file',str(f)]
  with (a.out/(session.name+'.log')).open('x') as log:rc=subprocess.call(command+['--']+row['argv'],cwd=E,env=env,stdout=log,stderr=subprocess.STDOUT)
  result=session/'episode/result.json';receipt=json.loads((session/'budget_receipt.json').read_text());assert receipt['source_unchanged'] and not receipt['leftover_descendants'];row.update(status='terminal',return_code=rc,budget_status=receipt['status'],receipt_sha256=sha(session/'budget_receipt.json'),result=json.loads(result.read_text()) if result.exists() else None);assert all(sha(Path(f))==h for f,h in pin.items());save();print(json.dumps({k:row[k] for k in ['case','status','return_code','result']}),flush=True)
if __name__=='__main__':main()
