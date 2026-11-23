"""Second, separately registered cohort after the clipping-rule replay failure."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/stage17/lighting_v2'

def main():
 OUT.mkdir(parents=True,exist_ok=True);plan=[]
 for light in (1.,.4,1.6):
  for case in ('A','B'):
   name=f'{case}_L{light:g}_hs_v2';argv=[sys.executable,str(HERE/'run_stage17.py'),'--session',str(OUT/name),'--case-file',str(HERE/f'stage17_evidence/cases/{case}.json'),'--asset-root','/home/CNS2025331827/docs/fingereye_stage16/FingerEyeLab','--color-mode','hs','--light-multiplier',str(light),'--evaluation-mode','deferred','--mode','minimal'];plan.append(dict(name=name,case=case,light=light,argv=argv))
 target=HERE/'stage17_evidence/lighting_plan_v2.json'
 if target.exists():assert json.loads(target.read_text())==plan
 else:target.write_text(json.dumps(plan,indent=2)+'\n')
 expected=hashlib.sha256((HERE/'stage17_evidence/controllers/hs_v2.py').read_bytes()).hexdigest()
 for row in plan:
  session=OUT/row['name']
  if session.exists():continue
  start=time.perf_counter()
  with (OUT/(row['name']+'.launch.log')).open('w') as log:
   try:rc=subprocess.call(row['argv'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),timeout=1200)
   except subprocess.TimeoutExpired:rc=124
  result_path=session/'episode/result.json';result=json.loads(result_path.read_text()) if result_path.exists() else {};actual=hashlib.sha256((session/'submitted.py').read_bytes()).hexdigest();assert actual==expected
  record=dict(name=row['name'],returncode=rc,launch_wall_s=time.perf_counter()-start,success=result.get('success'),error=result.get('error'),controller_sha256=actual,normal_entry_generated=True)
  with (OUT/'launches.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
if __name__=='__main__':main()
