"""Preregistered small lighting comparison; keeps all failed launches/results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/stage17/lighting'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    b=dict(object_id='abc_176242',dx=.001,dy=-.001,yaw=.2,mass_kg=.007,friction=.82,noise_seed=15002,rgb_noise_std=0)
    (HERE/'stage17_evidence/cases/B.json').write_text(json.dumps(b,indent=2)+'\n')
    plan=[]
    for light in (1.,.4,1.6):
        for case in ('A','B'):
            for mode in ('legacy','hs'):
                name=f'{case}_L{light:g}_{mode}'
                argv=[sys.executable,str(HERE/'run_stage17.py'),'--session',str(OUT/name),'--case-file',str(HERE/f'stage17_evidence/cases/{case}.json'),'--controller',str(HERE/f'stage17_evidence/controllers/{"stage16_frozen" if mode=="legacy" else "hs"}.py'),'--asset-root','/home/CNS2025331827/docs/fingereye_stage16/FingerEyeLab','--color-mode',mode,'--light-multiplier',str(light),'--evaluation-mode','deferred','--mode','minimal']
                plan.append(dict(name=name,case=case,light=light,color_mode=mode,argv=argv))
    target=HERE/'stage17_evidence/lighting_plan.json'
    if target.exists():assert json.loads(target.read_text())==plan
    else:target.write_text(json.dumps(plan,indent=2)+'\n')
    for row in plan:
        session=OUT/row['name']
        if (session/'episode/result.json').exists():continue
        if session.exists():
            print('INCOMPLETE_RETAINED',row['name'],flush=True);continue
        before=time.perf_counter()
        with (OUT/(row['name']+'.launch.log')).open('w') as log:
            try:rc=subprocess.call(row['argv'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),timeout=1200)
            except subprocess.TimeoutExpired:rc=124
        result_path=session/'episode/result.json';result=json.loads(result_path.read_text()) if result_path.exists() else {}
        record=dict(name=row['name'],returncode=rc,launch_wall_s=time.perf_counter()-before,success=result.get('success'),error=result.get('error'))
        with (OUT/'launches.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
if __name__=='__main__':main()
