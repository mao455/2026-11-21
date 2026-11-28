"""Re-run selected normal-entry panel rows with immutable source fingerprints."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

E=Path(__file__).resolve().parents[3]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-state',type=Path,required=True)
    p.add_argument('--select',nargs='+',required=True,help='hand:case rows')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[1,2],default=1)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    base=json.loads(a.from_state.read_text());rows=[]
    for key in a.select:
        hand,case=key.split(':',1)
        original=next(r for r in base['commands']if r['hand']==hand and r['case']==case)
        session=a.out/(case+'_'+hand);argv=list(original['argv'])
        argv[argv.index('--session')+1]=str(session)
        argv[argv.index('--controller')+1]=str(E/'mainline/strategy.py')
        rows.append(dict(hand=hand,case=case,session=str(session),argv=argv,status='planned'))
    files=[*E.glob('*.py'),*(E/'mainline').glob('*.py'),*(E/'hand_calibrations').glob('*.json'),Path(__file__)]
    pin={str(f):sha(f)for f in files}
    state=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=E,text=True).strip(),files=pin,rows=rows)
    lock=threading.Lock()
    def save():
        temp=a.out/'STATE.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(a.out/'STATE.json')
    def verify():assert all(sha(Path(f))==h for f,h in pin.items()),'frozen source changed'
    save();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def run(row):
        verify()
        with lock:row['status']='running';save()
        session=Path(row['session'])
        command=[sys.executable,str(E/'round2_evidence/budget_run.py'),'--host-timeout-s','1800','--receipt',str(session/'budget_receipt.json'),'--cwd',str(E)]
        for f in files:command+=['--fingerprint-file',str(f)]
        with(a.out/(session.name+'.log')).open('x')as log:rc=subprocess.call(command+['--']+row['argv'],cwd=E,env=env,stdout=log,stderr=subprocess.STDOUT)
        verify();receipt=json.loads((session/'budget_receipt.json').read_text());assert receipt['source_unchanged']and not receipt['leftover_descendants']
        f=session/'episode/result.json';result=json.loads(f.read_text())if f.exists()else None
        closures=0;trace=session/'episode/agent_trace.jsonl'
        if trace.exists():
            for line in trace.open():
                e=json.loads(line)
                if e.get('op')=='note':closures=max(closures,int(json.loads(e['text']).get('s13_recovery',{}).get('actual_grasps',0)))
        with lock:
            row.update(status='terminal',return_code=rc,result=result,actual_closures=closures);save()
            print(json.dumps(dict(hand=row['hand'],case=row['case'],closures=closures,success=bool(result and result['success']))),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)as pool:
        tasks=[pool.submit(run,row)for row in rows]
        for task in concurrent.futures.as_completed(tasks):task.result()
    state['completed']=True;save()

if __name__=='__main__':main()
