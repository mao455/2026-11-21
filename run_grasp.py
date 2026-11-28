"""Run one observation-only shared grasp episode in Isaac Lab."""
from pathlib import Path
import argparse,json,os,sys
from datetime import datetime
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-urdf',type=Path,required=True)
    p.add_argument('--case-file',type=Path,default=HERE/'examples/box.json')
    p.add_argument('--objects',type=Path,default=HERE/'examples/objects.json')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    output=(a.output or HERE/'outputs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True,exist_ok=False)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['FINGEREYE_OBJECT_MANIFEST']=str(a.objects.resolve())
    os.environ['FINGEREYE_LIGHT_SCALE']='1.0'
    from hand_runtime import prepare
    hand=prepare(a.robot_urdf,output/'hand_model')
    hand['contact_mode']='auto'
    description=output/'hand_model/description.json'
    description.write_text(json.dumps(hand,indent=2)+'\n')
    os.environ['FINGEREYE_HAND_DESCRIPTION']=str(description)
    from objects import create
    case=json.loads(a.case_file.read_text());case.pop('name',None)
    create(case['object_id'])
    print('OUTPUT',output,flush=True)
    import host
    sys.argv=['host','--session',str(output),'--name','episode','--controller',str(HERE/'strategy.py'),
        '--case-json',json.dumps(case),'--max-episodes','1']
    return host.main()

if __name__=='__main__':raise SystemExit(main())
