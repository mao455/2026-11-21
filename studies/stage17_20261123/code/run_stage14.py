"""Normal shared FSM entry; V9 remains the default recommended profile."""
import argparse,ast,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
V7_SHA='12647be9c1ced3a2f370930d96768ae25c438d86cc816600696348db4d176f91'
def controller_settings(path):
    """Read final literal assignments from the program actually submitted."""
    assignments={}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('S11_CONFIG','S5_ATTEMPTS','S9_EARLY_SETTLE'):
                    assignments[target.id]=node.value
    # Earlier dynamic defaults are allowed; the effective final assignments
    # must be literal, otherwise configuration provenance would be misleading.
    return {name:ast.literal_eval(value) for name,value in assignments.items()}

def build(path,early=None,profile="v9",attempts=3,fingers="two"):
    cfg=json.loads((HERE/'stage9_config.json').read_text())
    if early is None:early=cfg['early_settle']
    if profile not in cfg['profiles']:raise ValueError('unknown control profile')
    if attempts not in (1,2,3):raise ValueError('physical attempt budget 1..3')
    if fingers not in ('two','three','auto'):raise ValueError('unsupported finger group')
    settings=dict(cfg['profiles'][profile],profile=profile,fingers=fingers,
                  task_ranking=profile!='v9' or fingers!='two',evidence_recovery=profile=='fast' and fingers=='auto',task_wrench=cfg.get('task_wrench',{}))
    source='\n'.join((HERE/name).read_text() for name in ('visual_controller.py','stage9_frames.py','stage5_control.py','stage13_recovery.py','stage9_control.py'))
    source+='\nS9_MAX_FRAME_AGE=.1\nS9_EARLY_SETTLE='+repr(early)+'\n'
    source+='\nS5_ATTEMPTS='+repr(attempts)+'\nS11_CONFIG='+repr(settings)+'\n'
    compile(source,str(path),'exec');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
    return hashlib.sha256(source.encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True,type=Path);p.add_argument('--case-file',required=True,type=Path)
    p.add_argument('--fingers',choices=['two','three','auto'],default='two');p.add_argument('--profile',choices=['v9','fast','retry'],default='v9');p.add_argument('--max-attempts',type=int,choices=[1,2,3],default=3);p.add_argument('--controller',type=Path);p.add_argument('--early-settle',action=argparse.BooleanOptionalAction,default=None);p.add_argument('--mode',choices=['minimal','full'],default='minimal');p.add_argument('--phase',choices=['explore','heldout'],default='explore');a=p.parse_args()
    if (a.session/'episode').exists():p.error('immutable episode exists')
    a.session.mkdir(parents=True,exist_ok=True);controller=a.controller or a.session/'submitted.py'
    if a.controller:
        if a.early_settle is not None:p.error('frozen settings already present')
        sha=hashlib.sha256(controller.read_bytes()).hexdigest()
    else:sha=build(controller,a.early_settle,a.profile,a.max_attempts,a.fingers)
    runtime_manifest=controller.parent/'runtime_manifest.json'
    if a.controller and runtime_manifest.exists():
        for name,digest in json.loads(runtime_manifest.read_text())['sha256'].items():
            assert hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest()==digest,'frozen runtime differs: '+name
    case=json.loads(a.case_file.read_text());case.pop('name',None)
    if a.phase=='heldout':
        manifest=json.loads((controller.parent/'freeze.json').read_text());assert manifest['sha256']==sha and case in list(manifest['validation_cases'].values())
    (a.session/'frozen_controller.json').write_text(json.dumps(dict(sha256=sha)))
    cfg=json.loads((HERE/'stage9_config.json').read_text());assert cfg['physics_hz']==120 and cfg['control_hz']==20 and cfg['camera_fps']==cfg['video_fps']==30 and cfg['control_steps']==6 and cfg['camera_steps']==4
    actual=controller_settings(controller)
    selected=actual.get('S11_CONFIG',dict(profile='frozen_v9',fast_close=False,local_retry=False))
    cfg.update(profile=selected.get('profile','frozen_v9'),max_attempts=actual.get('S5_ATTEMPTS',3),recording_mode=a.mode,early_settle=actual.get('S9_EARLY_SETTLE',False),controller_sha256=sha,base_sha256=V7_SHA,case=case,selected_control_settings=selected)
    (a.session/'stage9_configuration.json').write_text(json.dumps(cfg,indent=2))
    from stage14_resources import admission_slot
    with admission_slot(a.session):
        return subprocess.call([sys.executable,str(HERE/'stage14_episode.py'),'--session',str(a.session.resolve()),'--name','episode','--max-episodes','1','--phase',a.phase,'--controller',str(controller.resolve()),'--case-json',json.dumps(case)],env=dict(os.environ,S9_MODE=a.mode))
if __name__=='__main__':raise SystemExit(main())
