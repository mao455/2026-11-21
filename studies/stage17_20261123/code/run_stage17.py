"""Maintained Stage17 entry: shared color policy and measured same-color light scaling."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_stage14 import controller_settings

HERE = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--color-mode',choices=('legacy','hs'),default='hs')
    p.add_argument('--light-multiplier',type=float,default=1.)
    p.add_argument('--evaluation-mode',choices=('inline','deferred'),default='deferred')
    p.add_argument('--normal-weight',type=float,default=None)
    p.add_argument('--margin-tie',choices=('historical','bounded_relative'),default='historical')
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--case-file', required=True, type=Path)
    p.add_argument('--asset-root', type=Path)
    p.add_argument('--registry-manifest', type=Path)
    p.add_argument('--controller', type=Path)
    p.add_argument('--profile', choices=('v9','fast','retry'), default='fast')
    p.add_argument('--fingers', choices=('two','three','auto'), default='two')
    p.add_argument('--max-attempts', type=int, choices=(1,2,3), default=3)
    p.add_argument('--preload', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--mode', choices=('minimal','full'), default='minimal')
    p.add_argument('--phase', choices=('explore','heldout'), default='explore')
    a=p.parse_args()
    if not .1 <= a.light_multiplier <= 3.:p.error('light multiplier outside .1..3')
    os.environ['FINGEREYE_LIGHT_SCALE']=str(a.light_multiplier)
    if a.preload and a.profile != 'fast' and not a.controller:
        p.error('--preload requires fast profile or an explicit diagnostic controller')
    if a.asset_root:
        os.environ['FINGEREYE_ASSET_ROOT']=str(a.asset_root.resolve())
    if a.registry_manifest:
        os.environ['FINGEREYE_OBJECT_MANIFEST']=str(a.registry_manifest.resolve())
    from stage15_runtime import asset_root, external_objects
    selected_root=asset_root()
    external_objects(selected_root)  # fail before expensive launch on changed assets
    if (a.session/'episode').exists():
        p.error('immutable episode exists; select a new session')
    a.session.mkdir(parents=True, exist_ok=True)
    controller=a.controller.resolve() if a.controller else a.session.resolve()/'submitted.py'
    if a.controller is None:
        if a.profile=='fast':
            from stage17_build_controller import build
            build(controller, fingers=a.fingers, attempts=a.max_attempts, preload=a.preload, normal_weight=a.normal_weight, margin_tie=a.margin_tie, color_mode=a.color_mode)
        else:
            from run_stage14 import build
            build(controller, profile=a.profile, fingers=a.fingers, attempts=a.max_attempts)
    sha=hashlib.sha256(controller.read_bytes()).hexdigest()
    case=json.loads(a.case_file.read_text());case.pop('name',None)
    # Paired archived public RGB can be relative to its case file for portability.
    reference=case.get('initial_overhead_reference')
    if reference:
        for key in ('rgb_path','metadata_path'):
            if not Path(reference[key]).is_absolute():
                reference[key]=str((a.case_file.parent/reference[key]).resolve())
    freeze=controller.parent/'freeze.json'
    if freeze.exists():
        manifest=json.loads(freeze.read_text())
        if manifest['controller_sha256']!=sha:
            raise RuntimeError('frozen controller hash mismatch')
        for name,digest in manifest.get('runtime_sha256',{}).items():
            path=HERE.parents[1]/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise RuntimeError('frozen runtime changed: '+name)
        for name,digest in manifest.get('asset_sha256',{}).items():
            if hashlib.sha256((selected_root/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('frozen static asset changed: '+name)
        if a.phase=='heldout' and case not in manifest['acceptance_cases']:
            raise RuntimeError('case absent from preregistered freeze')
    elif a.phase=='heldout':
        raise RuntimeError('heldout requires a frozen controller manifest')
    settings=controller_settings(controller)
    effective={}
    for node in ast.parse(controller.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('S15_PRELOAD','S15_LIGHT_FIRST'):
                    effective[target.id]=ast.literal_eval(node.value)
    cfg=json.loads((HERE/'stage9_config.json').read_text())
    assert (cfg['physics_hz'],cfg['control_hz'],cfg['camera_fps'])==(120,20,30)
    cfg.update(case=case,controller_sha256=sha,selected_control_settings=settings.get('S11_CONFIG'),
        max_attempts=settings.get('S5_ATTEMPTS'),recording_mode=a.mode,
        asset_root=str(selected_root),protocol='lift30_no_drop_v1',
        entry='run_stage17.py',light_multiplier=a.light_multiplier,color_mode_requested=a.color_mode,evaluation_mode=a.evaluation_mode,preload_requested=a.preload,effective_stage15_settings=effective,phase=a.phase)
    (a.session/'stage9_configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (a.session/'frozen_controller.json').write_text(json.dumps(dict(sha256=sha))+'\n')
    from stage14_resources import admission_slot
    with admission_slot(a.session):
        return subprocess.call([sys.executable,str(HERE/'stage16_episode.py'),'--session',str(a.session.resolve()),
            '--name','episode','--max-episodes','1','--phase',a.phase,
            '--controller',str(controller),'--case-json',json.dumps(case)],env=dict(os.environ,S9_MODE=a.mode,S16_EVALUATION_MODE=a.evaluation_mode))


if __name__=='__main__':
    raise SystemExit(main())
