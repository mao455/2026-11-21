"""Explicit, asset-free Stage16 public report export. Run in the full private checkout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--here',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args();h=a.here.resolve();repo=h.parents[2]
    out=h/'publication/stage16_20261122';out.mkdir(parents=True,exist_ok=True)
    import sys
    sys.path.insert(0,str(h))
    import publish_github
    provenance=[]
    def sanitize(blob):
        text=blob.decode('utf-8')
        text=text.replace('<LOCAL_DOCS>','<LOCAL_DOCS>')
        text=text.replace('<ISAAC_PYTHON>','<ISAAC_PYTHON>')
        text=text.replace('<LOCAL_HOME>','<LOCAL_HOME>')
        return text.encode('utf-8')
    def copy(source,dest):
        source=h/source;blob=source.read_bytes();public=sanitize(blob)
        if dest.startswith('validation/'):
            public=('\n'.join(line.rstrip(' \t') for line in public.decode().splitlines())+'\n').encode()
        target=out/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(public)
        provenance.append(dict(public_file=dest,source_file=str(source.relative_to(repo)),
                               source_sha256=digest(blob),public_sha256=digest(public),
                               source_bytes=len(blob),public_bytes=len(public),
                               transformation='UTF-8 local-root redaction; test-log trailing spaces trimmed if applicable' if public!=blob else 'byte exact'))
    # Already audited structured packages. No images, private assets, or raw step streams.
    for folder, dest in [('publication/stage16_closeout/noise','noise'),('stage16_evidence/closeout/gates_recovery','gates_recovery')]:
        for src in sorted((h/folder).glob('*')):
            if src.is_file() and src.suffix in ('.csv','.json','.txt','.md','.py'):
                copy(str(src.relative_to(h)),dest+'/'+src.name)
    for src in sorted((h/'stage16_evidence/closeout/contact_capture').glob('*')):
        if src.is_file() and src.suffix in ('.csv','.json','.txt','.py','.patch'):
            copy(str(src.relative_to(h)),'contact_capture/'+(src.name+'.txt' if src.suffix=='.patch' else src.name))
    for name in ('STAGE16_HANDOFF.txt','STAGE16_REPRODUCE.txt','STAGE16_EVIDENCE.txt','STAGE16_NUMERICS.txt','STAGE16_TIMING.txt'):
        copy(name,'reports/'+name)
    for name in ('episodes.csv','timing.csv','groups.csv','hold_contacts.csv','phase_physics.csv','resume_verification.json','A_commanded_lift_reference.json'):
        src='stage16_evidence/independent_acceptance_final/'+name
        if (h/src).exists():copy(src,'final_acceptance/'+name)
    for name in ('finalA_dn0.json','finalB_dn3.json','finalD_dn20.4.json','manifest.json'):
        src='stage16_evidence/acceptance_final/'+name
        if (h/src).exists():copy(src,'configurations/'+name)
    for name in ('simulation_phase_timing.csv','simulation_timing.csv','frozen_replay_timing.csv','frozen_replay_phases.csv','timing_readout.py','physical_regression.json','physical_regression.csv','audit_fix.json','validation.json'):
        src='stage16_evidence/closeout/'+name
        if (h/src).exists():copy(src,'timing/'+name)
    for label in ('A','B'):
        copy('stage16_evidence/closeout/paced_final_'+label+'/timing_summary.json','timing/paced_final_'+label+'.json')
    for name in ('historical_boundaries.csv','savings.csv','placement.csv','placement_boundaries.csv','public_replay_functions.csv','paced_summary.csv','hardware_availability.json'):
        copy('stage16_evidence/timing/'+name,'timing/historical/'+name)
    for name in ('version_gate_diff.csv','acceptance_versions_commands.csv','acceptance.csv'):
        copy('stage16_evidence/evidence/'+name,'history/'+name)
    copy('stage16_evidence/final/stage16_version_diff.csv','history/stage16_version_diff.csv')
    for name in ('run_stage16.py','stage16_backend.py','stage16_episode.py','stage16_build_controller.py','stage16_numerics.py','stage16_numerics_overlay.py','stage16_evidence_audit.py','stage16_realtime_timing.py','stage16_replay_profile.py','stage16_timing_audit.py','stage15_batch.py','stage15_readout.py','stage15_control.py','stage15_runtime.py','stage15_camera_noise.py','stage14_control.py','stage14_geometry.py','stage14_resources.py','stage13_recovery.py','general_grasp_contacts.py','general_grasp_integration.py','evaluator.py','publish_github.py','test_github_publish.py','test_stage16_evidence_audit.py','test_stage16_backend.py','test_stage16_numerics.py','test_stage16_realtime_timing.py'):
        if (h/name).exists():copy(name,'code/'+name)
    copy('stage16_evidence/timing/recorded_adapter.py','code/recorded_adapter.py')
    copy('stage16_evidence/timing/test_recorded_adapter.py','code/test_recorded_adapter.py')
    copy('stage16_evidence/closeout/release/controller.py','code/frozen_controller.py')
    copy('stage16_evidence/closeout/build_public.py','code/build_public.py')
    for src in sorted((h/'stage16_evidence/closeout/validation').glob('*.txt')):
        copy(str(src.relative_to(h)),'validation/'+src.name)
    copy('stage16_evidence/closeout/physical_readout.py','code/physical_readout.py')
    copy('stage16_evidence/closeout/PUBLIC_REPORT.md','REPORT.md')
    copy('stage16_evidence/closeout/PUBLIC_REPRODUCE.txt','REPRODUCE.txt')
    # One index, distinct populations; repeated evidence is never a new independent trial.
    tables=[('numerics','numerics/paired_modes.csv'),('noise','noise/episodes.csv'),('gates_recovery','gates_recovery/episodes.csv'),('stage15_original_six','history/acceptance.csv'),('final_acceptance','final_acceptance/episodes.csv'),('new_physical_regression','timing/physical_regression.csv'),('new_contact_diagnostic','contact_capture/episode.csv')]
    rows=[]
    for population,path in tables:
        if not (out/path).exists():continue
        with (out/path).open() as f:
            for i,r in enumerate(csv.DictReader(f)):
                rows.append(dict(evidence_population=population,source_table=path,source_row=i+2,**r))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
    freeze=json.loads((h/'stage16_evidence/closeout/release/freeze.json').read_text())
    reduced=dict(source_execution_commit=freeze['source_parent_commit'],controller_sha256=freeze['controller_sha256'],
                 protocol=freeze['protocol'],physics_hz=120,control_hz=20,rgb_fps=30,
                 recommended=dict(fingers='two',recovery=True,recovery_preload=False,normal_weight='historical 4/7',margin_tie='historical',evaluation_mode='deferred'),
                 runtime_file_count=len(freeze['runtime_sha256']),asset_calibration_file_count=len(freeze['asset_sha256']),
                 evaluator_sha256=digest((h/'evaluator.py').read_bytes()),
                 full_private_freeze_sha256=digest((h/'stage16_evidence/closeout/release/freeze.json').read_bytes()),
                 assets='Not published. Full authorized asset/calibration root and matching hashes required.',
                 historical_frozen_acceptance_source='979b5a87cf7a3f47f4f7e8b59d1e8f974ac0682b',
                 starting_handoff='d584af11938b382a2dd7852c53a7b7802239812d')
    (out/'FREEZE.json').write_text(json.dumps(reduced,ensure_ascii=False,indent=2)+'\n')
    (out/'EXPORT_PROVENANCE.json').write_text(json.dumps(dict(scope='Existing physical episodes plus explicitly marked new replay/audit/regression. Local roots redacted; raw data and assets excluded.',files=provenance),ensure_ascii=False,indent=2)+'\n')
    # Preserve the actual current remote README; old reports/manifest are left intact by additive publish.
    previous=(h/'stage16_evidence/closeout/previous_public_README.md').read_text()
    note='# Stage16 收尾核查\n\n最新：[五项逐项报告](studies/stage16_20261122/REPORT.md) · [逐例表](studies/stage16_20261122/cases.csv) · [复跑命令](studies/stage16_20261122/REPRODUCE.txt)。\n\nStage16推荐固定两指、恢复开、恢复预载关；真机40秒与正常三指垫尚未通过。下文为保留的Stage13历史对照。\n\n'
    (out/'PUBLIC_ROOT_README.md').write_text(note+previous)
    files={str(p.relative_to(repo)):p.read_bytes() for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    publish_github.validate_files('research',files)
    public_files={('README.md' if Path(src).name=='PUBLIC_ROOT_README.md' else 'studies/stage16_20261122/'+str(Path(src).relative_to(out.relative_to(repo)))):src for src in files}
    config=dict(schema=1,author=dict(name='mao455',email='194568355+mao455@users.noreply.github.com'),
                research=dict(remote='https://github.com/mao455/2026-11-21.git',branch='main',manifest_path='studies/stage16_20261122/PUBLICATION_MANIFEST.json',files=public_files))
    (h/'stage16_publication.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=sum(map(len,files.values())),case_rows=len(rows)),indent=2))


if __name__=='__main__':
    main()
