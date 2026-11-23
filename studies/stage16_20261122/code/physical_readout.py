"""Re-evaluate the new ordinary-entry physical regression and bind it to raw hashes."""
import csv
import hashlib
import json
from pathlib import Path
from stage15_readout import read

h=Path(__file__).resolve().parents[2]
ep=h.parents[1]/'outputs/stage16_closeout_physical/A_regression/episode'
r=read(ep)
score=json.loads((ep/'result.json').read_text())
b=json.loads((ep/'wall_profile_backend.json').read_text())
rpc=[x for x in json.loads((ep/'rpc_timing.json').read_text()) if x['op']=='step']
ready_to_last=rpc[-1]['start_wall_s']+rpc[-1]['inclusive_s']-b['marks']['ready']
row=dict(id='closeout_A_regression',execution_source='60b5a7b007fe5c8d52f6a11c31fa66c406238c7a',
         controller_sha256=r['controller_sha256'],configuration_sha256=hashlib.sha256((ep.parent/'stage9_configuration.json').read_bytes()).hexdigest(),
         case=json.dumps(r['case'],sort_keys=True),execution_valid=r['execution_valid'],success=r['success'],legacy_strict_success=r['legacy_strict_success'],
         actual_grasps=r['actual_grasps'],physical_steps=r['physical_steps'],physical_s=r['physical_s'],host_elapsed_s=r['host_elapsed_s'],
         ready_to_terminal_rpc_s=ready_to_last,formal_success_wall_within40=r['success'] and ready_to_last<=40,
         first_invalid_step=r['first_invalid_step'],failure=r['failure'],raw_score_exact_match=r['physics']==score['physics'],
         geometry_passed=r['geometry_passed'],hold_min_lift_m=r['physics']['hold_min_lift_m'],hold_min_clearance_m=r['physics']['hold_min_table_clearance_m'])
payload=dict(scope='ONE new actual closed-loop simulator regression after an offline-audit/publication-only fix; no hardware; not independent979bheldout acceptance',
             row=row,episode=str(ep),command='OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 <ISAAC_PYTHON> run_stage16.py --session ../../outputs/stage16_closeout_physical/A_regression --case-file stage16_evidence/acceptance_final/finalA_dn0.json --asset-root <STAGE16_TREE>/FingerEyeLab --controller stage16_evidence/closeout/release/controller.py --evaluation-mode deferred --phase heldout --mode minimal',
             source_sha256={n:hashlib.sha256((ep/n).read_bytes()).hexdigest() for n in ('result.json','steps.jsonl','agent_trace.jsonl','image_manifest.jsonl','public_responses.jsonl','wall_profile_backend.json','rpc_timing.json')})
raw=[json.loads(line) for line in (ep/'steps.jsonl').open()]
speed=dict(physics_hz=120,max_linear_component_m_s=max(max(abs(x) for x in r['actual_qvel'][:3]) for r in raw),
           max_angular_joint_rad_s=max(max(abs(x) for x in r['actual_qvel'][3:]) for r in raw),
           linear_violating_steps=sum(any(abs(x)>.15 for x in r['actual_qvel'][:3]) for r in raw),
           angular_violating_steps=sum(any(abs(x)>1.5 for x in r['actual_qvel'][3:]) for r in raw))
payload['actual_speed_diagnostic']=speed
old=Path('<LOCAL_DOCS>/fingereye_stage16_accept_final/FingerEyeLab/outputs/stage16_frozen_acceptance_final/episodes/finalA_dn0_seed16111/launch_001/episode')
def commands(path):
 return [{k:r[k] for k in ('target_q','steps','phase')} for r in map(json.loads,(path/'agent_trace.jsonl').read_text().splitlines()) if r.get('op')=='step']
payload['historical_final_A_comparison']=dict(all_commands_equal=commands(ep)==commands(old),
    steps_bytes_equal=(ep/'steps.jsonl').read_bytes()==(old/'steps.jsonl').read_bytes(),
    old_steps_sha256=hashlib.sha256((old/'steps.jsonl').read_bytes()).hexdigest())

out=h/'stage16_evidence/closeout'
(out/'physical_regression.json').write_text(json.dumps(payload,indent=2)+'\n')
with (out/'physical_regression.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
print(json.dumps(row,indent=2))
