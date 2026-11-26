"""Compare recorded renderer cameras against FK at every RGB exposure.

Offline diagnostics only; reads original source records, never launches a
simulator or exposes native camera/body state to the online controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

EXPERIMENT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(EXPERIMENT))
from mainline.hand_description import HandDescription


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(ep):
    desc=json.loads((ep.parent/'hand_model/description.json').read_text())
    hand=HandDescription(desc['urdf_path'],joint_names=desc['joint_names'],
                         base_joint_names=desc['base_joint_names'])
    manifest={f['step']:f for f in map(json.loads,(ep/'image_manifest.jsonl').open())}
    phase_by_step={r['step']+1:r['phase'] for r in map(json.loads,(ep/'steps.jsonl').open())}
    optics=json.loads((ep/'cross_hand_camera_optics.json').read_text())
    all_rows=[];phases={};exposure_q_error=0.;missing=[]
    for frame in map(json.loads,(ep/'cross_hand_camera_frames.jsonl').open()):
        step=frame['step'];order={name:i for i,name in enumerate(frame['simulator_joint_names'])}
        q=np.asarray([frame['actual_q'][order[name]] for name in hand.joint_names])
        poses=hand.fk(q);image=manifest.get(step)
        if image is None:missing.append(step)
        else:exposure_q_error=max(exposure_q_error,float(np.max(abs(q-np.asarray(image['actual_q'])))))
        phase=phase_by_step.get(step,'initial')
        for camera in frame['cameras']:
            name=camera['name'];cal=camera['calibration'];wanted=poses[cal['camera_link']]
            native_rot=Rotation.from_quat(np.asarray(camera['quaternion_opengl'])[[1,2,3,0]])
            position_error=float(np.linalg.norm(wanted[:3,3]-camera['position']))
            rotation_error=float((native_rot.inv()*Rotation.from_matrix(wanted[:3,:3])).magnitude())
            k_error=float(np.max(abs(np.asarray(camera['intrinsic_matrix'])-cal['renderer_intrinsic_matrix'])))
            row=dict(step=step,phase=phase,camera=name,position_error_m=position_error,
                     rotation_error_rad=rotation_error,renderer_k_error_px=k_error)
            all_rows.append(row)
            phase_data=phases.setdefault(phase,dict(exposures=set(),camera_comparisons=0,
                max_position_error_m=0.,max_rotation_error_rad=0.,max_renderer_k_error_px=0.))
            phase_data['exposures'].add(step);phase_data['camera_comparisons']+=1
            phase_data['max_position_error_m']=max(phase_data['max_position_error_m'],position_error)
            phase_data['max_rotation_error_rad']=max(phase_data['max_rotation_error_rad'],rotation_error)
            phase_data['max_renderer_k_error_px']=max(phase_data['max_renderer_k_error_px'],k_error)
    for p in phases.values():
        p['first_exposure_step']=min(p['exposures']);p['last_exposure_step']=max(p['exposures'])
        p['exposures']=len(p['exposures'])
    worst={key:max(all_rows,key=lambda r:r[key]) for key in
           ('position_error_m','rotation_error_rad','renderer_k_error_px')}
    max_pos=worst['position_error_m']['position_error_m'];max_rot=worst['rotation_error_rad']['rotation_error_rad']
    max_k=worst['renderer_k_error_px']['renderer_k_error_px']
    native_optics_pass=bool(optics['passed'] and all(row['passed'] and all(c['passed'] for c in row['checks']) for row in optics['cameras']))
    passed=bool(not missing and exposure_q_error<=1e-9 and max_pos<=2e-5 and max_rot<=2e-4 and max_k<=1e-5 and native_optics_pass)
    return dict(episode=str(ep),passed=passed,
        exposures=len(manifest),camera_comparisons=len(all_rows),camera_names=[c['name'] for c in optics['cameras']],
        max_position_error_m=max_pos,max_rotation_error_rad=max_rot,max_renderer_k_error_px=max_k,
        max_exposure_joint_difference_rad_or_m=exposure_q_error,missing_exposure_steps=missing,
        native_usd_optics_passed=native_optics_pass,native_optics_checks=sum(len(c['checks']) for c in optics['cameras']),
        tolerances=dict(position_m=2e-5,rotation_rad=2e-4,renderer_k_px=1e-5,exposure_q=1e-9),
        worst=worst,by_phase=phases,
        source_sha256={str(path):sha(path) for path in (ep.parent/'hand_model/description.json',
            Path(desc['urdf_path']),ep/'cross_hand_camera_frames.jsonl',ep/'cross_hand_camera_optics.json',ep/'image_manifest.jsonl')},
        limitation='Validates declared camera transforms, exposure synchronization and actual USD optical parameters; not hardware optical calibration or an independent rendered target pixel calibration.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[read(ep.resolve()) for ep in a.episode]
    result=dict(schema='fingereye.recorded_camera_audit.v1',passed=all(r['passed'] for r in rows),episodes=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],episodes=[{k:r[k] for k in
        ('episode','exposures','camera_comparisons','max_position_error_m','max_rotation_error_rad','max_renderer_k_error_px','max_exposure_joint_difference_rad_or_m','native_usd_optics_passed')} for r in rows])))


if __name__=='__main__':main()
