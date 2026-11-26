"""Offline native-body FK, real encoder response and rendered RGB admission."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
E=Path(__file__).resolve().parents[3];sys.path.insert(0,str(E))
from mainline.hand_description import HandDescription
sys.path.insert(0,str(E/'cross_evidence'));from audit_recorded_cameras import read as camera_read

def main():
 p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ep=a.session/'episode';desc=json.loads((a.session/'hand_model/description.json').read_text());h=HandDescription(desc['urdf_path'],joint_names=desc['joint_names']);result=json.loads((ep/'result.json').read_text());assert result['controller_completed'] and not result['error'];errors=[]
 for line in (ep/'cross_hand_camera_frames.jsonl').open():
  row=json.loads(line);order={n:i for i,n in enumerate(row['simulator_joint_names'])};fk=h.fk([row['actual_q'][order[n]] for n in h.joint_names])
  for name,pose in zip(row['native_body_names'],row['native_body_poses']):
   if name not in fk or name.startswith('left_palm_floating_'):continue
   T=fk[name];rot=Rotation.from_quat(np.asarray(pose[3:])[[1,2,3,0]]).as_matrix();errors.append([float(np.linalg.norm(T[:3,3]-pose[:3])),float(Rotation.from_matrix(T[:3,:3].T@rot).magnitude())])
 notes=[]
 for line in (ep/'agent_trace.jsonl').open():
  row=json.loads(line)
  if row.get('op')=='note':
   n=json.loads(row['text'])
   if 'admission_motion' in n:notes.append(n)
 note=notes[-1];motion=[dict(index=r['index'],target=r['target'][r['index']],actual=r['actual_q'][r['index']],error=abs(r['target'][r['index']]-r['actual_q'][r['index']])) for r in note['admission_motion']];camera=camera_read(ep);assert camera['passed'];assert max(r[0] for r in errors)<2e-5 and max(r[1] for r in errors)<2e-4;assert all(r['error']<.02 for r in motion);assert len(note['rgb'])==9 and all(r['shape']==[120,160,3] and r['std']>1 for r in note['rgb'].values())
 report=dict(passed=True,actual_closures=0,probe='small independent palm translations, rotations and three sensor-bearing finger flexions away from the object',native_body_FK_comparisons=len(errors),maximum_native_body_position_error_m=max(r[0] for r in errors),maximum_native_body_rotation_error_rad=max(r[1] for r in errors),joint_response=motion,rendered_views=note['rgb'],camera_audit=camera,limitation='declared nominal simulation optics; native pose/K checks, not measured hardware lens calibration')
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='camera_audit'}))
if __name__=='__main__':main()
