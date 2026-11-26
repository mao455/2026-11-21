"""Offline native-contact readout. Never imported by the controller."""
from pathlib import Path
from collections import defaultdict, deque
import argparse, json
import numpy as np

def audit(session):
    episode=session/'episode'
    result=json.loads((episode/'result.json').read_text())
    physics=result['physics']
    hold=deque(maxlen=240)
    valid=True;steps=0;previous=None;time_errors=0
    for line in (episode/'steps.jsonl').open():
        row=json.loads(line);steps+=1
        valid &= bool(row['penetration']['passed'])
        if previous is not None and abs(row['time_s']-previous-1/120)>1e-8:time_errors+=1
        previous=row['time_s'];hold.append(row)
    forces=defaultdict(list);vectors=defaultdict(list)
    for row in hold:
        for link,contacts in row['contacts'].items():
            force=float(contacts['normal_force_sum_n'])
            forces[link].append(force)
            normals=np.asarray(contacts['normals_world'],float)
            weights=np.asarray(contacts['normal_forces_n'],float)
            if len(weights):vectors[link].append((weights[:,None]*normals).sum(0))
    contacts=[]
    for link,values in forces.items():
        if max(values,default=0)<=0:continue
        contacts.append(dict(link=link,positive_hold_steps=sum(v>0 for v in values),minimum_normal_force_n=min(values),mean_normal_force_n=float(np.mean(values)),maximum_normal_force_n=max(values),mean_weighted_normal_vector=np.mean(vectors[link],axis=0).tolist()))
    return dict(session=str(session),success=result['success'],geometry_valid_every_step=valid,physics_steps=steps,physics_time_increment_errors=time_errors,hold_steps=len(hold),hold_contacts=contacts,hold_max_abs_joint_velocity=float(np.max(np.abs([r['actual_qvel']for r in hold]))),hold_min_lift_m=physics.get('hold_min_lift_m'),online_use=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--sessions',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[audit(s)for s in a.sessions];a.output.write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
if __name__=='__main__':main()
