"""Private all-hand contact audit. Never exposed in the policy observation."""
import json
from pathlib import Path
import numpy as np


def array(value):
    return value.detach().cpu().numpy().copy()


class NativeSelfContacts:
    def __init__(self, env, out):
        names = list(env.hand.body_names)
        paths = ['/World/envs/env_0/Robot/' + n for n in names]
        self.view = env.hand._physics_sim_view.create_rigid_contact_view(
            paths, [paths] * len(paths), max_contact_data_count=16384)
        manifest = json.loads((Path(__file__).parent/'assets/round7b_xhand_selfcollisions.json').read_text())
        self.excluded = {tuple(sorted((p['body_a'],p['body_b']))) for p in manifest['excluded_pairs']}
        self.names = [Path(p).name for p in self.view.sensor_paths]
        assert self.names == names, (self.names, names)
        assert self.view.filter_count == len(names)
        self.stream = (out/'native_self_contacts.jsonl').open('w',buffering=1)
        (out/'native_self_contact_manifest.json').write_text(json.dumps({
            'sensor_paths':self.view.sensor_paths, 'filter_paths':self.view.filter_paths,
            'sensor_count':self.view.sensor_count,'filter_count':self.view.filter_count,
            'private_diagnostic_only':True},indent=2)+'\n')

    def record(self, dt, step, phase):
        if step >= 12 and step % 12:
            return
        forces, points, normals, separations, counts, starts = [array(x) for x in self.view.get_contact_data(dt)]
        pairs=[]
        for i in range(len(self.names)):
            for j in range(len(self.names)):
                if i == j:continue
                n=int(counts[i,j]);start=int(starts[i,j])
                if not n:continue
                sl=slice(start,start+n)
                a,b=self.names[i],self.names[j]
                pairs.append({'a':a,'b':b,'directed_query':True,'excluded_in_usd':tuple(sorted((a,b))) in self.excluded,
                              'count':n,'normal_force_sum_n':float(forces[sl].sum()),
                              'min_separation_m':float(separations[sl].min()),
                              'points_world':points[sl].tolist(),'normals_world':normals[sl].tolist()})
        self.stream.write(json.dumps({'step':step,'phase':phase,'pairs':pairs})+'\n')
