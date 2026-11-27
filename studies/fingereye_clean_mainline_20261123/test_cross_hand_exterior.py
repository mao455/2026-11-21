"""Fixed assembly exterior, native motion reserve and active view witnesses."""
import ast
import json
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import fcl
import numpy as np
import trimesh
from cross_hand_runtime import pad_catalog

SOURCE=Path(__file__).parent/'mainline/strategy.py'
def function(name,classname=None):
    tree=ast.parse(SOURCE.read_text());scope=tree.body
    if classname:scope=next(n for n in scope if isinstance(n,ast.ClassDef)and n.name==classname).body
    node=next(n for n in scope if isinstance(n,ast.FunctionDef)and n.name==name)
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns[name]

class ExteriorTest(unittest.TestCase):
    def test_fixed_shell_is_first_contact(self):
        with tempfile.TemporaryDirectory()as temp:
            transform=np.eye(4);transform[1,3]=.007
            hand=SimpleNamespace(joint_names=('curl',),lower=np.array([-1.]),upper=np.array([1.]),
                joints=[('tip','sensor_shell',transform,np.zeros(3),'fixed',None)],
                fk=lambda q:dict(tip=np.eye(4),sensor_shell=transform),collision_meshes=[])
            meshes=[]
            for link,extent in [('tip',[.03,.008,.01]),('sensor_shell',[.024,.01,.012])]:
                mesh=trimesh.creation.box(extent);path=Path(temp)/(link+'.stl');mesh.export(path)
                hand.collision_meshes.append(dict(link=link,path=str(path),scale=np.ones(3),origin=np.eye(4)))
                mesh.apply_transform(hand.fk([0])[link]);meshes.append(mesh)
            whole=trimesh.util.concatenate(meshes)
            chain=dict(finger='curl',terminal_link='tip',joint_names=['curl'],links=['tip','sensor_shell'])
            regions=pad_catalog(hand,[chain],'synthetic_unregistered')[0]['regions']
            self.assertTrue(any(r['surface_link']=='sensor_shell'for r in regions))
            for region in regions:
                point=np.array(region['point']);normal=np.array(region['normal']);start=point+normal*.04
                hits,_,_=whole.ray.intersects_location([start],[-normal],multiple_hits=True)
                first=hits[np.argmin(np.linalg.norm(hits-start,axis=1))]
                np.testing.assert_allclose(first,point,atol=1e-7)
            calibration=Path(temp)/'hand_calibrations';calibration.mkdir()
            rows=pad_catalog(hand,[chain],'synthetic_unregistered')
            (calibration/'fixture.json').write_text(json.dumps(dict(pads=rows)))
            with patch('cross_hand_runtime.HERE',Path(temp)):
                frozen=pad_catalog(hand,[chain],'fixture')
            self.assertEqual(len(frozen[0]['regions']),len(regions))

    def test_clearance_covers_contact_offsets(self):
        mesh=trimesh.creation.box([.01]*3);model=fcl.BVHModel();model.beginModel(len(mesh.vertices),len(mesh.faces));model.addSubModel(mesh.vertices,mesh.faces.astype(np.int32));model.endModel()
        def poses(q):
            b=np.eye(4);b[0,3]=.01+float(q[0]);return dict(a=np.eye(4),b=b)
        checker=SimpleNamespace(robot=SimpleNamespace(fk=poses),names=['a','b'],pairs=np.array([[0,1]]),hulls=dict(a=mesh.vertices,b=mesh.vertices),models=dict(a=model,b=model),fcl=fcl,check=lambda q,object_check:dict(passed=True))
        check=function('motion_check','SelfAwareGeometry');refused=check(checker,np.array([.0006]))
        self.assertFalse(refused['passed']);self.assertAlmostEqual(refused['motion_clearance_failures'][0]['clearance_m'],.0006)
        self.assertTrue(check(checker,np.array([.0012]))['passed'])
        self.assertTrue(check(checker,np.array([.000799928]))['passed'])
    def test_active_view_requires_independent_baselines(self):
        classify=function('temporal_grounded_support')
        def report(xy,step=10,label='grounded_evidence',before=0):
            return dict(before_step=before,step=step,measured_offset_m=[*xy,0.],views=[dict(camera='one_lens',classification=label)])
        first=report((.016,.016));self.assertTrue(classify([first,report((.016,.024),20)]))
        for rows in [[first],[first,report((.024,.024),20)],[first,report((.016,.024),20,'carrying_evidence')],[first,report((.016,.024),20,before=5)]]:self.assertFalse(classify(rows))
if __name__=='__main__':unittest.main()
