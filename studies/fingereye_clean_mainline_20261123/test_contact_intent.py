"""Metamorphic geometry and finite-facet intent checks, independent of IK."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import trimesh
from cross_hand_runtime import pad_catalog
from mainline.contact_intent import ContactIntent

class ContactIntentTest(unittest.TestCase):
    def test_rigid_split_does_not_remove_support(self):
        with tempfile.TemporaryDirectory() as temp:
            catalogs=[]
            for split in (False,True):
                meshes=[];joints=[];poses=dict(root=np.eye(4),tip=np.eye(4))
                joints.append(('root','tip',np.eye(4),np.array([0.,0.,1.]),'revolute',0))
                for i,(extent,position) in enumerate(([.03,.014,.01],[.015,0.,0.]) for _ in range(1)) if not split else enumerate([([.0285,.014,.01],[.01425,0.,0.]),([.0015,.014,.01],[.02925,0.,0.])]):
                    name='tip' if i==0 else 'fixed_piece'
                    mesh=trimesh.creation.box(extent);mesh.apply_translation(position)
                    path=Path(temp)/f'{split}_{i}.stl';mesh.export(path)
                    meshes.append(dict(link=name,path=str(path),scale=np.ones(3),origin=np.eye(4)))
                    if i:joints.append(('tip',name,np.eye(4),np.zeros(3),'fixed',None));poses[name]=np.eye(4)
                hand=SimpleNamespace(joint_names=('curl',),lower=np.array([-1.]),upper=np.array([1.]),joints=joints,fk=lambda q:poses,collision_meshes=meshes)
                chain=dict(finger='curl',terminal_link='tip',joint_names=['curl'])
                catalogs.append(pad_catalog(hand,[chain],'synthetic_split_fixture')[0]['regions'])
            a,b=[np.array([r['point']for r in regions])for regions in catalogs]
            self.assertGreater(len(a),0);self.assertEqual(len(a),len(b))
            distance=np.linalg.norm(a[:,None]-b[None,:],axis=2)
            self.assertLess(max(distance.min(0).max(),distance.min(1).max()),1e-7)
            self.assertTrue(any(len(r['surface_links'])>1 for r in catalogs[1]))

    def test_tangent_shift_on_facet_is_not_a_miss(self):
        mesh=trimesh.creation.box([.02,.02,.02]);points=mesh.vertices
        robot=SimpleNamespace(selected_patches=[dict(support_radius_m=.001)])
        intent=ContactIntent(robot,[[.0103,0.,0.]],[[-1.,0.,0.]],dict(collision_points=points))
        self.assertLess(np.linalg.norm(intent.residual([[.0103,.005,0.]])),1e-9)
        self.assertLess(np.linalg.norm(intent.residual([[.0103,.005,.005]])),1e-9)
        self.assertAlmostEqual(np.linalg.norm(intent.residual([[.0113,.005,0.]])),1.5)
        self.assertAlmostEqual(np.linalg.norm(intent.residual([[.0103,.012,0.]])),3.)
        # A 1mm source footprint need not fit wholly inside the object facet.
        self.assertLess(np.linalg.norm(intent.residual([[.0103,.0099,0.]])),1e-9)

if __name__=='__main__':unittest.main()
