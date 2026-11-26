"""Independent topology, order and optical-contract regressions, no simulator."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mainline.hand_description import HandDescription
from mainline import strategy
class SharedHandTests(unittest.TestCase):
 def test_closure_observation_uses_object_views_independent_of_camera_owner(self):
  evidence=dict(wrist_fixed=True,views={'view_a':dict(finger='palm',object_identity_association=True),'view_b':dict(finger='unused_digit',object_identity_association=True)})
  first=strategy.closure_object_observation(evidence)
  self.assertTrue(first['associated']);self.assertFalse(first['contact_observable'])
  evidence['views']['view_a']['finger']='grasping_digit'
  self.assertEqual(first,strategy.closure_object_observation(evidence))
  evidence['views']['view_b'].update(registered=True,motion_evidence=True,confidence=.7)
  self.assertTrue(strategy.closure_object_observation(evidence)['registered_motion_veto'])
 def test_closure_observation_holds_for_missing_views_or_moving_wrist(self):
  for evidence in [dict(wrist_fixed=True,views={}),dict(wrist_fixed=True,views={'a':dict(object_identity_association=True)}),dict(wrist_fixed=False,views={'a':dict(object_identity_association=True),'b':dict(object_identity_association=True)})]:
   self.assertFalse(strategy.closure_object_observation(evidence)['associated'])
 def test_moving_finger_camera_preserves_identity_but_not_stationary_probe(self):
  from copy import deepcopy
  mask=np.ones((8,8),dtype=bool)
  view=dict(status='ok',visible=True,size=(8,8),camera=np.eye(4),mask=mask,centroid_px=[3.5,3.5],area_px=64)
  before=dict(step=0,views={'a':deepcopy(view),'b':deepcopy(view)})
  after=deepcopy(before);after['step']=120
  for row in after['views'].values():row['camera'][0,3]=.00035
  rows={n:dict(finger='digit',registered=True,motion_evidence=False,confidence=.8) for n in before['views']}
  with patch.object(strategy,'_S15_COMPARE_PRELOAD',return_value=dict(wrist_fixed=True,views=rows)):
   report=strategy.compare_preload(before,after)
  self.assertTrue(report['object_observation']['associated'])
  self.assertFalse(report['bounded_probe_association'])
  rows['a']['motion_evidence']=True
  self.assertTrue(strategy.closure_object_observation(report)['registered_motion_veto'])
  after['views']['a']['visible']=False;after['views']['b']['mask']=np.zeros((8,8),dtype=bool)
  with patch.object(strategy,'_S15_COMPARE_PRELOAD',return_value=dict(wrist_fixed=True,views={n:dict(finger='digit') for n in rows})):
   self.assertFalse(strategy.compare_preload(before,after)['object_observation']['associated'])
 def test_recovery_diagnostics_accept_tuple_joint_groups(self):
  from types import SimpleNamespace
  s=dict(robot=SimpleNamespace(finger_names=['a','b']),obs=dict(actual_q=[0,0,0],actual_qvel=[0,0,0],step=12),last=np.zeros(3),squeeze=np.zeros(2))
  with patch.object(strategy,'s5_static',return_value=dict(finger_joint_indices={'a':(0,1),'b':(2,)})):
   proposal,reason=strategy.s13_preload_proposal(s,{})
  self.assertIsNone(proposal);self.assertTrue(reason['settled'])
 def test_refinement_reuses_hand_bvhs_and_rebuilds_object_envelope(self):
  import trimesh
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'hand.urdf';path.write_text('<robot name="toy"><link name="a"><inertial/></link><link name="b"><inertial/></link></robot>')
   mesh=trimesh.creation.box(extents=[.01,.01,.01])
   robot=SimpleNamespace(urdf_path=path,description=SimpleNamespace(structural_collision_exclusions=lambda _:set()),surfaces={'a':mesh,'b':mesh.copy()},meshes=[('a',mesh.vertices),('b',mesh.vertices)])
   info=dict(table_z=0.,axes=np.eye(2),center=np.array([0.,0.,.01]),half=np.array([.01,.01,.01]))
   first=strategy.SelfAwareGeometry(robot,info)
   second=strategy.SelfAwareGeometry(robot,dict(info,center=np.array([.01,0.,.01])))
   self.assertIs(first.models,second.models)
   for name in first.models:self.assertIs(first.models[name],second.models[name])
   self.assertIsNot(first.object,second.object)
   np.testing.assert_allclose(second.object_bounds-first.object_bounds,[[.01,0,0],[.01,0,0]])
 def test_static_optical_frame_is_part_of_the_shared_fk(self):
  import xml.etree.ElementTree as ET
  from cross_hand_runtime import install_optical_frames
  from scipy.spatial.transform import Rotation
  root=ET.fromstring('<robot name="camera"><link name="palm"/><link name="CAM"/><joint name="mount" type="fixed"><parent link="palm"/><child link="CAM"/><origin xyz=".03 .02 .01" rpy=".4 .2 -.3"/></joint></robot>')
  mount=dict(joint_name='mount',camera_link='CAM',parent_link='palm',translation=[.03,.02,.01],rpy=[.4,.2,-.3])
  install_optical_frames(root,[mount],dict(camera_optical_frames={'*':dict(translation_m=[0,0,.0002],rpy_rad=[np.pi,0,0],source='static test')}))
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'hand.urdf';ET.ElementTree(root).write(p)
   fk=HandDescription(p).fk([])
  optical=fk[mount['camera_link']];mechanical=fk['CAM']
  np.testing.assert_allclose(optical[:3,3],mechanical[:3,3]+mechanical[:3,2]*.0002,atol=1e-14)
  np.testing.assert_allclose(-optical[:3,2],mechanical[:3,2],atol=1e-14)
  np.testing.assert_allclose(optical[:3,:3],Rotation.from_euler('xyz',mount['rpy']).as_matrix(),atol=1e-14)
  np.testing.assert_allclose(optical[:3,3],mount['translation'],atol=1e-14)
 def test_calibrated_identity_camera_keeps_original_mount(self):
  import xml.etree.ElementTree as ET
  from cross_hand_runtime import install_optical_frames
  root=ET.fromstring('<robot name="camera"><link name="camera"/></robot>')
  mount=dict(joint_name='mount',camera_link='camera',translation=[0,0,0],rpy=[0,0,0])
  original=ET.tostring(root);install_optical_frames(root,[mount],{})
  self.assertEqual(ET.tostring(root),original);self.assertEqual(mount['camera_link'],'camera')
 def test_structural_exclusions_keep_distant_and_other_finger_pairs(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'hand.urdf';links=('root','mount','a','a_fixed','tip','other')
   rows=[('fixed','root','mount'),('revolute','mount','a'),('fixed','a','a_fixed'),('revolute','a_fixed','tip'),('revolute','mount','other')]
   p.write_text('<robot name="toy">'+''.join(f'<link name="{x}"/>' for x in links)+''.join(f'<joint name="j{i}" type="{kind}"><parent link="{pa}"/><child link="{ch}"/><axis xyz="0 0 1"/><origin xyz="0 0 .01"/><limit lower="-1" upper="1" effort="2" velocity="1"/></joint>' for i,(kind,pa,ch) in enumerate(rows))+'</robot>')
   m=HandDescription(p,joint_names=['j4','j1','j3']);ex=m.structural_collision_exclusions()
   self.assertIn(tuple(sorted(('root','a_fixed'))),ex)
   self.assertIn(tuple(sorted(('mount','a'))),ex)
   self.assertNotIn(tuple(sorted(('root','tip'))),ex)
   self.assertNotIn(tuple(sorted(('a','other'))),ex)
   self.assertNotIn(tuple(sorted(('tip','other'))),ex)
   fk=m.fk([.3,.2,.1]);other=HandDescription(p);q=[.2,.1,.3]
   for n,T in other.fk(q).items():np.testing.assert_allclose(fk[n],T,atol=1e-14)
 def test_support_pair_accepts_arbitrary_view_names_without_global_calibration(self):
  names=('palm_view_A','digit_view_B');before=dict(step=0,palm=np.eye(4),selected_fingers=['digit_a','digit_b'],depth_bounds_m=[.01,.2],views={n:dict(visible=False,finger=f,status='missing_rgb') for n,f in zip(names,['digit_a','digit_b'])})
  after=dict(before,step=6,palm=np.eye(4));after['palm'][2,3]=.003
  with patch.object(strategy,'STAGE3_SUPPORT_CAMERAS',names):
   report=strategy.stage3_support_pair(before,after)
  self.assertEqual(report['state'],'unknown')
 def test_reported_K_matches_spawn_focal_and_aperture(self):
  path=Path(__file__).resolve().parents[2]/'source/fingereye/fingereye/tasks/direct/fingereye/xhand_camera_config.py'
  spec=importlib.util.spec_from_file_location('_test_cross_camera_config',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  cfg=module.FINGEREYE_CAMERA_CFG
  for width,height in [(160,120),(320,240)]:
   k=np.asarray(module.renderer_intrinsic_matrix(width,height));self.assertAlmostEqual(k[0,0],width*cfg.focal_length/cfg.horizontal_aperture);self.assertEqual(k[1,1],k[0,0]);self.assertEqual(k[0,2],width/2)
if __name__=='__main__':unittest.main()
