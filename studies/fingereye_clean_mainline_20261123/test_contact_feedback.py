"""Mechanical and geometric invariances of shared contact feedback."""
import unittest
from types import SimpleNamespace
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from mainline.contact_intent import ContactIntent
from general_grasp_contacts import task_wrench_margin

class ContactFeedbackTest(unittest.TestCase):
 def test_best_pair_ablation_is_independent_of_which_digit_is_named_third(self):
  points=np.array([[-.01,0,0],[0,.01,0],[.01,0,0]])
  normals=np.array([[-1.,0,0],[0,1.,0],[1.,0,0]])
  common=dict(reference=[0,0,0],mass_range_kg=(.005,.015),friction=.3,
   normal_uncertainty=.05,pad_radius_m=.001,max_normal_force=1.,lateral_force_n=.02,torque_nm=.0001)
  results=[]
  for ids in ([0,1,2],[1,2,0],[2,0,1]):
   result=task_wrench_margin(points[ids],normals[ids],**common)
   self.assertEqual(len(result['same_pose_pair_subsets']),3)
   best=result['without_third_contact']
   self.assertTrue(best['gravity_feasible'])
   self.assertGreaterEqual(result['minimum_task_margin']+1e-7,best['minimum_task_margin'])
   results.append([best['minimum_task_margin'],best['lateral_margin']])
  np.testing.assert_allclose(results,np.tile(results[0],(3,1)),atol=1e-7)

 def test_facet_normal_change_is_visible_with_identical_object_center(self):
  points=trimesh.creation.box([.02,.02,.02]).vertices
  robot=SimpleNamespace(selected_patches=[dict(support_radius_m=.001)])
  old=ContactIntent(robot,[[.0103,0,0]],[[-1.,0,0]],dict(collision_points=points))
  rotation=Rotation.from_euler('z',.22).as_matrix()
  new=ContactIntent(robot,[[.0103,0,0]],[[-1.,0,0]],dict(collision_points=points@rotation.T))
  np.testing.assert_allclose(points.mean(0),(points@rotation.T).mean(0),atol=1e-12)
  angle=np.arccos(np.dot(old.normals[0],new.normals[0]))
  self.assertAlmostEqual(angle,.22,places=8)

 def test_partial_edge_contact_reduces_load_footprint_without_rejection(self):
  robot=SimpleNamespace(selected_patches=[dict(support_radius_m=.001)])
  intent=ContactIntent(robot,[[.0103,0,0]],[[-1.,0,0]],dict(collision_points=trimesh.creation.box([.02,.02,.02]).vertices))
  points=np.array([[.0103,.0099,0.]])
  self.assertLess(intent.errors(points)[0],1e-10)
  self.assertAlmostEqual(intent.support_radii(points,[.001])[0],.0001,places=8)
  report=intent.report(points+[[.001,0,0]])
  self.assertAlmostEqual(report['normal_gap_m'][0],.001,places=8)
  self.assertLess(report['tangential_outside_region_m'][0],1e-10)

if __name__=='__main__':unittest.main()
