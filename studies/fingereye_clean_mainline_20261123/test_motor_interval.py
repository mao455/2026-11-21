import unittest
import numpy as np
from mainline.motor_feedback import interval_feedback,saturated_pd,matched_extra_effort

class MotorIntervalTest(unittest.TestCase):
 def test_held_target_is_not_a_new_command_despite_static_error(self):
  before=dict(actual_q=np.array([.1]),groups={'digit':[0]})
  after=dict(actual_q=np.array([.1]),actual_qvel=np.array([0.]))
  row=interval_feedback(before,after,np.array([0.]))['fingers']['digit']
  self.assertIsNone(row['drive_deficit'])
  self.assertEqual(row['evidence_state'],'information_insufficient_zero_command')
 def test_matching_unit_command_and_measured_increment(self):
  before=dict(actual_q=np.array([0.,0.]),groups={'digit':[0,1]},prismatic_indices=[1])
  after=dict(actual_q=np.array([.005,.001]),actual_qvel=np.zeros(2))
  row=interval_feedback(before,after,np.array([.01,.001]))['fingers']['digit']
  self.assertAlmostEqual(row['unit_groups']['rad']['drive_deficit'],.5)
  self.assertAlmostEqual(row['unit_groups']['m']['drive_deficit'],0.)
  self.assertEqual(row['evidence_state'],'following_command')
 def test_single_blocked_digit_remains_a_proxy(self):
  before=dict(actual_q=np.zeros(2),groups={'a':[0],'b':[1]})
  after=dict(actual_q=np.array([0.,.01]),actual_qvel=np.zeros(2))
  rows=interval_feedback(before,after,np.array([.01,.01]))['fingers']
  self.assertEqual(rows['a']['evidence_state'],'settled_resistance_proxy')
  self.assertEqual(rows['b']['evidence_state'],'following_command')
  self.assertFalse(rows['a']['contact_confirmed'])
 def test_pd_saturates_before_baseline_subtraction(self):
  tau,sat=saturated_pd([.2],[0.],[0.],[100.],[2.],[3.])
  np.testing.assert_allclose(tau,[3.]);self.assertTrue(sat[0])
 def test_unmatched_empty_response_does_not_create_a_force(self):
  cfg=dict(pose_tolerance=.01,speed_tolerance=.01,kp=[100],kd=[2],effort_limit=[3])
  a=dict(q=[.1],qvel=[0.],target=[.2],delta=[.01]);b=dict(q=[0.],qvel=[0.],target=[.2],delta=[.01])
  self.assertFalse(matched_extra_effort(a,b,cfg)['available'])

if __name__=='__main__':unittest.main()
