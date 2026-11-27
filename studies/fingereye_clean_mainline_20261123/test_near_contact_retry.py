"""Near contact refinement preserves accepted poses and strict safety checks."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from mainline import strategy as m

class NearContactRetryTest(unittest.TestCase):
    def setup_case(self,error):
        q=np.array([.1,.2]);meta=dict(per_finger_error_m=[error,error],
            normal_error_rad=[.1,.2],source_clearance=dict(passed=True),q=q.tolist())
        return SimpleNamespace(ik_seed_hints=None,defer_finite_contact=False),q,meta
    def test_existing_success_is_unchanged(self):
        robot,q,meta=self.setup_case(.0001)
        with patch.object(m,'optimize') as solve:
            result=m._s12_retry_near_contact(robot,q,None,None,q,meta)
        self.assertIs(result[0],q);self.assertIs(result[1],meta);solve.assert_not_called()
    def test_collision_and_far_miss_do_not_trigger_local_contact_retry(self):
        for error,clear in ((.003,True),(.0012,False)):
            robot,q,meta=self.setup_case(error);meta['source_clearance']['passed']=clear
            with patch.object(m,'optimize') as solve:m._s12_retry_near_contact(robot,q,None,None,q,meta)
            solve.assert_not_called()
    def test_near_miss_uses_own_pose_and_restores_solver_context(self):
        robot,q,meta=self.setup_case(.0012);expected=q+.01;report=copy.deepcopy(meta);report['per_finger_error_m']=[.0002,.0003]
        def solve(*args):
            np.testing.assert_array_equal(robot.ik_seed_hints[0],q)
            self.assertFalse(robot.defer_finite_contact)
            return expected,report
        with patch.object(m,'optimize',side_effect=solve):out,detail=m._s12_retry_near_contact(robot,q,None,None,q,meta)
        np.testing.assert_array_equal(out,expected);self.assertTrue(detail['near_contact_retry']['accepted'])
        self.assertIsNone(robot.ik_seed_hints);self.assertFalse(robot.defer_finite_contact)
    def test_retry_does_not_accept_unchanged_bad_error(self):
        robot,q,meta=self.setup_case(.0012)
        with patch.object(m,'optimize',return_value=(q,copy.deepcopy(meta))):out,detail=m._s12_retry_near_contact(robot,q,None,None,q,meta)
        self.assertIs(out,q);self.assertFalse(detail['near_contact_retry']['accepted'])

if __name__=='__main__':unittest.main()
