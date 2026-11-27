"""Contact count alone never earns mechanical benefit in shared ranking."""
import copy
import unittest
import numpy as np
from mainline.contact_regions import sort_task_candidates

class CandidateReserveTest(unittest.TestCase):
    def candidate(self, count, reserve=.0025, torque=0.):
        model=dict(minimum_task_margin=0.,gravity_feasible=False,lateral_margin=0.,torque_margin=torque,outward_normals=[[1,0,0],[-1,0,0]] if count==2 else [[1,0,0],[-1,0,0],[0,1,0]])
        return dict(points=np.zeros((count,3)),meta=dict(task_quality=dict(nominal=copy.deepcopy(model),robust=copy.deepcopy(model)),
           lift_tail_audits=dict(servo=dict(achievable_squeeze_m=reserve)),
           contact_regions=[dict(position_uncertainty_m=.0015)]*count,
           quality_terms=dict(planned_motion_s=10. if count==3 else 11.)))
    def test_no_load_gain_does_not_buy_third_finger_with_shorter_path(self):
        pair=self.candidate(2);triad=self.candidate(3)
        self.assertIs(sort_task_candidates([triad,pair])[0],pair)
    def test_real_antitorque_benefit_can_select_third_finger(self):
        pair=self.candidate(2);triad=self.candidate(3,torque=1.2)
        self.assertIs(sort_task_candidates([pair,triad])[0],triad)
    def test_closed_loop_motion_reserve_breaks_equal_load_tie(self):
        insufficient=self.candidate(2,reserve=.00025);sufficient=self.candidate(2)
        insufficient['meta']['quality_terms']['planned_motion_s']=9.
        self.assertIs(sort_task_candidates([insufficient,sufficient])[0],sufficient)

if __name__=='__main__':unittest.main()
