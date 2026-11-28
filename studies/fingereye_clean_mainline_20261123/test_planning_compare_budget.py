"""Budget cancellation may restore complete plans, never partial or mutable work."""
import unittest,json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from mainline import strategy as m


def candidate(verified=True):
    return dict(q=np.zeros(8),points=np.zeros((2,3)),inward=np.array([[1.,0.,0.],[-1.,0.,0.]]),prefix=[],
        meta=dict(feasibility_passed=verified,task_quality={},contact_fingers=['a','b'],
                  hand_contact_regions=[0,0],targets=np.zeros((2,3)).tolist(),candidate_id='verified'))


def guard():
    return dict(active=True,choices={},first_verified_cpu_s=None,deadline_cpu_s=float('inf'),rejected=[])


class PlanningCompareBudget(unittest.TestCase):
    def test_unfinished_candidate_cannot_start_or_spend_verified_window(self):
        g=guard();robot=SimpleNamespace(_planning_comparison=g)
        with patch.object(m.time,'process_time',return_value=10000.):
            m._remember_verified_candidate(robot,candidate(False))
            m.planning_progress(robot,'unfinished')
        self.assertEqual(g['choices'],{});self.assertIsNone(g['first_verified_cpu_s'])

    def test_window_begins_when_first_complete_plan_arrives(self):
        g=guard();robot=SimpleNamespace(_planning_comparison=g)
        with patch.object(m.time,'process_time',return_value=400.):m._remember_verified_candidate(robot,candidate())
        self.assertEqual(g['first_verified_cpu_s'],400.)
        self.assertEqual(g['deadline_cpu_s'],520.)
        with patch.object(m.time,'process_time',return_value=521.):
            with self.assertRaises(m._PlanningComparisonComplete):m.planning_progress(robot,'next_unfinished_IK')

    def test_cancel_restores_immutable_complete_snapshot_and_clears_control_guard(self):
        clock=[100.];notes=[];selected=[]
        robot=SimpleNamespace(select_pair=lambda *args:selected.append(args))
        ctx=SimpleNamespace(note=lambda value:notes.append(json.loads(value)))
        c=candidate()
        def search(s,robot,*args,**kwargs):
            m._remember_verified_candidate(robot,c)
            c['q'][0]=99. # Unfinished subsequent work must not modify the saved plan.
            clock[0]=221.
            m.planning_progress(robot,'subsequent_search')
        with patch.object(m.time,'process_time',side_effect=lambda:clock[0]), \
             patch.object(m,'_rank_contact_candidates_search',side_effect=search), \
             patch.object(m,'s12_order_candidates',side_effect=lambda rows:rows), \
             patch.object(m,'apply_contact_geometry'):
            rows,rejected=m.rank_contact_candidates(dict(ctx=ctx),robot,None,None,None,None,[],[],0.,'fixture')
        self.assertEqual(rows[0]['q'][0],0.)
        self.assertEqual(rejected,[]);self.assertEqual(selected, [(['a','b'],[0,0])])
        self.assertIsNone(robot._planning_comparison)
        self.assertTrue(notes[-1]['planning_comparison_budget']['restored_only_fully_verified_snapshots'])
        with patch.object(m.time,'process_time',return_value=10000.):m.planning_progress(robot,'physical_control')


    def test_local_timeout_has_explicit_unknown_optimality_and_never_claims_convergence(self):
        # A real residual followed by a deterministic expired clock exercises
        # the same synthetic result that interrupted the recovery surface IK.
        with patch.object(m.time, 'process_time', side_effect=[0., 19.]):
            fit = m.least_squares(lambda x: x-1., np.array([0.]))
        self.assertIsNone(fit.optimality)
        self.assertEqual(fit.status, 0)
        self.assertFalse(fit.success)
        self.assertTrue(np.isfinite(fit.x).all())
        self.assertEqual(fit.x.tolist(), [0.])

    def test_candidate_timeout_preserves_unknown_optimality(self):
        robot = SimpleNamespace(ik_call_deadline=1.)
        with patch.object(m.time, 'process_time', side_effect=[0., 0., 2.]):
            fit = m._slice_least_squares(robot, lambda x: x-1., np.array([0.]))
        self.assertIsNone(fit.optimality)
        self.assertFalse(fit.success)
        self.assertEqual(fit.status, 0)


if __name__=='__main__':unittest.main()
