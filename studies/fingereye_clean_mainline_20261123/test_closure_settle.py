"""A transient motor hold must be re-observed before closure can finish."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from mainline import strategy as m

class ClosureSettleTest(unittest.TestCase):
    def run_servo(self, resume=True, motion_veto=False, tracking_error=None, deficit=0., cap=.0011, support_radii=(.0012,.0012)):
        points=np.array([[.01,0,.02],[-.01,0,.02]])
        robot=SimpleNamespace(finger_names=['a','b'], selected_patches=[dict(support_radius_m=r) for r in support_radii],
                              tips=lambda q:(points,np.ones_like(points),None,None))
        s=dict(c=dict(meta={},q=np.zeros(6),points=points,inward=np.zeros_like(points)),
               robot=robot,ctx=None,volume=SimpleNamespace(reset_fusion=lambda:None),
               last=np.zeros(6),obs=dict(step=0,actual_q=np.zeros(6),actual_qvel=np.zeros(6),image_state=dict(step=0)),attempt=1,checker=None)
        if tracking_error is not None:s['obs']['actual_q'][:2]=-tracking_error
        actions=[];responses=0
        def note(ctx,key,value):
            if key=='s5_closure_action':actions.append(value)
        def response(*args,**kwargs):
            nonlocal responses
            responses+=1
            settled=bool(resume and responses>=2)
            return dict(motor=dict(fingers={name:dict(settled=settled,drive_deficit=deficit,joint_indices=[] if tracking_error is None else [j])for j,name in enumerate(['a','b'])}),moving_fingers=[])
        def step(state,target,*args,**kwargs):state['last']=np.asarray(target).copy()
        functions=dict(apply_contact_geometry=lambda *a:None,s5_note=note,s9_frame=lambda a:0,
          _s5_initial_close_motion=lambda *a:([],.0005,dict(passed=True)),
          s5_move=lambda *a,**k:None,s5_step=step,
          _s5_stationary_closure_evidence=lambda s,e:(e,False),
          capture_preload_snapshot=lambda *a:dict(actual_q=np.zeros(6)),s11_motor_snapshot=lambda *a:dict(actual_q=np.zeros(6)),
          s5_static=lambda *a:None,_s5_close_reference=lambda *a:np.zeros(6),close_ik=lambda r,q,p:q,
          _s5_incremental_close_plan=lambda r,ch,actual,last,target,**k:([(target,1,'close')],dict(passed=True)),
          compare_preload=response,_s2v_motor=lambda *a:response()['motor'],
          _s5_closure_motion_veto=lambda e:motion_veto,_s5_record_closure_reference=lambda *a:None)
        with patch.multiple(m,**functions):m.s5_close(s,mode='servo',cap=cap)
        return actions
    def test_motor_settle_pause_is_followed_by_new_decision_and_resume(self):
        actions=self.run_servo()
        self.assertGreaterEqual(len(actions),3)
        self.assertEqual(actions[1]['squeeze_m'],actions[0]['squeeze_m'])
        self.assertGreater(actions[2]['squeeze_m'][0],actions[1]['squeeze_m'][0])
    def test_continuously_unsettled_response_obeys_finite_budget(self):
        self.assertEqual(len(self.run_servo(resume=False)),15)
    def test_registered_object_motion_still_ends_probe(self):
        self.assertEqual(len(self.run_servo(motion_veto=True)),2)

    def test_stationary_zero_command_has_velocity_evidence_not_force(self):
        snapshot=dict(actual_q=np.zeros(2),actual_qvel=np.zeros(2),groups={'a':[0,1]})
        row=m._s2v_motor(snapshot,snapshot,np.zeros(2))['fingers']['a']
        self.assertTrue(row['settled'])
        self.assertIsNone(row['drive_deficit'])
    def test_settle_checks_entire_digit_including_undriven_joint(self):
        before=dict(actual_q=np.zeros(2),groups={'a':[0,1]})
        after=dict(actual_q=np.array([.01,0]),actual_qvel=np.array([0.,.1]))
        self.assertFalse(m._s2v_motor(before,after,np.array([.01,0.]))['fingers']['a']['settled'])
    def test_unchanged_rgb_preserves_checked_remaining_route(self):
        candidate=object();robot=object();refined=object();checker=object()
        state=dict(robot=robot,c=candidate,ctx=None)
        with patch.multiple(m,s5_stationary_rgb=lambda *a:{},s5_note=lambda *a:None,
            s5_rgb_envelope=lambda *a:(refined,0.),
            _s5_selected_surface_change=lambda *a:dict(material=False),
            SelfAwareGeometry=lambda *a:checker,
            s5_full_rgb_replan=lambda *a:self.fail('unchanged RGB replaced verified route')):
            self.assertEqual(m.s5_refine(state)['status'],'unchanged')
        self.assertIs(state['c'],candidate)
        self.assertIs(state['checker'],checker)

    def test_tracked_unsettled_digit_can_continue_a_small_bounded_probe(self):
        actions=self.run_servo(resume=False,tracking_error=.006)
        self.assertGreater(actions[1]['squeeze_m'][0],actions[0]['squeeze_m'][0])
        self.assertLessEqual(actions[1]['squeeze_m'][0]-actions[0]['squeeze_m'][0],.00025+1e-10)
        self.assertLessEqual(max(actions[-1]['squeeze_m']),.0011)
    def test_unsettled_probe_has_finite_total_not_only_finite_steps(self):
        actions=self.run_servo(resume=False,tracking_error=.006,cap=.0045)
        self.assertEqual(len(actions),15)
        self.assertLessEqual(max(actions[-1]['squeeze_m']),.0011+1e-10)
        self.assertEqual(actions[-1]['squeeze_m'],actions[-2]['squeeze_m'])

    def test_first_probe_uses_selected_small_surface_support(self):
        actions=self.run_servo(support_radii=(.00022,.0012))
        self.assertAlmostEqual(actions[0]['squeeze_m'][0]-.0005,.00022/3)
        self.assertAlmostEqual(actions[0]['squeeze_m'][1]-.0005,.00025)
    def test_every_probe_respects_the_selected_surface_step(self):
        radii=np.array([.00022,.00088]);actions=self.run_servo(cap=.0045,support_radii=radii)
        amounts=np.array([[.0005,.0005]]+[x['squeeze_m'] for x in actions])
        self.assertTrue(np.all(np.diff(amounts,axis=0)<=radii/3+1e-10))
    def test_broad_surface_does_not_increase_inherited_probe_step(self):
        actions=self.run_servo(support_radii=(.012,.012))
        self.assertEqual(actions[0]['squeeze_m'],[.00075,.00075])

    def test_unsettled_digit_with_real_tracking_lag_keeps_waiting(self):
        actions=self.run_servo(resume=False,tracking_error=.05)
        self.assertEqual(len(actions),15)
        self.assertEqual(actions[-1]['squeeze_m'],actions[0]['squeeze_m'])
    def test_unsettled_residual_does_not_create_contact_or_force_evidence(self):
        actions=self.run_servo(resume=False,tracking_error=.006,deficit=.9)
        self.assertTrue(all(x['contact_proxy_at_m']==[None,None] for x in actions))
        self.assertTrue(all(not row['contact_or_force_inferred'] for x in actions for row in x['unsettled_tracking']))

class MeasuredSpeedLimitsTest(unittest.TestCase):
    def robot(self):
        return SimpleNamespace(lo=np.zeros(8),base_indices=list(range(6)),
          description=SimpleNamespace(urdf_velocity=np.array([10.]*6+[13.578,.08]),
            joint_types=['prismatic']*3+['revolute']*4+['prismatic']))
    def test_digit_uses_declared_angular_velocity_instead_of_palm_task_rate(self):
        r=self.robot();limits=m.measured_speed_limits(r)
        self.assertLess(2.0028486251831055,limits[6])
        self.assertEqual(limits[6],13.578)
    def test_palm_task_rates_still_bound_large_declared_urdf_velocity(self):
        limits=m.measured_speed_limits(self.robot())
        np.testing.assert_array_equal(limits[:3],[.15]*3)
        np.testing.assert_array_equal(limits[3:6],[1.5]*3)
    def test_prismatic_digit_retains_linear_urdf_limit(self):
        self.assertEqual(m.measured_speed_limits(self.robot())[7],.08)
    def test_missing_digit_velocity_uses_conservative_unit_appropriate_fallback(self):
        r=self.robot();r.description.urdf_velocity[6:]=np.nan
        np.testing.assert_array_equal(m.measured_speed_limits(r)[6:],[1.5,.15])

if __name__=='__main__':unittest.main()
