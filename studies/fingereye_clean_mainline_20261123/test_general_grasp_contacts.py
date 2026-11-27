"""CPU-only contracts, kinematics and contact equilibrium; no simulator import."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from general_grasp_contacts import (
    ContactPlanner, ContactTarget, ObjectPatch, URDFHand, friction_wrench_feasibility,
    task_wrench_margin,
)


def radial_hand(path, count):
    pieces = ['<robot name="synthetic"><link name="root"/>']
    for index in range(count):
        angle = 2. * np.pi * index / count
        pieces.append(f'''
<link name="tip_{index}"/><link name="skin_{index}">
 <collision><geometry><box size="0.008 0.012 0.012"/></geometry></collision>
</link>
<joint name="slide_{index}" type="prismatic">
 <parent link="root"/><child link="tip_{index}"/>
 <origin xyz="{.04 * np.cos(angle)} {.04 * np.sin(angle)} 0" rpy="0 0 {angle}"/>
 <axis xyz="-1 0 0"/><limit lower="0" upper=".03" velocity=".2" effort="10"/>
</joint>
<joint name="mount_{index}" type="fixed"><parent link="tip_{index}"/>
 <child link="skin_{index}"/><origin xyz="0 0 .002"/></joint>''')
    path.write_text(''.join(pieces) + '</robot>')
    return path


def synthetic_contacts(hand, q, provenance='measured'):
    poses = hand.fk(q)
    contacts, patches = [], []
    for patch_index, finger in enumerate(hand.fingers):
        anchor = next(anchor for anchor in finger.anchors if np.allclose(anchor.normal, [-1, 0, 0]))
        transform = poses[anchor.link]
        point = transform[:3, :3] @ anchor.point + transform[:3, 3]
        normal = -(transform[:3, :3] @ anchor.normal)
        patch = ObjectPatch(point, normal, np.eye(3) * 1e-10, provenance,
                            normal_uncertainty=.02, friction_lower=.6, depth_bounded=True,
                            support=('synthetic_camera_a', 'synthetic_camera_b'))
        patches.append(patch)
        contacts.append(ContactTarget(finger.name, anchor, patch_index, patch))
    return contacts, patches


class WrenchTests(unittest.TestCase):
    points = np.array([[-.02, 0, 0], [.02, 0, 0]])
    normals = np.array([[-1., 0, 0], [1., 0, 0]])

    def test_opposed_contacts_balance_force_not_axial_torque(self):
        force = friction_wrench_feasibility(self.points, self.normals, friction=.6,
                                           normal_uncertainty=.02, external_wrenches=[[0, 0, -.5, 0, 0, 0]])
        self.assertTrue(force['feasible'])
        self.assertGreater(force['minimum_load_scale'], 4.)
        self.assertLess(force['normalized_residual'], 1e-8)
        self.assertEqual(force['nominal_wrench_rank'], 5)
        self.assertFalse(force['full_force_closure_certified'])
        forces = np.array(force['worst_witness']['contact_forces'])
        load = np.array(force['worst_witness']['external_wrench'])
        equilibrium = forces.sum(axis=0) + force['minimum_load_scale'] * load[:3]
        np.testing.assert_allclose(equilibrium, 0, atol=1e-8)
        torque = friction_wrench_feasibility(self.points, self.normals, friction=.6,
                                            external_wrenches=[[0, 0, 0, .01, 0, 0]])
        self.assertFalse(torque['feasible'])

    def test_same_side_and_uncertain_normals_are_rejected(self):
        same = friction_wrench_feasibility(self.points, [[1, 0, 0], [1, 0, 0]], friction=.6)
        self.assertFalse(same['feasible'])
        uncertain = friction_wrench_feasibility(self.points, self.normals, normal_uncertainty=.5)
        self.assertFalse(uncertain['feasible'])
        self.assertIn('erases', uncertain['reason'])

    def test_more_contacts_support_missing_torque_direction(self):
        angles = np.arange(3) * 2 * np.pi / 3
        normals = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))
        loads = np.diag([.2, .2, .2, .002, .002, .002])
        result = friction_wrench_feasibility(normals * .02, normals, friction=.6,
                                            normal_uncertainty=.02, external_wrenches=np.vstack((loads, -loads)))
        self.assertTrue(result['feasible'])
        self.assertEqual(result['nominal_wrench_rank'], 6)

    def test_uncertainty_scenarios_and_effort_are_not_ignored(self):
        result = friction_wrench_feasibility(self.points, self.normals, friction=.6,
                                            covariances=np.tile(np.eye(3) * 1e-8, (2, 1, 1)),
                                            external_wrenches=[[0, 0, -.2, 0, 0, 0]])
        self.assertEqual(result['uncertainty_scenarios'], 13)
        self.assertFalse(result['feasible'])
        angles = np.arange(3) * 2 * np.pi / 3
        radial = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(3)))
        stable = friction_wrench_feasibility(radial * .02, radial, friction=.6,
                                            covariances=np.tile(np.eye(3) * 1e-8, (3, 1, 1)),
                                            external_wrenches=[[0, 0, -.2, 0, 0, 0]])
        self.assertEqual(stable['uncertainty_scenarios'], 19)
        self.assertTrue(stable['feasible'])
        limited = friction_wrench_feasibility(self.points, self.normals, friction=.6,
                                             torque_map=np.eye(6), effort_limits=np.full(6, .01),
                                             external_wrenches=[[0, 0, -1, 0, 0, 0]])
        self.assertFalse(limited['feasible'])

    def test_rigid_frame_change_preserves_feasibility(self):
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('xyz', [.2, -.5, .8]).as_matrix()
        translation = np.array([.4, -.1, .2])
        load = np.array([0, 0, -.4, 0, 0, 0.])
        reference = np.array([0., 0., 0.])
        first = friction_wrench_feasibility(self.points, self.normals, friction=.6,
                                           external_wrenches=[load], reference=reference)
        second = friction_wrench_feasibility(self.points @ rotation.T + translation, self.normals @ rotation.T,
                                            friction=.6, external_wrenches=[np.r_[rotation @ load[:3], rotation @ load[3:]]],
                                            reference=rotation @ reference + translation)
        self.assertEqual(first['feasible'], second['feasible'])
        self.assertAlmostEqual(first['minimum_load_scale'], second['minimum_load_scale'], places=6)

    def test_gravity_is_fixed_and_reserve_witness_balances_total_load(self):
        load = np.array([0., 0., -.5, 0., 0., 0.])
        gravity = np.array([0., 0., -1., 0., 0., 0.])
        result = friction_wrench_feasibility(
            self.points, self.normals, friction=.6, external_wrenches=[load],
            gravity_wrench=gravity, reference=np.zeros(3), minimum_normal_fraction=0.)
        old = friction_wrench_feasibility(self.points, self.normals, friction=.6,
                                         external_wrenches=[load], reference=np.zeros(3))
        self.assertTrue(result['gravity_feasible'])
        self.assertAlmostEqual(old['minimum_load_scale'] - result['minimum_load_scale'], 2., places=6)
        forces = np.array(result['worst_witness']['contact_forces'])
        np.testing.assert_allclose(forces.sum(0) + gravity[:3] + result['minimum_load_scale'] * load[:3], 0., atol=1e-8)
        np.testing.assert_allclose(np.cross(self.points, forces).sum(0), 0., atol=1e-8)

    def test_weight_support_and_disturbance_reserve_are_distinct(self):
        result = friction_wrench_feasibility(
            self.points, self.normals, friction=.6, external_wrenches=[[0, 0, -.5, 0, 0, 0]],
            gravity_wrench=[0, 0, -2., 0, 0, 0], minimum_normal_fraction=0.)
        self.assertTrue(result['gravity_feasible'])
        self.assertFalse(result['feasible'])
        self.assertAlmostEqual(result['minimum_load_scale'], .8, places=6)
        failed = friction_wrench_feasibility(
            self.points, self.normals, friction=.6, external_wrenches=[[0, 0, -.5, 0, 0, 0]],
            gravity_wrench=[0, 0, -3., 0, 0, 0], minimum_normal_fraction=0.)
        self.assertFalse(failed['gravity_feasible'])
        self.assertFalse(failed['feasible'])
        self.assertEqual(failed['gravity_failure_scenarios'][0]['scenario'], 0)

    def test_task_reports_missing_two_point_torsion_and_third_contribution(self):
        pair = task_wrench_margin(self.points, self.normals, reference=np.zeros(3),
                                  friction=.6, normal_uncertainty=.02, mass_range_kg=(.01, .02))
        self.assertTrue(pair['lift_feasible'])
        self.assertFalse(pair['task_feasible'])
        self.assertAlmostEqual(pair['torque_margin'], 0., places=6)
        radial = np.c_[np.cos(np.arange(3) * 2 * np.pi / 3),
                       np.sin(np.arange(3) * 2 * np.pi / 3), np.zeros(3)]
        triple = task_wrench_margin(radial * .02, radial, reference=np.zeros(3),
                                    friction=.6, normal_uncertainty=.02, mass_range_kg=(.01, .02))
        self.assertTrue(triple['task_feasible'])
        self.assertGreater(triple['torque_margin'], 1.)
        self.assertGreater(triple['third_contact_increment']['torque_margin'], 0.)
        self.assertEqual(triple['reference'], pair['reference'])

    def test_task_effort_and_uncertainty_affect_gravity_support(self):
        limited = task_wrench_margin(self.points, self.normals, reference=np.zeros(3),
                                     friction=.6, normal_uncertainty=.02,
                                     torque_map=np.eye(6), effort_limits=np.full(6, .001))
        self.assertFalse(limited['gravity_feasible'])
        uncertain = task_wrench_margin(self.points, self.normals, reference=np.zeros(3),
                                       friction=.6, normal_uncertainty=.02,
                                       covariances=np.tile(np.eye(3) * 1e-8, (2, 1, 1)))
        self.assertEqual(uncertain['mass_trials'][0]['uncertainty_scenarios'], 13)
        self.assertFalse(uncertain['gravity_feasible'])

    def test_finite_pad_supports_small_com_offset_and_has_bounded_moment(self):
        arguments = dict(reference=[0., .0006, 0.], friction=.6, normal_uncertainty=.02,
                         mass_range_kg=(.01, .02))
        point = task_wrench_margin(self.points, self.normals, **arguments)
        footprint = task_wrench_margin(self.points, self.normals, pad_radius_m=.0012, **arguments)
        self.assertFalse(point['gravity_feasible'])
        self.assertTrue(footprint['task_feasible'])
        self.assertGreater(footprint['torque_margin'], 1.)
        arguments['reference'] = [0., .02, 0.]
        outside = task_wrench_margin(self.points, self.normals, pad_radius_m=.0012, **arguments)
        self.assertFalse(outside['gravity_feasible'])

    def test_footprint_sites_share_capacity_instead_of_multiplying_it(self):
        load = [[0., 0., -.5, 0., 0., 0.]]
        original = friction_wrench_feasibility(self.points, self.normals, friction=.6,
                                               external_wrenches=load, minimum_normal_fraction=0.)
        repeated = friction_wrench_feasibility(np.repeat(self.points, 4, axis=0),
                                               np.repeat(self.normals, 4, axis=0), friction=.6,
                                               external_wrenches=load, minimum_normal_fraction=0.,
                                               contact_groups=np.repeat(np.arange(2), 4))
        self.assertAlmostEqual(original['minimum_load_scale'], repeated['minimum_load_scale'], places=6)
        self.assertEqual(repeated['max_normal_force'], [2., 2.])
        self.assertEqual(len(repeated['worst_witness']['normal_forces']), 2)
        self.assertLessEqual(max(repeated['worst_witness']['normal_forces']), 2. + 1e-8)

    def test_pad_uncertainty_moves_whole_footprints_and_third_removal_is_grouped(self):
        angles = np.arange(3) * 2 * np.pi / 3
        radial = np.c_[np.cos(angles), np.sin(angles), np.zeros(3)]
        result = task_wrench_margin(.02 * radial, radial, reference=np.zeros(3),
                                    pad_radius_m=.0012, friction=.6, normal_uncertainty=.02,
                                    covariances=np.tile(np.eye(3) * 1e-8, (3, 1, 1)))
        self.assertEqual(result['mass_trials'][0]['uncertainty_scenarios'], 19)
        self.assertEqual(result['footprint_contact_groups'], [0]*4 + [1]*4 + [2]*4)
        self.assertIn('without_third_contact', result)
        self.assertEqual(result['mass_trials'][0]['max_normal_force'], [2., 2., 2.])


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_discovery_varies_with_topology_and_joint_order(self):
        for count in (2, 3, 5):
            order = [f'slide_{index}' for index in reversed(range(count))]
            hand = URDFHand.from_urdf(radial_hand(self.path / f'{count}.urdf', count), joint_order=order)
            self.assertEqual(len(hand.fingers), count)
            self.assertEqual(hand.joint_order, tuple(order))
            self.assertTrue(all(len(finger.anchors) > 0 for finger in hand.fingers))
            self.assertTrue(all(len(finger.distal_links) == 2 for finger in hand.fingers))
            np.testing.assert_allclose(hand.q_vector({name: .01 for name in order}), .01)
            with self.assertRaises(ValueError):
                URDFHand.from_urdf(hand.path, joint_order=order[:-1])

    def test_jacobian_and_fixed_attachment(self):
        hand = URDFHand.from_urdf(radial_hand(self.path / 'hand.urdf', 3))
        q = np.full(3, .01)
        contact = synthetic_contacts(hand, q)[0][0]
        analytic = hand.contact_jacobian(q, contact.anchor.link, contact.anchor.point)
        finite = np.zeros_like(analytic)
        for index in range(len(q)):
            lower, upper = q.copy(), q.copy()
            lower[index] -= 1e-6
            upper[index] += 1e-6
            poses = [hand.fk(value)[contact.anchor.link] for value in (lower, upper)]
            points = [pose[:3, :3] @ contact.anchor.point + pose[:3, 3] for pose in poses]
            finite[:, index] = (points[1] - points[0]) / 2e-6
        np.testing.assert_allclose(analytic, finite, atol=1e-9)
        self.assertTrue(np.allclose(hand.fk(q)['skin_0'][:3, 3], [.03, 0, .002]))

    def test_mimic_bounds_fk_and_jacobian(self):
        path = self.path / 'mimic.urdf'
        path.write_text('''<robot name="mimic"><link name="root"/><link name="middle"/>
<link name="tip"><collision><geometry><sphere radius=".004"/></geometry></collision></link>
<joint name="follower" type="prismatic"><parent link="middle"/><child link="tip"/>
<axis xyz="0 1 0"/><limit lower="0" upper=".015" velocity=".05" effort="1"/>
<mimic joint="drive" multiplier="-.5" offset=".02"/></joint>
<joint name="drive" type="prismatic"><parent link="root"/><child link="middle"/>
<axis xyz="1 0 0"/><limit lower="0" upper=".05" velocity=".2" effort="2"/></joint></robot>''')
        hand = URDFHand.from_urdf(path, joint_order=['drive'])
        np.testing.assert_allclose(hand.lower, [.01])
        np.testing.assert_allclose(hand.upper, [.04])
        np.testing.assert_allclose(hand.velocity, [.1])
        np.testing.assert_allclose(hand.fk([.02])['tip'][:3, 3], [.02, .01, 0])
        np.testing.assert_allclose(hand.contact_jacobian([.02], 'tip', np.zeros(3)), [[1], [-.5], [0]])

    def test_bounded_ik_respects_actual_q_limits_and_velocity(self):
        hand = URDFHand.from_urdf(radial_hand(self.path / 'ik.urdf', 3))
        contacts, _ = synthetic_contacts(hand, np.full(3, .012))
        planner = ContactPlanner(hand, position_tolerance=.0001)
        q = np.full(3, .002)
        solution = planner.solve_ik(q, contacts)
        self.assertTrue(solution.feasible)
        np.testing.assert_allclose(solution.q, .012, atol=1e-5)
        limited = planner.solve_ik(q, contacts, time_horizon=.01)
        self.assertFalse(limited.feasible)
        self.assertTrue(np.all(limited.q <= q + .002 + 1e-10))
        held = planner.solve_ik(q, contacts, max_delta=0.)
        np.testing.assert_array_equal(held.q, q)
        with self.assertRaises(ValueError):
            planner.solve_ik([-.1, .002, .002], contacts)
        violations = hand.bound_violations([-.0001, .002, .002])
        self.assertEqual(violations[0]['joint'], 'slide_0')
        self.assertAlmostEqual(violations[0]['excess'], .0001)

    def test_missing_robot_mesh_is_unknown_not_an_invented_pad(self):
        path = self.path / 'missing.urdf'
        path.write_text('''<robot name="missing"><link name="root"/>
<link name="tip"><collision><geometry><mesh filename="missing.stl"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="root"/><child link="tip"/>
<axis xyz="1 0 0"/><limit lower="0" upper=".1" velocity=".1" effort="1"/></joint></robot>''')
        hand = URDFHand.from_urdf(path)
        self.assertEqual(len(hand.fingers), 1)
        self.assertEqual(len(hand.fingers[0].anchors), 0)
        self.assertTrue(hand.geometry_warnings)

    def test_unselected_joints_are_held(self):
        hand = URDFHand.from_urdf(radial_hand(self.path / 'unselected.urdf', 5))
        contacts, _ = synthetic_contacts(hand, np.full(5, .012))
        actual = np.full(5, .002)
        result = ContactPlanner(hand).solve_ik(actual, contacts[:2])
        self.assertTrue(result.feasible)
        np.testing.assert_array_equal(result.q[2:], actual[2:])

    def test_unknown_geometry_and_B_adapter_remain_explicit(self):
        hand = URDFHand.from_urdf(radial_hand(self.path / 'unknown.urdf', 2))
        _, patches = synthetic_contacts(hand, [.01, .01])
        unknown = ObjectPatch(None, None, None, 'unknown')
        result = ContactPlanner(hand).plan([.01, .01], [patches[0], unknown])
        self.assertFalse(result.feasible)
        self.assertEqual(result.diagnostics['rejected_patches'][0]['reason'], 'unknown_geometry')
        silhouette = ObjectPatch.from_mapping(dict(point=[0, 0, 0], normal=[1, 0, 0],
                                                  covariance=np.eye(3) * 1e-8, provenance='inferred',
                                                  depth_unknown=True, normal_uncertainty=np.pi))
        self.assertEqual(silhouette.rejection_reason(), 'unbounded_depth')
        supported = ObjectPatch.from_mapping(dict(point=[0, 0, 0], normal=[1, 0, 0],
                                                 covariance=np.eye(3) * 1e-8, provenance='inferred',
                                                 depth_unknown=False, source_views=['one', 'two'], normal_uncertainty=.1))
        self.assertIsNone(supported.rejection_reason())
        self.assertEqual(supported.provenance, 'inferred')
        self.assertEqual(supported.support, ('one', 'two'))
        with self.assertRaises(ValueError):
            ObjectPatch([0, 0, 0], [1, 0, 0], np.diag([1, -1, 1]), 'measured')

    def test_planner_returns_real_two_and_three_finger_candidates(self):
        for count in (2, 3):
            hand = URDFHand.from_urdf(radial_hand(self.path / f'plan{count}.urdf', count))
            _, patches = synthetic_contacts(hand, np.full(count, .012), provenance='inferred')
            if count == 2:
                for patch in patches:
                    patch.covariance = np.zeros((3, 3))
            loads = [[0, 0, -.2, 0, 0, 0]] if count == 2 else [[0, 0, 0, .002, 0, 0]]
            result = ContactPlanner(hand, position_tolerance=.0003).plan(
                np.full(count, .002), patches, external_wrenches=loads,
                safety_check=lambda q: dict(passed=bool(np.all(q >= 0))))
            self.assertTrue(result.feasible, result.diagnostics)
            self.assertEqual(len(result.contacts), count)
            self.assertFalse(result.diagnostics['execution_ready'])
            self.assertFalse(result.diagnostics['collision_certified'])
            self.assertTrue(all(contact.patch.provenance == 'inferred' for contact in result.contacts))
            json.dumps(result.as_dict(), allow_nan=False)
            refused = ContactPlanner(hand).plan(np.full(count, .002), patches,
                                                 external_wrenches=loads, safety_check=lambda q: False)
            self.assertFalse(refused.feasible)

    def test_fixed_finger_groups_use_same_solver_and_hold_unselected_joints(self):
        hand = URDFHand.from_urdf(radial_hand(self.path / 'groups.urdf', 4))
        _, patches = synthetic_contacts(hand, np.full(4, .012))
        for patch in patches:
            patch.covariance = np.zeros((3, 3))
        names = [finger.name for finger in hand.fingers]
        actual = np.full(4, .002)
        planner = ContactPlanner(hand, position_tolerance=.0003)
        for selected in ((names[0], names[2]), (names[0], names[1], names[2])):
            result = planner.plan(actual, patches, external_wrenches=[[0, 0, -.2, 0, 0, 0]],
                                  contact_counts=(len(selected),), finger_groups=(selected,),
                                  safety_check=lambda q: {'passed': True})
            self.assertTrue(result.feasible, result.diagnostics)
            self.assertEqual(set(contact.finger for contact in result.contacts), set(selected))
            self.assertEqual(result.diagnostics['requested_contact_counts'], [len(selected)])
            self.assertEqual(result.diagnostics['requested_finger_groups'], [sorted(selected)])
            self.assertLessEqual(max(result.diagnostics['position_errors']), planner.position_tolerance)
            self.assertLessEqual(max(result.diagnostics['normal_errors']), planner.normal_tolerance)
            unselected = [i for i, name in enumerate(names) if name not in selected]
            np.testing.assert_array_equal(result.q[unselected], actual[unselected])

    def test_finger_groups_and_counts_fail_clearly_before_search(self):
        hand = URDFHand.from_urdf(radial_hand(self.path / 'bad_groups.urdf', 3))
        _, patches = synthetic_contacts(hand, np.full(3, .012))
        planner = ContactPlanner(hand)
        names = [finger.name for finger in hand.fingers]
        for settings in ({'finger_groups': [('absent', names[0])]},
                         {'finger_groups': [(names[0],)]},
                         {'contact_counts': [True]}, {'contact_counts': []},
                         {'contact_counts': [2], 'finger_groups': [names]}):
            with self.assertRaises(ValueError):
                planner.plan(np.full(3, .002), patches, **settings)

    def test_ranking_does_not_sum_more_residuals_for_three_fingers(self):
        from types import SimpleNamespace
        hand = URDFHand.from_urdf(radial_hand(self.path / 'quality.urdf', 3))
        contacts, _ = synthetic_contacts(hand, np.full(3, .012))
        planner = ContactPlanner(hand)
        actual = np.full(3, .002)
        qualities = []
        for count in (2, 3):
            solution = SimpleNamespace(q=np.full(3, .012), position_errors=np.full(count, .0004),
                                       normal_errors=np.full(count, .1))
            qualities.append(planner.candidate_quality(actual, contacts[:count], solution,
                                                       {'minimum_load_scale': 2.}))
        for key in qualities[0]:
            self.assertAlmostEqual(qualities[0][key], qualities[1][key])
        self.assertAlmostEqual(qualities[0]['minimum_command_travel_s'], .05)

    def test_actual_project_urdfs_discovery_and_revolute_jacobian(self):
        root = Path(__file__).resolve().parents[2]
        physical = URDFHand.from_urdf(root / 'assets/xhand3camFE_left/xhand3camFE_left.urdf')
        planning = URDFHand.from_urdf(root / 'assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf')
        self.assertEqual(len(physical.fingers), 5)
        self.assertEqual(len(physical.joint_order), 12)
        self.assertEqual(len(planning.joint_order), 18)
        self.assertEqual(len(planning.fingers), len(physical.fingers))
        self.assertTrue(any('open/inconsistent' in message for message in physical.geometry_warnings))
        q = (planning.lower + planning.upper) / 2
        anchor = next(anchor for finger in planning.fingers for anchor in finger.anchors if anchor.normal_reliable)
        analytic = planning.contact_jacobian(q, anchor.link, anchor.point)
        for index in range(len(q)):
            shifted = q.copy()
            shifted[index] += 1e-6
            poses = [planning.fk(value)[anchor.link] for value in (q, shifted)]
            points = [pose[:3, :3] @ anchor.point + pose[:3, 3] for pose in poses]
            np.testing.assert_allclose(analytic[:, index], (points[1] - points[0]) / 1e-6, atol=2e-6)

    def test_planner_can_require_five_contacts_instead_of_fixed_pair(self):
        hand = URDFHand.from_urdf(radial_hand(self.path / 'five.urdf', 5))
        _, patches = synthetic_contacts(hand, np.full(5, .012))
        result = ContactPlanner(hand).plan(np.full(5, .002), patches,
                                           external_wrenches=[[0, 0, -5, 0, 0, 0]])
        self.assertTrue(result.feasible, result.diagnostics)
        self.assertEqual(len(result.contacts), 5)
        self.assertEqual(result.diagnostics['tested_contact_counts'], [2, 3, 4, 5])
        self.assertGreaterEqual(result.diagnostics['wrench']['minimum_load_scale'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
