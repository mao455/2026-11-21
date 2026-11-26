"""Low-memory route tests: AST-extracted retimers and a synthetic exact checker.

Usage:
  python /tmp/test_fe_joint_routes.py --strategy /path/mainline/strategy.py
  python /tmp/test_fe_joint_routes.py --strategy /path/mainline/strategy.py --rrt-module /path/module.py

No strategy module, URDF, mesh, or simulator is imported.
"""
import argparse
import ast
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

DEFAULT = str(Path(__file__).resolve().parent/'mainline/strategy.py')
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', default=DEFAULT)
parser.add_argument('--rrt-module', default=str(Path(__file__).resolve().parent/'mainline/joint_path.py'))
args, test_args = parser.parse_known_args()


def load_retimers(path):
    tree = ast.parse(Path(path).read_text())
    wanted = {'_s3_steps', 'smooth_segment', '_s3_retime_approach'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise AssertionError('Missing expected retimer function')
    namespace = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


spec = importlib.util.spec_from_file_location('rrt_under_test', args.rrt_module)
rrt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rrt)


class RouteRetimerTests(unittest.TestCase):
    def test_route_fingers_keeps_every_collision_avoiding_turn(self):
        ns = load_retimers(args.strategy)
        initial = np.zeros(8)
        initial[[6, 7]] = [-.8, 0.]
        coords = [(-.8, .5), (.8, .5), (.8, 0.)]
        acts = []
        for xy in coords:
            q = initial.copy()
            q[[6, 7]] = xy
            acts.append([q.tolist(), 24, 'route_fingers'])
        # The original route goes around the disk. Collapsing a run of equal
        # phase labels to its final target cuts straight through its interior.
        for fast in (True,):
            with self.subTest(fast=fast):
                result = ns['_s3_retime_approach'](acts, initial)
                self.assertGreaterEqual(len(result), len(acts))
                route = np.asarray([node[0] for node in result])
                cursor = 0
                for xy in coords:
                    matches = np.flatnonzero(np.all(np.isclose(route[cursor:, [6, 7]], xy, rtol=0, atol=1e-12), axis=1))
                    self.assertGreater(len(matches), 0, 'A planned route waypoint was discarded')
                    cursor += int(matches[0]) + 1
                self.assertTrue(np.all(np.linalg.norm(route[:, [6, 7]], axis=1) > .3), 'Retiming creates a collision by collapsing route turns')
                self.assertTrue(np.allclose(route[:, :6], initial[:6], rtol=0, atol=0))
                self.assertTrue(all(row[1] > 0 and row[1] % 6 == 0 for row in result))

    def test_existing_close_and_probe_commands_remain_exact(self):
        ns = load_retimers(args.strategy)
        initial = np.zeros(8)
        close_q = initial.copy(); close_q[6] = .01
        probe_q = initial.copy(); probe_q[6] = .02
        acts = [[close_q.tolist(), 24, 'close'], [probe_q.tolist(), 240, 'probe_hold']]
        for fast in (True,):
            with self.subTest(fast=fast):
                self.assertEqual(ns['_s3_retime_approach'](acts, initial), acts)


class Model:
    # Deliberately noncontiguous, permuted group and unequal bounds.
    lo = np.array([-1., -.2, -1.2, -.3])
    hi = np.array([1., .2, 1.2, .3])
    all_finger_indices = np.array([2, 0])


class ExactChecker:
    def __init__(self, obstacle=True):
        self.obstacle = obstacle
        self.calls = []

    def check(self, q, object_check=True):
        if not np.array_equal(q[[1, 3]], [.123, -.234]):
            raise AssertionError('A nonplanned coordinate moved')
        if object_check:
            raise AssertionError('Unexpected object-check semantics')
        if np.any(q < Model.lo) or np.any(q > Model.hi):
            raise AssertionError('Checker received an out-of-bounds state')
        self.calls.append(q.copy())
        passed = not self.obstacle or np.linalg.norm(q[[0, 2]]) > .30
        return {'passed': bool(passed)}


class RRTTests(unittest.TestCase):
    def setUp(self):
        self.start = np.array([-.8, .123, 0., -.234])
        self.goal = np.array([.8, .123, 0., -.234])

    def plan(self, checker=None, **kwargs):
        return rrt.plan_finger_rrt_connect(Model(), checker or ExactChecker(), self.start, self.goal, **kwargs)

    def test_direct_path_is_exactly_checked_at_required_resolution(self):
        checker = ExactChecker(obstacle=False)
        path, counts = self.plan(checker)
        self.assertIsNotNone(path, counts)
        path = np.asarray(path)
        self.assertEqual(counts['reason'], 'direct_path')
        self.assertEqual(counts['extensions'], 0)
        self.assertEqual(counts['verified_returned_waypoints'], len(path))
        self.assertGreater(counts['collision_checks'], 2)
        self.assertLessEqual(np.linalg.norm(np.diff(path, axis=0), axis=1).max(), .04 + 1e-12)
        np.testing.assert_array_equal(path[0], self.start[Model.all_finger_indices])
        np.testing.assert_array_equal(path[-1], self.goal[Model.all_finger_indices])
        self.assertEqual(len(checker.calls), counts['collision_checks'])

    def test_obstacle_path_reproducible_exact_and_bounded(self):
        first_checker, second_checker = ExactChecker(), ExactChecker()
        first, a = self.plan(first_checker, seed=17001, max_extensions=1500)
        second, b = self.plan(second_checker, seed=17001, max_extensions=1500)
        self.assertIsNotNone(first, a)
        self.assertEqual(a['reason'], 'rrt_connected')
        first = np.asarray(first)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(a['extensions'], b['extensions'])
        self.assertEqual(a['collision_checks'], b['collision_checks'])
        self.assertTrue(np.all(np.linalg.norm(first, axis=1) > .30))
        self.assertLessEqual(np.linalg.norm(np.diff(first, axis=0), axis=1).max(), .04 + 1e-12)
        self.assertTrue(np.all(first >= Model.lo[Model.all_finger_indices]))
        self.assertTrue(np.all(first <= Model.hi[Model.all_finger_indices]))
        self.assertLessEqual(a['extensions'], 1500)
        self.assertLessEqual(a['collision_checks'], a['max_collision_checks'])

    def test_extension_and_collision_budgets_return_none(self):
        for kwargs, reason in [({'max_extensions': 0}, 'extension_budget_exhausted'),
                               ({'max_extensions': 1}, 'extension_budget_exhausted'),
                               ({'max_collision_checks': 1}, 'collision_check_budget_exhausted')]:
            with self.subTest(kwargs=kwargs):
                path, counts = self.plan(**kwargs)
                self.assertIsNone(path)
                self.assertEqual(counts['reason'], reason)
                self.assertLessEqual(counts['extensions'], counts['max_extensions'])
                self.assertLessEqual(counts['collision_checks'], counts['max_collision_checks'])

    def test_invalid_endpoint_and_fixed_coordinate_change(self):
        invalid = self.goal.copy(); invalid[[0, 2]] = 0.
        path, counts = rrt.plan_finger_rrt_connect(Model(), ExactChecker(), self.start, invalid)
        self.assertIsNone(path)
        self.assertEqual(counts['reason'], 'invalid_goal')
        out_of_limits = self.goal.copy(); out_of_limits[0] = 1.1
        path, counts = rrt.plan_finger_rrt_connect(Model(), ExactChecker(), self.start, out_of_limits)
        self.assertIsNone(path)
        self.assertEqual(counts['reason'], 'invalid_goal')
        changed = self.goal.copy(); changed[1] += .001
        with self.assertRaises(ValueError):
            rrt.plan_finger_rrt_connect(Model(), ExactChecker(), self.start, changed)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *test_args])
