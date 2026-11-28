"""Bounded, deterministic RRT-Connect over declared joint coordinates.

This module imports no hand assets or simulator. The caller supplies the same
exact checker used by the final command-path guard. It never changes physical
filters, clearances, joint limits, or object-overlap rules.
"""
from __future__ import annotations

import time
import numpy as np


def plan_finger_rrt_connect(
    robot,
    checker,
    start_q,
    goal_q,
    *,
    joint_indices=None,
    seed=17001,
    max_extensions=1500,
    step_size=0.25,
    edge_resolution=0.04,
    goal_bias=0.15,
    max_collision_checks=20000,
    max_wall_s=None,
    object_check=False,
):
    """Return ``(finger_waypoints_or_None, statistics)``.

    ``joint_indices`` defaults to ``robot.all_finger_indices``. Both full input
    vectors must have identical coordinates outside that group: the palm and
    every other coordinate are fixed at ``start_q`` throughout the search.
    Sampling uses the group's actual ``robot.lo``/``robot.hi`` limits.

    Each extension advances at most ``step_size`` in Euclidean joint distance.
    Every edge is sampled at Euclidean spacing <= ``edge_resolution`` and every
    sample calls ``checker.check(full_q, object_check=object_check)`` (or reuses
    exactly the same state previously checked). Returned edges are densified
    at that resolution and the complete returned sequence is checked again.
    This is discrete geometric validation, not continuous-collision proof; the
    caller must retain its original final path/command preflight.

    Angles are radians for the present hands. If a caller includes prismatic
    coordinates, it must deliberately choose a resolution/step in those units.
    Seed and count budgets make the search reproducible; an optional wall
    budget can terminate earlier and is reported explicitly.
    """
    started = time.perf_counter()
    lo = np.asarray(robot.lo, dtype=float)
    hi = np.asarray(robot.hi, dtype=float)
    start = np.asarray(start_q, dtype=float).copy()
    goal = np.asarray(goal_q, dtype=float).copy()
    ids = np.asarray(robot.all_finger_indices if joint_indices is None else joint_indices, dtype=int)
    if start.shape != lo.shape or goal.shape != lo.shape or hi.shape != lo.shape or lo.ndim != 1:
        raise ValueError('Inputs must follow the full declared joint order')
    if ids.ndim != 1 or len(ids) == 0 or len(set(ids.tolist())) != len(ids) or np.any(ids < 0) or np.any(ids >= len(lo)):
        raise ValueError('Joint indices must be a nonempty unique declared group')
    fixed = np.array([i for i in range(len(lo)) if i not in set(ids.tolist())], dtype=int)
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError('Endpoints must be finite')
    if len(fixed) and not np.allclose(start[fixed], goal[fixed], rtol=0., atol=1e-12):
        raise ValueError('Coordinates outside the planning group must be fixed')
    if not np.all(np.isfinite(lo[ids])) or not np.all(np.isfinite(hi[ids])) or np.any(lo[ids] > hi[ids]):
        raise ValueError('Sampling requires finite ordered bounds from the hand model')
    if int(max_extensions) != max_extensions or max_extensions < 0:
        raise ValueError('max_extensions must be a nonnegative integer')
    if int(max_collision_checks) != max_collision_checks or max_collision_checks < 1:
        raise ValueError('max_collision_checks must be a positive integer')
    if not np.isfinite(step_size) or step_size <= 0 or not np.isfinite(edge_resolution) or edge_resolution <= 0:
        raise ValueError('Step size and resolution must be finite and positive')
    if not 0 <= goal_bias <= 1:
        raise ValueError('goal_bias must be in [0, 1]')
    if max_wall_s is not None and (not np.isfinite(max_wall_s) or max_wall_s <= 0):
        raise ValueError('Wall budget must be finite and positive')

    stats = dict(seed=int(seed), joint_indices=ids.tolist(), extensions=0, random_samples=0,
                 collision_checks=0, state_checks=0, cache_hits=0, edge_checks=0,
                 rejected_edges=0, tree_nodes_start=1, tree_nodes_goal=1,
                 returned_waypoints=0, verified_returned_waypoints=0,
                 max_extensions=int(max_extensions), max_collision_checks=int(max_collision_checks),
                 step_size=float(step_size), edge_resolution=float(edge_resolution),
                 object_check=bool(object_check), reason=None)
    rng = np.random.default_rng(seed)
    cache = {}
    group_lo, group_hi = lo[ids], hi[ids]
    span = group_hi - group_lo
    metric_scale = np.where(span > 0., span, 1.)
    x_start, x_goal = start[ids].copy(), goal[ids].copy()

    class BudgetExpired(Exception):
        pass

    def budget():
        if max_wall_s is not None and time.perf_counter() - started >= max_wall_s:
            raise BudgetExpired('wall_budget_exhausted')

    def full_q(x):
        q = start.copy()
        q[ids] = x
        return q

    def state_valid(x):
        budget()
        stats['state_checks'] += 1
        q = full_q(x)
        if np.any(q < lo) or np.any(q > hi):
            return False
        key = np.asarray(x, dtype=np.float64).tobytes()
        if key in cache:
            stats['cache_hits'] += 1
            return cache[key]
        if stats['collision_checks'] >= max_collision_checks:
            raise BudgetExpired('collision_check_budget_exhausted')
        stats['collision_checks'] += 1
        passed = bool(checker.check(q, object_check=object_check)['passed'])
        cache[key] = passed
        return passed

    def edge_samples(a, b):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / edge_resolution)))
        # Assign the exact supplied endpoint rather than recomputing a+(b-a).
        return [b.copy() if i == count else a + (b - a) * (i / count)
                for i in range(1, count + 1)]

    def edge_valid(a, b):
        stats['edge_checks'] += 1
        for x in edge_samples(a, b):
            if not state_valid(x):
                stats['rejected_edges'] += 1
                return False
        return True

    def finish(path, reason):
        stats['reason'] = reason
        stats['elapsed_s'] = time.perf_counter() - started
        return path, stats

    def validate_returned(sparse):
        dense = [sparse[0].copy()]
        for a, b in zip(sparse, sparse[1:]):
            dense.extend(edge_samples(a, b))
        for x in dense:
            if not state_valid(x):
                return None
            stats['verified_returned_waypoints'] += 1
        stats['returned_waypoints'] = len(dense)
        return [x.copy() for x in dense]

    class Tree:
        def __init__(self, root, from_start):
            self.points = [root.copy()]
            self.parents = [-1]
            self.from_start = from_start

        def extend(self, target):
            budget()
            if stats['extensions'] >= max_extensions:
                return 'budget', None
            stats['extensions'] += 1
            points = np.asarray(self.points)
            nearest = int(np.argmin(np.sum(((points - target) / metric_scale) ** 2, axis=1)))
            base = self.points[nearest]
            delta = target - base
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-12:
                return 'reached', nearest
            reached = distance <= step_size
            candidate = target.copy() if reached else base + delta * (step_size / distance)
            if not edge_valid(base, candidate):
                return 'trapped', nearest
            self.points.append(candidate)
            self.parents.append(nearest)
            stats['tree_nodes_start' if self.from_start else 'tree_nodes_goal'] += 1
            return ('reached' if reached else 'advanced'), len(self.points) - 1

        def trace(self, index):
            path = []
            while index != -1:
                path.append(self.points[index])
                index = self.parents[index]
            return list(reversed(path))

    try:
        if not state_valid(x_start):
            return finish(None, 'invalid_start')
        if not state_valid(x_goal):
            return finish(None, 'invalid_goal')
        if edge_valid(x_start, x_goal):
            path = validate_returned([x_start, x_goal])
            return finish(path, 'direct_path' if path is not None else 'final_verification_failed')
        first = Tree(x_start, True)
        second = Tree(x_goal, False)
        while stats['extensions'] < max_extensions:
            budget()
            stats['random_samples'] += 1
            target = second.points[0].copy() if rng.random() < goal_bias else rng.uniform(group_lo, group_hi)
            status, first_index = first.extend(target)
            if status in ('advanced', 'reached'):
                connection = first.points[first_index]
                while stats['extensions'] < max_extensions:
                    status, second_index = second.extend(connection)
                    if status == 'reached':
                        if first.from_start:
                            sparse = first.trace(first_index) + list(reversed(second.trace(second_index)))[1:]
                        else:
                            sparse = second.trace(second_index) + list(reversed(first.trace(first_index)))[1:]
                        path = validate_returned(sparse)
                        return finish(path, 'rrt_connected' if path is not None else 'final_verification_failed')
                    if status in ('trapped', 'budget'):
                        break
            first, second = second, first
        return finish(None, 'extension_budget_exhausted')
    except BudgetExpired as exc:
        return finish(None, str(exc))
