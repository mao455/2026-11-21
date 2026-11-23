import numpy as np

from scipy.optimize import least_squares, linprog

def _vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'Expected finite vector of length {size}')
    return result

def _unit(value):
    result = _vector(value)
    length = np.linalg.norm(result)
    if length < 1e-12:
        raise ValueError('A known normal/axis must be nonzero')
    return result / length

def _tangents(normal, hints=()):
    for hint in hints:
        projection = hint - normal * np.dot(hint, normal)
        if np.linalg.norm(projection) > 1e-10:
            tangent = _unit(projection)
            return tangent, np.cross(normal, tangent)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent = _unit(np.cross(normal, axis))
    return tangent, np.cross(normal, tangent)

def friction_wrench_feasibility(points, normals, *, friction=0.3, normal_uncertainty=0.,
                                covariances=None, sigma=2., max_normal_force=2.,
                                external_wrenches=None, reference=None, cone_edges=8,
                                torque_map=None, effort_limits=None, check_uncertainty=True,
                                gravity_wrench=None, minimum_normal_fraction=.01,
                                load_scale_cap=10., contact_groups=None):
    """Bounded point-contact LP with conservative inscribed friction pyramids.

    Wrenches are [force, torque] expressed about reference. Default tests are
    +/-1 N along each base axis, zero torque: NOT six-dimensional force closure.
    Position scenarios perturb one patch/eigenaxis at a time; passing is not a
    certificate over the full joint covariance ellipsoid. Forces are planned,
    never measured. URDF effort bounds can constrain a supplied J.T map.
    With gravity_wrench supplied, gravity stays fixed while only the task
    disturbance is scaled: W f + gravity + alpha * disturbance = 0. A separate
    alpha=0 solve checks support before any disturbance reserve is reported.
    contact_groups can bind multiple footprint sites to one shared normal-force
    limit. Their identical covariance is applied as a rigid patch translation.
    """
    points, normals = np.asarray(points, dtype=float), np.asarray(normals, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or normals.shape != points.shape:
        raise ValueError('Need two or more point/normal pairs')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError('Nonfinite contact geometry')
    normals = np.array([_unit(normal) for normal in normals])
    count = len(points)
    friction = np.broadcast_to(friction, (count,)).astype(float)
    uncertainty = np.broadcast_to(normal_uncertainty, (count,)).astype(float)
    limits = np.broadcast_to(max_normal_force, (count,)).astype(float)
    groups = np.arange(count) if contact_groups is None else np.asarray(contact_groups)
    if groups.shape != (count,) or groups.dtype.kind not in 'iu' or np.any(groups < 0):
        raise ValueError('contact_groups must contain one nonnegative integer per site')
    group_ids = np.unique(groups)
    group_limits = np.array([limits[np.flatnonzero(groups == group)[0]] for group in group_ids])
    if any(not np.all(limits[groups == group] == limit) for group, limit in zip(group_ids, group_limits)):
        raise ValueError('Sites in one contact group must declare the same shared normal-force limit')
    if (cone_edges < 4 or not np.all(np.isfinite([*friction, *uncertainty, *limits, sigma]))
            or np.any(friction < 0) or np.any(uncertainty < 0) or np.any(limits <= 0) or sigma < 0):
        raise ValueError('Invalid friction, uncertainty, load, sigma or cone edge budget')
    if not np.isfinite(minimum_normal_fraction) or not 0 <= minimum_normal_fraction <= 1:
        raise ValueError('minimum_normal_fraction must be in [0, 1]')
    if not np.isfinite(load_scale_cap) or load_scale_cap <= 0:
        raise ValueError('load_scale_cap must be finite and positive')
    gravity = np.zeros(6) if gravity_wrench is None else _vector(gravity_wrench, 6)
    angle = np.arctan(friction) - uncertainty
    if np.any(angle <= 0):
        return dict(feasible=False, reason='normal_uncertainty_erases_robust_friction_cone', minimum_load_scale=0.)
    robust_friction = np.tan(angle)
    center = points.mean(axis=0) if reference is None else _vector(reference)
    length_scale = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-3)
    loads = np.column_stack((np.vstack((np.eye(3), -np.eye(3))), np.zeros((6, 3)))) if external_wrenches is None else np.atleast_2d(external_wrenches).astype(float)
    if loads.shape[1] != 6 or not len(loads) or not np.all(np.isfinite(loads)) or np.any(np.linalg.norm(loads, axis=1) < 1e-12):
        raise ValueError('Supply nonzero finite external wrenches about reference')
    blocks = []
    basis_hints = [*(points - center), *normals, *loads[:, :3], *loads[:, 3:], gravity[:3], gravity[3:]]
    for normal, coefficient in zip(normals, robust_friction):
        inward = -normal
        tangent, second = _tangents(inward, basis_hints)
        blocks.append(np.array([inward + coefficient * (np.cos(angle) * tangent + np.sin(angle) * second)
                                for angle in np.linspace(0, 2 * np.pi, cone_edges, endpoint=False)]).T)
    force_map = np.zeros((3 * count, count * cone_edges))
    normal_map = np.zeros((len(group_ids), count * cone_edges))
    for index, block in enumerate(blocks):
        force_map[3 * index:3 * index + 3, cone_edges * index:cone_edges * (index + 1)] = block
        group_row = int(np.flatnonzero(group_ids == groups[index])[0])
        normal_map[group_row, cone_edges * index:cone_edges * (index + 1)] = 1.
    inequality = [normal_map, -normal_map]
    bounds = [group_limits, -group_limits * minimum_normal_fraction]
    if torque_map is not None:
        torque_map, effort_limits = np.asarray(torque_map, dtype=float), np.asarray(effort_limits, dtype=float)
        if torque_map.ndim != 2 or torque_map.shape[1] != count * 3 or effort_limits.shape != (torque_map.shape[0],):
            raise ValueError('Invalid joint effort map/limits')
        if not np.all(np.isfinite(torque_map)) or np.any(effort_limits <= 0) or np.any(np.isnan(effort_limits)):
            raise ValueError('Nonfinite torque map or invalid effort limits')
        finite = np.isfinite(effort_limits)
        actuator_map = torque_map[finite] @ force_map
        inequality.extend((actuator_map, -actuator_map))
        bounds.extend((effort_limits[finite], effort_limits[finite]))
    inequality = np.column_stack((np.vstack(inequality), np.zeros(sum(len(bound) for bound in bounds))))
    bounds = np.concatenate(bounds)
    scenarios = [points]
    if covariances is not None:
        covariances = np.asarray(covariances, dtype=float)
        if covariances.shape != (count, 3, 3) or not np.all(np.isfinite(covariances)):
            raise ValueError('Expected finite covariance per contact')
        for group in group_ids:
            members = np.flatnonzero(groups == group)
            covariance = covariances[members[0]]
            if not all(np.allclose(covariances[index], covariance, rtol=0., atol=1e-15) for index in members):
                raise ValueError('Footprint sites in one contact group must share covariance')
            if not np.allclose(covariance, covariance.T, atol=1e-12):
                raise ValueError('Covariance must be symmetric')
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            if eigenvalues.min() < -1e-12:
                raise ValueError('Covariance must be positive semidefinite')
            if check_uncertainty:
                for axis in np.flatnonzero(eigenvalues > 1e-16):
                    for sign in (-1., 1.):
                        scenario = points.copy()
                        scenario[members] += sign * sigma * np.sqrt(eigenvalues[axis]) * eigenvectors[:, axis]
                        scenarios.append(scenario)
    minimum_scale, worst_residual, witness, nominal_rank = np.inf, 0., None, None
    solver_statuses = set()
    gravity_supported = True
    gravity_failures = []
    gravity_checks = []
    solve_count = 0
    per_load = [dict(external_wrench=load.tolist(), minimum_load_scale=float('inf'), scenario=None)
                for load in loads]
    for scenario_index, scenario in enumerate(scenarios):
        wrench_map = np.zeros((6, count * cone_edges))
        for index, block in enumerate(blocks):
            wrench_map[:3, index * cone_edges:(index + 1) * cone_edges] = block
            wrench_map[3:, index * cone_edges:(index + 1) * cone_edges] = np.cross(scenario[index] - center, block.T).T / length_scale
        if scenario_index == 0:
            nominal_rank = int(np.linalg.matrix_rank(wrench_map))
        scaled_gravity = gravity.copy()
        scaled_gravity[3:] /= length_scale
        if gravity_wrench is not None:
            support = linprog(np.full(count * cone_edges, 1e-8), A_ub=inequality[:, :-1],
                              b_ub=bounds, A_eq=wrench_map, b_eq=-scaled_gravity,
                              bounds=[(0., None)] * (count * cone_edges), method='highs')
            solve_count += 1
            solver_statuses.add(int(support.status))
            gravity_checks.append(dict(scenario=scenario_index, feasible=bool(support.success),
                                       solver_status=int(support.status),
                                       contact_forces=None if not support.success else (force_map @ support.x).reshape(-1, 3).tolist()))
            if not support.success:
                gravity_supported = False
                gravity_failures.append(dict(scenario=scenario_index, solver_status=int(support.status)))
                # Disturbance support is undefined if the unperturbed lift
                # load is infeasible. A helping disturbance must not masquerade
                # as reserve available while maintaining gravity equilibrium.
                for item in per_load:
                    item.update(minimum_load_scale=0., scenario=scenario_index)
                minimum_scale = 0.
                worst_residual = float('inf')
                witness = dict(scenario=scenario_index, reason='gravity_not_supported',
                               external_wrench=None, contact_forces=None, normal_forces=None)
                continue
        for load_index, load in enumerate(loads):
            scaled_load = load.copy()
            scaled_load[3:] /= length_scale
            equality = np.column_stack((wrench_map, scaled_load))
            objective = np.r_[np.full(count * cone_edges, 1e-8), -1.]
            solution = linprog(objective, A_ub=inequality, b_ub=bounds, A_eq=equality,
                               b_eq=-scaled_gravity, bounds=[(0., None)] * (count * cone_edges) + [(0., load_scale_cap)],
                               method='highs')
            solve_count += 1
            solver_statuses.add(int(solution.status))
            scale = float(solution.x[-1]) if solution.success else 0.
            residual = float(np.max(np.abs(equality @ solution.x + scaled_gravity))) if solution.success else float('inf')
            worst_residual = max(worst_residual, residual)
            if scale < per_load[load_index]['minimum_load_scale']:
                per_load[load_index].update(minimum_load_scale=scale, scenario=scenario_index)
            if scale < minimum_scale:
                minimum_scale = scale
                witness = dict(scenario=scenario_index, external_wrench=load.tolist(),
                               contact_forces=None if not solution.success else (force_map @ solution.x[:-1]).reshape(-1, 3).tolist(),
                               normal_forces=None if not solution.success else (normal_map @ solution.x[:-1]).tolist())
    return dict(feasible=bool(gravity_supported and minimum_scale >= 1. - 1e-6 and worst_residual <= 1e-6),
                minimum_load_scale=float(max(0., minimum_scale)), scale_cap=float(load_scale_cap),
                normalized_residual=worst_residual if np.isfinite(worst_residual) else None,
                nominal_wrench_rank=nominal_rank, full_force_closure_certified=False,
                reference=center.tolist(), external_wrenches=loads.tolist(), robust_friction=robust_friction.tolist(),
                max_normal_force=group_limits.tolist(), contact_groups=groups.tolist(),
                normal_force_scope='shared sum over footprint sites within each contact group',
                uncertainty_scenarios=len(scenarios),
                uncertainty_scope='one-patch principal-axis scenarios; not joint uncertainty certification',
                solver_statuses=sorted(solver_statuses), worst_witness=witness,
                gravity_feasible=gravity_supported, gravity_wrench=gravity.tolist(),
                gravity_failure_scenarios=gravity_failures, gravity_checks=gravity_checks,
                lp_solve_count=solve_count, per_load=per_load,
                load_semantics='fixed gravity plus scaled disturbance' if gravity_wrench is not None else 'scaled entire external wrench',
                force_semantics='hypothetical LP forces at the reported load scale, not measurements',
                effort_semantics='quasistatic contact effort; hand own gravity/dynamics reserve unavailable')

def task_wrench_margin(points, normals, *, reference, mass_range_kg=(.01, .05),
                       friction=.3, normal_uncertainty=.1, covariances=None,
                       max_normal_force=2., torque_map=None, effort_limits=None,
                       gravity_m_s2=(0., 0., -9.81), lift_acceleration_m_s2=(0., 0., .5),
                       lateral_force_n=.05, torque_nm=.0005, sigma=2.,
                       cone_edges=8, remove_third=True, load_scale_cap=100., pad_radius_m=0.):
    """Common, explicitly conditional lift/rotation reserve for any finger group.

    The COM is *assumed* to be ``reference`` (a public geometry estimate), never
    read from simulation. Both mass endpoints are tested. Force/torque tests use
    that same reference and world/root axes for every candidate. Contact normal
    bounds shrink the friction cones; supplied metric covariance adds the same
    one-contact principal-axis scenarios as friction_wrench_feasibility.

    ``lift_feasible`` requires weight plus commanded lift acceleration and the
    specified horizontal perturbation. ``task_feasible`` additionally requires
    all three torque axes. These are separate: two ideal point contacts cannot
    resist torque about their connecting line, even if they can lift. No soft
    finger torsion or measured contact force/pressure is inferred.
    Missing force/mass/friction calibration is made explicit in the result.
    An optional finite pad is represented by four rim sites sharing one total
    normal-force bound. Moments arise only from distributed Coulomb forces,
    with no independent torsion parameter. Its radius must come from static
    pad calibration and is conditional on that footprint actually contacting.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or normals.shape != points.shape:
        raise ValueError('Need two or more contact point/normal pairs')
    radii = np.broadcast_to(pad_radius_m, (len(points),)).astype(float)
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError('pad_radius_m must contain finite nonnegative radii')
    sites, site_normals, group_indices = [], [], []
    for index, (point, normal, radius) in enumerate(zip(points, normals, radii)):
        unit_normal = _unit(normal)
        tangent, second = _tangents(unit_normal, (np.array([0., 0., 1.]),))
        offsets = [np.zeros(3)] if radius == 0 else [radius * tangent, -radius * tangent,
                                                               radius * second, -radius * second]
        for offset in offsets:
            sites.append(point + offset)
            site_normals.append(unit_normal)
            group_indices.append(index)
    sites, site_normals = np.array(sites), np.array(site_normals)
    group_indices = np.array(group_indices, dtype=int)
    site_friction = np.broadcast_to(friction, (len(points),))[group_indices]
    site_normal_uncertainty = np.broadcast_to(normal_uncertainty, (len(points),))[group_indices]
    site_force_limits = np.broadcast_to(max_normal_force, (len(points),))[group_indices]
    if covariances is not None:
        covariances = np.asarray(covariances)
        if covariances.shape != (len(points), 3, 3):
            raise ValueError('Expected one covariance per original pad contact')
    site_covariances = None if covariances is None else covariances[group_indices]
    site_torque_map = None
    if torque_map is not None:
        torque_map = np.asarray(torque_map, float)
        if torque_map.ndim != 2 or torque_map.shape[1] != 3 * len(points):
            raise ValueError('torque_map must contain three force columns per original contact')
        site_torque_map = np.column_stack([torque_map[:, 3 * index:3 * index + 3] for index in group_indices])
    center = _vector(reference)
    masses = np.asarray(mass_range_kg, float)
    if (masses.shape != (2,) or not np.all(np.isfinite(masses)) or
            masses[0] <= 0 or masses[1] < masses[0]):
        raise ValueError('mass_range_kg must contain ordered positive endpoints')
    if not np.isfinite(lateral_force_n) or not np.isfinite(torque_nm) or lateral_force_n <= 0 or torque_nm <= 0:
        raise ValueError('Task force and torque disturbances must be positive')
    gravity = _vector(gravity_m_s2)
    acceleration = _vector(lift_acceleration_m_s2)
    # D'Alembert load in the accelerating hand frame: m * (g - a).
    effective_gravity = gravity - acceleration
    if np.linalg.norm(effective_gravity) < 1e-9:
        raise ValueError('Lift load must have nonzero effective gravity')
    force_axes = np.eye(3)[:2] * lateral_force_n
    force_loads = np.c_[np.vstack((force_axes, -force_axes)), np.zeros((4, 3))]
    torque_axes = np.eye(3) * torque_nm
    torque_loads = np.c_[np.zeros((6, 3)), np.vstack((torque_axes, -torque_axes))]
    loads = np.vstack((force_loads, torque_loads))
    names = ['force_x+', 'force_y+', 'force_x-', 'force_y-',
             'torque_x+', 'torque_y+', 'torque_z+', 'torque_x-', 'torque_y-', 'torque_z-']
    trials = []
    for mass in np.unique(masses):
        result = friction_wrench_feasibility(
            sites, site_normals, friction=site_friction, normal_uncertainty=site_normal_uncertainty,
            covariances=site_covariances, sigma=sigma, max_normal_force=site_force_limits,
            external_wrenches=loads, reference=center, cone_edges=cone_edges,
            torque_map=site_torque_map, effort_limits=effort_limits,
            gravity_wrench=np.r_[mass * effective_gravity, np.zeros(3)],
            minimum_normal_fraction=0., load_scale_cap=load_scale_cap, contact_groups=group_indices)
        trials.append(dict(mass_kg=float(mass), **result))
    per_load = []
    for index, name in enumerate(names):
        records = [dict(mass_kg=trial['mass_kg'], **trial['per_load'][index])
                   for trial in trials if 'per_load' in trial]
        worst = min(records, key=lambda item: item['minimum_load_scale']) if records else dict(minimum_load_scale=0.)
        per_load.append(dict(name=name, **worst))
    lateral_margin = min(item['minimum_load_scale'] for item in per_load[:4])
    torque_margin = min(item['minimum_load_scale'] for item in per_load[4:])
    gravity_ok = all(trial.get('gravity_feasible', False) for trial in trials)
    result = dict(
        gravity_feasible=gravity_ok,
        lift_feasible=bool(gravity_ok and lateral_margin >= 1. - 1e-6),
        task_feasible=bool(gravity_ok and min(lateral_margin, torque_margin) >= 1. - 1e-6),
        minimum_task_margin=float(min(lateral_margin, torque_margin)),
        lateral_margin=float(lateral_margin), torque_margin=float(torque_margin),
        margin_scale_cap=float(load_scale_cap),
        reference=center.tolist(), contact_points=points.tolist(), outward_normals=normals.tolist(),
        footprint_points=sites.tolist(), footprint_contact_groups=group_indices.tolist(),
        per_load=per_load, mass_trials=trials,
        assumptions=dict(mass_range_kg=masses.tolist(), friction_lower=np.asarray(friction).tolist(),
                         normal_uncertainty_rad=np.asarray(normal_uncertainty).tolist(),
                         max_normal_force_n=np.asarray(max_normal_force).tolist(),
                         gravity_m_s2=gravity.tolist(), lift_acceleration_m_s2=acceleration.tolist(),
                         horizontal_disturbance_n=float(lateral_force_n), torque_disturbance_nm=float(torque_nm),
                         com='assumed at common public geometry reference',
                         calibration='mass, friction, normal-force limits and acceleration are declared assumptions, not measurements',
                         pad_radius_m=radii.tolist(),
                         force_model='four rim Coulomb sites per finite pad; shared per-pad normal-force sum; no intrinsic torsion' if np.any(radii) else 'bounded point contacts; no pad torsional friction',
                         footprint='assumed planar contacting footprint within static calibrated pad; not observed contact area',
                         jacobian='center-contact Jacobian repeated at rim sites; local kinematic approximation',
                         effort='URDF incremental contact effort only; actuator gravity/dynamics reserve unavailable',
                         uncertainty='normal bound plus one-contact covariance axes; not a full joint uncertainty certificate'),
        input_scope='public contact geometry, candidate-q Jacobians, static hand limits; no object truth',
        ranking_scope='feasibility gates first; compare common task margin and lateral/torque terms; retain uncertainty caveats')
    if remove_third and len(points) == 3:
        arguments = dict(reference=center, mass_range_kg=masses, gravity_m_s2=gravity,
                         lift_acceleration_m_s2=acceleration, lateral_force_n=lateral_force_n,
                         torque_nm=torque_nm, sigma=sigma, cone_edges=cone_edges, remove_third=False,
                         load_scale_cap=load_scale_cap, pad_radius_m=radii[:2])
        for key, value in [('friction', friction), ('normal_uncertainty', normal_uncertainty),
                           ('max_normal_force', max_normal_force)]:
            arguments[key] = np.broadcast_to(value, (3,))[:2]
        if covariances is not None:
            arguments['covariances'] = np.asarray(covariances)[:2]
        if torque_map is not None:
            arguments['torque_map'] = np.asarray(torque_map)[:, :6]
            arguments['effort_limits'] = effort_limits
        baseline = task_wrench_margin(points[:2], normals[:2], **arguments)
        result['without_third_contact'] = {key: baseline[key] for key in
                                           ('gravity_feasible', 'lift_feasible', 'task_feasible',
                                            'minimum_task_margin', 'lateral_margin', 'torque_margin')}
        result['third_contact_increment'] = {key: result[key] - baseline[key] for key in
                                             ('minimum_task_margin', 'lateral_margin', 'torque_margin')}
    return result

