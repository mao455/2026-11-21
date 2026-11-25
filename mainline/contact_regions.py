"""Contact search on the public RGB envelope; no object ground truth inputs.

An overhead silhouette supplies an extruded, convex *hypothesis*, not measured
side geometry. Live RGB refinement can replace ``collision_points`` with a 3-D
envelope. Every returned point is on that envelope, before any IK stand-off.
"""

from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


def rgb_facet_height_bounds(cloud, facet_plane_tolerance_m=2e-7):
    """Upper bounds on the actual SVD-projected IK facet, before radius inset.

    Mirror both SVD stages in solve_surface_ik/finite_support_region. Projection
    of a nearly coplanar cloud can exceed its raw max_z. The pre-inset polygon
    contains every positive-radius inset, so bounds do not depend on pad choice.
    Unknown or degenerate projections remain searchable.
    """
    cloud = np.asarray(cloud, float)
    planes = ConvexHull(cloud).equations
    maxima = []
    for plane in planes:
        maximum = None
        try:
            on = cloud[np.abs(cloud @ plane[:3] + plane[3]) < facet_plane_tolerance_m]
            on = np.unique(np.round(on, 10), axis=0)
            if len(on) < 3:
                raise ValueError('RGB facet has fewer than three surface vertices')
            center = on.mean(0)
            _, _, basis = np.linalg.svd(on-center, full_matrices=False)
            on = on[ConvexHull((on-center) @ basis[:2].T).vertices]
            # finite_support_region recomputes its origin and basis after the
            # first hull selection. These frames must match exactly.
            origin = on.mean(0)
            _, _, basis = np.linalg.svd(on-origin, full_matrices=False)
            basis = basis[:2]
            polygon = (on-origin) @ basis.T
            hull = ConvexHull(polygon)
            projected = polygon @ basis + origin
            maximum = float(projected[:, 2].max())
            # finite_support_region accepts inset intersections with <=1e-10
            # halfspace slack and rounds 2-D coordinates to 12 decimals.
            # A homothetic expansion about an interior point bounds this slack
            # even for acute corners; adding a fixed world-z epsilon does not.
            interior = polygon[hull.vertices].mean(0)
            slack = float(np.min(-(hull.equations[:, :2] @ interior + hull.equations[:, 2])))
            if not np.isfinite(slack) or slack <= 0:
                raise ValueError('No certified interior point')
            center_z = float(interior @ basis[:, 2] + origin[2])
            maximum += 1e-10 / slack * max(0., maximum-center_z)
            # Include the same radius-independent LP fallback point used if
            # finite_support_region finds no pairwise inset intersections.
            fit = linprog([0., 0., -1.],
                A_ub=np.c_[hull.equations[:, :2], np.ones(len(hull.equations))],
                b_ub=-hull.equations[:, 2],
                bounds=[(None,None), (None,None), (0.,None)], method='highs')
            if not fit.success:
                raise ValueError('Uncertified finite support fallback')
            maximum = max(maximum, float(fit.x[:2] @ basis[:, 2] + origin[2]))
            maximum += 5e-13 * float(np.abs(basis[:, 2]).sum())
            maximum += 64*np.finfo(float).eps * max(1., float(np.abs(projected).max()))
            maximum = float(np.nextafter(maximum, np.inf))
        except Exception:
            maximum = None
        maxima.append(maximum)
    return planes, maxima


def necessary_table_witnesses(facet_bounds, outward_normals, pad_hull_distances,
                              table_z, position_limit_m, table_clearance_m,
                              standoff_m):
    """Prove conflict with EXISTING position/table gates; unknown stays eligible.

    A contact target is a convex combination on its finite RGB face plus the
    outward standoff. A pad at distance <=delta from its same-link mesh hull
    inherits mesh clearance minus delta under every rigid transform. Therefore
    upper_target_z + position_limit < table_z + clearance - delta is impossible.
    This uses the full facet max_z, never its start centroid or normal sign.
    """
    planes, maxima = facet_bounds
    witnesses = []
    for i, (normal, distance) in enumerate(zip(outward_normals, pad_hull_distances)):
        maximum = maxima[int(np.argmax(planes[:, :3] @ normal))]
        if maximum is None or distance is None:
            continue
        upper = maximum + float(normal[2]) * standoff_m
        lower = table_z + table_clearance_m - distance
        if upper + position_limit_m < lower:
            witnesses.append(dict(contact=i, facet_max_z_m=maximum,
                                  target_upper_z_m=upper, pad_lower_z_m=lower,
                                  pad_hull_distance_bound_m=distance,
                                  contradiction_gap_m=lower-upper-position_limit_m))
    return witnesses


def finite_support_region(vertices, pad_radius_m):
    """Inset an RGB facet by its usable finite contact radius.

    Use the static pad radius where it fits. On a narrower facet use its
    largest inscribed disc and return that smaller radius for wrench scoring.
    This describes envelope support, not a measured physical contact patch.
    """
    vertices = np.asarray(vertices, dtype=float)
    origin = vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(vertices - origin, full_matrices=False)
    basis = basis[:2]
    polygon = (vertices - origin) @ basis.T
    planes = ConvexHull(polygon).equations
    fit = linprog([0., 0., -1.], A_ub=np.c_[planes[:, :2], np.ones(len(planes))],
                  b_ub=-planes[:, 2], bounds=[(None, None), (None, None), (0., None)],
                  method='highs')
    if not fit.success or fit.x[2] <= 0:
        raise ValueError('RGB facet has no positive finite support area')
    inradius = float(fit.x[2])
    radius = min(float(pad_radius_m), inradius)
    if radius <= 0:
        raise ValueError('Static contact radius must be positive')
    offsets = planes[:, 2] + radius
    points = []
    for a, b in combinations(range(len(planes)), 2):
        matrix = planes[[a, b], :2]
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        point = np.linalg.solve(matrix, -offsets[[a, b]])
        if np.max(planes[:, :2] @ point + offsets) <= 1e-10:
            points.append(point)
    if not points:
        points = [fit.x[:2]]
    points = np.unique(np.round(points, 12), axis=0)
    return points @ basis + origin, inradius, radius


def _section(hull, cloud, z):
    """Interior ray origin in the horizontal convex-hull cross section."""
    vertices = []
    for face in hull.simplices:
        for a, b in combinations(face, 2):
            p, q = cloud[a], cloud[b]
            if abs(p[2] - z) < 1e-10:
                vertices.append(p[:2])
            if (p[2] - z) * (q[2] - z) < 0:
                vertices.append((p + (q - p) * (z - p[2]) / (q[2] - p[2]))[:2])
    if len(vertices) < 3:
        return None
    xy = np.unique(np.round(vertices, 12), axis=0)
    if len(xy) < 3:
        return None
    polygon = xy[ConvexHull(xy).vertices]
    return np.r_[polygon.mean(axis=0), z], polygon


def _hit(planes, origin, direction):
    denominators = planes[:, :3] @ direction
    eligible = np.flatnonzero(denominators > 1e-10)
    distances = -(planes[eligible, :3] @ origin + planes[eligible, 3]) / denominators[eligible]
    valid = distances >= -1e-10
    if not valid.any():
        return None
    eligible, distances = eligible[valid], distances[valid]
    k = int(np.argmin(distances))
    facet = int(eligible[k])
    return origin + max(0., float(distances[k])) * direction, planes[facet, :3], facet


def _horizontal_contact_sets(info, count, axes=None, heights=None, max_sets=96):
    """Return deterministic two- or three-contact surface hypotheses.

    ``points`` and ``outward_normals`` are Nx3 arrays, ordered around the
    envelope; they are NOT named fingers. The caller assigns every selected
    digit to every contact permutation and lets hand reachability choose.
    Two-contact chords use exact facet intersections. Three-contact sets are
    generated as independent surrounding rays and independently sampled
    heights, rather than augmenting a chosen two-contact grasp.
    """
    if count not in (2, 3):
        raise ValueError('contact count must be two or three')
    if max_sets <= 0:
        return []
    cloud = np.asarray(info['collision_points'], dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
        raise ValueError('collision_points must be finite Nx3 RGB-envelope vertices')
    hull = ConvexHull(cloud)
    bottom = max(float(cloud[:, 2].min()), float(info.get('table_z', cloud[:, 2].min())))
    top = float(cloud[:, 2].max())
    if top <= bottom:
        return []
    fractions = tuple(float(np.clip(h, .05, .95)) for h in (heights if heights is not None else (.6, .35, .8)))
    if not fractions:
        return []
    # Uniform headings ensure a near-round silhouette never fixes world X.
    # Input headings and actual observed side normals augment this coverage.
    headings = list(np.arange(8) * np.pi / 4)
    for axis in (() if axes is None else axes):
        headings.append(float(np.arctan2(axis[1], axis[0])))
    for plane in hull.equations:
        if np.linalg.norm(plane[:2]) > .5:
            headings.append(float(np.arctan2(plane[1], plane[0])))
    headings = list(dict.fromkeys(round(float(a % (2 * np.pi)), 10) for a in headings))
    sections = {}

    def section(fraction):
        if fraction not in sections:
            sections[fraction] = _section(hull, cloud, bottom + fraction * (top - bottom))
        return sections[fraction]

    # Outer loop interleaves directions before spending budget on additional
    # heights/shapes, avoiding a first-axis search monopoly.
    if count == 2:
        families = [(np.array([0., np.pi]), (h, h), offset)
                    for h in fractions for offset in (0., -.25, .25)]
    else:
        angle_families = (np.deg2rad([0, 120, 240]),
                          np.deg2rad([0, 90, 225]),
                          np.deg2rad([0, 100, 200]))
        height_families = [(h, h, h) for h in fractions]
        if len(fractions) > 1:
            height_families += [tuple(fractions[(k + j) % len(fractions)] for j in range(3))
                                for k in range(min(len(fractions), 3))]
        families = [(angles, hs, 0.) for hs in height_families for angles in angle_families]
    output, seen = [], set()
    for angles, hs, offset in families:
        for heading in headings:
            hits = []
            for angle, h in zip(angles + heading, hs):
                cross_section = section(h)
                if cross_section is None:
                    break
                origin, polygon = cross_section
                origin = origin.copy()
                direction = np.array([np.cos(angle), np.sin(angle), 0.])
                if offset:
                    tangent = np.array([-np.sin(heading), np.cos(heading), 0.])
                    extent = np.ptp(polygon @ tangent[:2])
                    origin += tangent * offset * extent
                    if np.any(hull.equations[:, :3] @ origin + hull.equations[:, 3] > 1e-9):
                        break
                hit = _hit(hull.equations, origin, direction)
                if hit is None:
                    break
                hits.append(hit)
            if len(hits) != count:
                continue
            points = np.array([hit[0] for hit in hits])
            normals = np.array([hit[1] for hit in hits])
            if np.min(np.linalg.norm(points[:, None] - points[None, :] + np.eye(count)[:, :, None], axis=2)) < 1e-6:
                continue
            # Distinct points on a shared facet can add moment support. Keep
            # them in this independent-ray stream; common force, IK and full
            # path checks decide whether a hand can execute the contact set.
            identity = tuple(sorted(tuple(np.round(np.r_[p, n], 8)) for p, n in zip(points, normals)))
            if identity in seen:
                continue
            seen.add(identity)
            output.append(dict(points=points, outward_normals=normals,
                               height_fractions=list(hs), heading_rad=float(heading),
                               facet_indices=[hit[2] for hit in hits], chord_offset_fraction=float(offset),
                               geometry_source='public_RGB_convex_envelope_surface_intersections',
                               provenance=dict(input='info.collision_points from RGB',
                                               height='RGB envelope height; initial overhead uses extrusion prior',
                                               visibility='side surface hypothesis, not certified observed surface',
                                               point_standoff_m=0., independent_contact_count=count)))
            if len(output) >= max_sets:
                return output
    return output


def _rgb_finite_facets(cloud):
    hull = ConvexHull(cloud)
    merged = {}
    for equation, simplex in zip(hull.equations, hull.simplices):
        key = tuple(np.round(equation, 9))
        if key not in merged:
            merged[key] = dict(equation=equation, vertex_ids=set())
        merged[key]['vertex_ids'].update(int(i) for i in simplex)
    facets = []
    for key, row in sorted(merged.items()):
        vertices = cloud[sorted(row['vertex_ids'])]
        # The mean of polygon vertices lies in its finite relative interior.
        facets.append(dict(point=vertices.mean(axis=0), normal=row['equation'][:3],
                           plane_identity=key))
    return facets


def _balanced_facet_combinations(facets, count):
    """Cover every finite face first, then interleave complete combination streams.

    The order schedules search only; no contact index denotes a named finger,
    an opposition pair, or a designated third contact. No ray samples repeat
    a plane tuple. The caller applies all digit assignments and force checks.
    """
    size = len(facets)
    if size < count:
        return
    normals = np.array([f['normal'] for f in facets])
    # Start above and then visit the most distinct remaining normals. This
    # spreads top/bottom/side orientations without a hand/shape identifier.
    order = [int(np.argmax(normals[:, 2]))]
    available = set(range(size)) - set(order)
    while available:
        nxt = min(available, key=lambda i: (max(float(normals[i] @ normals[j])
                                              for j in order), i))
        order.append(nxt)
        available.remove(nxt)
    seen = set()
    # Each face anchors a complete independently chosen contact set before
    # extra tuples. Angular diversity only schedules starts; equal diversity
    # prefers greater RGB height above the table before exposure-count ties.
    # This affects order only, not face eligibility or any physical gate.
    uses = np.zeros(size, dtype=int)
    for anchor in order:
        chosen = [anchor]
        while len(chosen) < count:
            partner = min((i for i in range(size) if i not in chosen),
                          key=lambda i: (max(float(normals[i] @ normals[j])
                                             for j in chosen), -float(facets[i]['point'][2]), uses[i], i))
            chosen.append(partner)
        ids = tuple(sorted(chosen))
        if ids not in seen:
            seen.add(ids)
            uses[list(ids)] += 1
            yield ids
    streams = [(lead, iter(combinations(range(lead + 1, size), count - 1)))
               for lead in range(size - count + 1)]
    while streams:
        active = []
        for lead, stream in streams:
            try:
                tail = next(stream)
            except StopIteration:
                continue
            active.append((lead, stream))
            ids = tuple(sorted(order[i] for i in (lead,) + tail))
            if ids not in seen:
                seen.add(ids)
                yield ids
        streams = active


def generate_contact_sets(info, count, axes=None, heights=None, max_sets=96):
    """Interleave independent finite RGB facets and preserved side hypotheses.

    Each new facet tuple appears once. All old ray starts remain available
    for seed scoring; IK independently slides over each complete finite facet.
    Bottom/top/slanted facets remain hypotheses and receive the same downstream
    force, reach, table, self-collision and path checks as side facets.
    """
    if heights is not None:
        heights = tuple(heights)
    horizontal = _horizontal_contact_sets(info, count, axes, heights, max_sets)
    if max_sets <= 0 or heights == ():
        return []
    # Preserve every legacy start. The caller scores all heights before
    # ordered-plane deduplication, retaining its best old initial seed.
    old = horizontal
    for region in old:
        region['provenance']['family'] = 'legacy_RGB_horizontal'
    cloud = np.asarray(info['collision_points'], dtype=float)
    facets = _rgb_finite_facets(cloud)
    bottom = max(float(cloud[:, 2].min()), float(info.get('table_z', cloud[:, 2].min())))
    height = float(cloud[:, 2].max()) - bottom
    if height <= 0:
        return []
    added = []
    # Two new plane combinations per finite face is a computation allocation,
    # not a geometry gate or calibration. It covers facets early without
    # repeating thousands of ray origins/heights for the same plane tuple.
    new_budget = min(max_sets, 2 * len(facets))
    for ids in _balanced_facet_combinations(facets, count):
        points = np.array([facets[i]['point'] for i in ids])
        normals = np.array([facets[i]['normal'] for i in ids])
        if count == 3 and max(float(normals[i] @ normals[j])
                              for i, j in combinations(range(count), 2)) > .995:
            continue
        region = dict(points=points, outward_normals=normals,
                      height_fractions=((points[:, 2] - bottom) / height).tolist(),
                      facet_indices=list(ids), heading_rad=float(np.arctan2(normals[0, 1], normals[0, 0])),
                      chord_offset_fraction=None,
                      geometry_source='public_RGB_independent_finite_facets',
                      provenance=dict(input='info.collision_points from RGB',
                                      height='independent finite RGB facet interior starts',
                                      visibility='RGB envelope hypotheses; facet visibility not certified',
                                      point_standoff_m=0., independent_contact_count=count,
                                      family='independent_RGB_facets'))
        added.append(region)
        if len(added) >= new_budget:
            break
    output, seen = [], set()
    for index in range(max(len(old), len(added))):
        for family in (added, old):
            if index >= len(family):
                continue
            region = family[index]
            key = tuple(sorted(tuple(np.round(np.r_[p, n], 9))
                               for p, n in zip(region['points'], region['outward_normals'])))
            if key in seen:
                continue
            seen.add(key)
            output.append(region)
            if len(output) >= max_sets:
                return output
    return output


def interleave_surface_families(variants):
    """Keep common seed-score order within each family under one shared budget.

    Call after optimistic-force pruning and ordered-plane deduplication. A
    family cannot monopolize the first IK slice merely through a seed bias.
    All variants retain the same downstream IK, force, path and task gates.
    """
    families = {}
    for variant in variants:
        family = variant[5]['provenance'].get('family', 'legacy_RGB_horizontal')
        families.setdefault(family, []).append(variant)
    order = sorted(families, key=lambda name: (name != 'independent_RGB_facets', name))
    return [families[name][index]
            for index in range(max((len(v) for v in families.values()), default=0))
            for name in order if index < len(families[name])]


def sort_task_candidates(candidates, tolerance=.1):
    """Rank the SAME task load for every digit count, with a hard margin gate.

    A nominal-minus-robust difference is not a confidence interval. It cannot
    turn a task-failing 0.8 into a tie with a task-passing 1.2. Robust margin is
    monotonic within each class; motion breaks numerical margin ties only.
    ``tolerance`` remains recorded for compatibility, not as permission to
    cross the task boundary or erase a measured margin difference.
    """
    values = []
    for index, candidate in enumerate(candidates):
        meta = candidate['meta']
        task = meta['task_quality']
        robust = float(task['robust']['minimum_task_margin'])
        nominal = float(task['nominal']['minimum_task_margin'])
        if not np.isfinite(robust):
            robust = -float('inf')
        gravity = bool(task['robust'].get('gravity_feasible', task['nominal'].get('gravity_feasible', False)))
        meets_task = gravity and robust >= 1.
        terms = meta.get('quality_terms', {})
        # Twelve decimals eliminate solver round-off ties without grouping
        # meaningful differences in reserve under an uncertainty-width rule.
        key = (not meets_task, not gravity, -round(robust, 12),
               float(terms.get('planned_motion_s', float('inf'))),
               -float(terms.get('min_table_clearance_m', 0.)),
               float(meta.get('quality_score', 0.)), index)
        values.append((key, candidate, robust, nominal, gravity, meets_task))
    two_margins = [r for _, c, r, _, g, _ in values if g and len(c.get('points', ())) == 2]
    best_two = max(two_margins, default=None)
    ordered = []
    for _, candidate, robust, nominal, gravity, meets_task in sorted(values, key=lambda item: item[0]):
        candidate['meta']['rank_reason'] = dict(
            rule='robust task >=1; robust gravity; monotonic robust reserve; motion for numerical ties',
            load_gate='robust_task_supported' if meets_task else 'conditional_task_not_met',
            robust_task_margin=robust, nominal_task_margin=nominal,
            gravity_feasible=gravity, task_requirement_margin=1.,
            supplied_legacy_tolerance=float(tolerance),
            best_two_contact_margin=best_two,
            third_contact_margin_increment=(robust - best_two if best_two is not None and len(candidate.get('points', ())) == 3 else None),
            shared_load_requirement='caller task_quality must use identical external loads for every contact count')
        ordered.append(candidate)
    return ordered


def force_support_possible(normals, *, friction, normal_uncertainty, max_normal_force, required_force, cone_edges=16):
    """Necessary force balance under the shared static contact assumptions.

    A circumscribed friction pyramid contains the circular Coulomb cone, so
    this optimistic LP may admit impossible grasps. Only proven infeasibility
    prunes a surface assignment. Torque, footprint, joint effort and covariance
    remain the responsibility of the full task score after contact IK.
    """
    from scipy.optimize import linprog
    normals=np.asarray(normals,float)
    if normals.ndim!=2 or normals.shape[1]!=3 or len(normals)<2:
        raise ValueError('Expected at least two finite contact normals')
    if np.any(np.linalg.norm(normals,axis=1)<1e-12):raise ValueError('Zero contact normal')
    count=len(normals)
    mu=np.broadcast_to(friction,(count,)).astype(float)
    uncertainty=np.broadcast_to(normal_uncertainty,(count,)).astype(float)
    caps=np.broadcast_to(max_normal_force,(count,)).astype(float)
    force=np.asarray(required_force,float)
    if (not np.isfinite(np.r_[normals.ravel(),mu,uncertainty,caps,force]).all()
            or np.any(mu<0) or np.any(uncertainty<0) or np.any(caps<=0)
            or force.shape!=(3,) or cone_edges<4):
        raise ValueError('Invalid force-balance calibration or task load')
    angle=np.arctan(mu)-uncertainty
    if np.any(angle<=0):return dict(possible=False,reason='normal_uncertainty_erases_robust_friction_cone')
    blocks=[]
    for normal,coefficient in zip(normals,np.tan(angle)/np.cos(np.pi/cone_edges)):
        inward=-normal/np.linalg.norm(normal)
        axis=np.eye(3)[np.argmin(abs(inward))]
        tangent=np.cross(inward,axis);tangent/=np.linalg.norm(tangent)
        second=np.cross(inward,tangent)
        blocks.append(np.array([inward+coefficient*(np.cos(t)*tangent+np.sin(t)*second)
            for t in np.arange(cone_edges)*2*np.pi/cone_edges]).T)
    matrix=np.concatenate(blocks,axis=1);normal_map=np.zeros((count,count*cone_edges))
    for i in range(count):normal_map[i,i*cone_edges:(i+1)*cone_edges]=1.
    result=linprog(np.zeros(count*cone_edges),A_ub=normal_map,b_ub=caps,
        A_eq=matrix,b_eq=force,bounds=(0,None),method='highs')
    return dict(possible=result.status!=2,solver_status=int(result.status),
        reason='optimistic force balance; passing is not task or path proof',
        cone='circumscribed outer approximation',required_force_n=force.tolist())


def region_surface_schedule(regions, max_surfaces=12):
    """Visit each region's own ranked facets in successive shared sweeps.

    Each region has its own hand/FK-ranked surface priority. Its first visit
    must search index zero rather than skip that association by region ordinal.
    Every region/facet pair still appears once when the budget permits.
    """
    for sweep in range(max_surfaces):
        for region_number, region_ids in enumerate(regions):
            yield region_number, region_ids, sweep


# Source-face domains for the unchanged finite contact disc.
import time
import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def plane_component(triangles_link, origin_link, normal_link,
                    plane_tolerance_m=5e-6, normal_dot_min=.99999):
    """Return unchanged-pad connected source plane as an exact triangle union.

    Coplanarity tolerance describes the static input, not execution success.
    Never joins disconnected faces by convex hull and never buffers the face.
    """
    tri = np.asarray(triangles_link, float)
    origin = np.asarray(origin_link, float)
    normal = np.asarray(normal_link, float)
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    B = np.stack((u, np.cross(normal, u)))
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    mask = ((normals @ normal > normal_dot_min) &
            (np.abs((tri-origin) @ normal).max(axis=1) <= plane_tolerance_m))
    if not np.any(mask):
        return None, B
    polys = [Polygon(t) for t in (tri[mask]-origin) @ B.T]
    union = unary_union([p for p in polys if p.area > 1e-20])
    components = list(union.geoms) if hasattr(union, 'geoms') else [union]
    containing = [p for p in components if p.geom_type=='Polygon' and p.distance(Point(0., 0.)) <= 1e-10]
    return (max(containing, key=lambda p: p.area) if containing else None), B


def _convex_merge(cells, domain, max_vertices=8, rounds=4):
    """Merge adjacent triangles only when their union stays convex and covered.

    A convex hull is only a proposed merge, and is rejected if it adds any area
    or leaves the original inset domain. This cannot bridge a notch or hole.
    """
    current = list(cells)
    for _ in range(rounds):
        changed = False
        tree = shapely.STRtree(current)
        consumed = set()
        updated = []
        for i, cell in enumerate(current):
            if i in consumed:
                continue
            choices = []
            for j in tree.query(cell):
                j = int(j)
                if j <= i or j in consumed:
                    continue
                other = current[j]
                shared = cell.boundary.intersection(other.boundary).length
                if shared > 1e-12:
                    choices.append((shared, j))
            chosen = None
            for _, j in sorted(choices, reverse=True):
                union = cell.union(current[j])
                if union.geom_type!='Polygon' or len(union.interiors):
                    continue
                hull = union.convex_hull
                if len(hull.exterior.coords)-1 > max_vertices:
                    continue
                # Retain exact source/domain containment. No area-tolerance
                # convexification across source concavities or holes.
                if not domain.covers(hull) or not union.covers(hull):
                    continue
                chosen = (j, hull)
                break
            if chosen is None:
                updated.append(cell)
            else:
                consumed.add(chosen[0])
                updated.append(chosen[1])
                changed = True
        current = updated
        if not changed:
            break
    return current


def contact_center_domain(source_plane_xy, origin_link, tangent_basis_link,
                          support_radius_m, quadrant_segments=32, merge=True):
    """Return convex hand-contact-center cells without changing the pad radius.

    A center u in cell i satisfies A_i @ u <= b_i; its 3-D position is
    origin_link + tangent_basis_link.T @ u. Every point of every cell has at
    least support_radius_m clearance to the original source-region boundary,
    including holes. Input/output are static geometry only.
    """
    started = time.perf_counter()
    radius = float(support_radius_m)
    if radius <= 0. or quadrant_segments < 4:
        raise ValueError('Positive unchanged disc radius and >=4 arc segments required')
    if source_plane_xy is None or source_plane_xy.is_empty:
        return dict(cells=[], reason='no continuous source plane', build_wall_s=time.perf_counter()-started)
    if not source_plane_xy.is_valid or source_plane_xy.geom_type!='Polygon':
        raise ValueError('A valid connected source polygon with explicit holes is required')
    # GEOS round offsets use inscribed arc chords. Its short-arc segment
    # rounding can exceed the nominal angle, so an analytic starting bound is
    # followed by an exact distance-to-original-boundary check and conservative
    # corrections. The declared mechanical disc radius is unchanged.
    offset = radius / np.cos(np.pi/(4*quadrant_segments))
    corrections = []
    for _ in range(10):
        inset = source_plane_xy.buffer(-offset, quad_segs=quadrant_segments, join_style='round')
        if inset.is_empty:
            triangles = []
            break
        triangles = list(shapely.constrained_delaunay_triangles(inset).geoms)
        clearance = min(t.distance(source_plane_xy.boundary) for t in triangles)
        if clearance >= radius-1e-12 and all(source_plane_xy.covers(t) for t in triangles):
            break
        correction = max(radius-clearance, 1e-10)*1.05
        corrections.append(float(correction))
        offset += correction
    else:
        raise RuntimeError('Cannot certify conservative finite-disc inset')
    answer = dict(origin_link_m=np.asarray(origin_link, float).tolist(),
                  tangent_basis_link=np.asarray(tangent_basis_link, float).tolist(),
                  support_radius_m=radius, computational_offset_m=float(offset),
                  maximum_extra_inset_m=float(offset-radius), quadrant_segments=quadrant_segments,
                  additional_distance_proof_corrections_m=corrections,
                  source_area_m2=float(source_plane_xy.area), inset_area_m2=float(inset.area),
                  inset_components=0 if inset.is_empty else (len(inset.geoms) if hasattr(inset, 'geoms') else 1),
                  source_holes=len(source_plane_xy.interiors), cells=[])
    if inset.is_empty:
        answer.update(reason='unchanged finite disc has no center inside this continuous source plane',
                      build_wall_s=time.perf_counter()-started)
        return answer
    cells = _convex_merge(triangles, inset) if merge else triangles
    accepted = []
    for cell in cells:
        clearance = float(cell.distance(source_plane_xy.boundary))
        if not source_plane_xy.covers(cell) or clearance < radius-1e-12:
            raise RuntimeError('Finite-disc containment proof failed')
        polygon = orient(cell, sign=1.)
        uv = np.asarray(polygon.exterior.coords)[:-1]
        edges = np.roll(uv, -1, axis=0)-uv
        A = np.stack((edges[:, 1], -edges[:, 0]), axis=1)
        A /= np.linalg.norm(A, axis=1)[:, None]
        b = np.einsum('ij,ij->i', A, uv)
        accepted.append(dict(vertices_uv_m=uv.tolist(), A=A.tolist(), b_m=b.tolist(),
                             area_m2=float(polygon.area), minimum_source_boundary_clearance_m=clearance))
    union = unary_union(cells)
    if inset.symmetric_difference(union).area > max(1e-16, inset.area*1e-9):
        raise RuntimeError('Convex partition omitted or added a material area of the conservative inset')
    answer.update(cells=accepted, initial_triangle_cells=len(triangles), convex_cells=len(accepted),
                  cell_area_m2=sum(c['area_m2'] for c in accepted),
                  minimum_cell_boundary_clearance_m=min(c['minimum_source_boundary_clearance_m'] for c in accepted),
                  partition_symmetric_difference_m2=float(inset.symmetric_difference(union).area),
                  build_wall_s=time.perf_counter()-started)
    return answer


def source_contact_center_cells(triangles_link, origin_link, normal_link,
                               support_radius_m, **kwargs):
    """Minimal source-triangles -> fixed-region convex-center-domain interface."""
    component, B = plane_component(triangles_link, origin_link, normal_link)
    return contact_center_domain(component, origin_link, B, support_radius_m, **kwargs)


def planar_contact_cells(robot, patch):
    """Drop-in static proposal API; does not alter Robot contacts or patches.

    Output cells_local_m are convex N-by-3 polygons in patch['link'] frame.
    Absent trustworthy planar source geometry -> fixed_center and no cells.
    The same immutable pad tuple is cached; callers must not mutate results.
    """
    terminal = patch['link']
    surface = patch.get('surface_link', terminal)
    p = np.asarray(patch['point_m'], float)
    n = np.asarray(patch['outward_normal'], float)
    radius = float(patch['support_radius_m'])
    key = (terminal, surface, tuple(p), tuple(n), radius)
    cache = robot.__dict__.setdefault('_planar_contact_center_cache', {})
    if key in cache:
        return cache[key]
    result = dict(status='fixed_center', cells_local_m=[], original_point_local_m=p.tolist(),
                  normal_local=n.tolist(), support_radius_m=radius, finger=patch.get('finger'),
                  link=terminal, surface_link=surface, source=patch.get('source'),
                  source_geometry_class=patch.get('geometry_class'),
                  scope='static source geometry only; contact-center domain, not physical pad/material calibration')
    children = {}
    for parent, child, _, _, kind, _ in robot.description.joints:
        if kind=='fixed':
            children.setdefault(parent, []).append(child)
    queue = [terminal]
    fixed_descendants = set()
    while queue:
        link = queue.pop()
        if link in fixed_descendants:
            continue
        fixed_descendants.add(link)
        queue.extend(children.get(link, ()))
    if surface not in fixed_descendants:
        result['reason'] = 'source component is not the contact link or a fixed descendant'
    elif surface not in robot.surfaces:
        result['reason'] = 'source component has no original collision mesh'
    elif not np.isfinite(radius) or radius<=0. or not np.all(np.isfinite(p)) or not np.all(np.isfinite(n)) or np.linalg.norm(n)<1e-12:
        result['reason'] = 'invalid static contact geometry'
    else:
        desc = robot.description
        neutral = desc.fk(np.clip(np.zeros(len(desc.joint_names)), desc.lower, desc.upper))
        relative = np.linalg.inv(neutral[terminal]) @ neutral[surface]
        triangles = np.asarray(robot.surfaces[surface].triangles, float)
        triangles = triangles @ relative[:3, :3].T + relative[:3, 3]
        try:
            domain = source_contact_center_cells(triangles, p, n, radius)
            result.update(domain)
            if domain['cells']:
                B = np.asarray(domain['tangent_basis_link'])
                result.update(status='planar_cells', cells_local_m=[(p+np.asarray(c['vertices_uv_m'])@B).tolist() for c in domain['cells']],
                              reason='unchanged finite disc fits inside exact continuous source plane',
                              source_component_fixed_verified=True, source_to_contact_transform=relative.tolist())
            else:
                result.setdefault('reason', 'no conservative finite-disc center domain')
        except (ValueError, RuntimeError) as error:
            result['reason'] = 'static domain unavailable: '+str(error)
    cache[key] = result
    return result



def ordered_surface_planes(goals, inward):
    """Same ordered-plane identity as finite-facet deduplication (9 decimals)."""
    return tuple(tuple(np.r_[normal, np.dot(normal, point)].round(9))
                 for point, normal in zip(goals, -np.asarray(inward)))


def prioritize_surface_hint(variants, hint):
    """Promote one already surviving plane tuple; never add or remove variants.

    Invoke after original force/table pruning, deduplication and family
    interleaving, before the original ordinal offset/count slice. This carries
    the region ranker's FK/target association into its first nonlinear search.
    A target removed by an existing check stays removed; unknown hints preserve
    the whole current order. No saved target/seed is used as a command.
    """
    rows = list(variants)
    report = dict(status='no_hint', all_input_variants_retained=True,
                  input_count=len(rows), promoted=False)
    if hint is None:
        return rows, report
    wanted = tuple(tuple(row) for row in hint['ordered_outward_planes'])
    report.update(status='hint_not_in_surviving_planes',
                  ordered_outward_planes=[list(row) for row in wanted],
                  ranking_score=hint['score'], ranking_seed_index=hint['seed_index'],
                  ranking_target_set=hint['target_set'],
                  ranking_permutation=hint['permutation'])
    for index, variant in enumerate(rows):
        if ordered_surface_planes(variant[3], variant[4]) != wanted:
            continue
        report.update(status='matched_surviving_plane_tuple', original_index=index,
                      promoted=index != 0,
                      selected_surface_set_number=variant[1],
                      selected_point_assignment=list(variant[2]))
        return [rows[index]] + rows[:index] + rows[index + 1:], report
    return rows, report



# Shared across every group/region ranked by this worker. No hand identity key:
# only the complete public contact geometry/model/task determines an LP result.
_SURFACE_TASK_PRIORITY_CACHE = {}

# One current public RGB geometry at a time. Values are immutable finite-face
# geometry only; nearest supported centers still depend on each target query.
_SURFACE_FACET_GEOMETRY_CACHE = dict(geometry=None, facets={}, builds=0, hits=0)


def _surface_geometry_content_key(cloud):
    import hashlib
    cloud = np.asarray(cloud)
    return (cloud.shape, cloud.dtype.str,
            hashlib.sha256(cloud.tobytes(order='C')).digest())


def _finite_surface_facet_geometry(cloud, hull, normal, radius, geometry_key):
    cache = _SURFACE_FACET_GEOMETRY_CACHE
    if cache['geometry'] != geometry_key:
        cache['facets'].clear()
        cache['geometry'] = geometry_key
        cache['builds'] = cache['hits'] = 0
    # Normals are the caller's declared unit outward vectors. Keep their exact
    # values and the original argmax arithmetic; no rounding/renormalization.
    key = (tuple(np.asarray(normal, float)), float(radius))
    if key not in cache['facets']:
        plane = hull.equations[np.argmax(hull.equations[:, :3] @ normal)]
        on = np.unique(np.round(cloud[abs(cloud @ plane[:3] + plane[3]) < 2e-7], 10), axis=0)
        center = on.mean(0)
        _, _, basis = np.linalg.svd(on - center, full_matrices=False)
        on = on[ConvexHull((on - center) @ basis[:2].T).vertices]
        inset, _, usable = finite_support_region(on, radius)
        projection = None
        if len(inset) > 2:
            center = inset.mean(0)
            _, _, basis = np.linalg.svd(inset - center, full_matrices=False)
            xy = (inset - center) @ basis[:2].T
            polygon = ConvexHull(xy)
            projection = (center, basis[:2], polygon.equations.copy())
            inset = inset[polygon.vertices]
        inset.setflags(write=False)
        if projection is not None:
            for array in projection:
                array.setflags(write=False)
        # Insert only a completely built geometry. Preserve existing error
        # handling on a degenerate face rather than caching a provisional row.
        cache['facets'][key] = (inset, usable, projection)
        cache['builds'] += 1
    else:
        cache['hits'] += 1
    return cache['facets'][key]


def _finite_surface_task_priority(info, points, outward, radii, models, task, facets):
    """Conditional six-dimensional nominal task priority; never a reject gate.

    Mirror the solver's radius-inset finite RGB facets. Move each old start to
    its nearest supported center on that same facet for this LP only; original
    FK targets remain unchanged. Ignore J.T/effort until a real q is available.
    All subsequent original IK, path, robust and effort checks remain binding.
    """
    from general_grasp_contacts import task_wrench_margin
    cloud, hull = facets[:2]
    # Production passes this once-prepared content key. Two-item callers keep
    # the old interface and receive content-safe caching as well.
    geometry_key = facets[2] if len(facets) > 2 else _surface_geometry_content_key(cloud)
    centers, supported = [], []
    for point, normal, radius in zip(points, outward, radii):
        inset, usable, projection = _finite_surface_facet_geometry(
            cloud, hull, normal, radius, geometry_key)
        query = np.asarray(point) - normal * .0003
        if len(inset) == 1:
            nearest = inset[0]
        else:
            candidates = []
            if projection is not None:
                center, basis, equations = projection
                projected = (query - center) @ basis.T
                if np.max(equations[:, :2] @ projected + equations[:, 2]) <= 1e-10:
                    candidates.append(center + projected @ basis)
            for first, last in zip(inset, np.roll(inset, -1, axis=0)):
                delta = last - first
                fraction = np.clip((query - first) @ delta / max(delta @ delta, 1e-30), 0., 1.)
                candidates.append(first + fraction * delta)
            nearest = min(candidates, key=lambda value: np.linalg.norm(value - query))
        centers.append(nearest + normal * .0003)
        supported.append(usable)
    # Preserve contact order: the existing inscribed friction pyramid uses
    # ordered basis hints, so merging permutations could change its numerical
    # approximation. Identical geometry/models still share work across groups.
    # Include every model/task value that can affect the LP.
    count = len(points)
    uncertainty = np.broadcast_to(task['normal_uncertainty'], (count,))
    rows = [tuple(np.r_[point, normal, radius,
                  model['friction_coefficient'], model['normal_force_limit_n'], angle])
                  for point, normal, radius, model, angle in
                  zip(centers, outward, supported, models, uncertainty)]
    parameters = {key: value for key, value in task.items()
                  if key not in ('friction', 'max_normal_force', 'pad_radius_m',
                                 'normal_uncertainty', 'position_sigma_m', 'margin_tolerance')}
    parameters.update(reference=np.asarray(info['center']).tolist(), remove_third=False)
    def frozen(value):
        if isinstance(value, dict):return tuple(sorted((key, frozen(item)) for key, item in value.items()))
        if isinstance(value, (tuple, list, np.ndarray)):return tuple(frozen(item) for item in value)
        return value
    key = (tuple(rows), frozen(parameters))
    hit = key in _SURFACE_TASK_PRIORITY_CACHE
    if not hit:
        data = np.asarray(rows)
        result = task_wrench_margin(data[:, :3], data[:, 3:6], pad_radius_m=data[:, 6],
            friction=data[:, 7], max_normal_force=data[:, 8], normal_uncertainty=data[:, 9], **parameters)
        summary = {name: result[name] for name in ('gravity_feasible', 'task_feasible',
            'minimum_task_margin', 'lateral_margin', 'torque_margin')}
        summary['classification'] = 'conditional_nominal_public_geometry_static_model_no_q_effort'
        _SURFACE_TASK_PRIORITY_CACHE[key] = summary
    return dict(_SURFACE_TASK_PRIORITY_CACHE[key], supported_centers_world_m=np.asarray(centers).tolist(),
                supported_radius_m=supported, cache_hit=hit)

class _RegionRankDeadline(RuntimeError):
    def __init__(self, stage):
        super().__init__(stage)
        self.stage = stage


def _region_rank_checkpoint(deadline_cpu_s, stage, audit):
    # Cooperative boundary only: one existing LP/domain/FK unit is atomic.
    audit['stage'] = stage
    if deadline_cpu_s is not None and time.process_time() >= deadline_cpu_s:
        raise _RegionRankDeadline(stage)


def _factorized_region_proposals(point_cloud, normal_cloud, targets, normals,
                                 lower, upper, default, *, limit, deadline_cpu_s,
                                 audit):
    """Bounded coarse admission; no IK, LP, FCL or robot mutation here.

    Per anchor, fit the default tuple, choose each region at that fixed common
    pose, and jointly refit once. This is not a full-universe search or a
    reachability certificate. Every winner still needs the original checks.
    """
    seed_count, target_count = len(point_cloud[0]), len(targets)
    default = tuple(default)
    pool = {}
    default_witness = None
    proposal = dict(method='default_anchor_independent_regions_one_joint_refit',
        retained_limit=limit, default_protected=True, default_regions=list(default),
        anchors_expected=seed_count*target_count, anchors_processed=0,
        retained_pool_repeat_hits=0, retained_pool_replacements=0,
        complete=False, full_tuple_universe_searched=False,
        unselected_status='not_selected_by_bounded_proposal_not_proven_unreachable',
        coarse_anchor_not_dispatched_or_added_to_ik_bank=True)
    audit['region_admission'] = proposal

    def rotate(values, cosine, sine):
        result = np.broadcast_to(values, cosine.shape+values.shape[-2:]).copy()
        result[..., 0] = values[..., 0]*cosine[..., None]-values[..., 1]*sine[..., None]
        result[..., 1] = values[..., 0]*sine[..., None]+values[..., 1]*cosine[..., None]
        return result

    def fit(points, ns, goals, inward):
        center = points.mean(-2)
        centered = points-center[..., None, :]
        destination = goals-goals.mean(-2)[..., None, :]
        cosine = 1500.**2*np.sum(centered[..., :2]*destination[..., :2], axis=(-2, -1))
        cosine += 12.**2*np.sum(ns[..., :2]*inward[..., :2], axis=(-2, -1))
        sine = 1500.**2*np.sum(centered[..., 0]*destination[..., 1]-centered[..., 1]*destination[..., 0], axis=-1)
        sine += 12.**2*np.sum(ns[..., 0]*inward[..., 1]-ns[..., 1]*inward[..., 0], axis=-1)
        theta = np.arctan2(sine, cosine)
        cosine, sine = np.cos(theta), np.sin(theta)
        rotated = rotate(points, cosine, sine)
        translation = np.clip(goals.mean(-2)-rotated.mean(-2), lower, upper)
        fitted = rotated+translation[..., None, :]
        fitted_ns = rotate(ns, cosine, sine)
        # Score the actual clamped pose; no monotonic/global-optimum claim.
        cost = (1500.**2*np.sum((fitted-goals)**2, axis=(-2, -1))
                +12.**2*np.sum((fitted_ns-inward)**2, axis=(-2, -1)))/len(default)
        return cost, cosine, sine, translation

    points0 = np.stack([cloud[:, region] for cloud, region in zip(point_cloud, default)], axis=1)[:, None]
    normals0 = np.stack([cloud[:, region] for cloud, region in zip(normal_cloud, default)], axis=1)[:, None]
    for first in range(0, target_count, 6):
        _region_rank_checkpoint(deadline_cpu_s, 'factorized_target_chunk', audit)
        goals, inward = targets[None, first:first+6], normals[None, first:first+6]
        default_cost, cosine, sine, translation = fit(points0, normals0, goals, inward)
        ids, chosen_points, chosen_normals = [], [], []
        for finger, (points, ns) in enumerate(zip(point_cloud, normal_cloud)):
            transformed = rotate(points[:, None], cosine, sine)+translation[..., None, :]
            transformed_ns = rotate(ns[:, None], cosine, sine)
            residual = (1500.**2*np.sum((transformed-goals[..., finger, None, :])**2, axis=-1)
                        +12.**2*np.sum((transformed_ns-inward[..., finger, None, :])**2, axis=-1))
            region = np.argmin(residual, axis=-1)
            ids.append(region)
            chosen_points.append(points[np.arange(seed_count)[:, None], region])
            chosen_normals.append(ns[np.arange(seed_count)[:, None], region])
        ids = np.stack(ids, axis=-1)
        cost, _, _, _ = fit(np.stack(chosen_points, axis=-2),
                            np.stack(chosen_normals, axis=-2), goals, inward)
        for seed in range(seed_count):
            for local_target in range(cost.shape[1]):
                target = first+local_target
                default_key = (float(default_cost[seed, local_target]), default, seed, target)
                if default_witness is None or default_key < default_witness:
                    default_witness = default_key
                regions = tuple(int(value) for value in ids[seed, local_target])
                if regions == default or limit == 1:
                    continue
                key = (float(cost[seed, local_target]), regions, seed, target)
                if regions in pool:
                    proposal['retained_pool_repeat_hits'] += 1
                    if key < pool[regions]:pool[regions] = key
                elif len(pool) < limit-1:
                    pool[regions] = key
                else:
                    worst = max(pool, key=pool.get)
                    if key < pool[worst]:
                        del pool[worst]
                        pool[regions] = key
                        proposal['retained_pool_replacements'] += 1
        proposal['anchors_processed'] += seed_count*cost.shape[1]
        retained = [default_witness]+sorted(pool.values())
        proposal['retained'] = [dict(regions=list(row[1]), score=row[0],
            seed_index=row[2], target_index=row[3]) for row in retained]
        _region_rank_checkpoint(deadline_cpu_s, 'factorized_target_chunk_complete', audit)
    proposal['complete'] = True
    return [default]+[row[1] for row in sorted(pool.values())]


def rank_region_tuples(robot,prior,info,regions,axes,heights,*,task_wrench=None,
                       proposal_limit=None,deadline_cpu_s=None):
    """Original rank, optionally restricted by explicit bounded admission."""
    started = time.process_time()
    counts = [len(options) for options in robot.patch_options]
    universe = 1
    for count in counts:universe *= count
    audit = dict(context_complete=False, catalog_region_counts=counts,
                 full_tuple_universe=universe)
    if proposal_limit is not None and (not isinstance(proposal_limit, int) or proposal_limit < 1):
        raise ValueError('proposal_limit must be a positive integer')
    try:
        ordered, result = _rank_region_tuples(robot,prior,info,regions,axes,heights,
            task_wrench=task_wrench,proposal_limit=proposal_limit,
            deadline_cpu_s=deadline_cpu_s,progress=audit)
    except _RegionRankDeadline as error:
        return [], dict(audit,status='group_cpu_deadline_before_solver',
            deadline_stage=error.stage,cpu_s=time.process_time()-started,
            region_scores=[],no_solver_admitted=True,
            provisional_records_are_diagnostics_only=True)
    result.update(audit)
    if proposal_limit is not None:
        result.update(all_input_tuples_retained_scope='admitted_pool_only',
            full_tuple_universe_searched=False,
            unselected_status='not_selected_by_bounded_proposal_not_proven_unreachable')
        if not audit.get('region_admission',{}).get('complete',False):
            result.update(status='bounded_admission_unavailable',
                original_status=result['status'],region_scores=[],no_solver_admitted=True)
            return [], result
    return ordered, result


def _rank_region_tuples(robot,prior,info,regions,axes,heights,*,task_wrench=None,
                        proposal_limit=None,deadline_cpu_s=None,progress=None):
    """Return (all ordered tuples,audit);leave contact selection/catalog intact.

    Pass the same task_wrench used by axis_contact_candidates (or set
    robot.region_ranking_task_wrench). Missing task/hypotheses preserve input
    order. Mean1500² position+12² normal cost matches current seed weighting.
    Existing table-proven impossible hints receive last priority; unknown
    stays unproven. Nominal task feasibility then original FK cost order
    hints within each class. This never
    drops a region/target or substitutes for final q/effort/robust scoring.
    Same measured,.4/.7 uniform and128 canonical-name Sobol articulation bank;
    articulations clipped legal before yaw fit to avoid boundary discrepancies.
    """
    import itertools
    from scipy.stats import qmc
    started=time.process_time();r=robot;prior=np.asarray(prior,float)
    progress={} if progress is None else progress
    checkpoint=lambda stage:_region_rank_checkpoint(deadline_cpu_s,stage,progress)
    checkpoint('rank_preparation_start')
    original=[tuple(ids)for ids in regions];count=len(r.finger_names)
    selected=tuple(r.contact_region_indices)
    task=task_wrench if task_wrench is not None else getattr(r,'region_ranking_task_wrench',None)
    def fallback(reason):
        return original,dict(status=reason,cpu_s=time.process_time()-started,all_input_tuples_retained=True)
    if not original:return fallback('empty')
    if task is None:return fallback('unchanged_missing_shared_task')
    models=[patch['contact_model']for patch in r.selected_patches]
    friction=tuple(model['friction_coefficient']for model in models)
    caps=tuple(model['normal_force_limit_n']for model in models)
    force=-max(task['mass_range_kg'])*(np.array([0.,0.,-9.81])-np.asarray(task.get('lift_acceleration_m_s2',[0.,0.,.5])))
    force_cache=r.__dict__.setdefault('_surface_force_cache',{})
    sets=generate_contact_sets(info,count,axes=axes,heights=heights,max_sets=4096)
    checkpoint('target_sets_generated')
    progress['target_sets']=len(sets)
    progress['target_rows_prepared']=0
    target_rows=[];force_refusals=0
    for si,region in enumerate(sets):
        for perm in itertools.permutations(range(count)):
            checkpoint('prepare_force_target')
            normals=np.asarray(region['outward_normals'])[list(perm)]
            key=(tuple(normals.round(9).ravel()),friction,caps,task['normal_uncertainty'],tuple(force))
            if key not in force_cache:
                force_cache[key]=force_support_possible(normals,friction=friction,
                    normal_uncertainty=task['normal_uncertainty'],max_normal_force=caps,required_force=force)
            if not force_cache[key]['possible']:
                force_refusals+=1;continue
            target_rows.append((np.asarray(region['points'])[list(perm)]+normals*.0003,-normals,si,perm))
            progress['target_rows_prepared']=len(target_rows)
    checkpoint('force_targets_complete')
    if not target_rows:return fallback('unchanged_no_task_possible_public_targets')
    targets=np.array([row[0]for row in target_rows]);normals=np.array([row[1]for row in target_rows])
    destination=targets-targets.mean(1)[:,None,:]
    active=sorted(r.active_finger_indices,key=lambda i:r.names[i]);spans=r.hi[active]-r.lo[active]
    configs=[prior.copy()]
    for fraction in (.4,.7):
        q=prior.copy();q[active]=r.lo[active]+fraction*spans;configs.append(q)
    for fractions in qmc.Sobol(len(active),scramble=False).random_base2(7):
        q=prior.copy();q[active]=r.lo[active]+fractions*spans;configs.append(q)
    poses=[]
    for q in configs:
        checkpoint('prepare_articulation_fk')
        q=np.clip(q,r.lo+2e-6,r.hi-2e-6);q[r.base_indices]=0.;poses.append(r.fk(q))
    checkpoint('articulation_fk_complete')
    # Transform every local region once per finger/seed,never rebuild FK per tuple.
    point_cloud=[];normal_cloud=[]
    for link,options in zip(r.contact_links,r.patch_options):
        rotations=np.array([fk[link][:3,:3]for fk in poses]);translations=np.array([fk[link][:3,3]for fk in poses])
        local_points=np.array([opt['point']for opt in options]);local_normals=np.array([opt['normal']for opt in options])
        point_cloud.append((np.einsum('sij,rj->sri',rotations,local_points)+translations[:,None,:])@r.upright_reference.T)
        normal_cloud.append(np.einsum('sij,rj->sri',rotations,local_normals)@r.upright_reference.T)
    progress['context_complete']=True
    checkpoint('region_clouds_complete')
    if proposal_limit is not None:
        original=_factorized_region_proposals(point_cloud,normal_cloud,targets,normals,
            r.lo[r.base_indices[:3]]+2e-6,r.hi[r.base_indices[:3]]-2e-6,
            tuple(r.contact_region_indices),limit=proposal_limit,
            deadline_cpu_s=deadline_cpu_s,audit=progress)
    target_norm=np.sum(destination**2,axis=(1,2));normal_norm=np.sum(normals**2,axis=(1,2))
    records=[]
    cloud=np.asarray(info['collision_points'],float)
    public_facets=(cloud,ConvexHull(cloud),_surface_geometry_content_key(cloud))
    priority_rows={};priority_cache_hits=0;priority_cache_misses=0;priority_errors=[]
    # Reuse the downstream table relation for priority only. Missing/unknown
    # information never proves rejection and never removes a target or region.
    table_context=None;bound_robot=None;table_priorities={};table_priority_errors=[]
    try:
        checker=getattr(r,'ik_checker',None)
        if checker is not None:
            from copy import copy
            # The original bound method only reads meshes/contact links/points.
            # A shallow proxy shares its cache but leaves every selected contact
            # field, including a dynamic refined center, untouched on r.
            bound_robot=copy(r)
            bound_robot._pad_hull_distance_cache=r.__dict__.setdefault('_pad_hull_distance_cache',{})
            cloud=public_facets[0]
            cache=r.__dict__.setdefault('_rgb_facet_height_cache',{})
            key=(cloud.shape,cloud.tobytes())
            if key not in cache:
                cache.clear();cache[key]=rgb_facet_height_bounds(cloud)
            table_z=float(checker.table_z)
            if not np.isfinite(table_z):raise ValueError('Nonfinite table height leaves priority unknown')
            table_context=(cache[key],table_z)
    except Exception as error:
        table_priority_errors.append(str(error))
    for ids in original:
        checkpoint('rank_admitted_tuple')
        points=np.stack([point_cloud[i][:,k]for i,k in enumerate(ids)],axis=1)
        ns=np.stack([normal_cloud[i][:,k]for i,k in enumerate(ids)],axis=1)
        center=points.mean(1);centered=points-center[:,None,:]
        cpos=np.einsum('sci,tci->st',centered[:,:,:2],destination[:,:,:2])
        spos=np.einsum('sc,tc->st',centered[:,:,0],destination[:,:,1])-np.einsum('sc,tc->st',centered[:,:,1],destination[:,:,0])
        cn=np.einsum('sci,tci->st',ns[:,:,:2],normals[:,:,:2])
        sn=np.einsum('sc,tc->st',ns[:,:,0],normals[:,:,1])-np.einsum('sc,tc->st',ns[:,:,1],normals[:,:,0])
        cosine=1500.**2*cpos+12.**2*cn;sine=1500.**2*spos+12.**2*sn
        pconst=(np.sum(centered**2,axis=(1,2))[:,None]+target_norm[None,:]-2*np.einsum('sc,tc->st',centered[:,:,2],destination[:,:,2]))
        nconst=(np.sum(ns**2,axis=(1,2))[:,None]+normal_norm[None,:]-2*np.einsum('sc,tc->st',ns[:,:,2],normals[:,:,2]))
        cost=(1500.**2*pconst+12.**2*nconst-2*np.hypot(cosine,sine))/count
        theta=np.arctan2(sine,cosine);co=np.cos(theta);si=np.sin(theta)
        rotated_center=np.broadcast_to(center[:,None,:],(len(configs),len(targets),3)).copy()
        rotated_center[:,:,0]=center[:,None,0]*co-center[:,None,1]*si
        rotated_center[:,:,1]=center[:,None,0]*si+center[:,None,1]*co
        translation=targets.mean(1)[None,:,:]-rotated_center
        clamped=np.clip(translation,r.lo[r.base_indices[:3]]+2e-6,r.hi[r.base_indices[:3]]-2e-6)
        cost+=1500.**2*np.sum((clamped-translation)**2,axis=2)
        radii=tuple(float(r.patch_options[i][region].get('radius_m',r.selected_patches[i]['support_radius_m'])) for i,region in enumerate(ids))
        if radii not in priority_rows:
            evaluations=[]
            for target,inward in zip(targets,normals):
                checkpoint('admitted_nominal_surface_task')
                try:
                    evaluation=_finite_surface_task_priority(info,target,-inward,radii,models,task,public_facets)
                    priority_cache_hits+=int(evaluation['cache_hit']);priority_cache_misses+=int(not evaluation['cache_hit'])
                except Exception as error:
                    # Missing/degenerate geometry cannot drop a target. It
                    # keeps the original FK priority behind known task passes.
                    evaluation=dict(task_feasible=False,gravity_feasible=False,classification='priority_unknown_retained',reason=str(error))
                    priority_errors.append(str(error))
                evaluations.append(evaluation)
                progress['nominal_surface_task_rows_completed']=len(evaluations)
                progress['nominal_surface_facet_cache']=dict(
                    cached_faces=len(_SURFACE_FACET_GEOMETRY_CACHE['facets']),
                    builds=_SURFACE_FACET_GEOMETRY_CACHE['builds'],
                    hits=_SURFACE_FACET_GEOMETRY_CACHE['hits'],
                    scope='cumulative for current exact public RGB geometry; query centers remain per target')
            priority_rows[radii]=evaluations
        evaluations=priority_rows[radii]
        distances=(None,)*count
        checkpoint('admitted_pad_hull_priority')
        if table_context is not None:
            # Query the original bound method at candidate catalog points;
            # never select_regions or modify the real robot's contact state.
            try:
                bound_robot.contact_points=np.array([r.patch_options[i][region]['point']for i,region in enumerate(ids)])
                values=bound_robot.pad_hull_distance_bounds()
                if len(values)!=count:raise ValueError('Incomplete static pad/hull bounds')
                distances=tuple(float(value) if value is not None and np.isfinite(value) and value>=0. else None for value in values)
            except Exception as error:
                table_priority_errors.append(str(error))
        if distances not in table_priorities:
            proofs=[];unknown=[]
            for inward in normals:
                checkpoint('admitted_table_priority')
                witnesses=[];unavailable=table_context is None or any(value is None for value in distances)
                if table_context is not None:
                    try:
                        facet_bounds,table_z=table_context
                        planes,maxima=facet_bounds
                        unavailable=unavailable or any(maxima[int(np.argmax(planes[:,:3]@normal))] is None for normal in -inward)
                        witnesses=necessary_table_witnesses(facet_bounds,-inward,distances,
                            table_z=table_z,position_limit_m=.0008,
                            table_clearance_m=.00065,standoff_m=.0003)
                    except Exception as error:
                        unavailable=True;table_priority_errors.append(str(error))
                proofs.append(witnesses);unknown.append(unavailable)
            table_priorities[distances]=(proofs,unknown)
        proofs,unknown=table_priorities[distances]
        target_seeds=np.argmin(cost,axis=0)
        ti=min(range(len(target_rows)),key=lambda index:(bool(proofs[index]),not evaluations[index]['task_feasible'],float(cost[target_seeds[index],index]),int(target_seeds[index]),index))
        seed=int(target_seeds[ti])
        angle=theta[seed,ti];c=np.cos(angle);s=np.sin(angle)
        rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        fitted=points[seed]@rotation.T+clamped[seed,ti];fitted_normals=ns[seed]@rotation.T
        errors=np.linalg.norm(fitted-targets[ti],axis=1)
        angles=np.arccos(np.clip(np.sum(fitted_normals*normals[ti],axis=1),-1.,1.))
        records.append(dict(regions=list(ids),nominal_surface_task=evaluations[ti],table_priority_proven_impossible=bool(proofs[ti]),table_priority_witnesses=proofs[ti],table_priority_unknown_inputs=unknown[ti],table_priority_pad_hull_distances_m=list(distances),score=float(max(0.,cost[seed,ti])),seed_index=int(seed),target_index=int(ti),target_set=int(target_rows[ti][2]),permutation=list(target_rows[ti][3]),ordered_outward_planes=[list(plane) for plane in ordered_surface_planes(targets[ti],normals[ti])],max_position_m=float(errors.max()),normal_error_rad=angles.tolist()))
    checkpoint('admitted_region_ranking_complete')
    ranked=sorted(records,key=lambda row:(row['table_priority_proven_impossible'],not row['nominal_surface_task']['task_feasible'],row['score']));ordered=[tuple(row['regions'])for row in ranked]
    assert len(ordered)==len(original)and sorted(ordered)==sorted(original)
    assert tuple(r.contact_region_indices)==selected
    return ordered,dict(status='ranked',priority_rule='existing table-proven impossible last; conditional nominal public task pass first within each class; original FK cost next; unknown retained; no new rejection',table_priority_context_available=table_context is not None,table_priority_errors=table_priority_errors,all_existing_table_impossible_targets_retained_for_downstream=True,nominal_surface_task_cache_hits=priority_cache_hits,nominal_surface_task_cache_misses=priority_cache_misses,nominal_surface_task_cache_size=len(_SURFACE_TASK_PRIORITY_CACHE),nominal_surface_task_priority_errors=priority_errors,all_existing_force_surviving_targets_retained=True,nominal_surface_task_target_count_per_radius=len(target_rows),nominal_surface_task_radius_variants=len(priority_rows),approximation='fixed catalog center,upright+yaw;nominal finite RGB footprint task priority has no q/effort/covariance;necessary table/FCL/path and final robust task remain downstream',count=count,tuple_count=len(original),seed_count=len(configs),target_sets=len(sets),target_permutations=len(target_rows),existing_force_necessary_refusals=force_refusals,region_scores=ranked,cpu_s=time.process_time()-started,all_input_tuples_retained=True,normalization='mean1500² position error²+12² unit-normal chord error²',seed_boundary_rule='same131 bank clipped legal before fitting yaw;uniform/Sobol definitions unchanged')


def _compile_source_rank_cell(vertices):
    cell = np.asarray(vertices, float)
    if len(cell) == 1:
        return dict(point=cell[0])
    origin = cell.mean(0)
    _, _, basis = np.linalg.svd(cell-origin, full_matrices=False)
    basis = basis[:2]
    xy = (cell-origin) @ basis.T
    hull = ConvexHull(xy)
    polygon = cell[hull.vertices]
    edges = np.roll(polygon, -1, axis=0)-polygon
    return dict(origin=origin, basis=basis, equations=hull.equations,
                polygon=polygon, edges=edges, lengths2=np.sum(edges*edges, axis=1))


def _closest_source_rank_cell(cell, query):
    if 'point' in cell:
        return cell['point']
    uv = (query-cell['origin']) @ cell['basis'].T
    equations = cell['equations']
    if np.max(equations[:, :2] @ uv+equations[:, 2]) <= 1e-12:
        return cell['origin']+uv @ cell['basis']
    t = np.clip(np.sum((query-cell['polygon'])*cell['edges'], axis=1)
                / cell['lengths2'], 0., 1.)
    points = cell['polygon']+t[:, None]*cell['edges']
    return points[np.argmin(np.sum((points-query)**2, axis=1))]


def source_domain_corrected_rank(robot, prior, info, records, axes, heights,
                                 *, grasp_seed, deadline_cpu_s=None):
    started = time.process_time()
    audit = dict(input_records=len(records), processed_records=0)
    try:
        return _source_domain_corrected_rank(robot, prior, info, records, axes,
            heights, grasp_seed=grasp_seed,deadline_cpu_s=deadline_cpu_s,progress=audit)
    except _RegionRankDeadline as error:
        return [], dict(audit,status='group_cpu_deadline_before_solver',
            deadline_stage=error.stage,cpu_s=time.process_time()-started,
            no_solver_admitted=True,provisional_records_are_diagnostics_only=True)


def _source_domain_corrected_rank(robot, prior, info, records, axes, heights,
                                  *, grasp_seed,deadline_cpu_s=None,progress=None):
    """Return all original rank records, reordered by one source-domain refit.

    Call after rank_region_tuples and before creating its region hints.
    Every original guard/target record remains attached. This bounded correction only
    evaluates the ONE previously retained seed/target per tuple, not all
    343*131*1752 combinations. Missing source planes retain their fixed point.
    Candidate centers/q are initializer metadata; downstream certification is
    unchanged and must independently rerun on the real current robot state.
    """
    from copy import copy
    import time
    from scipy.stats import qmc
    started = time.process_time()
    progress={} if progress is None else progress
    checkpoint=lambda stage:_region_rank_checkpoint(deadline_cpu_s,stage,progress)
    checkpoint('source_rank_start')
    proxy = copy(robot)
    proxy._planar_contact_center_cache = robot.__dict__.setdefault("_planar_contact_center_cache", {})
    proxy.description = copy(robot.description)
    group = tuple(robot.finger_names)
    selected_before = tuple(robot.contact_region_indices)
    points_before = robot.contact_points.copy()
    pads_before = robot.description.pads
    prior = np.asarray(prior, float)
    active = sorted(robot.active_finger_indices, key=lambda i:robot.names[i])
    spans = robot.hi[active]-robot.lo[active]
    bank = [prior.copy()]
    for fraction in (.4, .7):
        q = prior.copy(); q[active] = robot.lo[active]+fraction*spans
        bank.append(q)
    for fractions in qmc.Sobol(len(active), scramble=False).random_base2(7):
        q = prior.copy(); q[active] = robot.lo[active]+fractions*spans
        bank.append(q)
    sets = generate_contact_sets(info, len(group), axes=axes,
                                                 heights=heights, max_sets=4096)
    checkpoint('source_rank_sets_ready')
    cache = {}
    result = []
    max_reproduction_error = 0.
    source_domain_cpu = 0.
    for old_rank, original in enumerate(records):
        checkpoint('source_rank_record')
        row = dict(original)
        proxy.select_pair(group, row['regions'])
        local_before = proxy.contact_points.copy()
        normal_before = proxy.contact_normals.copy()
        surface = sets[row['target_set']]
        order = row['permutation']
        outward = np.asarray(surface['outward_normals'])[order]
        targets = np.asarray(surface['points'])[order]+outward*.0003
        inward = -outward
        assert ordered_surface_planes(targets, inward) == tuple(
            tuple(plane) for plane in row['ordered_outward_planes'])
        seed = grasp_seed(proxy, np.clip(bank[row['seed_index']],
                          proxy.lo+2e-6, proxy.hi-2e-6), targets, inward, None, 12.)
        points, normals, _, poses = proxy.tips(seed)
        baseline = float((1500.**2*np.sum((points-targets)**2)
                          +12.**2*np.sum((normals-inward)**2))/len(group))
        max_reproduction_error = max(max_reproduction_error, abs(baseline-row['score']))
        projected, sources = [], []
        for link, patch, target, fallback in zip(proxy.contact_links,
                proxy.selected_patches, targets, local_before):
            key = (patch['link'], patch['surface_link'], tuple(patch['point_m']),
                   tuple(patch['outward_normal']), patch['support_radius_m'])
            checkpoint('source_rank_domain')
            if key not in cache:
                domain_start = time.process_time()
                domain = planar_contact_cells(proxy, patch)
                cells = domain['cells_local_m'] or [fallback[None, :]]
                cache[key] = (domain, [_compile_source_rank_cell(cell) for cell in cells])
                source_domain_cpu += time.process_time()-domain_start
            checkpoint('source_rank_domain_ready')
            domain, cells = cache[key]
            query = poses[link][:3, :3].T @ (target-poses[link][:3, 3])
            candidates = np.array([_closest_source_rank_cell(cell, query) for cell in cells])
            chosen = int(np.argmin(np.sum((candidates-query)**2, axis=1)))
            point = candidates[chosen]
            projected.append(point)
            sources.append(dict(status=domain['status'], cells=len(domain['cells_local_m']),
                chosen_cell=chosen if domain['cells_local_m'] else None,
                query_local_m=query.tolist(), projected_local_m=point.tolist(),
                center_shift_m=float(np.linalg.norm(point-fallback)),
                query_distance_m=float(np.linalg.norm(point-query)),
                unchanged_radius_m=domain['support_radius_m']))
        proxy.contact_points = np.asarray(projected)
        fitted = grasp_seed(proxy, seed, targets, inward, None, 12.)
        newpoints, newnormals, _, _ = proxy.tips(fitted)
        newscore = float((1500.**2*np.sum((newpoints-targets)**2)
                         +12.**2*np.sum((newnormals-inward)**2))/len(group))
        assert np.array_equal(seed[proxy.all_finger_indices], fitted[proxy.all_finger_indices])
        assert np.array_equal(normal_before, proxy.contact_normals)
        row.update(score=newscore, source_domain_original_score=original['score'],
            source_domain_original_rank=old_rank,
            source_domain_one_step=dict(sources=sources,
                q_initializer=fitted.tolist(), local_centers_m=proxy.contact_points.tolist(),
                original_fk_max_position_m=float(np.linalg.norm(points-targets, axis=1).max()),
                max_position_m=float(np.linalg.norm(newpoints-targets, axis=1).max()),
                normal_error_rad=np.arccos(np.clip(np.sum(newnormals*inward, axis=1), -1., 1.)).tolist(),
                target_points_world_m=targets.tolist(), inward_world=inward.tolist(),
                articulation_unchanged=True, source_normals_unchanged=True,
                scope='one projection/refit at existing best static seed/target; ranking producer stores initializer; consumer must exactly bind current region and ordered planes; no endpoint/path or feasibility certificate'))
        result.append(row)
        progress['processed_records']=len(result)
        checkpoint('source_rank_record_complete')
    result.sort(key=lambda r:(r['table_priority_proven_impossible'],
                              not r['nominal_surface_task']['task_feasible'], r['score']))
    for rank, row in enumerate(result):
        row['source_domain_new_rank'] = rank
    assert len(result) == len(records)
    assert sorted(tuple(r['regions']) for r in result) == sorted(tuple(r['regions']) for r in records)
    assert tuple(robot.contact_region_indices) == selected_before
    assert np.array_equal(robot.contact_points, points_before)
    assert robot.description.pads is pads_before
    return result, dict(cpu_s=time.process_time()-started,
        source_domain_build_and_compile_cpu_s=source_domain_cpu,
        distinct_static_domains=len(cache), original_score_reproduction_max_error=max_reproduction_error,
        all_original_region_rows_retained=True, all_original_target_assignments_retained=True,
        original_table_and_task_priority_classes_retained=True,
        search_scope='one existing best static seed/target per region; one projection and one common palm refit',
        new_optimizer_calls=0, new_gates=0, simulation_started=False)


CONVEX_COORDINATE_EPS = 1e-8


def normalized_convex_point(vertices, coordinates):
    """Decode raw coordinates bounded by [CONVEX_COORDINATE_EPS, 1].

    The lower bound is dimensionless numerical conditioning, not clearance.
    No extra floor is added by the decoder. Feasible one-sided coordinate
    changes retain nonzero point derivatives at the numerical lower bound.
    """
    vertices = np.asarray(vertices, float)
    coordinates = np.asarray(coordinates, float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or coordinates.shape != (len(vertices),)
            or not np.all(np.isfinite(vertices))
            or not np.all(np.isfinite(coordinates))
            or np.any(coordinates < CONVEX_COORDINATE_EPS)
            or np.any(coordinates > 1.)):
        raise ValueError('Finite convex vertices and coordinates in [1e-8, 1] required')
    weights = coordinates/coordinates.sum()
    # Center first so a small domain translated far from zero keeps precision.
    center = vertices.mean(axis=0)
    return center+weights@(vertices-center)


def convex_nearest_weights(vertices, query):
    """Return bounded raw coordinates for the nearest existing convex point.

    Vertex order is unrestricted; duplicates and interior vertices are allowed.
    A centroid fan gives barycentric weights for planar polygon interiors; line
    and singleton domains are supported. The returned initializer is
    (1 - n*eps)*barycentric + eps, with eps=CONVEX_COORDINATE_EPS. It sums to one
    and satisfies the decoder bounds. Its displacement from the exact nearest
    point is bounded by n*eps times the domain diameter upper bound.

    This changes the optimization coordinates as well as their initialization;
    any existing coordinate regularizer consequently changes its center/units.
    No FK, LP, IK or physical gate occurs. The existing source cell/facet and
    its support radius remain the caller's responsibility.
    """
    vertices = np.asarray(vertices, float)
    query = np.asarray(query, float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or query.shape != (3,) or not np.all(np.isfinite(vertices))
            or not np.all(np.isfinite(query))):
        raise ValueError('Finite Nx3 convex-domain vertices and one query required')
    count = len(vertices)
    if count*CONVEX_COORDINATE_EPS >= 1.:
        raise ValueError('Too many vertices for bounded convex coordinates')
    center = vertices.mean(axis=0)
    local = vertices-center
    _, singular, basis = np.linalg.svd(local, full_matrices=False)
    numerical = 64*np.finfo(float).eps
    if count == 1 or singular[0] == 0.:
        weights = np.full(count, 1./count)
        kind = 'singleton'
    elif len(singular) < 2 or singular[1] <= numerical*singular[0]:
        line = local@basis[0]
        first, last = int(np.argmin(line)), int(np.argmax(line))
        fraction = np.clip(((query-center)@basis[0]-line[first])
                           /(line[last]-line[first]), 0., 1.)
        weights = np.zeros(count)
        weights[first], weights[last] = 1-fraction, fraction
        kind = 'segment'
    else:
        polygon = local@basis[:2].T
        point = (query-center)@basis[:2].T
        # Normalize for unit-independent hull orientation tests.
        scale = float(np.max(np.ptp(polygon, axis=0)))
        uv = polygon/scale
        target = point/scale
        order = np.lexsort((uv[:, 1], uv[:, 0]))
        unique = [int(order[0])]
        for index in order[1:]:
            if not np.array_equal(uv[index], uv[unique[-1]]):
                unique.append(int(index))
        def turn(a, b, c):
            ab, ac = uv[b]-uv[a], uv[c]-uv[a]
            return ab[0]*ac[1]-ab[1]*ac[0]
        lower, upper = [], []
        for index in unique:
            while len(lower) >= 2 and turn(lower[-2], lower[-1], index) <= 0.:
                lower.pop()
            lower.append(index)
        for index in reversed(unique):
            while len(upper) >= 2 and turn(upper[-2], upper[-1], index) <= 0.:
                upper.pop()
            upper.append(index)
        hull = lower[:-1]+upper[:-1]
        if len(hull) < 3:
            raise ValueError('Planar convex domain lost numerical rank')
        ring = uv[hull]
        edges = np.roll(ring, -1, axis=0)-ring
        delta = target-ring
        signed = edges[:, 0]*delta[:, 1]-edges[:, 1]*delta[:, 0]
        weights = None
        if np.all(signed >= -numerical):
            for offset, first in enumerate(hull):
                last = hull[(offset+1) % len(hull)]
                matrix = np.stack([uv[first], uv[last]], axis=1)
                if abs(np.linalg.det(matrix)) <= numerical:
                    continue
                beta, gamma = np.linalg.solve(matrix, target)
                alpha = 1-beta-gamma
                if min(alpha, beta, gamma) >= -numerical:
                    weights = np.full(count, max(0., alpha)/count)
                    weights[first] += max(0., beta)
                    weights[last] += max(0., gamma)
                    weights /= weights.sum()
                    break
        if weights is None:
            fractions = np.clip(np.sum(delta*edges, axis=1)
                                /np.sum(edges**2, axis=1), 0., 1.)
            candidates = ring+fractions[:, None]*edges
            edge = int(np.argmin(np.sum((candidates-target)**2, axis=1)))
            weights = np.zeros(count)
            weights[hull[edge]] = 1-fractions[edge]
            weights[hull[(edge+1) % len(hull)]] = fractions[edge]
        kind = 'planar_convex_polygon'
    projected = center+weights@local
    barycentric = weights.copy()
    coordinates = (1-count*CONVEX_COORDINATE_EPS)*barycentric+CONVEX_COORDINATE_EPS
    reconstructed = normalized_convex_point(vertices, coordinates)
    diameter_bound = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    floor_fraction = count*CONVEX_COORDINATE_EPS
    audit = dict(method='closest convex point; centroid-fan nonnegative coordinates',
        domain_kind=kind, vertex_count=count, query_m=query.tolist(),
        projected_point_m=projected.tolist(), reconstructed_point_m=reconstructed.tolist(),
        projection_distance_m=float(np.linalg.norm(query-projected)),
        encoding_displacement_m=float(np.linalg.norm(reconstructed-projected)),
        initialization_floor_displacement_bound_m=floor_fraction*diameter_bound,
        barycentric_weights=barycentric.tolist(),
        coordinate_epsilon=CONVEX_COORDINATE_EPS,
        initializer='(1 - n*eps)*barycentric + eps',
        decoder='coordinate / sum(coordinate)',
        coordinate_bounds=[CONVEX_COORDINATE_EPS, 1.],
        factor='bounded normalized convex optimization coordinates and initializer',
        existing_coordinate_regularizer_changes_coordinates_and_center=True)
    return coordinates, audit
