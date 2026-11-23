"""Recovery-only two-pad targets on current RGB hull facets.
Global extrema can belong to a different tangent coordinate on a rotated shape.
The original contact/IK/path/motion gates stay unchanged.
"""
_S14_EXTREMA_CANDIDATES=_s3_candidates_on_axes
_S14_SURFACE_ACTIVE=False
def _s14_surface_candidates(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, retime=False, refinement=False):
    if len(robot.contact_points) == 3:
        return _s12_three_candidates(robot, info, prior, checker, initial, axes, heights, shift, geometry_source, retime, refinement)
    results, rejected = [], []
    for axis_index, axis in enumerate(axes):
        axis = np.asarray(axis, dtype=float).copy()
        if axis[0] < 0:
            axis = -axis
        tangent = np.cross([0., 0., 1.], axis)
        pts = np.asarray(info['collision_points'])
        projections = pts[:, :2] @ np.array([axis[:2], tangent[:2]]).T
        lo, hi = projections.min(0), projections.max(0)
        center = axis * ((lo[0] + hi[0]) / 2) + tangent * ((lo[1] + hi[1]) / 2)
        half = (hi - lo) / 2
        rotation = Rotation.from_euler('z', np.arctan2(axis[1], axis[0])).as_matrix()
        seed = np.asarray(prior).copy()
        seed[:3] = rotation @ seed[:3] + np.r_[np.asarray(info['center'])[:2], 0.]
        seed[3:6] = Rotation.from_matrix(rotation @ Rotation.from_euler('ZYX', seed[3:6]).as_matrix()).as_euler('ZYX')
        seed[12:] = 0.
        for h in heights:
            shifted=dict(info,center=np.asarray(info['center'])+tangent*shift)
            regions=_s12_rgb_contact_regions(shifted,axis,(h,h),(0.,0.))
            if not regions:
                rejected.append(dict(first_rejection='no_public_surface_intersection',height_fraction=h))
                continue
            goals=np.array([r['point_world_m'] for r in regions])
            inward=-np.array([r['outward_normal_world'] for r in regions])
            q, meta = optimize(robot, seed, goals, inward)
            meta.update(contact_regions=regions,contact_target_method='ray_to_current_RGB_convex_surface',previous_method='global_axis_extrema_at_unrelated_tangent_coordinate')
            meta.update(height_fraction=h, along_fraction=shift / max(2 * half[1], 1.e-6),
                stage3_axis_world=axis.tolist(), axis_index=axis_index, stage3_shift_m=shift, geometry_source=geometry_source)
            _s12_candidate_identity(dict(q=q, points=robot.tips(q)[0], inward=inward, prefix=[], meta=meta))
            if not _s12_endpoint_gate(robot, q, meta, checker, initial, refinement):
                rejected.append(meta)
                continue
            candidate = _s12_complete_candidate(robot, checker, initial, q, inward, meta, retime)
            if candidate is not None:
                results.append(candidate)
                if len(results) >= (2 if globals().get('S11_CONFIG', {}).get('task_ranking', False) else 1):
                    return results, rejected
            else:
                rejected.append(meta)
        if results:
            break
    return results, rejected
def _s3_candidates_on_axes(robot,info,prior,checker,initial,axes,heights,shift,geometry_source,retime=False,refinement=False):
    if len(robot.contact_points)!=2 or not _S14_SURFACE_ACTIVE:
        return _S14_EXTREMA_CANDIDATES(robot,info,prior,checker,initial,axes,heights,shift,geometry_source,retime,refinement)
    return _s14_surface_candidates(robot,info,prior,checker,initial,axes,heights,shift,geometry_source,retime,refinement)

_S14_CONTACT_CANDIDATES=s11_contact_candidates
def s11_contact_candidates(s,*args,**kwargs):
    global _S14_SURFACE_ACTIVE
    previous=_S14_SURFACE_ACTIVE
    _S14_SURFACE_ACTIVE=bool(s.get('actual_grasps',0)>=1)
    try:return _S14_CONTACT_CANDIDATES(s,*args,**kwargs)
    finally:_S14_SURFACE_ACTIVE=previous
