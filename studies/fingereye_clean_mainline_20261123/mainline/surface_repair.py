"""Broader nearby observed facets for a pair whose two footprints are edge slivers.

This creates alternatives, never changes endpoint/path limits or certifies contact.
"""
import copy
import time
import numpy as np
from scipy.spatial import ConvexHull


def supported_pair_alternatives(m,s,robot,info,initial,candidates,rejected,retime):
    if not candidates:return candidates
    old=next((c for c in candidates if len(c['points'])==2),None)
    if old is None or m._s12_pair_sufficient(old['meta']):return candidates
    saved=old['meta'];real=saved.get('realized_contact_geometry',{})
    radii=np.asarray(real.get('support_radius_m',[]))
    if radii.shape!=(2,):return candidates
    robot.select_pair(saved['contact_fingers'],saved['hand_contact_regions']);m.apply_contact_geometry(robot,saved)
    caps=np.array([p['support_radius_m']for p in robot.selected_patches])
    # Only spend this small slice if both footprints could improve substantially.
    # Fourfold footprint growth is a scheduling preference, not an admission gate.
    if not np.all(caps>4.*np.maximum(radii,1e-7)):return candidates
    cloud=np.asarray(info['collision_points']);hull=ConvexHull(cloud);vertices=cloud[hull.vertices]
    normals=np.array(real['object_normals']);options=[[],[]]
    for plane in np.unique(np.round(hull.equations,9),axis=0):
        face=vertices[abs(vertices@plane[:3]+plane[3])<3e-7]
        if len(face)<3:continue
        origin=face.mean(0);_,_,basis=np.linalg.svd(face-origin,full_matrices=False);basis=basis[:2]
        polygon=ConvexHull((face-origin)@basis.T)
        radius=float(-polygon.equations[:,2].max())
        for k in range(2):
            # Neighbouring normals within twice the declared RGB normal noise.
            sigma=max(float(r.get('normal_uncertainty_rad',.15))for r in saved.get('contact_regions',[])or[{}])
            if plane[:3]@normals[k]<np.cos(2.*sigma):continue
            if min(radius,caps[k])<=4.*max(radii[k],1e-7):continue
            options[k].append((radius,origin+plane[:3]*.0003,plane[:3]))
    if not all(options):return candidates
    combinations=sorted(((a,b)for a in options[0]for b in options[1]),key=lambda p:-min(p[0][0]/caps[0],p[1][0]/caps[1]))
    deadline=time.process_time()+28.;audits=[]
    for a,b in combinations[:2]:
        if time.process_time()>=deadline:break
        goals=np.array([a[1],b[1]]);inward=-np.array([a[2],b[2]])
        robot.ik_surface_info=None;robot.ik_contact_info=info;robot.ik_checker=s['checker']
        robot.ik_seed_hints=[np.asarray(old['q'])];robot.defer_finite_contact=False
        try:
            q,meta=m.optimize(robot,initial,goals,inward)
            # Reuse the just-computed near solution on this exact finite face.
            # No new target or relaxed gate is introduced by the restart.
            if max(meta['per_finger_error_m'])>=.0008:
                first=dict(per_finger_error_m=meta['per_finger_error_m'],normal_error_rad=meta['normal_error_rad'])
                robot.ik_seed_hints=[q]
                q,meta=m.optimize(robot,initial,goals,inward)
                meta['same_surface_restart']=first
        finally:robot.ik_seed_hints=None
        for key in ('contact_fingers','hand_contact_regions','hand_contact_points_local_m','hand_contact_cells_local_m','contact_regions'):
            if key in saved:meta[key]=copy.deepcopy(saved[key])
        meta.update(candidate_id=str(saved['candidate_id'])+'-broad-'+str(len(audits)),
            contact_target_method='nearby_broader_finite_RGB_facets',replaces_object_targets=True,
            previous_targets_m=saved['targets'],proposed_targets_m=goals.tolist())
        passed=m._s12_endpoint_gate(robot,q,meta,s['checker'],initial)
        candidate=m._s12_complete_candidate(robot,s['checker'],initial,q,inward,meta,retime)if passed else None
        if candidate is not None:
            score,terms=m.s11_contact_quality(robot,candidate,initial)
            meta.update(quality_score=score,quality_terms=terms,task_quality=m.s12_task_quality(robot,candidate,info))
            candidates.append(candidate)
            if hasattr(m,'_remember_verified_candidate'):m._remember_verified_candidate(robot,candidate)
        else:rejected.append(meta)
        audits.append(dict(candidate_id=meta['candidate_id'],complete=candidate is not None,
            previous_targets_m=saved['targets'],new_targets_m=goals.tolist(),
            per_finger_error_m=meta['per_finger_error_m'],normal_error_rad=meta['normal_error_rad'],
            first_rejection=meta.get('first_rejection')))
    m.s5_note(s['ctx'],'broader_surface_comparison',dict(audits=audits,pair_retained=True))
    robot.select_pair(saved['contact_fingers'],saved['hand_contact_regions']);m.apply_contact_geometry(robot,saved)
    return candidates
