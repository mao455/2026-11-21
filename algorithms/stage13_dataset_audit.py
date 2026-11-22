"""Offline, geometry-only audit of the fixed Stage13 registration.

Does not replace objects, read success rates, run a policy, or change runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from trimesh.ray.ray_pyembree import RayMeshIntersector
from stage13_geometry_identity import primitive_signature, compare_signatures

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def compact_distance_audit(path):
    """Stream the old pretty JSON, replacing giant distance vectors by counts.

    A size/SHA plus row hash, vertex count, extrema and all duplicate summaries
    are retained. Numeric vectors are reproducible from the native mesh.
    """
    compact, summaries = [], []
    active = None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for raw in f:
            h.update(raw)
            line = raw.decode()
            if active is not None:
                value = line.strip()
                if value.startswith(']'):
                    summaries.append(active)
                    compact.append('      "distances": []' + (',' if value.endswith(',') else '') + '\n')
                    active = None
                elif value:
                    x = float(value.rstrip(','))
                    active['count'] += 1
                    if active['minimum_m'] is None: active['minimum_m'] = x
                    active['maximum_m'] = x
            elif re.match(r'\s*"distances":\s*\[\s*$', line):
                active = dict(count=0, minimum_m=None, maximum_m=None)
            else:
                compact.append(line)
    if active is not None:
        raise ValueError('Truncated distance vector')
    value = json.loads(''.join(compact))
    if len(value['rows']) != len(summaries):
        raise ValueError('Unexpected distance audit format')
    for row, summary in zip(value['rows'], summaries):
        native = Path('/home/CNS2025331827/docs/grasp_small_obj/FingerEyeLab/assets/objects/manifest_native_obj')/(row['object_id']+'_native.obj')
        row['reproduction_native_asset'] = str(native)
        row['reproduction_native_sha256'] = sha(native) if native.exists() else None
        row.pop('distances')
        row['distance_summary'] = summary
        row['expected_distance_count_passed'] = summary['count'] == row['vertices'] * (row['vertices'] - 1) // 2
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest(), **value)


def patch_audit(mesh, radius=.0012, max_sag=.0002, min_z=.002):
    """Check tangent support disks on native surfaces, never grasp outcomes.

    This is a static 2.4mm-diameter support-footprint assumption, not calibrated
    compliant-contact physics or an assertion that a particular hand can reach.
    """
    normals = mesh.face_normals
    centers = mesh.triangles_center
    z0, z1 = mesh.bounds[:, 2]
    idx = np.flatnonzero((abs(normals[:, 2]) < .5) &
                        (centers[:, 2] >= z0 + .2 * (z1-z0)) &
                        (centers[:, 2] <= z0 + .8 * (z1-z0)))
    if not len(idx):
        return dict(passed=False, tested=0, reason='no_midheight_side_face')
    # Highest area faces plus azimuth bins avoid favoring triangulation density.
    chosen = set(idx[np.argsort(mesh.area_faces[idx])[-12:]].tolist())
    angle = np.arctan2(normals[idx, 1], normals[idx, 0])
    for a in np.linspace(-np.pi, np.pi, 16, endpoint=False):
        cost = abs(np.angle(np.exp(1j*(angle-a)))) + abs(centers[idx, 2]-(z0+z1)/2) / max(z1-z0, 1e-8)
        chosen.add(int(idx[np.argmin(cost)]))
    ray = RayMeshIntersector(mesh)
    accepted, details = [], []
    for face in sorted(chosen):
        n = normals[face]
        tangent = np.cross(n, [0, 0, 1.]); tangent /= np.linalg.norm(tangent)
        other = np.cross(n, tangent)
        angles = np.arange(8) * np.pi/4
        offsets = np.r_[np.zeros((1, 3)), radius*(np.cos(angles)[:, None]*tangent + np.sin(angles)[:, None]*other)]
        p = centers[face]
        origins = p + offsets + n*.01
        locations, rays, faces = ray.intersects_location(origins, np.tile(-n, (9, 1)), multiple_hits=False)
        if len(set(rays)) != 9:
            details.append(dict(face=face, passed=False, hits=len(set(rays))))
            continue
        order = np.argsort(rays); locations, faces = locations[order], faces[order]
        sag = float(np.max(abs((locations-(p+offsets)) @ n)))
        normal_cos = float(np.min(normals[faces] @ n))
        clearance = float(locations[:, 2].min() - z0 + .0001)
        passed = bool(sag <= max_sag and normal_cos >= np.cos(np.pi/6) and clearance >= min_z)
        record = dict(face=face, point_native_m=p.tolist(), outward_normal=n.tolist(), maximum_sag_m=sag,
                      minimum_normal_cosine=normal_cos, minimum_footprint_table_clearance_m=clearance, passed=passed)
        details.append(record)
        if passed: accepted.append(record)
    azimuth = [np.arctan2(r['outward_normal'][1], r['outward_normal'][0]) for r in accepted]
    opposed = any(np.cos(a-b) < -.5 for a in azimuth for b in azimuth)
    return dict(passed=bool(accepted and opposed), tested=len(details), accepted=len(accepted),
                opposed_sides=bool(opposed), support_radius_m=radius, maximum_sag_threshold_m=max_sag,
                minimum_footprint_table_clearance_threshold_m=min_z, witnesses=accepted[:4],
                rejected_examples=[r for r in details if not r['passed']][:3])


def source_geometry_audit(item, installed_mesh, tolerance_m=1e-8):
    """Verify source bytes and any explicitly declared uniform source transform.

    Vertex sets are compared in both directions, independently of OBJ ordering.
    An omitted transform retains the original identical-source-byte rule.
    """
    path=Path(item['source_asset']);transform=item.get('source_transform')
    expected_hash=item.get('source_sha256') if transform is not None else item['sha256']
    result=dict(source_asset=str(path),expected_source_sha256=expected_hash,
                actual_source_sha256=sha(path) if path.exists() else None,
                transform=transform,vertex_tolerance_m=tolerance_m,
                source_hash_passed=False,transformed_vertices_passed=False)
    result['source_hash_passed']=bool(expected_hash and result['actual_source_sha256']==expected_hash)
    if transform is None:
        result['transformed_vertices_passed']=result['source_hash_passed']
        result['verification']='untransformed native source must have identical bytes to installed asset'
        return result
    if not isinstance(transform,dict):
        result['failure']='source_transform must be an explicit object';return result
    scale=transform.get('uniform_scale');z=transform.get('z_translation_m')
    if (isinstance(scale,bool) or isinstance(z,bool) or
        not isinstance(scale,(int,float)) or not isinstance(z,(int,float)) or
        not np.isfinite(scale) or not np.isfinite(z) or scale<=0):
        result['failure']='source transform requires finite positive uniform_scale and finite z_translation_m';return result
    if not path.exists():
        result['failure']='source asset missing';return result
    source=trimesh.load(path,force='mesh',process=True)
    expected=np.asarray(source.vertices)*scale+np.array([0.,0.,z])
    actual=np.asarray(installed_mesh.vertices)
    forward=cKDTree(actual).query(expected)[0];backward=cKDTree(expected).query(actual)[0]
    delta=max(float(forward.max()),float(backward.max()))
    result.update(source_processed_vertex_count=len(expected),installed_processed_vertex_count=len(actual),
                  maximum_bidirectional_vertex_difference_m=delta,
                  transformed_vertices_passed=bool(len(expected)==len(actual) and delta<=tolerance_m),
                  verification='source vertices times uniform_scale plus world-Z translation, bidirectional vertex-set check')
    return result


def mesh_audit(item, lab, spectra, comparisons):
    path = lab/item['asset']
    mesh = trimesh.load(path, force='mesh', process=True)
    hull = mesh.convex_hull
    dims = mesh.extents
    side = float(mesh.area_faces[abs(mesh.face_normals[:, 2]) < .5].sum()/mesh.area)
    bottom = mesh.vertices[mesh.vertices[:, 2] <= mesh.bounds[0, 2]+1e-6, :2]
    center = np.asarray(mesh.center_mass)
    stable = False
    support_area = 0.
    support_margin = None
    if len(bottom) >= 3 and np.linalg.matrix_rank(bottom-bottom.mean(0)) == 2:
        support = ConvexHull(bottom)
        support_area = float(support.volume)
        eq = support.equations
        support_margin = float(np.min(-(eq[:, :2]@center[:2]+eq[:, 2])))
        stable = support_margin >= -1e-7
    # One full spectrum at a time: no giant arrays are retained in JSON.
    d = pdist(hull.vertices); d.sort(); diameter = float(d[-1])
    norm = d/diameter
    signatures = dict(vertices=len(hull.vertices), pairs=len(d), diameter_m=diameter,
                      native_distance_sha256=hashlib.sha256(np.rint(d/1e-7).astype('<i8').tobytes()).hexdigest(),
                      normalized_distance_sha256=hashlib.sha256(np.rint(norm/1e-5).astype('<i8').tobytes()).hexdigest())
    same_count = spectra.setdefault(len(norm), [])
    for previous_id, previous in same_count:
        delta = max(float(np.max(abs(norm[k:k+1000000]-previous[k:k+1000000])))
                    for k in range(0, len(norm), 1000000))
        comparisons.append(dict(a=previous_id, b=item['object_id'], maximum_normalized_distance_delta=delta,
                                duplicate_under_registered_tolerance=delta < 1e-5))
    same_count.append((item['object_id'], norm.astype(np.float32)))
    patch = patch_audit(mesh)
    ratio = abs(mesh.volume)/max(hull.volume, 1e-15)
    minimum_face_height = float(mesh.bounds[1, 2]-mesh.bounds[0, 2])
    source=source_geometry_audit(item,mesh)
    gate = dict(asset_hash=sha(path)==item['sha256'], native_dimensions=bool(.008-1e-8 <= minimum_face_height <= .05+1e-8 and np.all(dims[:2]>=.006-1e-8) and np.all(dims[:2]<=.06+1e-8)),
                declared_dimensions=bool(np.allclose(dims,item['dims_m'],rtol=0,atol=1e-8)),
                watertight_convex=bool(mesh.is_watertight and np.isclose(ratio, 1, rtol=1e-5)),
                side_area=side >= .25, source_asset_hash=source['source_hash_passed'],
                source_transformed_vertices=source['transformed_vertices_passed'])
    return dict(object_id=item['object_id'], gates=gate, native_height_m=minimum_face_height,
                source_provenance=source,primitive_identity=primitive_signature(mesh),
                side_area_fraction=side, static_initial_support=dict(center_mass_native_m=center.tolist(), support_area_m2=support_area,
                center_projection_margin_m=support_margin, native_orientation_static_stable=bool(stable) if support_area else None,
                classification='flat_support_COM_projection_certified' if stable else 'line_or_point_support_unresolved_without_settling',
                limitation='Static support polygon only; actual post-settle height/stability must be verified from each scene and recorded physics.'),
                tangent_pad_footprint=patch, shape=signatures)


def history_audit(ids, source_lab):
    """List prior declaration/report exposure without inferring success rates."""
    roots = [source_lab/'experiments']
    candidates = []
    for base in roots:
        for p in base.rglob('*'):
            if not p.is_file() or p.stat().st_size > 5*1024*1024:
                continue
            if p.suffix not in ('.json','.txt','.md') or any(x in p.parts for x in ('versions','archive','__pycache__')):
                continue
            if 'STAGE13' in p.name or 'stage13' in str(p): continue
            candidates.append(p)
    found = {k:dict(reports=[],configurations=[]) for k in ids}
    pattern = re.compile(r'abc_(\d+)(?!\d)')
    idset = set(ids)
    for p in candidates:
        try: matched=set('abc_'+m for m in pattern.findall(p.read_text(errors='ignore'))) & idset
        except OSError: continue
        kind = 'reports' if re.match(r'STAGE\d+_', p.name) else 'configurations'
        for obj in matched: found[obj][kind].append(str(p.relative_to(source_lab)))
    return dict(scope='Prior source experiments reports/configurations only; a mention is exposure, not evidence of completed physics or success. Output raw physics not scanned.',
                scanned_files=len(candidates), objects={k:dict(**v, exposed=bool(v['reports'] or v['configurations'])) for k,v in found.items()})


def private_candidate_geometry(audit):
    """True mesh only for a static refused-pose counterfactual, never policy input."""
    import fcl
    import visual_controller as v
    from scipy.spatial.transform import Rotation
    saved=json.loads((audit/'baseline_cube_first_rejection.json').read_text())
    ep=Path(saved['episode'])
    scene=json.loads((ep/'scene.json').read_text())
    initial=json.loads(next((ep/'public_responses.jsonl').open()))['observation']
    robot=v.Robot(scene['robot_description']['urdf_path']); robot.configure_contacts(3)
    checker=v.SelfAwareGeometry(robot,v._S3_ESTIMATE(initial))
    q=np.asarray(saved['recomputed_candidate']['q']); poses=robot.fk(q)
    mesh=trimesh.load(scene['mesh'],force='mesh',process=True)
    pose=np.asarray(scene['initial_object_pose']); transform=np.eye(4)
    transform[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix(); transform[:3,3]=pose[:3]
    mesh.apply_transform(transform)
    model=fcl.BVHModel(); model.beginModel(len(mesh.vertices),len(mesh.faces))
    model.addSubModel(np.asarray(mesh.vertices),np.asarray(mesh.faces,dtype=np.int32)); model.endModel()
    obj=fcl.CollisionObject(model); rows=[]
    for name in saved['recomputed_candidate']['endpoint']['object_collisions']:
        hand=fcl.CollisionObject(checker.models[name],fcl.Transform(poses[name][:3,:3],poses[name][:3,3]))
        result=fcl.CollisionResult()
        overlap=fcl.collide(hand,obj,fcl.CollisionRequest(enable_contact=True,num_max_contacts=32),result)
        rows.append(dict(link=name,true_mesh_overlap=bool(overlap),
                    penetration_depth_m=max([0.]+[float(c.penetration_depth) for c in result.contacts]),
                    public_envelope_rejected=True))
    return dict(scope='Offline static candidate pose against true initial cube mesh; zero executed commands and no physical contact claim.',
                native_mesh_sha256=sha(Path(scene['mesh'])),links=rows,
                limitation='Authored robot meshes and true initial object mesh; not simulator cooked convex pieces. Public-envelope overapproximation cannot excuse independent robot self overlap.')


def first_rejection(audit):
    path = audit/'baseline_cube_first_rejection.json'
    j=json.loads(path.read_text());c=j['recomputed_candidate'];e=c['endpoint']
    return dict(input_sha256=sha(path), source_commit=j['source_commit'], physical_commands=j['physical_commands'],
                first_rejection=c['first_rejection'], per_finger_error_m=c['per_finger_error_m'],
                normal_error_rad=c['normal_error_rad'], endpoint_table_clearance_m=e['table_clearance_m'],
                table_threshold_m=e['thresholds']['table_clearance_m'], witnesses=e['clearance_witnesses'],
                self_collision_is_exact_robot_mesh=True, object_collision_uses_public_envelope=True,
                explanation='Endpoint failure precedes any command. Robot self overlap is independent of object-envelope uncertainty. Envelope collision is a conservative public-model check, not physical contact or private object mesh overlap.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-lab',type=Path,default=Path('/home/CNS2025331827/docs/grasp_small_obj/FingerEyeLab'))
    parser.add_argument('--audit',type=Path,default=LAB/'outputs/three_regrasp_20261122_01/audit')
    parser.add_argument('--output',type=Path,default=HERE/'STAGE13_GEOMETRY_AUDIT.json')
    args=parser.parse_args();started=time.perf_counter()
    registry=HERE/'STAGE13_OBJECTS.json';j=json.loads(registry.read_text())
    rows=[]; spectra={}; comparisons=[]
    for item in j['objects']:
        rows.append(mesh_audit(item,LAB,spectra,comparisons))
        print(json.dumps(dict(object_id=item['object_id'],gates=rows[-1]['gates'],patch=rows[-1]['tangent_pad_footprint']['passed'])),flush=True)
    pair_audit=[]
    for item in j['objects']:
        runs=[r for r in j['runs'] if r['object_id']==item['object_id']]
        cases={r['condition']:json.loads((HERE/r['case']).read_text()) for r in runs}
        delta={k:[cases['clean'].get(k),cases['noise'].get(k)] for k in set(cases['clean'])|set(cases['noise']) if cases['clean'].get(k)!=cases['noise'].get(k)}
        pair_audit.append(dict(object_id=item['object_id'],differences=delta,passed=delta=={'rgb_noise_std':[0.,3.]},same_seed=cases['clean']['noise_seed']==cases['noise']['noise_seed']))
    compact=compact_distance_audit(args.audit/'geometry_duplicate_audit.json')
    old_ids={r['object_id'] for r in compact['rows']};ids=[r['object_id'] for r in rows]
    normalized={};native={}
    for row in rows:
        normalized.setdefault(row['shape']['normalized_distance_sha256'],[]).append(row['object_id'])
        native.setdefault(row['shape']['native_distance_sha256'],[]).append(row['object_id'])
    primitive_pairs=[];unresolved_primitive_pairs=0
    for index,first in enumerate(rows):
        for second in rows[index+1:]:
            identity=compare_signatures(first['primitive_identity'],second['primitive_identity'])
            if identity['resolved']:
                primitive_pairs.append(dict(a=first['object_id'],b=second['object_id'],**identity))
            else:unresolved_primitive_pairs+=1
    primitive_duplicates=[r for r in primitive_pairs if r['same_up_to_uniform_scale']]
    spectrum_duplicates=[r for r in comparisons if r['duplicate_under_registered_tolerance']]
    no_duplicates=not (primitive_duplicates or spectrum_duplicates or any(len(v)>1 for v in normalized.values()) or len(set(ids))!=len(ids))
    config=json.loads((HERE/'stage9_config.json').read_text())
    noise=json.loads((args.audit/'environment_data_audit.json').read_text())['noise']
    lp=json.loads((args.audit/'same_load_two_three.json').read_text())
    lp_summary={k:dict(geometry_feasible=v['geometry_feasible'],motion_scope=v['motion_scope'],
                task={s:{key:v['task'][s].get(key) for key in ('gravity_feasible','lift_feasible','task_feasible','minimum_task_margin','lateral_margin','torque_margin')} for s in ('nominal','robust')}) for k,v in lp.items()}
    out=dict(schema=1,scope='Offline static qualification and evidence audit; registry unchanged; no GPU/physics runs; no selection based on outcomes.',
             source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),registration_sha256=sha(registry),
             object_count=len(rows),case_pair_count=len(pair_audit),unique_id_count=len(set(ids)),
             declared_geometry_all_passed=bool(all(all(r['gates'].values()) for r in rows) and no_duplicates),
             per_object_geometry_gates_all_passed=all(all(r['gates'].values()) for r in rows),
             no_duplicate_gate=no_duplicates,
             flat_or_low_curvature_pad_footprint_all_passed=all(r['tangent_pad_footprint']['passed'] for r in rows),
             static_native_support_certified_count=sum(r['static_initial_support']['native_orientation_static_stable'] is True for r in rows),
             static_native_support_unresolved_ids=[r['object_id'] for r in rows if r['static_initial_support']['native_orientation_static_stable'] is None],
             stable_post_settle_height_verified=False,
             qualification_limit='Installed geometric dimensions and explicit source provenance/transforms are verified. Neither registration nor this CPU audit settles each object in physics; consult the separate settled qualification for post-settle height>=8mm and pose stability.',
             shape_hash_duplicate_groups=[v for v in normalized.values() if len(v)>1],
             shape_pairwise_comparisons=comparisons,
             shape_duplicate_pairs=spectrum_duplicates,
             primitive_identity_audit=dict(method='Circular-cylinder cap/side planes and fitted circle, ordinary-box support planes; native size and uniform-scale shape compared separately, independently of vertex count.',
                 resolved_pairs=primitive_pairs,unresolved_pair_count=unresolved_primitive_pairs,
                 unsupported_object_ids=[r['object_id'] for r in rows if r['primitive_identity'] is None],
                 duplicate_pairs=primitive_duplicates,
                 native_duplicate_pairs=[r for r in primitive_pairs if r['same_native_geometry']],
                 helper_sha256=sha(HERE/'stage13_geometry_identity.py')),
             native_duplicate_groups=[v for v in native.values() if len(v)>1],
             uniqueness_limit='Sorted hull distance spectra supplement cross-tessellation cylinder/box fits. Unsupported arbitrary surfaces remain unresolved by primitive fits; no-duplicate gate means no match under these declared checks, not a mathematical proof of noncongruence for all homometric or differently tessellated arbitrary surfaces.',
             declared_diagnostic_ids_excluded=not bool(set(ids)&set(j['diagnostic_pyramid_ids_excluded'])),
             pairs=pair_audit,all_pairs_only_rgb_sigma=all(r['passed'] for r in pair_audit),
             timing=dict(physics_hz=config['physics_hz'],control_hz=config['control_hz'],camera_fps=config['camera_fps'],video_fps=config['video_fps']),
             noise_audit=dict(**noise,source_defaults_verified="parameters['rgb_noise_std'] = case.get('rgb_noise_std', 0.)" in (HERE/'round14_backend.py').read_text(),
                 runtime_source_sha256={name:sha(HERE/name) for name in ('round14_backend.py','stage9_backend.py','stage9_config.json')}),objects=rows,history_exposure=history_audit(ids,args.source_lab),
             prior_large_distance_audit=dict(**compact,missing_registered_ids=sorted(set(ids)-old_ids),extra_ids=sorted(old_ids-set(ids)),
                 cleanup_recommendation='Do not delete this unique large audit until this compact row/hash evidence and native mesh hashes are reviewed. Distance vectors are regenerable; preserve compact metadata first.'),
             three_finger_baseline_first_rejection=first_rejection(args.audit),
             three_finger_true_mesh_offline_counterfactual=private_candidate_geometry(args.audit),same_load_two_three=lp_summary,
             elapsed_cpu_wall_s=time.perf_counter()-started)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),bytes=args.output.stat().st_size,qualification=out['declared_geometry_all_passed'],patches=out['flat_or_low_curvature_pad_footprint_all_passed'],static_support_certified=out['static_native_support_certified_count'],normalized_duplicate_pairs=out['shape_duplicate_pairs'],elapsed_s=out['elapsed_cpu_wall_s'])),flush=True)

if __name__=='__main__':main()
