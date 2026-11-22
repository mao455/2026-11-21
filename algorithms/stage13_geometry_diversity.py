"""Offline geometry diversity screen, separate from exact identity or policy.

These explicitly chosen family thresholds suppress practically identical
primitives. They do not certify arbitrary surfaces distinct. No outcomes are read.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.distance import pdist

from stage13_geometry_identity import primitive_signature


def near_sphere_signature(mesh, maximum_axis_ratio=1.01, maximum_shell_fraction=.01):
    """Fit a full ellipsoid; require a dense convex surface near a sphere.

    The continuous radial shell check prevents boxes whose eight corners lie
    on a sphere from becoming false sphere matches. Native size is ignored.
    """
    if len(mesh.faces) < 200 or not mesh.is_watertight or not mesh.is_convex:
        return None
    v = np.asarray(mesh.vertices)
    origin = v.mean(0)
    unit = np.linalg.norm(v-origin, axis=1).mean()
    if not unit > 0:
        return None
    x = (v-origin)/unit
    design = np.c_[x[:, 0]**2, x[:, 1]**2, x[:, 2]**2,
                   2*x[:, 0]*x[:, 1], 2*x[:, 0]*x[:, 2],
                   2*x[:, 1]*x[:, 2], x]
    coef, _, rank, _ = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)
    if rank < 9:
        return None
    matrix = np.array([[coef[0], coef[3], coef[4]],
                       [coef[3], coef[1], coef[5]],
                       [coef[4], coef[5], coef[2]]])
    try:
        center = -.5*np.linalg.solve(matrix, coef[6:])
        eigenvalues = np.linalg.eigvalsh(matrix/(1+center@matrix@center))
    except np.linalg.LinAlgError:
        return None
    if np.any(eigenvalues <= 0):
        return None
    axes = unit/np.sqrt(eigenvalues)
    ratio = float(axes.max()/axes.min())
    world_center = origin+unit*center
    delta = v-world_center
    radius = float(np.linalg.norm(delta, axis=1).mean())
    outer = float(np.linalg.norm(delta, axis=1).max())
    planes = np.einsum('ij,ij->i', mesh.face_normals,
                       mesh.triangles[:, 0]-world_center)
    if planes.min() <= 0:
        return None
    inner = float(planes.min())
    shell_fraction = max(abs(outer-radius), abs(inner-radius))/radius
    if ratio > maximum_axis_ratio or shell_fraction > maximum_shell_fraction:
        return None
    return dict(kind='near_sphere', center_m=world_center.tolist(),
                ellipsoid_semiaxes_m=axes.tolist(), axis_ratio=ratio,
                reference_radius_m=radius, inner_radius_m=inner,
                outer_radius_m=outer, radial_shell_fraction=shell_fraction,
                maximum_axis_ratio=maximum_axis_ratio,
                maximum_shell_fraction=maximum_shell_fraction)


def diversity_signature(mesh):
    return primitive_signature(mesh) or near_sphere_signature(mesh)


def compare_diversity(a, b, tolerance=.05):
    if not a or not b or a['kind'] != b['kind']:
        return dict(resolved=False, near_same_family=None)
    kind = a['kind']
    if kind == 'near_sphere':
        return dict(resolved=True, near_same_family=True, kind=kind,
                    metric='Both fit near-sphere gates; ignore native size.',
                    relative_difference=0.)
    if kind == 'circular_cylinder':
        av = np.array([a['aspect_ratio']]); bv = np.array([b['aspect_ratio']])
        metric = 'absolute aspect difference / larger aspect'
    elif kind == 'box':
        av = np.array(a['normalized_dimensions'])
        bv = np.array(b['normalized_dimensions'])
        metric = 'max component absolute difference / larger normalized dimension'
    else:
        return dict(resolved=False, near_same_family=None)
    difference = float(np.max(abs(av-bv)/np.maximum(abs(av), abs(bv))))
    return dict(resolved=True, near_same_family=difference <= tolerance,
                kind=kind, metric=metric, relative_difference=difference,
                tolerance=tolerance, a_values=av.tolist(), b_values=bv.tolist())


def components(ids, edges):
    """Transitive components retain earliest preregistration order."""
    parent = {key: key for key in ids}
    def find(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]; key = parent[key]
        return key
    for edge in edges:
        a, b = find(edge['a']), find(edge['b'])
        parent[b] = a
    groups = {}
    for key in ids:
        groups.setdefault(find(key), []).append(key)
    return [dict(members=group, retain=group[0], remove=group[1:])
            for group in groups.values() if len(group) > 1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(study, freeze_directory='frozen_v3'):
    lab = study.parents[1]
    plan_path = study/freeze_directory/'plan.json'
    plan = json.loads(plan_path.read_text())
    objects = plan['objects']; ids = [row['object_id'] for row in objects]
    meshes = {}; signatures = {}; spectra = {}
    for row in objects:
        path = lab/row['asset']; assert sha(path) == row['sha256']
        mesh = trimesh.load(path, force='mesh', process=False)
        meshes[row['object_id']] = mesh
        signatures[row['object_id']] = diversity_signature(mesh)
        spectrum = np.sort(pdist(mesh.convex_hull.vertices))
        spectra[row['object_id']] = spectrum/spectrum[-1]
    pairs = [dict(a=a, b=b, **compare_diversity(signatures[a], signatures[b]))
             for a, b in itertools.combinations(ids, 2)]
    edges = [row for row in pairs if row.get('near_same_family')]
    groups = components(ids, edges)
    remove = [key for group in groups for key in group['remove']]
    retained = [key for key in ids if key not in remove]
    source_root = Path(objects[0]['source_asset']).parent
    proposals = json.loads((study/'audit/rescaled_geometry_candidates.json').read_text())['candidates']
    from stage13_dataset_audit import patch_audit
    candidates = []
    for row in proposals:
        key = 'abc_'+row['id']
        if key in ids:
            continue
        source = source_root/(key+'_native.obj')
        mesh = trimesh.load(source, force='mesh', process=False)
        mesh.vertices *= row['scale']
        z_translation = -float(mesh.bounds[0, 2]); mesh.vertices[:, 2] += z_translation
        signature = diversity_signature(mesh)
        near = [dict(object_id=other, **compare_diversity(signature, signatures[other]))
                for other in retained]
        spectrum = np.sort(pdist(mesh.convex_hull.vertices)); spectrum /= spectrum[-1]
        spectrum_comparisons = [dict(object_id=other, max_normalized_difference=float(abs(spectrum-spectra[other]).max()))
                                for other in retained if len(spectrum) == len(spectra[other])]
        duplicate = [row for row in spectrum_comparisons if row['max_normalized_difference'] < 1e-5]
        candidates.append(dict(object_id=key, source_asset=str(source), source_sha256=sha(source),
                               uniform_scale=row['scale'], z_translation_m=z_translation,
                               scaled_dimensions_m=mesh.extents.tolist(), vertices=len(mesh.vertices),
                               faces=len(mesh.faces), watertight=bool(mesh.is_watertight),
                               convex=bool(mesh.is_convex), diversity_signature=signature,
                               near_retained=[r for r in near if r.get('near_same_family')],
                               exact_spectrum_matches=duplicate,
                               closest_equal_hull_vertex_spectrum=min(spectrum_comparisons, key=lambda r: r['max_normalized_difference']) if spectrum_comparisons else None,
                               native_orientation_patch=patch_audit(mesh, radius=.0012, max_sag=.0003, min_z=.002)))
        meshes[key] = mesh; signatures[key] = signature
    proposal_ids = [row['object_id'] for row in candidates]
    candidate_pairs = [dict(a=a, b=b, **compare_diversity(signatures[a], signatures[b]))
                       for a, b in itertools.combinations(proposal_ids, 2)]
    return dict(schema='stage13_offline_family_diversity_v1',
                scope='Geometry-only analysis. No policy/runtime, physical episode or outcomes read or altered.',
                freeze_directory=freeze_directory, frozen_source_commit=json.loads((study/freeze_directory/'freeze.json').read_text())['source_commit'],
                plan_sha256=sha(plan_path), helper_sha256=sha(Path(__file__)),
                rules=dict(near_sphere_axis_ratio_max=1.01, near_sphere_continuous_shell_fraction_max=.01,
                           primitive_relative_difference_max=.05, normalization='Ignore translation, rotation and uniform scale.',
                           transitive_graph=True, retention='Earliest frozen preregistration rank, independent of grasp outcomes.'),
                input_count=len(ids), signatures=signatures, near_pairs=edges,
                connected_components=groups, retained_ids=retained, removed_ids=remove,
                retained_count=len(retained), replacement_candidates=candidates,
                replacement_near_pairs=[row for row in candidate_pairs if row.get('near_same_family')],
                limitations=['Family thresholds are an explicit proposed diversity amendment, separate from exact geometric identity.',
                             'Unsupported arbitrary shapes remain unresolved by primitive comparison.',
                             'Native-orientation tangent patches are a geometric precheck; actual settled qualification remains required.',
                             'No replacement is approved/installed or frozen by this analysis.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--freeze-directory', default='frozen_v3')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.study, args.freeze_directory)
    output = args.output or args.study/'audit/diversity_v3_analysis.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), removed_ids=result['removed_ids'],
                          retained_count=result['retained_count'], near_pairs=result['near_pairs'])), flush=True)


if __name__ == '__main__':
    main()
