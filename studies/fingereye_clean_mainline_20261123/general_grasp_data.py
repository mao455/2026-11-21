"""Private asset survey and identity split. NEVER import into a policy.

Native dimensions and geometry are used only to construct evaluation scenes.
The sandbox exposes robot assets and public observations, not this registry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
MANIFEST = LAB.parent / "manifest.csv"
ASSETS = LAB / "assets/objects/manifest_native_obj"
SPLIT = HERE / "general_object_split.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def survey(out):
    rows, rejected = [], []
    source = list(csv.DictReader(MANIFEST.open()))
    for item in source:
        path = ASSETS / f"abc_{item['id']}_native.obj"
        if not path.exists():
            rejected.append(dict(id=item['id'], reason='missing_native_obj'))
            continue
        # Preselect only by physical scale/mesh resource cost, never grasp outcomes.
        if int(item['num_faces']) > 16000:
            rejected.append(dict(id=item['id'], reason='over_16000_faces'))
            continue
        mesh = trimesh.load(path, force='mesh', process=True)
        dims = mesh.extents
        if not (.008 <= max(dims) <= .04 and min(dims) >= .003
                and .006 <= dims[2] <= .035):
            rejected.append(dict(id=item['id'], reason='native_size_outside_stage1_budget',
                                 dims_m=dims.tolist()))
            continue
        hull = mesh.convex_hull
        volume_ratio = float(abs(mesh.volume) / max(hull.volume, 1e-15))
        if not mesh.is_watertight or not .03 <= volume_ratio <= 1.001:
            rejected.append(dict(id=item['id'], reason='nonwatertight_or_degenerate'))
            continue
        eig = np.linalg.eigvalsh(np.cov(mesh.vertices.T))
        descriptor = [*np.sort(dims) / max(dims), volume_ratio,
                      hull.volume / np.prod(dims), eig[0] / max(eig[-1], 1e-15),
                      mesh.area / max(hull.area, 1e-15),
                      float(mesh.euler_number < 2)]
        rows.append(dict(id=item['id'], object_id='abc_' + item['id'],
                         asset=str(path.relative_to(LAB)), sha256=sha(path),
                         source_parquet=item['source_parquet'],
                         source_global_row=int(item['global_row']),
                         source_label=item['label'], source_split=item['split'],
                         dims_m=dims.tolist(), faces=len(mesh.faces),
                         volume_ratio=volume_ratio, euler_number=int(mesh.euler_number),
                         geometry_group='concave_or_holed' if volume_ratio < .9 or mesh.euler_number < 2 else 'near_convex',
                         descriptor=np.asarray(descriptor).tolist()))
    out.write_text(json.dumps(dict(source_manifest=str(MANIFEST),
        manifest_sha256=sha(MANIFEST), manifest_rows=len(source),
        native_asset_count=len(list(ASSETS.glob('*.obj'))), eligible=rows,
        rejected=rejected, selection_uses_grasp_results=False), indent=2) + '\n')
    return rows


def select(rows, count=18):
    # Deterministic farthest-point sampling of shape descriptors, no success filtering.
    # The unchanged strict interior audit rejects concavity. Do not silently
    # convexify an evaluation object or weaken its penetration test.
    rows = sorted((r for r in rows if r['id'] != '90621'
                   and np.isclose(r['volume_ratio'], 1., rtol=1e-5)),
                  key=lambda r: int(r['id']))
    if len(rows) < count:
        raise ValueError('Not enough eligible native assets')
    x = np.asarray([r['descriptor'] for r in rows])
    x = (x - np.median(x, axis=0)) / np.maximum(np.std(x, axis=0), .05)
    selected = [int(np.argmin(np.asarray([r['descriptor'][4] for r in rows])))]
    while len(selected) < count:
        distance = np.min(np.linalg.norm(x[:, None] - x[selected][None], axis=2), axis=1)
        distance[selected] = -1
        selected.append(int(np.argmax(distance)))
    result = []
    for rank, index in enumerate(selected):
        row = dict(rows[index], selection_rank=rank,
                   split='development' if rank % 3 == 0 else 'frozen_test')
        row['native_asset'] = row['asset']
        source = LAB / row['native_asset']
        destination = HERE / 'assets/general_stage1' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != row['sha256']:
            raise RuntimeError('Existing fixture differs')
        if not destination.exists():
            shutil.copyfile(source, destination)
        row['asset'] = str(destination.relative_to(LAB))
        # Same conditions for both algorithms; identity never enters public data.
        row['case'] = dict(object_id=row['object_id'], dx=-.0012, dy=.0018,
                           yaw=.27 if rank % 2 == 0 else -.31,
                           mass_kg=.0052, friction=.82, noise_seed=1900 + rank)
        result.append(row)
    return dict(schema=1, objects=result, native_size_preserved=True,
        policy_can_read_split=False, training_objects=[],
        identity_disjoint=True, selection='deterministic farthest-point shape sampling',
        heldout_scope='Unseen in this stage; not a claim of absence from all historical projects',
        eligible_for_unchanged_convex_audit=len(rows),
        scope_limit='Concave/holed assets surveyed, excluded from physical tests because the original interior audit rejects them; no convexification',
        collision_scope='Original evaluator: authored object meshes / existing PhysX cooking')


def registry():
    return {r['object_id']: r for r in json.loads(SPLIT.read_text())['objects']}


def install():
    """Extend only the private scene registry, following the Round15 adapter pattern."""
    import round14_objects
    if getattr(round14_objects, '_general_installed', False):
        return
    objects = registry()
    previous = round14_objects.create
    round14_objects.OBJECTS.update({k: ('native_dataset', ()) for k in objects})
    def create(name):
        if name not in objects:
            return previous(name)
        path = LAB / objects[name]['asset']
        if sha(path) != objects[name]['sha256']:
            raise RuntimeError('Frozen native object changed: ' + name)
        return path
    round14_objects.create = create
    round14_objects._general_installed = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey', type=Path, required=True)
    parser.add_argument('--split', type=Path, default=SPLIT)
    args = parser.parse_args()
    if args.survey.exists() or args.split.exists():
        raise FileExistsError('Keep frozen survey/split; choose a new output')
    rows = survey(args.survey)
    args.split.write_text(json.dumps(select(rows), indent=2) + '\n')
    print('eligible', len(rows), 'selected', 18)
