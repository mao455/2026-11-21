"""Offline authored USD/URDF middle-module equivalence; no cooked-piece claim."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
from scipy.spatial import cKDTree
from pxr import Usd
R=Path(__file__).resolve().parents[2];E=R/'stage17_evidence';p=json.loads((E/'geometry_figure_provenance.json').read_text());ep=Path(p['episode']);scene=json.loads((ep/'scene.json').read_text())
def load(n,path):
 s=importlib.util.spec_from_file_location(n,path);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
m=load('cad_policy',E/'controllers/hs_v2.py');g=load('cad_geometry',ep/'harness_snapshot/geometry.py');robot=m.Robot(scene['robot_description']['urdf_path']);asset=Path(p['urdf']).parent/'xhand3camFE_left_usd_convex_decomposition/xhand3camFE_left.usd';stage=Usd.Stage.Open(str(asset));stage.SetEditTarget(stage.GetSessionLayer());[prim.SetInstanceable(False) for prim in list(stage.Traverse()) if prim.IsInstance()];names=['left_hand_middle_link2',*['left_hand_middle_camera_component'+str(i)+'_link' for i in (1,2,3)]];meshes,manifest=g.usd_collision_meshes(stage,str(stage.GetDefaultPrim().GetPath()),names);rows=[]
for n in names:
 a=robot.surfaces[n];b=meshes[n];error=max(cKDTree(a.vertices).query(b.vertices)[0].max(),cKDTree(b.vertices).query(a.vertices)[0].max());rows.append(dict(link=n,urdf_vertices=len(a.vertices),usd_vertices=len(b.vertices),urdf_faces=len(a.faces),usd_faces=len(b.faces),bidirectional_nearest_vertex_max_m=float(error),passed=bool(error<1e-7)))
assert all(r['passed'] for r in rows),rows
out=dict(scope='authored mesh geometry in each body local frame, not cooked PhysX pieces',usd=str(asset),usd_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),urdf=p['urdf'],urdf_sha256=p['urdf_sha256'],mesh_checks=rows,usd_collision_manifest=manifest);(E/'middle_usd_equivalence.json').write_text(json.dumps(out,indent=2)+'\n');print(rows)
