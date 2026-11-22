"""Pre-outcome geometry-only paired evaluation registration (private inputs)."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.distance import pdist
from stage13_geometry_identity import primitive_signature, compare_signatures
HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source-lab',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
 if (HERE/'STAGE13_OBJECTS.json').exists():raise RuntimeError('registration already exists')
 manifest=a.source_lab.parent/'manifest.csv';eligible=[];rejected=[];distance_groups=[]
 for item in sorted(csv.DictReader(manifest.open()),key=lambda x:int(x['id'])):
  if item['id'] in ('781904','781914'):
   rejected.append(dict(id=item['id'],reason='previously_declared_pyramid_diagnostic_only'));continue
  asset=a.source_lab/'assets/objects/manifest_native_obj'/('abc_'+item['id']+'_native.obj')
  if not asset.exists() or int(item['num_faces'])>16000:continue
  mesh=trimesh.load(asset,force='mesh',process=True);dims=mesh.extents
  if not (.008-1e-8<=dims[2]<=.050+1e-8 and np.all(dims[:2]>=.006-1e-8) and np.all(dims[:2]<=.060+1e-8)):
   rejected.append(dict(id=item['id'],reason='native_dimensions'));continue
  hull=mesh.convex_hull;ratio=abs(mesh.volume)/max(hull.volume,1e-15)
  if not mesh.is_watertight or not np.isclose(ratio,1.,rtol=1e-5):
   rejected.append(dict(id=item['id'],reason='original_convex_audit_scope'));continue
  # Reject tips/slopes-only sides using static geometry, never rollout outcomes.
  side=float(np.sum(mesh.area_faces[abs(mesh.face_normals[:,2])<.5])/mesh.area)
  if side<.25:
   rejected.append(dict(id=item['id'],reason='insufficient_ordinary_side_area',side_area_fraction=side));continue
  primitive=primitive_signature(mesh)
  primitive_duplicate=next(((g,compare_signatures(primitive,g['primitive'])) for g in distance_groups if compare_signatures(primitive,g['primitive']).get('same_up_to_uniform_scale')),None)
  if primitive_duplicate:
   representative,identity=primitive_duplicate
   rejected.append(dict(id=item['id'],reason='same_analytic_primitive_up_to_rotation_translation_uniform_scale',representative=representative['id'],identity=identity));continue
  distances=np.sort(pdist(hull.vertices));norm=distances/max(distances.max(),1e-12)
  duplicate=next((g for g in distance_groups if len(norm)==len(g['d']) and np.max(abs(norm-g['d']))<1e-5),None)
  if duplicate:
   rejected.append(dict(id=item['id'],reason='same_shape_up_to_rotation_translation_uniform_scale',representative=duplicate['id']));continue
  distance_groups.append(dict(id=item['id'],d=norm,primitive=primitive))
  eig=np.linalg.eigvalsh(np.cov(hull.vertices.T))
  descriptor=[*np.sort(dims)/max(dims),hull.volume/np.prod(dims),eig[0]/max(eig[-1],1e-15),side,len(hull.vertices)/100.]
  eligible.append(dict(id=item['id'],object_id='abc_'+item['id'],source_asset=str(asset),sha256=sha(asset),dims_m=dims.tolist(),faces=len(mesh.faces),volume_ratio=float(ratio),descriptor=descriptor,side_area_fraction=side,primitive_identity=primitive,shape_signature_sha256=hashlib.sha256(np.rint(norm/1e-5).astype('<i8').tobytes()).hexdigest()))
 if len(eligible)<40:raise RuntimeError('Only '+str(len(eligible))+' distinct eligible shapes; no physical outcomes have been run')
 x=np.asarray([r['descriptor'] for r in eligible]);x=(x-np.median(x,axis=0))/np.maximum(x.std(0),.05)
 indices=[0]
 while len(indices)<40:
  d=np.min(np.linalg.norm(x[:,None]-x[indices][None],axis=2),axis=1);d[indices]=-1;indices.append(int(np.argmax(d)))
 rows=[];runs=[];cases=HERE/'stage13_cases';cases.mkdir(exist_ok=True)
 for rank,i in enumerate(indices):
  row=dict(eligible[i],selection_rank=rank)
  dst=HERE/'assets/stage13'/Path(row['source_asset']).name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['source_asset'],dst);row['asset']=str(dst.relative_to(LAB))
  common=dict(object_id=row['object_id'],dx=-.0012,dy=.0018,yaw=.27 if rank%2==0 else -.31,mass_kg=.007,friction=.82,noise_seed=13000+rank)
  for condition,sigma in (('clean',0.),('noise',3.)):
   case=dict(common,rgb_noise_std=sigma);name=f'{rank:02d}_{row["object_id"]}_{condition}'
   path=cases/(name+'.json');path.write_text(json.dumps(case,indent=2)+'\n')
   runs.append(dict(name=name,object_id=row['object_id'],condition=condition,case=str(path.relative_to(HERE)),profile='fast',fingers='auto',max_attempts=3,mode='minimal',phase='heldout'))
  row['case_base']=common;rows.append(row)
 protocol=dict(schema=1,source_manifest=str(manifest),source_manifest_sha256=sha(manifest),selection_uses_results=False,minimum_initial_height_m=.008,maximum_initial_height_m=.050,xy_extents_m=[.006,.060],max_faces=16000,watertight_convex_only=True,minimum_side_area_fraction=.25,diagnostic_pyramid_ids_excluded=['abc_781904','abc_781914'],distinct_shape_test='cross-tessellation analytic-cylinder and ordinary-box fits plus extreme hull pairwise distances normalized by diameter at1e-5; rigid/uniform-scale matches excluded; unresolved arbitrary surfaces are not a mathematical noncongruence proof',selection='numeric-ID ordered survey; deterministic standardized-descriptor farthest point; 40 different fixtures under declared identity checks',eligible_distinct_shapes=len(eligible),object_count=40,episode_count=80,physics_hz=120,control_hz=20,rgb_video_fps=30,noise=dict(std_0_255=3.,six_fingereye_only=True,initial_overhead=False),success='unchanged evaluator: commanded >=29.5mm, actual lift>=28mm, object clearance>=5mm for continuous2s, no drop after qualifying window; nominal lift30mm',history_scope='may include historically seen objects; candidate80 frozen after development; no unseen-generalization claim',objects=rows,runs=runs)
 (HERE/'STAGE13_OBJECTS.json').write_text(json.dumps(protocol,indent=2)+'\n');a.audit.parent.mkdir(parents=True,exist_ok=True);a.audit.write_text(json.dumps(dict(eligible=eligible,rejected=rejected,registration_sha256=sha(HERE/'STAGE13_OBJECTS.json')),indent=2)+'\n')
 print(json.dumps(dict(eligible_distinct=len(eligible),registered=40,episodes=80,sha256=sha(HERE/'STAGE13_OBJECTS.json'))))
if __name__=='__main__':main()
