"""Recorded-image and full CAD evidence; does not alter policy or physics."""
from pathlib import Path
import csv,hashlib,importlib.util,json,sys
import cv2,numpy as np,trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'stage17_evidence';F=E/'figures'
def load(path):
 s=importlib.util.spec_from_file_location('offline_policy',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
m=load(E/'controllers/hs_v2.py')
# Annotation is a projection of full meshes at actual measured q, not a contact simulation.
p=json.loads((E/'geometry_figure_provenance.json').read_text());ep=Path(p['episode']);sc=json.loads((ep/'scene.json').read_text());r=[json.loads(x) for x in (ep/'steps.jsonl').read_text().splitlines()][p['step']];robot=m.Robot(sc['robot_description']['urdf_path']);robot.configure_contacts(3);fk=robot.fk(r['actual_q']);inv=np.linalg.inv(fk['left_hand_middle_link2']);shapes={}
for n,s in robot.surfaces.items():
 if n=='left_hand_middle_link2' or 'middle_camera' in n:
  sh=s.copy();sh.apply_transform(inv@fk[n]);shapes[n]=sh
colors={'left_hand_middle_link2':'#b1b5bb','left_hand_middle_camera_component1_link':'#4c91c6','left_hand_middle_camera_component2_link':'#f2bc6b','left_hand_middle_camera_component3_link':'#71b47e'}
fig,axes=plt.subplots(1,3,figsize=(16,7))
for ax,(title,horiz,depth,sign) in zip(axes,[('Palmar (+x)',1,0,1),('Side (+y)',0,1,1),('Dorsal (-x)',1,0,-1)]):
 tris=np.concatenate([s.triangles for s in shapes.values()])*1000;cs=np.concatenate([[colors[n]]*len(s.faces) for n,s in shapes.items()]);order=np.argsort(tris[:,:,depth].mean(1)*sign)
 ax.add_collection(PolyCollection(tris[order][:,:,[horiz,2]],facecolors=cs[order],edgecolors='none',rasterized=True,alpha=.85))
 def mark(v,label,color,target):
  v=np.array(v)*1000;ax.scatter(v[horiz],v[2],s=38,c=color,edgecolor='white',zorder=10);ax.annotate(label,xy=(v[horiz],v[2]),xytext=target,fontsize=9,color=color,arrowprops=dict(arrowstyle='-',color=color),zorder=11,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
 mark(p['loaded_contact_link2_m'],'Loaded component1\n2.47 mN; palmar face','red',(-34,7))
 mark(p['designated_pad_link2_m'],'component3\ndesignated pad','darkgreen',(17,27))
 for name,pos in p['lens_link2_m'].items():
  short=name.split('camera_')[-1];target={'front1':(-34,40),'front2':(-34,56),'back':(16,47)}[short];mark(pos,'lens '+short,'black',target)
 ax.set_xlim(-38,39);ax.set_ylim(-11,69);ax.set_aspect('equal');ax.grid(alpha=.25);ax.set_xlabel(('y' if horiz==1 else 'x')+' in middle_link2 (mm)');ax.set_ylabel('z (mm)');ax.set_title(title+'\nmarkers projected, including hidden side')
fig.suptitle('20_safe: actual recorded q + full URDF CAD, step 3576 / 29.8083 s\nBlue housing component1; green component3; orange component2; gray middle_link2. Red point is measured, not an IK target.',fontsize=12);fig.tight_layout(rect=(0,0,1,.91));fig.savefig(F/'middle_three_views.png',dpi=170);plt.close(fig)
p['component_bounds_link2_m']={n:s.bounds.tolist() for n,s in shapes.items()};usdroot=Path(sc['robot_description']['urdf_path']).parent/'xhand3camFE_left_usd_convex_decomposition';p['usd_files']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(usdroot.rglob('*.usd'))};p['usd_note']='Resolved backend convex-decomposition asset set, provenance from recorded harness; actual mesh equivalence checked separately; cooked internal pieces unavailable.';(E/'geometry_figure_provenance.json').write_text(json.dumps(p,indent=2)+'\n')
# First numerically feasible normal-three candidates, refused before dispatch.
data=json.loads((ROOT/'stage16_evidence/numerics/fixed_candidate_audit.json').read_text());fig=plt.figure(figsize=(14,7));out=[]
for i,c in enumerate(data['cases']):
 row=c['rows'][2];meta=row['meta'];episode=Path(c['episode']);scene=json.loads((episode/'scene.json').read_text());robot=m.Robot(scene['robot_description']['urdf_path']);robot.configure_contacts(3);poses=robot.fk(row['q']);obs=json.loads(next((episode/'public_responses.jsonl').open()))['observation'];info=m._S3_ESTIMATE(obs);hull=trimesh.convex.convex_hull(np.array(info['collision_points']));ax=fig.add_subplot(1,2,i+1,projection='3d');first=meta['endpoint']['first_failed_guard'];conflicts=set(sum([w['links'] for w in meta['endpoint']['clearance_witnesses'] if not w['passed']],[]));pts=[]
 for n,sh in robot.surfaces.items():
  if any(s in n for s in ('thumb','index','middle')):
   sh=sh.copy();sh.apply_transform(poses[n]);tri=sh.triangles;col=Poly3DCollection(tri*1000,facecolor=('#d63737' if n in conflicts else '#8baabd'),edgecolor='none',alpha=(.75 if n in conflicts else .14),rasterized=True);ax.add_collection3d(col);pts.append(sh.vertices)
 ax.add_collection3d(Poly3DCollection(hull.triangles*1000,facecolor='#e4bc25',edgecolor='none',alpha=.55,rasterized=True));case='20' if i==0 else '34';ax.set_xlim(-90,90);ax.set_ylim(-90,90);ax.set_zlim(0,160);ax.set_box_aspect((1,1,1));ax.view_init(16,-65);ax.set_xlabel('world x mm');ax.set_ylabel('world y mm');ax.set_zlabel('z mm');ax.set_title(f'{case}: normal three candidate {row["candidate_index"]}, NOT dispatched\nfirst: middle component1 vs RGB envelope\n{first["clearance_m"]*1000:.3f} mm vs 0 mm threshold',fontsize=10)
 out.append(dict(case=case,q=row['q'],candidate_index=row['candidate_index'],first_rejection=meta['first_rejection'],endpoint=meta['endpoint'],source_episode=str(episode),rgb_collision_points=np.asarray(info['collision_points']).tolist()))
fig.suptitle('Recorded first rejected feasible-normal candidate poses. Red = rejected component1/link2; yellow = RGB envelope.\nOffline pose rendering; not a physical trajectory or proof of true penetration.',fontsize=12);fig.tight_layout(rect=(0,0,1,.9));fig.savefig(F/'first_rejected_20_34.png',dpi=150);plt.close(fig);(E/'first_rejected_20_34.json').write_text(json.dumps(out,indent=2)+'\n')
# Real finger-camera video frames, before/after proof and consecutive frames.
import cv2,pathlib,json,numpy as np
p=pathlib.Path('/home/CNS2025331827/docs');out=p/'fingereye_stage17/FingerEyeLab/experiments/rax_dexbin_tinyobj/stage17_evidence/figures';out.mkdir(parents=True,exist_ok=True)
cases={'20_safe':('fingereye_stage16/FingerEyeLab/outputs/stage16/diagnostic/20_safe/episode',[0,14,19,24,29.8,31.5]),'B':('fingereye_stage15_accept_final/FingerEyeLab/outputs/stage15_acceptance/batch/episodes/recovery_poseB_seed15002/launch_001/episode',[27,28,29,35,36,49,55,57.8]),'C':('fingereye_stage15_accept_final/FingerEyeLab/outputs/stage15_acceptance/batch/episodes/hard_poseC_seed15003/launch_001/episode',[28,29.5,30,37,39,56,76,78]),'D0':('fingereye_stage15_accept_final/FingerEyeLab/outputs/stage15_acceptance/batch/episodes/divergence_poseD_dn0_seed15004/launch_001/episode',[14.3,14.333,14.367,23,25,30,39,41.8]),'D20.4':('fingereye_stage15_accept_final/FingerEyeLab/outputs/stage15_acceptance/batch/episodes/divergence_poseD_dn20.4_seed15004/launch_001/episode',[14.3,14.333,14.367,23,25,30,39,41.8])}

manifest=[]
for name,(rel,times) in cases.items():
 ep=Path('/home/CNS2025331827/docs')/rel;video=ep/'fingereye_30fps.mp4';cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));chosen=([894,895,896,942,943,944] if name=='20_safe' else [840,841,842,1650,1651,1652] if name=='B' else [885,886,887,2280,2281,2282] if name=='C' else [429,430,431,1197,1198,1199]);tiles=[]
 for frameid in chosen:
  frameid=min(count-1,frameid);cap.set(cv2.CAP_PROP_POS_FRAMES,frameid);ok,frame=cap.read()
  if not ok:continue
  frame=cv2.resize(frame,(640,240));cv2.putText(frame,f'{name} near RGB frame {frameid}, {frameid/fps:.3f}s',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);tiles.append(frame);manifest.append(dict(case=name,video=str(video),frame=frameid,time_s=frameid/fps))
 cap.release()
 if tiles:cv2.imwrite(str(F/(name+'_near_consecutive.png')),np.concatenate(tiles,axis=0))
(E/'near_video_frames.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('created CAD and near video evidence')
