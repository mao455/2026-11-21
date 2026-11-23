from pathlib import Path
import csv,hashlib,importlib.util,json
import cv2,numpy as np
R=Path(__file__).resolve().parents[2];E=R/'stage17_evidence';ep=R/'../../outputs/stage17/lighting/A_L1_legacy/episode';s=importlib.util.spec_from_file_location('vp',E/'controllers/hs_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);obs=[json.loads(x)['observation'] for x in (ep/'public_responses.jsonl').read_text().splitlines()];o=min(obs,key=lambda o:abs(o['step']-1200));hue=m.initial_hue(np.load(o['initial_overhead']['rgb_path']));rows=[];tiles=[]
for camera,slot in o['fingereye'].items():
 raw=np.load(slot['rgb_path']);hsv=cv2.cvtColor(raw,cv2.COLOR_RGB2HSV);m.S17_COLOR_MODE='legacy';ref=m.color_foreground(raw,hue)
 for scale in (.05,.1,.15,.2,.3,.4,.7,1.,1.3,1.6):
  h=hsv.copy();h[...,2]=np.clip(np.rint(h[...,2].astype(float)*scale),0,255).astype(np.uint8);rgb=cv2.cvtColor(h,cv2.COLOR_HSV2RGB)
  for mode in ('legacy','hs'):
   m.S17_COLOR_MODE=mode;mask=m.color_foreground(rgb,hue);rows.append(dict(step=o['step'],camera=camera,scale=scale,mode=mode,reference=int(ref.sum()),kept=int((mask&ref).sum()),below_legacy_V35=int((ref&(h[...,2]<=35)).sum()),reference_min_V=int(h[...,2][ref].min()) if ref.any() else None,reference_max_V=int(h[...,2][ref].max()) if ref.any() else None))
  if camera=='index_front1' and scale in (.1,.2,.4,1.,1.6):
   rgb=cv2.resize(rgb,(480,360),interpolation=cv2.INTER_NEAREST);cv2.putText(rgb,f'V x {scale}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,50,50),2);tiles.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
with (E/'v_floor.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if tiles:cv2.imwrite(str(E/'figures/v_floor.png'),np.concatenate(tiles,axis=1))
for scale in (.05,.1,.15,.2,.3,.4,.7,1.,1.3,1.6):
 print(scale,[(mode,sum(r['kept'] for r in rows if r['mode']==mode and r['scale']==scale)) for mode in ('legacy','hs')])
