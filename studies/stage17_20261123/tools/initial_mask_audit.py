"""Same raw overhead input, isolate initial legacy/HS behavior without dynamics."""
from pathlib import Path
import importlib.util,json
import cv2,numpy as np
R=Path(__file__).resolve().parents[2];E=R/'stage17_evidence';s=importlib.util.spec_from_file_location('mask_audit',E/'controllers/hs_v2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);rows=[]
for case in ('A','B'):
 for light in ('.4','1','1.6'):
  ep=R/f'../../outputs/stage17/lighting/{case}_L{light if light!=".4" else "0.4"}_legacy/episode';o=json.loads(next((ep/'public_responses.jsonl').open()))['observation'];rgb=np.load(o['initial_overhead']['rgb_path']);masks={}
  for mode in ('legacy','hs'):
   m.S17_COLOR_MODE=mode;mask=m.color_foreground(rgb,initial=True);masks[mode]=mask;info=m._S3_ESTIMATE(o);rows.append(dict(case=case,light=float(light),mode=mode,pixels=int(mask.sum()),initial_hue=float(m.initial_hue(rgb)),center_m=np.asarray(info['center']).tolist(),half_m=np.asarray(info['half']).tolist(),changed_pixels=int((mask^masks['legacy']).sum())))
(E/'initial_mask_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
