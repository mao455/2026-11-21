from pathlib import Path
import csv,json,numpy as np,cv2
R=Path(__file__).resolve().parents[2];E=R/'stage17_evidence';rows=[]
for cohort in ('lighting','lighting_v2'):
 for session in sorted((R/'../../outputs/stage17'/cohort).iterdir()):
  ep=session/'episode';f=ep/'public_responses.jsonl'
  if not f.exists():continue
  obs=json.loads(next(f.open()))['observation'];light=json.loads((ep/'light_intervention.json').read_text());cfg=json.loads((session/'stage9_configuration.json').read_text());assert cfg['case']['rgb_noise_std']==0;assert (cfg['physics_hz'],cfg['control_hz'],cfg['camera_fps'])==(120,20,30)
  for l in light['lights']:assert l['intensity']==l['original_intensity']*light['multiplier']
  for camera,slot in [('overhead',obs['initial_overhead']),*obs['fingereye'].items()]:
   rgb=np.load(slot['rgb_path']);v=rgb.max(-1);rows.append(dict(session=session.name,cohort=cohort,light=light['multiplier'],camera=camera,mean_V=float(v.mean()),max_V=int(v.max()),clipped_V_pixels=int((v==255).sum()),pixels=v.size))
if rows:
 with (E/'rendered_brightness.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# A aggregate digital replay, kept counts reference pixels, and geometry stage outcome.
for dirname in ('v_replay','v_replay_v2'):
 px=list(csv.DictReader((E/dirname/'pixels.csv').open()));geo=list(csv.DictReader((E/dirname/'geometry.csv').open()));out=[]
 for scope in ('near_only','all_rgb'):
  for scale in (.4,.7,1.,1.3,1.6):
   for mode in ('legacy','hs'):
    selected=[r for r in px if r['scope']==scope and float(r['scale'])==scale and r['mode']==mode and (scope=='all_rgb' or r['camera']!='overhead')];g=[r for r in geo if r['scope']==scope and float(r['scale'])==scale and r['mode']==mode];out.append(dict(scope=scope,scale=scale,mode=mode,reference=sum(int(r['reference_pixels']) for r in selected),kept=sum(int(r['kept_reference']) for r in selected),mean_V=float(np.mean([float(r['mean_V']) for r in selected])),clipped_pixels=sum(int(r['clipped_pixels']) for r in selected),mean_quantization_mae_DN=float(np.mean([float(r['quantization_mae_DN']) for r in selected])),geometry_valid=sum(r['valid']=='True' for r in g),geometry_total=len(g),geometry_failures=sorted(set(r['reason'] for r in g if r['valid']!='True'))))
 with (E/dirname/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(dirname,json.dumps(out[10:],indent=2))
print('brightness rows',len(rows))
