"""Paired HSV-value replay of unmodified public RGB; no physics claims."""
import argparse,copy,csv,hashlib,importlib.util,json,tempfile,time
from pathlib import Path
import cv2,numpy as np

def load_policy(path):
 s=importlib.util.spec_from_file_location('v_replay_policy',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 m=load_policy(a.controller);obs=[json.loads(x)['observation'] for x in (a.episode/'public_responses.jsonl').read_text().splitlines()];description=json.loads((a.episode/'public/initial.json').read_text())['robot_description']
 # Fixed exposure positions, covering pregrasp/refinement, proof and hold.
 chosen=[min(obs,key=lambda o:abs(o['step']-s)) for s in (1200,1600,2600,3800)]
 base=m._S3_ESTIMATE(obs[0]);hue=m.initial_hue(np.load(obs[0]['initial_overhead']['rgb_path']));robot=m.Robot(description['urdf_path']);rows=[];geometry=[];sources={}
 with tempfile.TemporaryDirectory(prefix='stage17-v-') as tmp:
  for source in chosen:
   for scope in ('near_only','all_rgb'):
    for scale in (.4,.7,1.,1.3,1.6):
     current=copy.deepcopy(source)
     for name,slot in list(current['fingereye'].items())+[('overhead',current['initial_overhead'])]:
      original=Path(slot['rgb_path']);raw=np.load(original);sources[str(original)]=hashlib.sha256(original.read_bytes()).hexdigest();hsv=cv2.cvtColor(raw,cv2.COLOR_RGB2HSV);mult=1. if name=='overhead' and scope=='near_only' else scale;value=hsv[...,2].astype(float)*mult;newhsv=hsv.copy();newhsv[...,2]=np.clip(np.rint(value),0,255).astype(np.uint8);rgb=cv2.cvtColor(newhsv,cv2.COLOR_HSV2RGB);target=Path(tmp)/f'{name}.npy';np.save(target,rgb);slot['rgb_path']=str(target)
      initial=name=='overhead';reference=m.color_foreground(raw,hue if not initial else None,initial=initial)
      for mode in ('legacy','hs'):
       m.S17_COLOR_MODE=mode;mask=m.color_foreground(rgb,hue if not initial else None,initial=initial);count=int(reference.sum());rows.append(dict(step=source['step'],scope=scope,scale=scale,camera=name,mode=mode,reference_pixels=count,kept_reference=int((mask&reference).sum()),foreground_pixels=int(mask.sum()),unknown_pixels=int((~m.color_reliability(rgb)).sum()),mean_V=float(newhsv[...,2].mean()),reference_mean_V=float(newhsv[...,2][reference].mean()) if count else None,clipped_pixels=int((value>255).sum()),zero_pixels=int((newhsv[...,2]==0).sum()),quantization_mae_DN=float(np.abs(np.clip(value,0,255)-newhsv[...,2]).mean())))
     for mode in ('legacy','hs'):
      m.S17_COLOR_MODE=mode;start=time.perf_counter()
      try:
       info=m._S3_ESTIMATE(current) if scope=='all_rgb' else base;m._S14_SHAPE_PRIOR=info;v=m.Stage3LiveVolume(robot,description['fingereye_calibration'],info);report=v.carve(current);geometry.append(dict(step=source['step'],scope=scope,scale=scale,mode=mode,valid=report['valid'],reason=report.get('reason'),views=len(report.get('views',{})),voxels=report.get('occupied_voxels'),compute_s=time.perf_counter()-start))
      except Exception as exc:geometry.append(dict(step=source['step'],scope=scope,scale=scale,mode=mode,valid=False,reason=repr(exc),compute_s=time.perf_counter()-start))
 for filename,data in [('pixels.csv',rows),('geometry.csv',geometry)]:
  with (a.output/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (a.output/'provenance.json').write_text(json.dumps(dict(scope='same-input perception replay only',episode=str(a.episode),controller_sha256=hashlib.sha256(a.controller.read_bytes()).hexdigest(),rgb_noise_std=0,hue=hue,transform='OpenCV uint8 HSV; H/S preserved before HSV-to-RGB conversion; V rint ties-to-even, clip 0..255; RGB requantization can shift recovered H/S',sources=sources),indent=2)+'\n')
 print('pixels',len(rows),'geometry',len(geometry))
if __name__=='__main__':main()
