"""30 fps private videos, asynchronous encoding; reuse exact policy pixels."""
import hashlib,json,queue,threading,time
import numpy as np
import imageio.v2 as imageio
class Recorder:
    def __init__(self,env,cfg,out,case,enabled=False):
        self.enabled=enabled;self.env=env;self.out=out;self.last=-1;self.count=0;self.error=None;self.encode_s=0.
        if not enabled:return
        (out/'recording_camera.json').write_text(json.dumps(dict(source='existing third_view RGB render product',resolution=[320,240],fps=30,policy_access='initial overhead only; later frames private',extra_render_products=0,scope='Fixed native overhead; six policy RGB copied after configured RGB noise. No additional camera changes the policy renderer graph.'),indent=2))
        self.q=queue.Queue();self.stream=(out/'recording_frames.jsonl').open('w',buffering=1048576)
        self.worker=threading.Thread(target=self.encode,daemon=True);self.worker.start()
    def encode(self):
        try:
            writers=[imageio.get_writer(self.out/n,fps=30,codec='libx264',quality=7,macro_block_size=2) for n in ('external_30fps.mp4','fingereye_30fps.mp4')]
            while True:
                item=self.q.get()
                if item is None:break
                start=time.perf_counter()
                for w,a in zip(writers,item):w.append_data(a)
                self.encode_s+=time.perf_counter()-start
            start=time.perf_counter()
            for w in writers:w.close()
            self.encode_s+=time.perf_counter()-start
        except BaseException as exc:self.error=repr(exc)
    def capture(self,step,phase,frames):
        if not self.enabled or step==self.last:return
        external=frames['third_view'][0][...,:3].copy()
        names=[f'{f}_{m}' for f in ('index','thumb') for m in ('front1','front2','back')]
        fingers=np.vstack([np.hstack([frames[n][0][...,:3] for n in names[i:i+3]]) for i in (0,3)])
        self.q.put((external,fingers.copy()))
        self.stream.write(json.dumps(dict(frame=self.count,step=step,time_s=step/120,phase=phase,sha256={n:hashlib.sha256(frames[n][0][...,:3].tobytes()).hexdigest() for n in names}))+'\n')
        self.count+=1;self.last=step
    def close(self):
        if not self.enabled:return
        self.q.put(None);self.worker.join();self.stream.close()
        (self.out/'video_timing.json').write_text(json.dumps(dict(encode_worker_s=self.encode_s,frames=self.count,fps=30,duration_s=self.count/30,first_sample_s=0,last_sample_s=self.last/120,error=self.error)))
        if self.error:raise RuntimeError(self.error)
