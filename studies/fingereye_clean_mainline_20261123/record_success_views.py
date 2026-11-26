"""Private presentation recorder. External RGB is never a policy observation."""
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


class Recorder:
    def __init__(self, env, cfg, out, case):
        from isaaclab.sensors import Camera, CameraCfg
        import isaaclab.sim as sim_utils
        import torch
        self.env, self.cfg, self.out, self.case = env, cfg, out, case
        self.names = [f'{f}_{m}' for f in ('index', 'thumb') for m in ('front1', 'front2', 'back')]
        camera_cfg = CameraCfg(prim_path='/World/RecordingExternalCamera', update_period=0.,
                              height=720, width=960, data_types=['rgb'],
                              spawn=sim_utils.PinholeCameraCfg(focal_length=32., horizontal_aperture=20.955,
                                                              clipping_range=(.005, 5.)))
        self.camera = Camera(camera_cfg)
        self.camera._initialize_impl()
        self.eye, self.target = [.145, -.19, .125], [0., 0., .043]
        self.camera.set_world_poses_from_view(torch.tensor([self.eye], device=env.device),
                                             torch.tensor([self.target], device=env.device))
        self.external = imageio.get_writer(out/'external_20fps.mp4', fps=20, codec='libx264',
                                          quality=8, macro_block_size=2)
        self.combined = imageio.get_writer(out/'combined_20fps.mp4', fps=20, codec='libx264',
                                          quality=8, macro_block_size=2)
        self.fingers = imageio.get_writer(out/'fingereye_20fps.mp4', fps=20, codec='libx264',
                                         quality=8, macro_block_size=2)
        self.stream = (out/'recording_frames.jsonl').open('w')
        self.count = 0
        self.last = -1
        (out/'recording_camera.json').write_text(json.dumps(dict(eye=self.eye,target=self.target,
            resolution=[960,720],fps=20,physics_hz=120,policy_access=False,
            fingereye_native_resolution=[160,120],fingereye_display_scale=2,
            scope='Live rendered RGB during a new physics run. Fixed external camera, no pose replay.'),indent=2))

    def capture(self, step, phase):
        if step == self.last:
            return
        from round7c_dynamics import capture_cameras
        from fingereye.tasks.direct.fingereye.xhand_camera_recorder import to_uint8_batch
        frames = capture_cameras(self.env, self.cfg, self.names)
        self.camera.update(0., force_recompute=True)
        rgb = to_uint8_batch(self.camera.data.output['rgb'])[0][...,:3].copy()
        self.external.append_data(rgb)
        tiles=[]
        # Each row pairs the index and thumb view with the same camera orientation.
        for mount in ('front1','front2','back'):
            row=[]
            for finger in ('index','thumb'):
                name=f'{finger}_{mount}'
                tile=cv2.resize(frames[name][0][...,:3],(320,240),interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(tile,(0,0),(319,25),(20,24,32),-1)
                cv2.putText(tile,name,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
                row.append(tile)
            tiles.append(np.hstack(row))
        fingers=np.vstack(tiles)
        self.fingers.append_data(fingers)
        canvas=np.zeros((840,1600,3),np.uint8);canvas[:]=(17,23,32)
        canvas[60:780,:960]=rgb;canvas[60:780,960:]=fingers
        title=f"XHAND | {self.case.get('object_id', 'original_box')} | t = {step/120:05.2f} s | {phase}"
        cv2.putText(canvas,title,(24,36),cv2.FONT_HERSHEY_SIMPLEX,.82,(245,245,245),2,cv2.LINE_AA)
        cv2.putText(canvas,'EXTERNAL VIEW (recording only)',(24,809),cv2.FONT_HERSHEY_SIMPLEX,.6,(210,220,230),1,cv2.LINE_AA)
        cv2.putText(canvas,'LIVE FINGEREYE | synchronized 20 fps | 1x speed',(970,809),cv2.FONT_HERSHEY_SIMPLEX,.51,(210,220,230),1,cv2.LINE_AA)
        self.combined.append_data(canvas)
        self.stream.write(json.dumps(dict(frame=self.count,step=step,time_s=step/120,phase=phase,
            external_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            finger_rgb_sha256={n:hashlib.sha256(frames[n][0].tobytes()).hexdigest() for n in self.names}))+'\n')
        if step in (0,1440,2400,3360,3840,4320,4560,4680):
            imageio.imwrite(self.out/f'preview_{step:04d}.jpg',canvas)
        self.count+=1;self.last=step

    def close(self):
        for writer in (self.external,self.combined,self.fingers):writer.close()
        self.stream.close()
