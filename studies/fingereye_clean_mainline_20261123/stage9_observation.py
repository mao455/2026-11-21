"""Allowlist public inputs while letting the controller handle incomplete arrivals."""
import hashlib,json,re
from pathlib import Path
import numpy as np
from visual_observation import CAMERAS,OBS_KEYS,FRAME_KEYS,TOP_KEYS,ROBOT_KEYS,CALIBRATION_KEYS
class ObservationGuard:
    def __init__(self,public):
        self.public=Path(public).resolve();self.last=-1;self.frame=-1;self.top=None;self.seen=set();self.cameras=None;self.dimension=None;self.initial_geometry=None;self.geometry_stat=None
    def validate(self,response,initial=False):
        assert set(response)==({'ok','observation','robot_description'} if initial else {'ok','observation'})
        if initial:
            r=response['robot_description']
            self.cameras=set(r['fingereye_calibration'])
            self.dimension=len(r['joint_names'])
            from mainline.hand_description import HandDescription
            model=HandDescription(r['urdf_path'],joint_names=r['joint_names'])
            assert len(set(r['joint_names']))==self.dimension
            limits=np.asarray(r['joint_limits'])
            assert limits.shape==(self.dimension,2) and np.all(limits[:,0]<limits[:,1])
            assert np.allclose(limits[:,0],model.lower,atol=1e-5) and np.allclose(limits[:,1],model.upper,atol=1e-5)
        o=response['observation']
        geometry=o.get('initial_rgb_geometry')
        assert set(o)==OBS_KEYS|{'image_state'}|({'initial_rgb_geometry'} if geometry is not None else set())
        if initial:
            self.table_z=float(response['robot_description']['table_z'])
            if geometry is not None:
                from initial_rgb_geometry import read_public_geometry
                read_public_geometry(geometry,self.public,self.table_z,o['initial_overhead'])
                self.initial_geometry=json.dumps(geometry,sort_keys=True)
                stat=Path(geometry['path']).stat();self.geometry_stat=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
        assert (None if geometry is None else json.dumps(geometry,sort_keys=True))==self.initial_geometry
        if geometry is not None:
            stat=Path(geometry['path']).stat()
            assert (stat.st_size,stat.st_mtime_ns,stat.st_ino)==self.geometry_stat
        assert o['step']>=self.last and o['step']>=0
        assert abs(o['time_s']-o['step']/120)<1e-9
        for k in ('actual_q','actual_qvel'):
            a=np.asarray(o[k]);assert a.shape==(self.dimension,) and np.isfinite(a).all()
        c=o['image_state'];frames=o['fingereye']
        assert frames is None or isinstance(frames,dict) and set(frames)<=self.cameras
        if c is not None:
            assert set(c)=={'step','time_s','actual_q','actual_qvel','capture_start_wall_s','capture_done_wall_s','available_wall_s'}
            assert c['step']>=0 and c['step']%4==0 and abs(c['time_s']-c['step']/120)<1e-9
            assert c['capture_start_wall_s']<=c['capture_done_wall_s']<=c['available_wall_s']
            for k in ('actual_q','actual_qvel'):
                a=np.asarray(c[k]);assert a.shape==(self.dimension,) and np.isfinite(a).all()
        complete=c is not None and frames is not None and set(frames)==self.cameras
        for name,f in (frames or {}).items():
            assert set(f)==FRAME_KEYS and c is not None
            assert f['frame_id']==c['step']//4 and f['capture_step']==c['step']
            assert Path(f['rgb_path']).name==f'{name}_{f["frame_id"]:05d}.npy'
            assert Path(f['rgb_path']).resolve().parent==self.public
            assert (f['height'],f['width'])==(120,160)
        if initial:assert complete and c['step']==0
        top=o['initial_overhead'];assert set(top)==TOP_KEYS and top['capture_step']==0
        top_path=Path(top['rgb_path']).resolve();assert top_path.parent==self.public and top_path.name=='overhead_initial.npy'
        a=np.load(top_path,allow_pickle=False,mmap_mode='r');assert a.dtype==np.uint8 and a.shape==(top['height'],top['width'],3)
        frozen=(json.dumps(top,sort_keys=True),hashlib.sha256(top_path.read_bytes()).hexdigest())
        assert self.top is None or self.top==frozen
        # Future metadata is delivered to the inbox for rejection; it never
        # grants permission to publish a future file. Missing camera bundles
        # are normalized to one missing exposure, never partial six-view data.
        if complete and c['step']<=o['step']:
            for f in frames.values():
                a=np.load(f['rgb_path'],allow_pickle=False,mmap_mode='r');assert a.dtype==np.uint8 and a.shape==(120,160,3)
        if not complete:o['image_state']=None;o['fingereye']=None
        for p in self.public.iterdir():
            if p.name in self.seen:continue
            if p.name in ('overhead_initial.npy','initial.json'):self.seen.add(p.name);continue
            if geometry is not None and p.name=='initial_rgb_geometry.json':self.seen.add(p.name);continue
            assert any(re.fullmatch(re.escape(n)+r'_\d{5}\.npy',p.name) for n in self.cameras)
            assert int(p.stem.rsplit('_',1)[1])<=o['step']//4
            a=np.load(p,allow_pickle=False,mmap_mode='r');assert a.dtype==np.uint8 and a.shape==(120,160,3)
            self.seen.add(p.name)
        if initial:
            r=response['robot_description'];assert ROBOT_KEYS<=set(r)<=ROBOT_KEYS|{'contact_mode','source_urdf_path','base_joint_names','palm_link','root_link','finger_chains','source_to_runtime_joint_names','pads'} and set(r['fingereye_calibration'])==self.cameras
            for n,v in r['fingereye_calibration'].items():assert set(v)==CALIBRATION_KEYS and v['logical_name']==n
        self.top=frozen;self.last=o['step']
        if complete:self.frame=max(self.frame,c['step']//4)
