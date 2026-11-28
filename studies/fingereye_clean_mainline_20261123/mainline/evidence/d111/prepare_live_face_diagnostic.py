"""Create an explicitly separate physical yaw intervention; policy is unchanged."""
from pathlib import Path
import argparse, hashlib, json

E=Path(__file__).resolve().parents[3]

def prepare(out, yaw):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    backend=out/'diagnostic_backend.py';episode=out/'diagnostic_episode.py';runner=out/'diagnostic_run_stage17.py'
    hook='''                    for _ in range(count):
                        if not _diag_applied and request.get('phase')=='s5_translate':
                            _diag_applied=True
                            _state=env.object.data.root_state_w[ids].clone()
                            _before=_state[:,:7].clone()
                            _q=_state[:,3:7].clone()
                            _cz=float(np.cos(_DIAG_YAW/2));_sz=float(np.sin(_DIAG_YAW/2))
                            _state[:,3]=_cz*_q[:,0]-_sz*_q[:,3]
                            _state[:,4]=_cz*_q[:,1]-_sz*_q[:,2]
                            _state[:,5]=_cz*_q[:,2]+_sz*_q[:,1]
                            _state[:,6]=_cz*_q[:,3]+_sz*_q[:,0]
                            env.object.write_root_pose_to_sim(_state[:,:7],env_ids=ids)
                            env.object.write_root_velocity_to_sim(torch.zeros_like(_state[:,7:13]),env_ids=ids)
                            save(out/'diagnostic_yaw_intervention.json',dict(step=len(records),phase=request.get('phase'),yaw_rad=_DIAG_YAW,
                                before_pose=_before.cpu().numpy().tolist(),assigned_pose=_state[:,:7].cpu().numpy().tolist(),
                                private_diagnostic_only=True,policy_receives_only_normal_RGB_q_qvel=True))
'''
    backend.write_text("import sys\nfrom pathlib import Path\nE=Path("+repr(str(E))+ ")\nsys.path.insert(0,str(E))\nimport stage16_backend as b\noriginal=b.transform\nHOOK="+repr(hook)+"\ndef transform(source,mode):\n    source=original(source,mode)\n    a='        for line in timed_requests(sys.stdin):'\n    assert source.count(a)==1\n    source=source.replace(a,'        _diag_applied=False\\n        _DIAG_YAW='+str("+repr(yaw)+")+'\\n'+a)\n    a='                    for _ in range(count):\\n'\n    assert source.count(a)==1\n    return source.replace(a,HOOK)\nb.transform=transform\nfrom stage14_resources import simulator_slot\nwith simulator_slot(Path(sys.argv[sys.argv.index('--output')+1])):b.main()\n")
    episode.write_text("from pathlib import Path\nimport sys\nE=Path("+repr(str(E))+")\nsys.path.insert(0,str(E))\np=E/'stage16_episode.py'\ns=p.read_text().replace(\"HERE / 'stage16_backend.py'\",\"Path(\"+repr("+repr(str(backend))+")+\")\")\nns={'__file__':str(p),'__name__':'__main__'}\nexec(compile(s,str(p),'exec'),ns)\n")
    runner.write_text("from pathlib import Path\nimport sys\nE=Path("+repr(str(E))+")\nsys.path.insert(0,str(E))\np=E/'run_stage17.py'\ns=p.read_text().replace(\"HERE/'stage16_episode.py'\",\"Path(\"+repr("+repr(str(episode))+")+\")\")\nns={'__file__':str(p),'__name__':'__main__'}\nexec(compile(s,str(p),'exec'),ns)\n")
    files=[backend,episode,runner]
    for f in files:compile(f.read_text(),str(f),'exec')
    manifest=dict(scope='separate diagnostic, never a fixed-panel score',yaw_rad=yaw,phase='first s5_translate physics step',
        policy=str(E/'mainline/strategy.py'),policy_sha256=hashlib.sha256((E/'mainline/strategy.py').read_bytes()).hexdigest(),
        files={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},
        source_unchanged=True,object_truth_and_pose_write_visible_to_policy=False,
        native_and_object_truth_for_postrun_diagnosis_only=True,rates_hz=[120,20,30])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--yaw',type=float,default=.25);a=p.parse_args();print(json.dumps(prepare(a.out,a.yaw)))
