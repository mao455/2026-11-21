"""Optional exact geometry acceleration; same Isaac loop and scorer."""
import json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rax_tinyobj_round1'))
import stage9_backend

def main():
    # The backend validates its private case in a separate interpreter.
    import run_round14_episode
    run_round14_episode.CASE_PARAMETER_RANGES['rgb_noise_std']=(0.,20.4)
    from stage14_geometry import install
    stats=(dict(reference=True) if os.environ.get('S14_GEOMETRY_REFERENCE')=='1'
           else install(verify=os.environ.get('S14_VERIFY_GEOMETRY')=='1'))
    out=Path(sys.argv[sys.argv.index('--output')+1])
    measured_original=stage9_backend.measured
    def measured_with_geometry_evidence(name,function,*args,**kwargs):
        result=measured_original(name,function,*args,**kwargs)
        if name=='video_finalize' or (name=='strict_geometry' and stats.get('calls',1)%240==0):
            (out/'stage14_geometry_equivalence.json').write_text(json.dumps(stats,indent=2))
        return result
    stage9_backend.measured=measured_with_geometry_evidence
    if os.environ.get('S14_DIAGNOSTIC_STOP')=='1':
        import round8_convex_audit
        original=round8_convex_audit.Audit.check
        def guarded(self,body_poses,object_pose):
            result=original(self,body_poses,object_pose)
            if not result['passed']:
                (out/'stage14_independent_geometry_stop.json').write_text(json.dumps(dict(check=result,object_pose=list(object_pose)),indent=2))
                raise RuntimeError('independent true geometry diagnostic guard')
            return result
        round8_convex_audit.Audit.check=guarded
    try:return stage9_backend.main()
    finally:(out/'stage14_geometry_equivalence.json').write_text(json.dumps(stats,indent=2))
if __name__=='__main__':
    from stage14_resources import simulator_slot
    with simulator_slot(Path(sys.argv[sys.argv.index('--output')+1])):main()
