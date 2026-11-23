"""Unchanged legal sensor sandbox/physics gates with Stage4 private registration."""
from pathlib import Path
import run_round14_episode as runtime
from stage4_registry import install,registry,SPLIT,ADDITIONAL_SPLIT
from general_grasp_data import LAB
HERE=Path(__file__).resolve().parent
install()
OriginalPeer=runtime.Peer
class Stage4Peer(OriginalPeer):
 def __init__(self,command,log,env):
  command=[str(HERE/'stage4_backend.py') if str(arg)==str(HERE/'round14_backend.py') else arg for arg in command]
  super().__init__(command,log,env)
runtime.Peer=Stage4Peer
original_fingerprint=runtime.fingerprint
def fingerprint(paths):
 extra=[*([ADDITIONAL_SPLIT] if ADDITIONAL_SPLIT.exists() else []),Path(__file__),HERE/'stage4_backend.py',HERE/'stage4_registry.py',SPLIT,HERE/'stage3_registry.py',HERE/'STAGE3_OBJECTS.json',HERE/'stage2_contact_audit.py',HERE/'stage2_registry.py',HERE/'STAGE2_OBJECTS.json',HERE/'general_object_split.json',*(LAB/r['asset'] for r in registry().values())]
 return original_fingerprint(list(dict.fromkeys([*paths,*extra])))
runtime.fingerprint=fingerprint
if __name__=='__main__':raise SystemExit(runtime.main())
