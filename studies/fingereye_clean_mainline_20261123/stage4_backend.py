"""Original physics, private native scene registry and passive contact audit."""
import sys
from pathlib import Path
if __name__=='__main__':
 from stage4_registry import install
 install()
 here=Path(__file__).resolve().parent
 sys.path.insert(0,str(here.parent/'rax_tinyobj_round1'))
 from stage2_contact_audit import install as audit_contacts
 audit_contacts(Path(sys.argv[sys.argv.index('--output')+1]))
 from round14_backend import main
 main()
