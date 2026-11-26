"""Private reserved objects. Never mounted in policy sandbox."""
from pathlib import Path
import numpy as np,trimesh
from round13_objects import OBJECTS as OLD,create as old_create
P=Path(__file__).resolve().parent
NEW={'r14_box':('box',(.0094,.021,.0245)), 'r14_cylinder':('cylinder',(.0112,.024)), 'r14_wedge':('wedge',(.0115,.022,.025)), 'r14_pentagon':('prism',(.012,.023,5))}
# Stage12 development fixtures: registered before observing grasp outcomes.
# Dimensions stay private to scene construction; policy receives rendered RGB.
NEW.update(s12_cylinder=('cylinder',(.030,.030)),s12_cube=('box',(.030,.030,.030)))
OBJECTS={**OLD,**NEW}
def create(name):
 if name not in NEW:return old_create(name)
 kind,d=NEW[name]
 if kind=='box':m=trimesh.creation.box(extents=d)
 elif kind in ('cylinder','prism'):m=trimesh.creation.cylinder(radius=d[0]/2,height=d[1],sections=int(d[2]) if kind=='prism' else 32)
 else:
  x,y,z=d;m=trimesh.convex.convex_hull(np.array([[-x/2,-y/2,0],[x/2,-y/2,0],[-x/2,y/2,0],[x/2,y/2,0],[-x/2,-y/2,z],[-x/2,y/2,z]]))
 m.apply_translation([0,0,-m.bounds[0,2]]);p=P/'assets/round14_objects'/f'{name}.obj';p.parent.mkdir(exist_ok=True,parents=True);content=m.export(file_type='obj')
 if p.exists():assert p.read_text()==content
 else:p.write_text(content)
 return p
