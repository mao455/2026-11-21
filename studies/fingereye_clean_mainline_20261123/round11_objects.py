"""Private frozen Round11 test identities; never execution inputs."""
from pathlib import Path
import numpy as np,trimesh
from round10_objects import OBJECTS as OLD,create as old_create
P=Path(__file__).resolve().parent
NEW={'r11_box':('box',(.0085,.0215,.023)), 'r11_cylinder':('cylinder',(.011,.0185)), 'r11_wedge':('wedge',(.010,.019,.021))}
OBJECTS={**OLD,**NEW}
def create(name):
 if name not in NEW:return old_create(name)
 kind,d=NEW[name]
 if kind=='box':m=trimesh.creation.box(extents=d)
 elif kind=='cylinder':m=trimesh.creation.cylinder(radius=d[0]/2,height=d[1],sections=32)
 else:
  x,y,z=d;m=trimesh.convex.convex_hull(np.array([[-x/2,-y/2,0],[x/2,-y/2,0],[-x/2,y/2,0],[x/2,y/2,0],[-x/2,-y/2,z],[-x/2,y/2,z]]))
 m.apply_translation([0,0,-m.bounds[0,2]]);p=P/'assets/round11_objects'/f'{name}.obj';p.parent.mkdir(exist_ok=True);content=m.export(file_type='obj')
 if p.exists():assert p.read_text()==content
 else:p.write_text(content)
 return p
