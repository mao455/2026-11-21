"""Private Round12 holdout fixtures, not mounted into standalone execution."""
from pathlib import Path
import numpy as np,trimesh
from round11_objects import OBJECTS as OLD,create as old_create
P=Path(__file__).resolve().parent
NEW={'r12_box_small':('box',(.0093,.019,.0205)), 'r12_box_large':('box',(.0125,.025,.024)), 'r12_cylinder':('cylinder',(.0125,.021)), 'r12_wedge':('wedge',(.0115,.022,.0225)), 'r12_hexagon':('hexagon',(.0145,.019))}
OBJECTS={**OLD,**NEW}
def create(name):
 if name not in NEW:return old_create(name)
 kind,d=NEW[name]
 if kind=='box':m=trimesh.creation.box(extents=d)
 elif kind in ('cylinder','hexagon'):m=trimesh.creation.cylinder(radius=d[0]/2,height=d[1],sections=32 if kind=='cylinder' else 6)
 else:
  x,y,z=d;m=trimesh.convex.convex_hull(np.array([[-x/2,-y/2,0],[x/2,-y/2,0],[-x/2,y/2,0],[x/2,y/2,0],[-x/2,-y/2,z],[-x/2,y/2,z]]))
 m.apply_translation([0,0,-m.bounds[0,2]]);p=P/'assets/round12_objects'/f'{name}.obj';p.parent.mkdir(exist_ok=True);content=m.export(file_type='obj')
 if p.exists():assert p.read_text()==content
 else:p.write_text(content)
 return p
