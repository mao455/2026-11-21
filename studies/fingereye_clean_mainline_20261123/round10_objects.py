"""PRIVATE fresh test identities; never mounted into execution worker."""
from pathlib import Path
import numpy as np
import trimesh
from round9_objects import OBJECTS as OLD,create as old_create
P=Path(__file__).resolve().parent
NEW={'r10_box':('box',(.011,.024,.019)), 'r10_hexagon':('hexagon',(.014,.022)), 'r10_ellipsoid':('ellipsoid',(.012,.018,.024))}
OBJECTS={**OLD,**NEW}
def create(name):
 if name not in NEW:return old_create(name)
 kind,d=NEW[name]
 if kind=='box':m=trimesh.creation.box(extents=d)
 elif kind=='hexagon':m=trimesh.creation.cylinder(radius=d[0]/2,height=d[1],sections=6)
 else:m=trimesh.creation.icosphere(subdivisions=2);m.apply_scale(np.array(d)/2)
 m.apply_translation([0,0,-m.bounds[0,2]]);out=P/'assets/round10_objects'/f'{name}.obj';out.parent.mkdir(exist_ok=True);content=m.export(file_type='obj')
 if out.exists():assert out.read_text()==content
 else:out.write_text(content)
 return out
