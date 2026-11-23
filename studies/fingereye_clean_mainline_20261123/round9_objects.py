"""Private new-size fixtures for a frozen Round9 evaluation; never policy inputs."""
from pathlib import Path
import numpy as np
import trimesh
from round8_objects import OBJECTS as OLD_OBJECTS,create as old_create
P=Path(__file__).resolve().parent
NEW_OBJECTS={'r9_box':('box',(.009,.017,.018)),
             'r9_cylinder':('cylinder',(.0125,.021)),
             'r9_wedge':('wedge',(.012,.022,.019))}
OBJECTS={**OLD_OBJECTS,**NEW_OBJECTS}
def create(name):
    if name not in NEW_OBJECTS:return old_create(name)
    kind,dims=NEW_OBJECTS[name]
    if kind=='box':mesh=trimesh.creation.box(extents=dims)
    elif kind=='cylinder':mesh=trimesh.creation.cylinder(radius=dims[0]/2,height=dims[1],sections=32)
    else:
        x,y,z=dims
        mesh=trimesh.convex.convex_hull(np.array([[-x/2,-y/2,0],[x/2,-y/2,0],[-x/2,y/2,0],[x/2,y/2,0],[-x/2,-y/2,z],[-x/2,y/2,z]]))
    mesh.apply_translation([0,0,-mesh.bounds[0,2]])
    path=P/'assets/round9_objects'/f'{name}.obj';path.parent.mkdir(exist_ok=True)
    text=mesh.export(file_type='obj')
    if path.exists() and path.read_text()!=text:raise RuntimeError('Fixture cannot change')
    if not path.exists():path.write_text(text)
    return path
if __name__=='__main__':
    for name in NEW_OBJECTS:print(create(name))
