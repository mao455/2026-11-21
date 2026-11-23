"""Private scene generation only: never mounted into policy sandbox."""
from pathlib import Path
import numpy as np
import trimesh
P=Path(__file__).resolve().parent
OBJECTS={'box_dev':('box',(.010,.020,.020)), 'cylinder_dev':('cylinder',(.014,.020)),
         'wedge_dev':('wedge',(.014,.020,.022)), 'box_test':('box',(.012,.018,.022)),
         'cylinder_test':('cylinder',(.016,.018)), 'wedge_test':('wedge',(.016,.018,.020))}
def create(name):
    kind, dims=OBJECTS[name]
    if kind=='box':mesh=trimesh.creation.box(extents=dims)
    elif kind=='cylinder':mesh=trimesh.creation.cylinder(radius=dims[0]/2,height=dims[1],sections=32)
    else:
        x,y,z=dims
        mesh=trimesh.convex.convex_hull(np.array([[-x/2,-y/2,0],[x/2,-y/2,0],[-x/2,y/2,0],[x/2,y/2,0],[-x/2,-y/2,z],[-x/2,y/2,z]]))
    mesh.apply_translation([0,0,-mesh.bounds[0,2]])
    path=P/'assets'/'round8_objects'/f'{name}.obj';path.parent.mkdir(exist_ok=True)
    if path.exists():
        if path.read_text()!=mesh.export(file_type='obj'):raise ValueError('Existing object changed')
    else:path.write_text(mesh.export(file_type='obj'))
    return path
if __name__=='__main__':
    for name in OBJECTS:print(create(name))
