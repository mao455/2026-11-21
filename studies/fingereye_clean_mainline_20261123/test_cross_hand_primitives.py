"""URDF admission and conservative primitive geometry for any hand."""
import json, tempfile, unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from cross_hand_runtime import materialize_collision_primitives, prepare

class PrimitiveTests(unittest.TestCase):
 def test_curved_meshes_enclose_primitives_and_origins_are_retained(self):
  root=ET.fromstring('<robot><link name="p"><collision><origin xyz="1 2 3"/><geometry><box size=".02 .03 .04"/></geometry></collision><collision><geometry><cylinder radius=".01" length=".03"/></geometry></collision><collision><geometry><sphere radius=".012"/></geometry></collision></link></robot>')
  with tempfile.TemporaryDirectory() as tmp:
   rows=materialize_collision_primitives(root,Path(tmp))
   self.assertEqual(len(rows),3)
   self.assertEqual(root.find('.//origin').get('xyz'),'1 2 3')
   box,cylinder,sphere=[trimesh.load(r['path'],force='mesh') for r in rows]
   np.testing.assert_allclose(box.extents,[.02,.03,.04],atol=2e-9)
   plane_radius=np.einsum('ij,ij->i',sphere.face_normals,sphere.triangles[:,0]).min()
   self.assertGreaterEqual(plane_radius,.012-2e-9)
   theta=np.linspace(0,2*np.pi,1024)
   samples=np.c_[.01*np.cos(theta),.01*np.sin(theta),np.zeros_like(theta)]
   from scipy.spatial import ConvexHull
   hull=ConvexHull(cylinder.vertices)
   self.assertLessEqual((samples@hull.equations[:,:3].T+hull.equations[:,3]).max(),2e-9)
 def test_no_camera_frame_cannot_be_silently_calibrated(self):
  source=b'<robot name="unknown"><link name="p"/></robot>'
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'hand.urdf';p.write_bytes(source);dest=Path(tmp)/'runtime'
   with self.assertRaisesRegex(ValueError,'missing_optical_calibration'):prepare(p,dest)
   self.assertEqual(p.read_bytes(),source)
   self.assertEqual(json.loads((dest/'readiness.json').read_text())['reason'],'missing_optical_calibration')
if __name__=='__main__':unittest.main()
