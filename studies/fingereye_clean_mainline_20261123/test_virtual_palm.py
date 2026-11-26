"""Preserve physical FK across noncommuting floating-coordinate charts."""
from pathlib import Path
import tempfile,unittest,xml.etree.ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation
from mainline.hand_description import HandDescription
from cross_hand_runtime import normalize_virtual_palm,BASE_NAMES
class VirtualPalmTests(unittest.TestCase):
 def test_noncommuting_rotation_chart_preserves_physical_subtree(self):
  root=E.Element('robot',name='generic');nodes=['world','x','y','z','roll','pitch','palm','tip']
  for n in nodes:E.SubElement(root,'link',name=n)
  E.SubElement(root.find("link[@name='palm']"),'visual')
  axes=[(1,0,0),(0,1,0),(0,0,1)]*2+[(0,1,0)]
  for i,(p,c,axis) in enumerate(zip(nodes,nodes[1:],axes)):
   j=E.SubElement(root,'joint',name='source'+str(i),type='prismatic' if i<3 else 'revolute');E.SubElement(j,'parent',link=p);E.SubElement(j,'child',link=c);E.SubElement(j,'axis',xyz=' '.join(map(str,axis)));E.SubElement(j,'origin',xyz='0 0 .03' if i==6 else '0 0 0');E.SubElement(j,'limit',lower='-3',upper='3',effort='1',velocity='2')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.urdf';E.ElementTree(root).write(p);source=HandDescription(p);mapping=normalize_virtual_palm(root);self.assertEqual(mapping['palm_link'],'palm');E.ElementTree(root).write(p);runtime=HandDescription(p,joint_names=[*BASE_NAMES,'source6']);self.assertEqual(len(source.joint_names),len(runtime.joint_names))
   for _ in range(30):
    q=np.random.default_rng(_).uniform(-.7,.7,7);old=q.copy();old[3:6]=Rotation.from_euler('ZYX',q[3:6]).as_euler('XYZ')
    for link in ['palm','tip']:np.testing.assert_allclose(source.fk(old)[link],runtime.fk(q)[link],atol=2e-14)
if __name__=='__main__':unittest.main()
