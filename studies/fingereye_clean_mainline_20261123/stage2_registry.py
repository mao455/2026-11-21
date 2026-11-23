"""Private stage2 native asset registration; forbidden to policy imports."""
import json
from pathlib import Path
import general_grasp_data as data
HERE=Path(__file__).resolve().parent
SPLIT=HERE/'STAGE2_OBJECTS.json'
def registry():
 rows=data.registry()
 rows.update({r['object_id']:r for r in json.loads(SPLIT.read_text())['objects']})
 return rows
def install():
 import round14_objects
 objects=registry();previous=round14_objects.create
 round14_objects.OBJECTS.update({k:('native_dataset',()) for k in objects})
 def create(name):
  if name not in objects:return previous(name)
  path=data.LAB/objects[name]['asset']
  if data.sha(path)!=objects[name]['sha256']:raise RuntimeError('Frozen asset SHA mismatch')
  return path
 round14_objects.create=create
