"""Private Stage3 identity registration. Never mounted into policy sandbox."""
import json
from pathlib import Path
from stage2_registry import registry as previous_registry,install as previous_install
from general_grasp_data import LAB,sha
HERE=Path(__file__).resolve().parent
SPLIT=HERE/'STAGE3_OBJECTS.json'
def registry():
 rows=previous_registry()
 rows.update({r['object_id']:r for r in json.loads(SPLIT.read_text())['objects']})
 return rows
def install():
 previous_install()
 import round14_objects
 objects=registry();previous=round14_objects.create
 round14_objects.OBJECTS.update({k:('native_dataset',()) for k in objects})
 def create(name):
  if name not in objects:return previous(name)
  path=LAB/objects[name]['asset']
  if sha(path)!=objects[name]['sha256']:raise RuntimeError('Frozen asset SHA mismatch')
  return path
 round14_objects.create=create
