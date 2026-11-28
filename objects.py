"""Scene asset registry, never mounted into the policy sandbox."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
MANIFEST=Path(__import__('os').environ.get('FINGEREYE_OBJECT_MANIFEST',str(HERE/'examples/objects.json')))
OBJECTS={r['object_id']:r for r in json.loads(MANIFEST.read_text())['objects']}
def create(name):
    row=OBJECTS[name];p=Path(row['asset'])
    if not p.is_absolute():p=MANIFEST.parent/p
    p=p.resolve()
    if not p.is_file():raise FileNotFoundError(p)
    if row.get('sha256') and hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Object asset checksum mismatch')
    return p
