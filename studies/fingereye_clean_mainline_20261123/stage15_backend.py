"""Portable root/private registry adapter for the unchanged Stage14 backend."""
import hashlib
import json
import os
from pathlib import Path
import sys

from stage15_runtime import install, install_backend_source


def main():
    configuration = install()
    root = Path(configuration['asset_root'])
    import xml.etree.ElementTree as element_tree
    from cross_hand_runtime import selected
    hand = selected()
    urdf = Path(hand['urdf_path']) if hand else root / 'assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf'
    paths = sorted({mesh.get('filename') for mesh in element_tree.parse(urdf).findall('.//mesh')})
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'stage15_static_asset_resolution.json').write_text(json.dumps(dict(configuration,
        urdf_path=str(urdf), urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        resolved_mesh_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths},
        resolution='selected URDF references rebased in memory; source URDF bytes unchanged'), indent=2))
    import stage9_backend
    install_backend_source(stage9_backend)
    import stage14_backend
    return stage14_backend.main()


if __name__ == '__main__':
    from stage14_resources import simulator_slot
    with simulator_slot(Path(sys.argv[sys.argv.index('--output') + 1])):
        main()
