"""Private portable Stage15 adapters; never mounted into the policy sandbox.

FINGEREYE_ASSET_ROOT selects a FingerEyeLab tree containing assets/scripts/source.
FINGEREYE_OBJECT_MANIFEST optionally adds private objects with relative assets
and mandatory SHA256. Bundled experimental fixtures remain in this checkout.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
CODE_LAB = HERE.parents[1]
_STATE = None


def rebase_robot_mesh(filename, root=None):
    """Resolve historical absolute URDF mesh exports inside selected static hand."""
    root = asset_root() if root is None else Path(root)
    marker = '/assets/xhand3camFE_left/'
    value = str(filename)
    if Path(value).is_absolute() and marker in value:
        selected = root / 'assets/xhand3camFE_left' / value.split(marker, 1)[1]
        if not selected.is_file():
            raise FileNotFoundError('Selected static robot mesh is missing: ' + str(selected))
        return str(selected)
    return filename


def install_robot_resolvers(root):
    """Rebase only the selected URDF in memory, preserving every source byte.

    ElementTree consumers read FK/calibration/mesh paths directly. yourdfpy uses
    lxml and its supported filename_handler. Both must select the same meshes
    even when the historical checkout remains present on this machine.
    """
    import xml.etree.ElementTree as element_tree
    import yourdfpy
    urdf = (root / 'assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf').resolve()

    def selected(source):
        return isinstance(source, (str, os.PathLike)) and Path(source).resolve() == urdf

    original_parse = element_tree.parse

    def parse(source, *args, **kwargs):
        tree = original_parse(source, *args, **kwargs)
        if selected(source):
            for mesh in tree.findall('.//mesh'):
                mesh.set('filename', str(rebase_robot_mesh(mesh.get('filename'), root)))
        return tree

    element_tree.parse = parse
    original_load = yourdfpy.URDF.load

    def load(source, **kwargs):
        if selected(source):
            previous = kwargs.get('filename_handler')

            def handler(fname):
                resolved = rebase_robot_mesh(fname, root)
                return previous(resolved) if previous else yourdfpy.filename_handler_magic(resolved, str(urdf.parent))

            kwargs['filename_handler'] = handler
        return original_load(source, **kwargs)

    yourdfpy.URDF.load = staticmethod(load)


def asset_root():
    return Path(os.environ.get('FINGEREYE_ASSET_ROOT', str(CODE_LAB))).expanduser().resolve()


def external_objects(root=None):
    root = asset_root() if root is None else Path(root).resolve()
    name = os.environ.get('FINGEREYE_OBJECT_MANIFEST')
    if not name:
        return {}, None
    manifest = Path(name).expanduser().resolve()
    payload = json.loads(manifest.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get('objects'), list):
        raise ValueError('Private manifest must contain objects: [...]')
    result = {}
    for item in payload['objects']:
        if not isinstance(item, dict):
            raise ValueError('Each private object must be a dictionary')
        identity, relative, digest = (item.get(key) for key in ('object_id', 'asset', 'sha256'))
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identity):
            raise ValueError('Invalid private object_id')
        if identity in result:
            raise ValueError('Duplicate private object_id: ' + identity)
        if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Object asset must be relative to FINGEREYE_ASSET_ROOT')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Every private object requires a lowercase SHA256')
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError('Private object SHA256 mismatch: ' + identity)
        result[identity] = dict(item, asset=str(path.absolute()))
    return result, manifest


def install(extra_files=()):
    """Install before importing stage4_episode or calling stage9_backend.main.

    Returns the selected root and manifest. Repeated calls with the same settings
    are idempotent; a process cannot silently change its immutable asset tree.
    """
    global _STATE
    root = asset_root()
    supplied, manifest = external_objects(root)
    key = (str(root), str(manifest) if manifest else None)
    if _STATE is not None:
        if _STATE['key'] != key:
            raise RuntimeError('Stage15 runtime root or private manifest changed in-process')
        _STATE['extra_files'].update(Path(p).resolve() for p in extra_files)
        return dict(asset_root=str(root), object_manifest=str(manifest) if manifest else None)
    for folder in ('assets', 'scripts', 'source'):
        if not (root / folder).is_dir():
            raise FileNotFoundError(f'Asset root requires {folder}: {root}')
    for folder in (root / 'scripts', root / 'source/fingereye'):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    install_robot_resolvers(root)
    import general_grasp_data
    import stage3_registry
    import stage4_registry
    import run_round14_episode
    import run_published
    general_grasp_data.LAB = root
    stage3_registry.LAB = root
    stage4_registry.LAB = root
    run_round14_episode.LAB = root
    run_published.LAB = root
    original_registry = stage4_registry.registry

    def registry():
        bundled = original_registry()
        merged = {}
        for identity, value in bundled.items():
            value = dict(value)
            relative = Path(value['asset'])
            value['asset'] = str((CODE_LAB if relative.parts[:1] == ('experiments',) else root) / relative)
            merged[identity] = value
        for identity, value in supplied.items():
            if identity in merged and (value['sha256'] != merged[identity]['sha256'] or
                    Path(value['asset']).resolve() != Path(merged[identity]['asset']).resolve()):
                raise ValueError('Private manifest conflicts with frozen object: ' + identity)
            merged[identity] = value
        return merged

    stage4_registry.registry = registry
    original_fingerprint = run_round14_episode.fingerprint
    extras = {Path(p).resolve() for p in extra_files}
    _STATE = dict(key=key, extra_files=extras)

    def portable_path(path):
        path = Path(path).absolute()
        try:
            relative = path.relative_to(CODE_LAB)
        except ValueError:
            relative = None
        if relative is not None and relative.parts[:1] in (('assets',), ('scripts',), ('source',)):
            return root / relative
        if path == CODE_LAB.parent / 'fingereye_camera_cfg.py':
            return root.parent / 'fingereye_camera_cfg.py'
        return path

    def fingerprint(paths):
        # Stage15 installs Stage4 directly. Include the Stage9/14 adapters too,
        # whose extra fingerprint wrapper is otherwise not imported. The same
        # closure is checked before and after physics, including recipe files.
        additional = [*HERE.glob('*.py'), *HERE.glob('*.json')]
        for directory in ('scripts', 'source'):
            additional.extend(path for path in (root / directory).rglob('*')
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
        additional.extend(_STATE['extra_files'])
        if manifest:
            additional.append(manifest)
        additional.extend(Path(value['asset']) for value in registry().values())
        additional.extend(path for path in (root / 'assets/xhand3camFE_left').rglob('*') if path.is_file())
        return original_fingerprint(list(dict.fromkeys(portable_path(path) for path in [*paths, *additional])))

    run_round14_episode.fingerprint = fingerprint
    return dict(asset_root=str(root), object_manifest=str(manifest) if manifest else None)


def portable_backend_source(source):
    old = 'LAB = HERE.parents[1]'
    if source.count(old) != 1:
        raise RuntimeError('Unexpected historical backend LAB declaration')
    return source.replace(old, "LAB = Path(os.environ.get('FINGEREYE_ASSET_ROOT', str(HERE.parents[1]))).expanduser().resolve()")


def install_backend_source(stage9_backend):
    """Change only root selection in the saved, executed historical backend.

    stage9_backend saves the complete generated source and its executed hash.
    The original simulator loop, evaluator and public observation code remain.
    """
    if getattr(stage9_backend, '_stage15_portable_source_installed', False):
        return
    import inspect
    source = inspect.getsource(stage9_backend.main)
    old = "source=(HERE/'round14_backend.py').read_text()"
    if source.count(old) != 1:
        raise RuntimeError('Unexpected Stage9 backend source loader')
    source = source.replace(old, "source=portable_backend_source((HERE/'round14_backend.py').read_text())")
    stage9_backend.portable_backend_source = portable_backend_source
    exec(compile(source, str(HERE / 'stage15_backend.py'), 'exec'), stage9_backend.__dict__)
    stage9_backend._stage15_portable_source_installed = True
