"""Freeze one policy from maintained source; generated controllers are artifacts."""
import hashlib
import json
from pathlib import Path
from stage16_build_controller import build as build_stage16

def build(output, *, color_mode='hs', **settings):
    if color_mode not in ('legacy', 'hs'):
        raise ValueError('unknown color mode')
    result = build_stage16(output, **settings)
    path = Path(output)
    source = path.read_text() + '\nS17_COLOR_MODE = ' + repr(color_mode) + '\n'
    compile(source, str(path), 'exec')
    path.write_text(source)
    result.update(controller_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  color_mode=color_mode, source_entry='run_stage17.py')
    path.with_suffix('.provenance.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
