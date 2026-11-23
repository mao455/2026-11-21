"""Stage15 safe default plus opt-in, separately controlled numerical ablations."""
from pathlib import Path
import hashlib
import json
from stage15_build_controller import build as stage15_build
HERE=Path(__file__).resolve().parent

def build(output, fingers='two', attempts=3, preload=False, light_first=None,
          normal_weight=None, margin_tie='historical'):
    if normal_weight is not None and (normal_weight<=0 or normal_weight>10):
        raise ValueError('normal weight outside supported diagnostic range')
    if margin_tie not in ('historical','bounded_relative'):raise ValueError('unknown tie mode')
    stage15_build(output,fingers=fingers,attempts=attempts,preload=preload,light_first=light_first)
    with Path(output).open('a') as f:
        f.write('\n# Explicit Stage16 ablations; these do not change hard safety gates.\n')
        f.write('S16_COMMON_NORMAL_WEIGHT = '+repr(normal_weight)+'\n')
        f.write('S16_MARGIN_TIE = '+repr(margin_tie)+'\n')
        f.write((HERE/'stage16_numerics_overlay.py').read_text())
    path=Path(output)
    provenance=json.loads(path.with_suffix('.provenance.json').read_text())
    provenance.update(controller_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        stage16_overlay_sha256=hashlib.sha256((HERE/'stage16_numerics_overlay.py').read_bytes()).hexdigest(),
        stage16_normal_weight=normal_weight,stage16_margin_tie=margin_tie)
    path.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance
