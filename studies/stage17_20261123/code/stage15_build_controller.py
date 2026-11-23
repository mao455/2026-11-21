"""Build the public-observation baseline and explicit development ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from run_stage14 import build as base_build, controller_settings
from stage14_build_recovery import OVERLAY

HERE = Path(__file__).resolve().parent


def build(output, *, fingers='two', attempts=3, preload=False, light_first=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('immutable controller exists: ' + str(output))
    with tempfile.TemporaryDirectory(prefix='fingereye-s15-build-') as folder:
        base = Path(folder) / 'base.py'
        base_build(base, profile='fast', fingers=fingers, attempts=attempts)
        settings = controller_settings(base)
        settings['S11_CONFIG']['evidence_recovery'] = True
        source = base.read_text()
    # Finger selection and retry direction are separate decisions. Historical
    # fixed-two mode silently replaced the requested third recovery direction
    # with a diagonal; the validated AUTO recovery retained the requested axis.
    # Preserve that recovery behavior while excluding unaccepted three-pad grasps.
    retry_condition = "if s['attempt'] >= 3 and globals().get('S11_CONFIG',{}).get('fingers','two')=='two':"
    if source.count(retry_condition) != 1:
        raise RuntimeError('unexpected historical retry-axis implementation')
    source = source.replace(retry_condition,
        retry_condition[:-1] + " and not globals().get('S15_KEEP_REQUESTED_RETRY_AXIS', False):")
    modules = ['stage14_shape_fit.py', 'stage14_surface_pair.py', 'stage15_control.py']
    source += '\n' + (HERE/modules[0]).read_text() + OVERLAY
    source += '\n' + (HERE/modules[1]).read_text()
    source += '\n' + (HERE/modules[2]).read_text()
    source += '\nS11_CONFIG = ' + repr(settings['S11_CONFIG']) + '\n'
    source += '\nS15_PRELOAD = ' + repr(preload) + '\n'
    source += '\nS15_KEEP_REQUESTED_RETRY_AXIS = True\n'
    if light_first is not None:
        if not .0005 <= light_first <= .0045:
            raise ValueError('light diagnostic squeeze must be 0.5..4.5mm')
        source += '''
_S15_NORMAL_CLOSE = s5_close
def s5_close(s, mode='servo', cap=.0045):
    if s.get('actual_grasps', 0):
        return _S15_NORMAL_CLOSE(s, mode, cap)
    _S15_NORMAL_CLOSE(s, 'fixed', S15_LIGHT_FIRST)
    # Diagnostic first closure is explicit. Recovery itself is autonomous.
    s['last_closure_cap'] = cap
    s5_note(s['ctx'], 's15_light_first_diagnostic', dict(step=s['obs']['step'],
        squeeze_m=s['squeeze'].tolist(), recovery_cap_m=cap, no_reset=True))
S15_LIGHT_FIRST = ''' + repr(light_first) + '\n'
    compile(source, str(output), 'exec')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    sha = lambda value: hashlib.sha256(value).hexdigest()
    provenance = dict(schema=1, controller_sha256=sha(source.encode()),
        modules_sha256={name: sha((HERE/name).read_bytes()) for name in modules},
        fingers=fingers, attempts=attempts, preload=preload, light_first_m=light_first,
        retry_axis='preserve_requested_as_validated_auto',
        diagnostic=light_first is not None, online_private_inputs=False,
        protocol='lift30_no_drop_v1', physical_hz=120, control_hz=20, rgb_fps=30)
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fingers', choices=('two','three','auto'), default='two')
    p.add_argument('--max-attempts', type=int, choices=(1,2,3), default=3)
    p.add_argument('--preload', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--light-first', type=float)
    a=p.parse_args()
    print(json.dumps(build(a.output, fingers=a.fingers, attempts=a.max_attempts,
                           preload=a.preload, light_first=a.light_first)))


if __name__ == '__main__':
    main()
