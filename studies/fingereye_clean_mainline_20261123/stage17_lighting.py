"""Private scene intervention; light metadata is never an online policy input."""
import json
import os

def configure_lights(env, out):
    from pxr import UsdLux
    scale = float(os.environ['FINGEREYE_LIGHT_SCALE'])
    if not .1 <= scale <= 3.:
        raise ValueError('light scale outside experiment range')
    rows = []
    for prim in env.sim.stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        light = UsdLux.LightAPI(prim)
        intensity = light.GetIntensityAttr()
        original = float(intensity.Get())
        color = list(light.GetColorAttr().Get())
        exposure = float(light.GetExposureAttr().Get())
        intensity.Set(original * scale)
        rows.append(dict(path=str(prim.GetPath()), type=prim.GetTypeName(),
                         original_intensity=original, intensity=float(intensity.Get()),
                         color=color, exposure=exposure))
    if not rows:
        raise RuntimeError('no light found; refusing an unverified lighting experiment')
    (out/'light_intervention.json').write_text(json.dumps(dict(multiplier=scale,
        lights=rows, same_color=True, rgb_postprocessing=False), indent=2)+'\n')
