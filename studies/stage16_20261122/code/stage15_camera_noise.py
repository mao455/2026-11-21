"""Collect an explicit static RGB camera sequence or estimate its temporal DN.

Capture opens only the device explicitly supplied with --device. No discovery
or device access occurs for the estimate subcommand. Output is processed RGB,
so statistics describe the current exposure/gain/ISP/output configuration and
are not a sensor read-noise or electron-gain calibration.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def capture(args):
    # This import, and all hardware operations, stay in the explicit subcommand.
    import cv2
    metadata = json.loads(args.metadata.read_text())
    required = {'camera_model', 'sensor', 'exposure_us', 'gain', 'illumination', 'isp', 'output_settings'}
    if not required <= metadata.keys():
        raise ValueError('metadata missing fields: ' + ', '.join(sorted(required - metadata.keys())))
    args.output.mkdir(parents=True, exist_ok=False)
    device = int(args.device) if args.device.isdecimal() else args.device
    camera = cv2.VideoCapture(device)
    if not camera.isOpened():
        raise RuntimeError('cannot open explicitly selected camera: ' + args.device)
    rows = []
    try:
        if args.width:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.fps:
            camera.set(cv2.CAP_PROP_FPS, args.fps)
        if args.native_exposure is not None:
            camera.set(cv2.CAP_PROP_EXPOSURE, args.native_exposure)
        if args.native_gain is not None:
            camera.set(cv2.CAP_PROP_GAIN, args.native_gain)
        for _ in range(args.warmup):
            ok, _ = camera.read()
            if not ok:
                raise RuntimeError('camera failed during warmup')
        for index in range(args.frames):
            started = time.monotonic_ns()
            ok, image = camera.read()
            completed = time.monotonic_ns()
            if not ok:
                raise RuntimeError('camera read failed at frame ' + str(index))
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if rgb.dtype != np.uint8 or rgb.shape[-1] != 3:
                raise ValueError('expected uint8 three-channel RGB output')
            path = args.output / ('frame_' + str(index).zfill(6) + '.npy')
            np.save(path, rgb)
            row = dict(frame_id=index, path=path.name, sha256=sha(path),
                       read_start_monotonic_ns=started, read_done_monotonic_ns=completed,
                       sensor_exposure_timestamp=None,
                       exposure_native_readback=camera.get(cv2.CAP_PROP_EXPOSURE),
                       gain_native_readback=camera.get(cv2.CAP_PROP_GAIN))
            rows.append(row)
            with (args.output / 'frames.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        metadata.update(device=args.device, capture_software='OpenCV ' + cv2.__version__,
                        source_color='driver BGR converted to RGB', dtype='uint8', dn_range=[0, 255],
                        width=camera.get(cv2.CAP_PROP_FRAME_WIDTH), height=camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        fps_readback=camera.get(cv2.CAP_PROP_FPS),
                        auto_exposure_native_readback=camera.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                        native_exposure_units='driver-defined; metadata exposure_us must come from camera configuration',
                        timing_scope='host read interval; sensor exposure timestamp unavailable through this interface',
                        frames=len(rows), static_scene_instruction='fixed camera and scene; fixed working illumination, exposure, gain and ISP')
    finally:
        camera.release()
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), frames=len(rows))))


def correlation(a, b):
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def estimate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    metadata_path = args.input / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    manifest = args.input / 'frames.jsonl'
    if manifest.exists():
        frames = [json.loads(l) for l in manifest.open() if l.strip()]
        paths = [args.input / r['path'] for r in frames]
        for row, path in zip(frames, paths):
            if row.get('sha256') and sha(path) != row['sha256']:
                raise ValueError('frame hash mismatch: ' + str(path))
    else:
        paths = sorted(args.input.glob('*.npy'))
        frames = [dict(frame_id=i, path=p.name, sha256=sha(p)) for i, p in enumerate(paths)]
    if len(paths) < 4:
        raise ValueError('at least four static RGB frames are needed')
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    shape = arrays[0].shape
    if any(a.dtype != np.uint8 or a.shape != shape or a.ndim != 3 or a.shape[-1] != 3 for a in arrays):
        raise ValueError('all frames must be equal-shape uint8 H x W x RGB')
    pixels = np.stack(arrays).astype(np.float32)
    mean = pixels.mean(axis=0)
    # Non-overlapping pairs prevent one frame appearing on both sides of two residuals.
    # Difference / sqrt(2) estimates single-frame temporal noise if stationary
    # and independent; drift, motion, compression and ISP remain in this result.
    residual = (pixels[1::2][:len(pixels)//2] - pixels[0::2][:len(pixels)//2]) / np.sqrt(2.)
    rows = []
    edges = [0, 16, 32, 64, 96, 128, 160, 192, 224, 240, 256]
    for channel, name in enumerate('RGB'):
        for low, high in [(0, 256)] + list(zip(edges[:-1], edges[1:])):
            selected = (mean[:, :, channel] >= low) & (mean[:, :, channel] < high)
            values = residual[:, :, :, channel][:, selected].ravel()
            if not values.size:
                continue
            rows.append(dict(channel=name, brightness_min_dn=low, brightness_max_dn_exclusive=high,
                             pixels=int(selected.sum()), independent_frame_pairs=len(residual),
                             residual_samples=len(values), mean_residual_dn=float(values.mean()),
                             temporal_std_dn=float(values.std()), rms_dn=float(np.sqrt(np.mean(values**2))),
                             mad_sigma_dn=float(np.median(np.abs(values - np.median(values))) / .6744897501960817),
                             p01_dn=float(np.quantile(values, .01)), p05_dn=float(np.quantile(values, .05)),
                             median_dn=float(np.median(values)), p95_dn=float(np.quantile(values, .95)),
                             p99_dn=float(np.quantile(values, .99)),
                             saturation_fraction=float(((pixels[:, :, :, channel][:, selected] == 0) |
                                                        (pixels[:, :, :, channel][:, selected] == 255)).mean())))
    diagnostics = []
    for channel, name in enumerate('RGB'):
        r = residual[:, :, :, channel]
        frame_mean = pixels[:, :, :, channel].mean(axis=(1, 2))
        diagnostics.append(dict(channel=name,
            horizontal_residual_correlation=correlation(r[:, :, :-1], r[:, :, 1:]),
            vertical_residual_correlation=correlation(r[:, :-1, :], r[:, 1:, :]),
            frame_mean_min_dn=float(frame_mean.min()), frame_mean_max_dn=float(frame_mean.max()),
            frame_mean_linear_slope_dn_per_frame=float(np.polyfit(np.arange(len(frame_mean)), frame_mean, 1)[0]),
            median_per_pixel_temporal_std_dn=float(np.median(pixels[:, :, :, channel].std(axis=0, ddof=1)))))
    required = {'camera_model', 'sensor', 'exposure_us', 'gain', 'illumination', 'isp', 'output_settings'}
    report = dict(scope='processed RGB temporal residual at the recorded working settings; static-scene diagnostic, EMVA-inspired, not a complete EMVA sensor characterization',
                  estimator='nonoverlapping adjacent-frame difference / sqrt(2); assumes stationary independent noise; drift and motion are not subtracted',
                  frames=len(pixels), shape=list(shape), dtype='uint8', dn_range=[0, 255],
                  camera_metadata=metadata, missing_provenance=sorted(required - metadata.keys()),
                  metadata_sha256=sha(metadata_path) if metadata_path.exists() else None,
                  input_frames=frames, diagnostics=diagnostics,
                  exposure_gain_grouping='one fixed metadata configuration per input directory; estimate each exposure/gain/ISP separately',
                  simulated_noise_equivalence='compare measured temporal_std_dn and quantiles to 3/6/12/20.4 DN; no claim of calibrated real-camera equivalence')
    csv_write(args.output / 'noise_by_brightness_channel.csv', rows)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(frames=len(pixels), output=str(args.output), missing_provenance=report['missing_provenance'])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--device', required=True, help='explicit camera index or device path')
    c.add_argument('--metadata', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--frames', type=int, default=128)
    c.add_argument('--warmup', type=int, default=30)
    c.add_argument('--width', type=int)
    c.add_argument('--height', type=int)
    c.add_argument('--fps', type=float)
    c.add_argument('--native-exposure', type=float)
    c.add_argument('--native-gain', type=float)
    c.set_defaults(function=capture)
    e = sub.add_parser('estimate')
    e.add_argument('--input', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    e.set_defaults(function=estimate)
    args = p.parse_args()
    if args.command == 'capture' and args.frames < 4:
        p.error('capture needs at least four frames')
    args.function(args)


if __name__ == '__main__':
    main()
