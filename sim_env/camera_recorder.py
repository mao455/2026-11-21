"""Streaming video recorder for the canonical ``xhand3camFE_left`` cameras.

The recorder consumes synchronized RGB batches from the three cameras already
mounted by :mod:`xhand_camera_config`.  It streams each camera and the composed
three-view image to MP4, writes per-frame metadata and event keyframes, and
keeps episode/run manifests valid even when a rollout aborts.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import time

import numpy as np

from .camera_config import (
    ALL_XHAND_CAMERA_NAMES,
    CALIBRATION_STATUS,
    CAMERA_WARNING,
    DISTORTION_MODEL,
)


ROBOT_FAMILY = "xhand3camFE_left"
XHAND_RECORDING_CAMERA_NAMES = ALL_XHAND_CAMERA_NAMES
THIRD_VIEW_CAMERA_NAME = "third_view"
COMPOSITE_VIDEO_NAME = "synchronized_three_view"

__all__ = [
    "COMPOSITE_VIDEO_NAME",
    "ROBOT_FAMILY",
    "THIRD_VIEW_CAMERA_NAME",
    "XHAND_RECORDING_CAMERA_NAMES",
    "XHandCameraRecorder",
    "capture_named_camera_batches",
    "compose_xhand_three_view",
    "to_uint8_batch",
]


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def to_uint8_batch(image) -> np.ndarray:
    """Convert one tiled-camera batch to contiguous RGB uint8."""

    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected camera batch shape: {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.max(finite)) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def capture_named_camera_batches(
    env,
    camera_names: tuple[str, ...] | list[str],
) -> dict[str, np.ndarray]:
    """Render once, then read a synchronized bundle from active sensors."""

    if env.sim.has_rtx_sensors():
        env.sim.render()
    env.scene.update(dt=0.0)
    output: dict[str, np.ndarray] = {}
    active = getattr(env, "_active_cameras", {})
    for name in camera_names:
        sensor = active.get(name)
        if sensor is None:
            continue
        rgb = getattr(sensor.data, "output", {}).get("rgb")
        if rgb is None:
            continue
        output[str(name)] = to_uint8_batch(rgb)
    return output


def compose_xhand_three_view(
    frames: dict[str, np.ndarray],
    *,
    simulation_step: int | None = None,
    simulation_time_s: float | None = None,
    phase: str | None = None,
    episode_id: int | None = None,
) -> np.ndarray:
    """Compose the three XHand streams with padding and optional audit text."""

    sensor_1 = np.asarray(frames["sensor_1"], dtype=np.uint8)
    sensor_2 = np.asarray(frames["sensor_2"], dtype=np.uint8)
    sensor_3 = np.asarray(frames["sensor_3_back"], dtype=np.uint8)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:480, :640] = sensor_1
    canvas[:480, 640:1280] = sensor_2
    canvas[480:720, 480:800] = sensor_3
    try:
        import cv2

        lines = []
        if episode_id is not None and simulation_step is not None:
            lines.append(f"episode={episode_id:06d} step={simulation_step}")
        if simulation_time_s is not None:
            lines.append(f"sim_time={simulation_time_s:.3f}s")
        if phase:
            lines.append(f"phase={phase}")
        for idx, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (12, 516 + 32 * idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    except Exception:
        pass
    return canvas


class XHandCameraRecorder:
    """Streaming recorder for the formal XHand three-camera module.

    Frames are piped to ffmpeg as they are captured, so vectorized rollouts do
    not retain several gigabytes of RGB arrays.  Each bundle is read after one
    render call and receives one shared simulation step/time record.
    """

    def __init__(
        self,
        *,
        output_root: Path,
        gravity_stems: list[str],
        candidate_ranks: list[int],
        primary_finger: str,
        camera_metadata: dict[str, dict],
        requested_fps: float,
        simulation_dt: float,
        render_interval: int,
        task_id: str,
        robot_usd_path: str | Path,
        robot_urdf_path: str | Path,
        robot_urdf_name: str,
        experiment_entrypoint: str | Path,
        codec: str = "libx264",
        strict: bool = True,
        save_keyframes: bool = True,
    ) -> None:
        self.output_root = Path(output_root)
        self.gravity_stems = list(gravity_stems)
        self.candidate_ranks = [int(value) for value in candidate_ranks]
        self.primary_finger = str(primary_finger)
        self.camera_metadata = copy.deepcopy(camera_metadata)
        self.requested_fps = float(requested_fps)
        self.simulation_dt = float(simulation_dt)
        self.render_interval = int(render_interval)
        self.task_id = str(task_id)
        self.robot_usd_path = str(Path(robot_usd_path).resolve())
        self.robot_urdf_path = str(Path(robot_urdf_path).resolve())
        self.robot_urdf_name = str(robot_urdf_name)
        self.experiment_entrypoint = str(Path(experiment_entrypoint).resolve())
        self.codec = str(codec)
        self.strict = bool(strict)
        self.save_keyframes = bool(save_keyframes)
        self._writers: list[dict[str, object]] = [dict() for _ in gravity_stems]
        self._metadata_files = []
        self._bundle_counts = [0 for _ in gravity_stems]
        self._captured_counts = [
            {
                name: 0
                for name in (
                    *XHAND_RECORDING_CAMERA_NAMES,
                    THIRD_VIEW_CAMERA_NAME,
                )
            }
            for _ in gravity_stems
        ]
        self._unavailable_counts = [
            {
                name: 0
                for name in (
                    *XHAND_RECORDING_CAMERA_NAMES,
                    THIRD_VIEW_CAMERA_NAME,
                )
            }
            for _ in gravity_stems
        ]
        self._encoding_errors: list[list[str]] = [[] for _ in gravity_stems]
        self._first_simulation_time: list[float | None] = [None for _ in gravity_stems]
        self._last_simulation_time: list[float | None] = [None for _ in gravity_stems]
        self._scheduler_dropped_counts = [
            {
                name: 0
                for name in (
                    *XHAND_RECORDING_CAMERA_NAMES,
                    THIRD_VIEW_CAMERA_NAME,
                )
            }
            for _ in gravity_stems
        ]
        self._runtime_intrinsics: dict[str, object] = {}
        self._closed = False
        self._saved_keyframe_events: list[set[str]] = [set() for _ in gravity_stems]
        run_manifest_path = self.output_root / "cameras" / "camera_manifest.json"
        next_episode_id = 0
        if run_manifest_path.is_file():
            try:
                existing_manifest = json.loads(run_manifest_path.read_text())
                existing_episode_ids = [
                    int(episode["episode_id"])
                    for episode in existing_manifest.get("episodes", [])
                    if episode.get("episode_id") is not None
                ]
                if existing_episode_ids:
                    next_episode_id = max(existing_episode_ids) + 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                next_episode_id = 0
        self._episode_ids = [
            next_episode_id + env_idx for env_idx in range(len(gravity_stems))
        ]
        self._dirs: list[Path] = []
        for stem in gravity_stems:
            base_directory = self.output_root / "cameras" / stem
            directory = base_directory
            retry = 0
            while directory.exists():
                retry += 1
                directory = self.output_root / "cameras" / f"{stem}_retry_{retry:03d}"
            directory.mkdir(parents=True, exist_ok=False)
            self._dirs.append(directory)
            self._metadata_files.append((directory / "frame_metadata.jsonl").open("w"))

    def _video_path(self, env_idx: int, name: str) -> Path:
        if name == THIRD_VIEW_CAMERA_NAME:
            return self._dirs[env_idx] / "third_view.mp4"
        return self._dirs[env_idx] / f"{name}.mp4"

    def _get_writer(self, env_idx: int, name: str):
        writer = self._writers[env_idx].get(name)
        if writer is not None:
            return writer
        import imageio.v2 as imageio

        path = self._video_path(env_idx, name)
        try:
            writer = imageio.get_writer(
                path,
                fps=self.requested_fps,
                codec=self.codec,
                quality=8,
                macro_block_size=None,
                output_params=["-preset", "veryfast", "-threads", "1"],
            )
        except Exception as exc:
            message = f"{name}: {type(exc).__name__}: {exc}"
            self._encoding_errors[env_idx].append(message)
            if self.strict:
                raise RuntimeError(f"CAMERA_RECORDING_INITIALIZATION_FAILED: {message}") from exc
            return None
        self._writers[env_idx][name] = writer
        return writer

    def update_runtime_intrinsics(self, env) -> None:
        """Cache renderer-reported intrinsics from the active XHand sensors."""

        for name in XHAND_RECORDING_CAMERA_NAMES:
            sensor = getattr(env, "_active_cameras", {}).get(name)
            matrices = getattr(getattr(sensor, "data", None), "intrinsic_matrices", None)
            if matrices is None:
                continue
            arr = matrices.detach().cpu().numpy() if hasattr(matrices, "detach") else np.asarray(matrices)
            if arr.size:
                self._runtime_intrinsics[name] = np.asarray(arr[0], dtype=np.float64).tolist()

    def append(
        self,
        batches: dict[str, np.ndarray],
        *,
        simulation_step: int,
        simulation_time_s: float,
        phase: str,
        event: str | None = None,
    ) -> None:
        """Append one synchronized bundle for every vectorized environment."""

        num_envs = len(self.gravity_stems)
        for env_idx in range(num_envs):
            if self._first_simulation_time[env_idx] is None:
                self._first_simulation_time[env_idx] = float(simulation_time_s)
            self._last_simulation_time[env_idx] = float(simulation_time_s)
            bundle_id = self._bundle_counts[env_idx]
            camera_frames: dict[str, np.ndarray] = {}
            camera_entries: dict[str, dict] = {}
            for name in (*XHAND_RECORDING_CAMERA_NAMES, THIRD_VIEW_CAMERA_NAME):
                batch = batches.get(name)
                available = bool(batch is not None and env_idx < int(batch.shape[0]))
                if not available:
                    self._unavailable_counts[env_idx][name] += 1
                    camera_entries[name] = {"available": False}
                    continue
                frame = np.asarray(batch[env_idx], dtype=np.uint8)
                camera_frames[name] = frame
                frame_id = self._captured_counts[env_idx][name]
                writer = self._get_writer(env_idx, name)
                if writer is not None:
                    try:
                        writer.append_data(frame)
                    except Exception as exc:
                        message = f"{name}: {type(exc).__name__}: {exc}"
                        self._encoding_errors[env_idx].append(message)
                        if self.strict:
                            raise RuntimeError(f"CAMERA_RECORDING_WRITE_FAILED: {message}") from exc
                self._captured_counts[env_idx][name] += 1
                spec = self.camera_metadata.get(name, {})
                camera_entries[name] = {
                    "available": True,
                    "frame_id": int(frame_id),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "camera_prim_path": str(spec.get("camera_prim_path_pattern", "")).replace(
                        "env_.*", f"env_{env_idx}"
                    ),
                    "calibration_status": spec.get("calibration_status", CALIBRATION_STATUS),
                    "intrinsic_matrix": spec.get("intrinsic_matrix"),
                    "renderer_intrinsic_matrix": spec.get("renderer_intrinsic_matrix"),
                }

            complete = all(name in camera_frames for name in XHAND_RECORDING_CAMERA_NAMES)
            any_camera = any(name in camera_frames for name in XHAND_RECORDING_CAMERA_NAMES)
            record = {
                "episode_id": int(self._episode_ids[env_idx]),
                "candidate_rank": int(self.candidate_ranks[env_idx]),
                "bundle_id": int(bundle_id),
                "simulation_step": int(simulation_step),
                "simulation_time_s": float(simulation_time_s),
                "fsm_state": str(phase),
                "task_stage": str(phase),
                "capture_wall_time_ns": int(time.time_ns()),
                "event": event,
                "synchronization_status": (
                    "COMPLETE" if complete else "PARTIAL" if any_camera else "UNAVAILABLE"
                ),
                "cameras": camera_entries,
            }
            self._metadata_files[env_idx].write(json.dumps(_jsonable(record)) + "\n")
            self._metadata_files[env_idx].flush()
            if event and self.save_keyframes and event not in self._saved_keyframe_events[env_idx]:
                self._saved_keyframe_events[env_idx].add(event)
                keyframe_dir = self._dirs[env_idx] / "keyframes"
                keyframe_dir.mkdir(parents=True, exist_ok=True)
                safe_event = re.sub(r"[^A-Za-z0-9_.-]+", "_", event).strip("_") or "event"
                import imageio.v2 as imageio

                for name in XHAND_RECORDING_CAMERA_NAMES:
                    frame = camera_frames.get(name)
                    if frame is None:
                        continue
                    imageio.imwrite(
                        keyframe_dir
                        / (
                            f"bundle_{bundle_id:06d}_step_{simulation_step:08d}_"
                            f"{safe_event}_{name}.jpg"
                        ),
                        frame,
                        quality=92,
                    )
            self._bundle_counts[env_idx] += 1

    def close(
        self,
        *,
        outcomes: list[dict] | None = None,
        aborted_error: str | None = None,
    ) -> None:
        """Finalize streams and manifests after success or an aborted rollout."""

        if self._closed:
            return
        self._closed = True
        for env_idx, writers in enumerate(self._writers):
            for name, writer in list(writers.items()):
                try:
                    writer.close()
                except Exception as exc:
                    self._encoding_errors[env_idx].append(
                        f"{name} close: {type(exc).__name__}: {exc}"
                    )
            try:
                self._metadata_files[env_idx].close()
            except Exception:
                pass

            outcome = outcomes[env_idx] if outcomes is not None and env_idx < len(outcomes) else {}
            cameras = copy.deepcopy(self.camera_metadata)
            first_time = self._first_simulation_time[env_idx]
            last_time = self._last_simulation_time[env_idx]
            duration = (
                max(0.0, float(last_time) - float(first_time))
                if first_time is not None and last_time is not None
                else 0.0
            )
            required_unavailable_frames = sum(
                self._unavailable_counts[env_idx][name]
                for name in XHAND_RECORDING_CAMERA_NAMES
            )
            required_scheduler_drops = sum(
                self._scheduler_dropped_counts[env_idx][name]
                for name in XHAND_RECORDING_CAMERA_NAMES
            )
            required_streams_have_frames = all(
                self._captured_counts[env_idx][name] > 0
                for name in XHAND_RECORDING_CAMERA_NAMES
            )
            cameras[THIRD_VIEW_CAMERA_NAME] = {
                "logical_name": THIRD_VIEW_CAMERA_NAME,
                "width": None,
                "height": None,
                "calibration_status": "not_applicable_third_view",
                "video_path": str(self._video_path(env_idx, THIRD_VIEW_CAMERA_NAME)),
                "captured_frames": int(self._captured_counts[env_idx][THIRD_VIEW_CAMERA_NAME]),
                "unavailable_frames": int(self._unavailable_counts[env_idx][THIRD_VIEW_CAMERA_NAME]),
                "recording_stats": self._recording_stats(
                    env_idx, THIRD_VIEW_CAMERA_NAME, duration
                ),
            }
            for name, spec in cameras.items():
                if name == THIRD_VIEW_CAMERA_NAME:
                    continue
                spec["runtime_intrinsic_matrix"] = self._runtime_intrinsics.get(name)
                spec["video_path"] = str(self._video_path(env_idx, name))
                spec["captured_frames"] = int(self._captured_counts[env_idx][name])
                spec["unavailable_frames"] = int(self._unavailable_counts[env_idx][name])
                spec["recording_stats"] = self._recording_stats(env_idx, name, duration)
            manifest = {
                "format": "fingereye.xhand_twelve_camera_recording.v1",
                "camera_warning": CAMERA_WARNING,
                "task_id": self.task_id,
                "robot_family": ROBOT_FAMILY,
                "robot_usd_path": self.robot_usd_path,
                "robot_urdf_path": self.robot_urdf_path,
                "robot_urdf_name": self.robot_urdf_name,
                "run_id": self.output_root.name,
                "primary_finger": self.primary_finger,
                "episode_id": int(self._episode_ids[env_idx]),
                "candidate_rank": int(self.candidate_ranks[env_idx]),
                "episode_directory": str(self._dirs[env_idx]),
                "requested_fps": float(self.requested_fps),
                "simulation_dt": float(self.simulation_dt),
                "render_interval": int(self.render_interval),
                "calibration_status": CALIBRATION_STATUS,
                "distortion_model": DISTORTION_MODEL,
                "distortion_coefficients": None,
                "cameras": cameras,
                "frame_metadata_path": str(self._dirs[env_idx] / "frame_metadata.jsonl"),
                "captured_bundles": int(self._bundle_counts[env_idx]),
                "required_camera_unavailable_frames": int(required_unavailable_frames),
                "required_camera_scheduler_dropped_frames": int(required_scheduler_drops),
                "encoding_errors": list(self._encoding_errors[env_idx]),
                "complete": bool(
                    aborted_error is None
                    and not self._encoding_errors[env_idx]
                    and required_unavailable_frames == 0
                    and required_scheduler_drops == 0
                    and required_streams_have_frames
                ),
                "initialization_status": "initialized",
                "shutdown_status": "closed",
                "aborted_error": aborted_error,
                "episode_success": outcome.get("success_strict"),
                "episode_failure_reason": outcome.get("failure_reason"),
            }
            (self._dirs[env_idx] / "camera_manifest.json").write_text(
                json.dumps(_jsonable(manifest), indent=2)
            )
        run_manifest_path = self.output_root / "cameras" / "camera_manifest.json"
        existing_episodes = []
        existing_modules = []
        if run_manifest_path.is_file():
            try:
                existing_manifest = json.loads(run_manifest_path.read_text())
                existing_episodes = list(existing_manifest.get("episodes", []))
                existing_modules = list(existing_manifest.get("primary_finger_modules", []))
            except Exception:
                existing_episodes = []
                existing_modules = []
        current_episodes = [
            json.loads((directory / "camera_manifest.json").read_text())
            for directory in self._dirs
        ]
        run_manifest = {
            "format": "fingereye.xhand_twelve_camera_run.v1",
            "robot_family": ROBOT_FAMILY,
            "task_id": self.task_id,
            "robot_usd_path": self.robot_usd_path,
            "robot_urdf_path": self.robot_urdf_path,
            "robot_urdf_name": self.robot_urdf_name,
            "run_id": self.output_root.name,
            "experiment_entrypoint": self.experiment_entrypoint,
            "primary_finger_modules": sorted(set(existing_modules + [self.primary_finger])),
            "requested_fps": float(self.requested_fps),
            "simulation_dt": float(self.simulation_dt),
            "render_interval": int(self.render_interval),
            "initialization_status": "initialized",
            "shutdown_status": "closed",
            "calibration_status": CALIBRATION_STATUS,
            "distortion_model": DISTORTION_MODEL,
            "distortion_coefficients": None,
            "episodes": existing_episodes + current_episodes,
        }
        run_manifest_path.write_text(json.dumps(_jsonable(run_manifest), indent=2))

    def _recording_stats(self, env_idx: int, name: str, duration: float) -> dict:
        captured = int(self._captured_counts[env_idx][name])
        dropped = int(self._scheduler_dropped_counts[env_idx][name])
        return {
            "requested_frames": int(self._bundle_counts[env_idx]) + dropped,
            "captured_frames": captured,
            "written_frames": captured if not self._encoding_errors[env_idx] else None,
            "unavailable_frames": int(self._unavailable_counts[env_idx][name]),
            "dropped_frames": dropped,
            "encoding_errors": int(len(self._encoding_errors[env_idx])),
            "actual_average_fps": (
                float((captured - 1) / duration)
                if captured > 1 and duration > 0.0
                else 0.0
            ),
            "maximum_queue_size": 0,
        }

    def camera_dir(self, env_idx: int) -> Path:
        """Return the allocated (possibly retry-suffixed) episode directory."""

        return self._dirs[env_idx]

    def record_scheduler_drop(self, count: int) -> None:
        """Record capture deadlines skipped by the simulation-time scheduler."""

        count = max(0, int(count))
        if count == 0:
            return
        for env_counts in self._scheduler_dropped_counts:
            for name in env_counts:
                env_counts[name] += count

    def video_path(self, env_idx: int, name: str) -> Path:
        """Return the path for one camera or composite MP4."""

        return self._video_path(env_idx, name)

    def captured_count(self, env_idx: int, name: str) -> int:
        """Return the number of captured frames for one named stream."""

        return int(self._captured_counts[env_idx][name])

    def encoding_errors(self, env_idx: int) -> list[str]:
        """Return a copy of the episode's accumulated encoding errors."""

        return list(self._encoding_errors[env_idx])
