"""Isaac launch and measured joint-name mapping for the shared hand interface."""
from __future__ import annotations
import os,sys,json,math,pickle,copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from hand_runtime import selected,configure_env as _configure_env
_hand=selected()
HAND_ONLY_JOINT_NAMES=list(_hand['joint_names'])
BASE_JOINT_NAMES=list(_hand['base_joint_names'])
XHAND_JOINT_NAMES=[n for n in HAND_ONLY_JOINT_NAMES if n not in BASE_JOINT_NAMES]
BODY_NAME_ALIASES={}
from isaaclab.app import AppLauncher
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--motion_pkl')
parser.add_argument('--save_video',action='store_true')
parser.add_argument('--video_width',type=int,default=320)
parser.add_argument('--video_height',type=int,default=240)
parser.add_argument('--camera-fps',dest='video_fps',type=float,default=30.)
parser.add_argument('--camera_eye',default='0,-0.001,0.5')
parser.add_argument('--camera_target',default='0,0,0')
AppLauncher.add_app_launcher_args(parser)
args_cli=parser.parse_args()
args_cli.enable_cameras=True
args_cli.robot_urdf=_hand['urdf_path']
args_cli.base_stiffness=1800.;args_cli.base_damping=120.;args_cli.base_max_effort=200.
args_cli.task='FingerEye-Shared-Grasp-v1'
app_launcher=AppLauncher(args_cli)
simulation_app=app_launcher.app
import torch
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab_tasks.utils import parse_env_cfg
gym.register(id=args_cli.task,entry_point='sim_env.hand_environment:FingerEyeXHandSyntheLabEnv',disable_env_checker=True,
 kwargs={'env_cfg_entry_point':'sim_env.hand_config:FingerEyeXHandSyntheLabEnvCfg'})
def _normalize_quat_wxyz_numpy(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1.0e-12)


def _base_rpy_to_quat_wxyz_numpy(yaw, pitch, roll) -> np.ndarray:
    """Return the quaternion for Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    yaw, pitch, roll = np.broadcast_arrays(
        np.asarray(yaw, dtype=np.float64),
        np.asarray(pitch, dtype=np.float64),
        np.asarray(roll, dtype=np.float64),
    )
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    quat = np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )
    return _normalize_quat_wxyz_numpy(quat)


def _env_joint_names(env) -> list[str]:
    return [str(name) for name in env.hand.joint_names]


def _canonical_to_env_q(env, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    src = {name: idx for idx, name in enumerate(HAND_ONLY_JOINT_NAMES)}
    missing = [name for name in _env_joint_names(env) if name not in src]
    if missing:
        raise RuntimeError("Env hand joints are missing from canonical q order: " + ", ".join(missing))
    return np.asarray([q[src[name]] for name in _env_joint_names(env)], dtype=np.float32)


def _env_to_canonical_q(env, q_env: np.ndarray) -> np.ndarray:
    q_env = np.asarray(q_env, dtype=np.float32).reshape(-1)
    src = {name: idx for idx, name in enumerate(_env_joint_names(env))}
    missing = [name for name in HAND_ONLY_JOINT_NAMES if name not in src]
    if missing:
        raise RuntimeError("Canonical hand joints are missing from env q order: " + ", ".join(missing))
    return np.asarray([q_env[src[name]] for name in HAND_ONLY_JOINT_NAMES], dtype=np.float32)


def _canonical_to_env_q_batch(env, q_batch: np.ndarray) -> np.ndarray:
    q_batch = np.asarray(q_batch, dtype=np.float32)
    if q_batch.ndim != 2:
        raise ValueError(f"Expected q_batch to be 2D, got {q_batch.shape}.")
    return np.stack([_canonical_to_env_q(env, q) for q in q_batch], axis=0).astype(np.float32)


def _control_target_from_q(q: np.ndarray) -> np.ndarray:
    by_name = {name: float(q[idx]) for idx, name in enumerate(HAND_ONLY_JOINT_NAMES)}
    return np.asarray(
        [by_name[name] for name in XHAND_JOINT_NAMES],
        dtype=np.float32,
    )


def _torch_quat_multiply_wxyz(left, right):
    import torch

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )


def _torch_quat_error_rotvec_wxyz(target, current):
    """Shortest world-frame rotvec taking current body orientation to target."""
    import torch

    eps = torch.finfo(target.dtype).eps
    target = target / torch.linalg.vector_norm(target, dim=-1, keepdim=True).clamp_min(eps)
    current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(eps)
    current_conjugate = torch.cat([current[..., :1], -current[..., 1:]], dim=-1)
    error = _torch_quat_multiply_wxyz(target, current_conjugate)
    error = error / torch.linalg.vector_norm(error, dim=-1, keepdim=True).clamp_min(eps)
    error = torch.where(error[..., :1] < 0.0, -error, error)
    vector = error[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, error[..., :1].clamp(0.0, 1.0))
    scale = torch.where(vector_norm > 1.0e-7, angle / vector_norm.clamp_min(eps), torch.full_like(vector_norm, 2.0))
    return vector * scale


def _torch_clamp_vector_norm(value, max_norm: float):
    import torch

    if float(max_norm) <= 0.0:
        return torch.zeros_like(value)
    eps = torch.finfo(value.dtype).eps
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = (float(max_norm) / norm.clamp_min(eps)).clamp(max=1.0)
    return value * scale


def _set_base_pose_overlay_wrench(env, forces, torques, body_id: int, env_ids) -> None:
    """Set the persistent palm wrench without emitting a warning every step.

    Newer Isaac Lab releases expose the wrench composer directly and mark the
    legacy articulation setter as deprecated.  Calling the legacy setter in a
    control loop logs once per physics step, which both floods the rollout log
    and makes the video visibly stutter.  Keep a compatibility fallback for
    older installations that do not yet expose the composer.
    """
    composer = getattr(env.hand, "permanent_wrench_composer", None)
    if composer is not None and hasattr(composer, "set_forces_and_torques"):
        composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=[int(body_id)],
            env_ids=env_ids,
            is_global=True,
        )
        return
    env.hand.set_external_force_and_torque(
        forces,
        torques,
        body_ids=[int(body_id)],
        env_ids=env_ids,
        is_global=True,
    )


def _clear_base_pose_overlay(env, env_ids) -> None:
    config = getattr(env, "_xhand_base_pose_overlay_config", None)
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return
    body_id = config.get("body_id")
    if body_id is None:
        return
    import torch

    count = len(env_ids)
    zeros = torch.zeros((count, 1, 3), device=env.device, dtype=torch.float32)
    _set_base_pose_overlay_wrench(env, zeros, zeros, int(body_id), env_ids)


def _apply_base_pose_overlay(env, q_batch: np.ndarray, env_ids) -> None:
    """Apply a world-frame Cartesian/quaternion PD wrench to the palm body."""
    config = getattr(env, "_xhand_base_pose_overlay_config", None)
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return
    body_id = config.get("body_id")
    if body_id is None:
        return
    import torch

    q_batch = np.asarray(q_batch, dtype=np.float32)
    if q_batch.ndim == 1:
        q_batch = q_batch.reshape(1, -1)
    if q_batch.ndim != 2 or q_batch.shape[1] < len(BASE_JOINT_NAMES):
        raise ValueError(f"Invalid base overlay q target shape: {q_batch.shape}")
    if q_batch.shape[0] != len(env_ids):
        raise ValueError(
            f"Base overlay q batch/env mismatch: q={q_batch.shape[0]}, env_ids={len(env_ids)}"
        )

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    target_pos_w = torch.as_tensor(q_batch[:, :3], device=env.device, dtype=torch.float32)
    target_pos_w = target_pos_w + env.scene.env_origins[env_ids_t]
    target_quat_wxyz = _base_rpy_to_quat_wxyz_numpy(
        q_batch[:, 3], q_batch[:, 4], q_batch[:, 5]
    ).astype(np.float32)
    target_quat_w = torch.as_tensor(target_quat_wxyz, device=env.device, dtype=torch.float32)

    current_pos_w = env.hand.data.body_pos_w[env_ids_t, int(body_id)]
    current_quat_w = env.hand.data.body_quat_w[env_ids_t, int(body_id)]
    current_lin_vel_w = env.hand.data.body_lin_vel_w[env_ids_t, int(body_id)]
    current_ang_vel_w = env.hand.data.body_ang_vel_w[env_ids_t, int(body_id)]

    force_w = float(config["position_kp"]) * (target_pos_w - current_pos_w)
    force_w -= float(config["position_kd"]) * current_lin_vel_w
    orientation_error_w = _torch_quat_error_rotvec_wxyz(target_quat_w, current_quat_w)
    torque_w = float(config["orientation_kp"]) * orientation_error_w
    torque_w -= float(config["orientation_kd"]) * current_ang_vel_w
    force_w = _torch_clamp_vector_norm(force_w, float(config["max_force"]))
    torque_w = _torch_clamp_vector_norm(torque_w, float(config["max_torque"]))

    _set_base_pose_overlay_wrench(
        env,
        force_w.unsqueeze(1).contiguous(),
        torque_w.unsqueeze(1).contiguous(),
        int(body_id),
        env_ids_t,
    )


def _motion_sample_for_dynamic_env(
    payload: dict,
    q_start: np.ndarray,
    object_mass: float,
    robot_friction: float,
    object_friction: float,
) -> dict:
    source = dict(payload.get("source_candidate", {}))
    sample = dict(source)
    sample.update(
        {
            "format": "fingereye.combined_motion_static_gravity_sample.v1",
            "object_mesh_path": str(payload.get("object_mesh_path", source.get("object_mesh_path", ""))),
            "object_pose_world": np.asarray(payload.get("object_pose_world", payload.get("object_pose_env")), dtype=np.float32),
            "object_pose_env": np.asarray(payload.get("object_pose_env", payload.get("object_pose_world")), dtype=np.float32),
            "object_scale": np.asarray(payload.get("object_scale", source.get("object_scale", [1.0, 1.0, 1.0])), dtype=np.float32),
            "object_mass": float(object_mass),
            "robot_friction": float(robot_friction),
            "object_friction": float(object_friction),
            "current_all_joint_values": np.asarray(q_start, dtype=np.float32),
            "target_all_joint_values": np.asarray(q_start, dtype=np.float32),
            "target_action": _control_target_from_q(np.asarray(q_start, dtype=np.float32)),
            "joint_names": list(HAND_ONLY_JOINT_NAMES),
            "control_joint_names": list(XHAND_JOINT_NAMES),
            "robot_root_pose_env": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
    )
    return sample


def _set_q_state_and_target_batch(env, q_batch: np.ndarray, env_ids, *, snap_state: bool) -> None:
    import torch

    q_tensor = torch.as_tensor(_canonical_to_env_q_batch(env, q_batch), device=env.device)
    if snap_state:
        q_vel = torch.zeros_like(q_tensor)
        env.hand.write_joint_state_to_sim(q_tensor, q_vel, env_ids=env_ids)
        _clear_base_pose_overlay(env, env_ids)
    else:
        _apply_base_pose_overlay(env, q_batch, env_ids)
    env.hand.set_joint_position_target(q_tensor, env_ids=env_ids)


def _write_object_pose_batch(
    env,
    object_pose_envs: np.ndarray,
    env_ids,
    *,
    validation_started: bool = False,
) -> None:
    if validation_started:
        raise RuntimeError(
            "OBJECT_POSE_WRITE_AFTER_VALIDATION: pose rewriting is forbidden during gravity validation"
        )
    if getattr(env, "object", None) is None:
        return
    import torch

    pose = torch.as_tensor(np.asarray(object_pose_envs, dtype=np.float32).reshape(len(env_ids), 7), device=env.device)
    pose_w = pose.clone()
    pose_w[:, :3] += env.scene.env_origins[env_ids]
    env.object.write_root_pose_to_sim(pose_w, env_ids)
    zero_vel = torch.zeros((len(env_ids), 6), device=env.device, dtype=torch.float32)
    if hasattr(env.object, "write_root_link_velocity_to_sim"):
        env.object.write_root_link_velocity_to_sim(zero_vel, env_ids)
    else:
        env.object.write_root_velocity_to_sim(zero_vel, env_ids)


def _remove_background_only(num_envs: int = 1) -> None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim_paths = []
    for env_id in range(max(1, int(num_envs))):
        prim_paths.extend(
            [
                f"/World/envs/env_{env_id}/Background",
            ]
        )
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            stage.RemovePrim(prim_path)


def _ensure_ground_table() -> None:
    import omni.usd
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    stage = omni.usd.get_context().get_stage()
    ground_path = "/World/ground"
    if not stage.GetPrimAtPath(ground_path).IsValid():
        spawn_ground_plane(prim_path=ground_path, cfg=GroundPlaneCfg())


def _physics_configuration_snapshot(env, num_envs: int, *, independent_ground: bool) -> list[dict]:
    """Read authored collision settings and live PhysX mass/inertia per slot."""
    from pxr import Usd, UsdPhysics, UsdShade

    stage = env.sim.stage
    snapshots = []
    object_view = env.object.root_physx_view
    object_masses = object_view.get_masses().detach().cpu().numpy()
    object_inertias = object_view.get_inertias().detach().cpu().numpy()
    hand_view = env.hand.root_physx_view
    hand_masses = hand_view.get_masses().detach().cpu().numpy()
    hand_inertias = hand_view.get_inertias().detach().cpu().numpy()

    def properties(root_path: str) -> list[dict]:
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return []
        entries = []
        for prim in Usd.PrimRange(root):
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    or prim.HasAPI(UsdPhysics.MassAPI)):
                continue
            attrs = {}
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if name.startswith(("physics:", "physx")) and attr.HasAuthoredValueOpinion():
                    attrs[name] = str(attr.Get())
            material_path = None
            material_properties = None
            effective_physics_material = None
            try:
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
                if material:
                    material_path = str(material.GetPath())
                    material_prim = material.GetPrim()
                    material_properties = {
                        "schemas": list(material_prim.GetAppliedSchemas()),
                        "attributes": {
                            attr.GetName(): str(attr.Get())
                            for attr in material_prim.GetAttributes()
                            if attr.GetName().startswith(("physics:", "physx")) and attr.HasAuthoredValueOpinion()
                        },
                    }
                    # USD may fall back to an all-purpose preview material
                    # when no physics binding exists. Report that explicitly
                    # rather than calling the preview material friction.
                    if material_prim.HasAPI(UsdPhysics.MaterialAPI):
                        effective_physics_material = material_properties
            except Exception:
                pass
            entries.append({
                "relative_path": str(prim.GetPath()).removeprefix(root_path),
                "type": prim.GetTypeName(),
                "schemas": list(prim.GetAppliedSchemas()),
                "attributes": attrs,
                "bound_physics_material": material_path,
                "bound_material_properties": material_properties,
                "effective_physics_material": effective_physics_material,
            })
        return entries

    for idx in range(num_envs):
        prefix = f"/World/envs/env_{idx}"
        ground = f"{prefix}/AsyncGround" if independent_ground else "/World/ground"
        snapshots.append({
            "env_index": idx,
            "object": properties(f"{prefix}/Object"),
            "ground": properties(ground),
            "object_physx_mass": np.asarray(object_masses[idx]).tolist(),
            "object_physx_inertia": np.asarray(object_inertias[idx]).tolist(),
            "hand_physx_mass": np.asarray(hand_masses[idx]).tolist(),
            "hand_physx_inertia": np.asarray(hand_inertias[idx]).tolist(),
        })
    return snapshots


def _object_pose_batch(env, env_ids) -> np.ndarray:
    import torch

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    pos = env.object.data.root_pos_w[env_ids_t].detach().cpu().numpy()
    quat = env.object.data.root_quat_w[env_ids_t].detach().cpu().numpy()
    return np.concatenate([pos, quat], axis=1).astype(np.float32)


def _object_velocity_batch(env, env_ids) -> np.ndarray:
    """Return object world linear/angular velocity as [vx, vy, vz, wx, wy, wz]."""
    import torch

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    linear = env.object.data.root_lin_vel_w[env_ids_t].detach().cpu().numpy()
    angular = env.object.data.root_ang_vel_w[env_ids_t].detach().cpu().numpy()
    return np.concatenate([linear, angular], axis=1).astype(np.float32)

