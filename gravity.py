"""World-frame object gravity; shared by the frozen physics loop."""
def world_to_body(force_w, quaternion_wxyz):
    """Inverse rotation for unit quaternions; accepts (..., 3), (..., 4)."""
    import torch
    # PhysX link quaternions use wxyz in IsaacLab's public data interface.
    q = quaternion_wxyz / torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    axis = -q[..., 1:]
    twice_cross = 2. * torch.cross(axis, force_w, dim=-1)
    return force_w + q[..., :1] * twice_cross + torch.cross(axis, twice_cross, dim=-1)


def set_world_gravity_force(env, env_ids, object_mass, gravity_scale, horizontal_force_xy=None):
    if getattr(env, 'object', None) is None:
        return
    import torch
    num_envs = len(env_ids)
    num_bodies = int(getattr(env.object, 'num_bodies', 1))
    force_w = torch.zeros((num_envs, num_bodies, 3), device=env.device, dtype=torch.float32)
    force_w[..., 2] = -9.81 * float(gravity_scale) * float(object_mass) / max(num_bodies, 1)
    if horizontal_force_xy is not None:
        horizontal = torch.as_tensor(horizontal_force_xy, dtype=torch.float32, device=env.device).reshape(num_envs, 2)
        force_w[..., :2] = horizontal[:, None, :] / max(num_bodies, 1)
    # The backend calls scene.update(dt) after every physics step, invalidating
    # the public data cache. This reads the current pose rather than the
    # permanent wrench composer's stale first quaternion.
    quaternion = env.object.data.body_link_quat_w[env_ids]
    force_b = world_to_body(force_w, quaternion)
    env.object.set_external_force_and_torque(
        force_b, torch.zeros_like(force_b), env_ids=env_ids, is_global=False)

