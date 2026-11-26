"""Explicit R3-G physics: unstabilized, per-step world-fixed object gravity.

Preserves the original backend and evaluator. The installed IsaacLab permanent
wrench composer caches its first body quaternion even across subsequent global
force calls. Supply a newly transformed LOCAL force each step to bypass that
cache, without touching installed IsaacLab or changing historical trial files.
"""
import builtins
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PHYSICS_VERSION = 'R3-G'


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


def backend_main():
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    original_import = builtins.__import__
    patched = False
    audit_stream = (output/'gravity_force_audit.jsonl').open('w', buffering=1)

    def importing(name, *args, **kwargs):
        nonlocal patched
        module = original_import(name, *args, **kwargs)
        if name == 'rollout_xhand_motion_dual' and not patched:
            patched = True
            original_configure = module._configure_env

            def configure(cfg, *args, **kwargs):
                original_configure(cfg, *args, **kwargs)
                cfg.sim.physx.enable_stabilization = False
                (output / 'physics_version.json').write_text(json.dumps({
                    'version': PHYSICS_VERSION,
                    'enable_stabilization': False,
                    'solver_type': cfg.sim.physx.solver_type,
                    'gravity_frame': 'world',
                    'gravity_implementation': 'current_link_inverse_rotation_then_local_force_each_step',
                    'evaluator_thresholds_unchanged': True,
                    'physics_changed_from_round2': True,
                    'physics_changed_from_R3_U': True,
                    'do_not_pool_with_prior_physics_results': True,
                }, indent=2) + '\n')
            module._configure_env = configure
            def audited_gravity(env, ids, mass, scale, horizontal_force_xy=None):
                import torch
                set_world_gravity_force(env, ids, mass, scale, horizontal_force_xy)
                q=env.object.data.body_link_quat_w[ids]
                inverse=q.clone();inverse[...,1:]*=-1
                local=env.object.permanent_wrench_composer.composed_force_as_torch[ids]
                actual=world_to_body(local,inverse)
                expected=torch.zeros_like(actual);expected[...,2]=-9.81*float(mass)*float(scale)/actual.shape[1]
                if horizontal_force_xy is not None:expected[...,:2]=torch.as_tensor(horizontal_force_xy,device=env.device)[:,None,:]/actual.shape[1]
                error=float(torch.max(torch.abs(actual-expected)))
                audit_stream.write(json.dumps({'time_s':env.object.data._sim_timestamp,'world_force_n':actual.detach().cpu().tolist(),'desired_world_force_n':expected.detach().cpu().tolist(),'max_error_n':error})+'\n')
                if error>1e-6:raise RuntimeError('Applied gravity is not world-fixed')
            module._set_object_gravity_force = audited_gravity
        return module

    builtins.__import__ = importing
    try:
        import backend
        backend.main()
    finally:
        builtins.__import__ = original_import
        audit_stream.close()


def runner():
    import run_episode as run
    original_peer = run.Peer

    class VersionedPeer(original_peer):
        def __init__(self, command, *args, **kwargs):
            backend_path = str(HERE / 'backend.py')
            if backend_path in command:
                command = command.copy()
                index = command.index(backend_path)
                command[index:index + 1] = [str(Path(__file__).resolve()), '--backend-probe']
                output = Path(command[command.index('--output') + 1])
                source = Path(__file__).read_bytes()
                (output / 'physics_wrapper_snapshot.py').write_bytes(source)
                (output / 'physics_wrapper_hash.json').write_text(json.dumps({
                    'sha256': hashlib.sha256(source).hexdigest(), 'version': PHYSICS_VERSION}) + '\n')
            super().__init__(command, *args, **kwargs)

    run.Peer = VersionedPeer
    return run.main()


if __name__ == '__main__':
    if '--backend-probe' in sys.argv:
        sys.argv.remove('--backend-probe')
        backend_main()
    else:
        raise SystemExit(runner())
