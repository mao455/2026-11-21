import copy
import hashlib
import json
import os
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_visual_material

from .base_environment import FingerEyeLabEnv
from .base_config import DIR_PATH
from .scene_config import FingerEyeSyntheLabEnvCfg
from .scene_tools import ensure_floor_uv


class FingerEyeSyntheLabEnv(FingerEyeLabEnv):
    """Environment that loads a synthesized final-grasp sample into IsaacSim."""

    cfg: FingerEyeSyntheLabEnvCfg

    def __init__(self, cfg: FingerEyeSyntheLabEnvCfg, render_mode: str | None = None, **kwargs):
        self._synthe_sample_path = self._resolve_sample_path(cfg)
        loaded_sample = self._load_sample(self._synthe_sample_path)
        if "batch_samples" in loaded_sample:
            self._synthe_samples = list(loaded_sample["batch_samples"])
            if not self._synthe_samples:
                raise ValueError(f"{self._synthe_sample_path} has empty batch_samples.")
            self._synthe_batch_mode = True
            self._synthe_object_spawn_template = copy.deepcopy(cfg.object_cfg.spawn)
            self._synthe_usd_cache = {}
            cfg.object_cfg = copy.deepcopy(cfg.object_cfg)
            cfg.object_cfg.spawn = None
        else:
            self._synthe_samples = [loaded_sample]
            self._synthe_batch_mode = False
            self._synthe_object_spawn_template = None
            self._synthe_usd_cache = {}
            self._configure_object_cfg_from_sample(cfg, loaded_sample, self._synthe_sample_path)
        self._synthe_sample = self._synthe_samples[0]
        robot_friction = self._synthe_sample.get("robot_friction")
        if robot_friction is not None:
            robot_friction = float(robot_friction)
            cfg.synthe_robot_static_friction = robot_friction
            cfg.synthe_robot_dynamic_friction = robot_friction
            cfg.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=robot_friction,
                dynamic_friction=robot_friction,
                friction_combine_mode="max",
            )
            cfg.robot_cfg = copy.deepcopy(cfg.robot_cfg)
            if hasattr(cfg.robot_cfg.spawn, "physics_material"):
                cfg.robot_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                    static_friction=robot_friction,
                    dynamic_friction=robot_friction,
                    friction_combine_mode="max",
                )
        super().__init__(cfg, render_mode, **kwargs)
        self._synthe_initial_object_z = np.asarray(
            [np.asarray(sample["object_pose_env"], dtype=np.float32).reshape(7)[2] for sample in self._synthe_samples],
            dtype=np.float32,
        )

    def _bind_synthe_robot_physics_material(self, stage) -> None:
        robot_friction = getattr(self.cfg, "synthe_robot_static_friction", None)
        if robot_friction is None:
            return
        robot_friction = float(robot_friction)
        if robot_friction <= 0.0:
            return
        self._synthe_robot_material_bound_count = 0
        self._synthe_robot_material_verified_count = 0

        material_path = "/World/Looks/SyntheRobotPhysicsMaterial"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=robot_friction,
            dynamic_friction=float(getattr(self.cfg, "synthe_robot_dynamic_friction", robot_friction)),
            friction_combine_mode="max",
        )
        material_cfg.func(material_path, material_cfg)

        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        bound_count = 0
        verified_count = 0
        bound_paths: set[str] = set()

        def deinstance_robot_prim(prim) -> None:
            try:
                if prim.IsInstance():
                    prim.SetInstanceable(False)
            except Exception:
                pass
            try:
                parent = prim.GetParent()
                while parent and parent.IsValid():
                    parent_path = str(parent.GetPath())
                    if "/Robot" not in parent_path:
                        break
                    if parent.IsInstance():
                        parent.SetInstanceable(False)
                    parent = parent.GetParent()
            except Exception:
                pass

        try:
            for prim in stage.Traverse():
                prim_path = str(prim.GetPath())
                if "/Robot" in prim_path:
                    deinstance_robot_prim(prim)
        except Exception:
            pass

        def bind_prim(prim) -> bool:
            nonlocal bound_count, verified_count
            if not prim or not prim.IsValid():
                return False
            prim_path = str(prim.GetPath())
            if prim_path in bound_paths:
                return False
            deinstance_robot_prim(prim)
            try:
                binding_api = (
                    UsdShade.MaterialBindingAPI(prim)
                    if prim.HasAPI(UsdShade.MaterialBindingAPI)
                    else UsdShade.MaterialBindingAPI.Apply(prim)
                )
                binding_api.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                bound_paths.add(prim_path)
                bound_count += 1
                try:
                    bound_material, _ = binding_api.ComputeBoundMaterial(materialPurpose="physics")
                    if bound_material and bound_material.GetPrim().GetPath() == material.GetPrim().GetPath():
                        verified_count += 1
                except Exception:
                    pass
                return True
            except Exception:
                return False

        iterables = [stage.Traverse()]
        try:
            iterables.append(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
        except Exception:
            pass
        seen_paths: set[str] = set()
        for iterable in iterables:
            for prim in iterable:
                prim_path = str(prim.GetPath())
                if prim_path in seen_paths:
                    continue
                seen_paths.add(prim_path)
                if "/Robot" not in prim_path:
                    continue
                is_robot_root = prim_path.endswith("/Robot")
                is_collision_path = "/collisions" in prim_path
                try:
                    has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
                except Exception:
                    has_collision_api = False
                if not (is_robot_root or is_collision_path or has_collision_api):
                    continue
                bind_prim(prim)
                try:
                    if prim.IsInstanceProxy():
                        bind_prim(prim.GetPrimInPrototype())
                except Exception:
                    pass
        self._synthe_robot_material_bound_count = int(bound_count)
        self._synthe_robot_material_verified_count = int(verified_count)
        if bound_count:
            print(
                "[INFO] Bound Synthe robot physics material: "
                f"mu={robot_friction:g}, collision_prims={bound_count}, verified={verified_count}"
            )
        else:
            print(f"[WARN] No Robot collision prims found for Synthe robot material binding (mu={robot_friction:g}).")

    def _bind_synthe_object_visual_material(self, stage, object_path: str) -> None:
        """Bind a conspicuous yellow preview material to the sample object.

        Sample meshes are referenced from converted USDs and may carry their
        original materials.  Binding at the object instance root gives the
        inspection cameras a stable, uniform color without touching any
        physics material or mesh/collision attributes.  The helper is
        deliberately best-effort: a malformed/custom USD should still be
        usable for a physics rollout.
        """

        if not bool(getattr(self.cfg, "object_visual_color_enabled", True)):
            return
        raw_color = getattr(self.cfg, "object_visual_color", (1.0, 0.85, 0.0))
        try:
            color = tuple(max(0.0, min(1.0, float(value))) for value in raw_color[:3])
            if len(color) != 3:
                raise ValueError("object_visual_color must have three channels")
        except Exception as exc:
            print(f"[WARN] invalid object_visual_color={raw_color!r}; keeping source material: {exc}")
            return
        material_path = "/World/Looks/SyntheObjectVisualYellow"
        try:
            material = UsdShade.Material.Define(stage, material_path)
            shader_path = f"{material_path}/Shader"
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
            diffuse = shader.GetInput("diffuseColor")
            if not diffuse:
                diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            diffuse.Set(Gf.Vec3f(*color))
            roughness_value = max(0.0, min(1.0, float(getattr(self.cfg, "object_visual_roughness", 0.65))))
            roughness = shader.GetInput("roughness")
            if not roughness:
                roughness = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
            roughness.Set(roughness_value)
            metallic = shader.GetInput("metallic")
            if not metallic:
                metallic = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
            metallic.Set(0.0)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            bind_visual_material(object_path, material_path)
            print(
                f"[SYNThe] bound object preview color rgb={tuple(round(v, 3) for v in color)} "
                f"to {object_path}",
                flush=True,
            )
        except Exception as exc:
            print(f"[WARN] failed to bind object preview material at {object_path}: {exc}")

    @staticmethod
    def _resolve_sample_path(cfg: FingerEyeSyntheLabEnvCfg) -> Path:
        sample = str(getattr(cfg, "sample_pkl", "") or os.environ.get("FINGEREYE_SYNTHE_SAMPLE_PKL", "")).strip()
        if not sample:
            raise ValueError(
                "FingerEyeSyntheLabEnv requires cfg.sample_pkl or FINGEREYE_SYNTHE_SAMPLE_PKL "
                "pointing to a grasp_sample_*.pkl file."
            )
        path = Path(sample).expanduser()
        if path.is_absolute():
            return path
        cwd_path = path.resolve()
        if cwd_path.exists():
            return cwd_path
        return (Path(DIR_PATH) / path).resolve()

    @staticmethod
    def _load_sample(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Synthetic grasp sample not found: {path}")
        with path.open("rb") as f:
            sample = pickle.load(f)
        if "batch_samples" in sample:
            batch_samples = list(sample["batch_samples"])
            for idx, item in enumerate(batch_samples):
                missing = [key for key in ("current_all_joint_values", "target_action", "object_pose_env") if key not in item]
                if missing:
                    raise KeyError(f"{path} batch_samples[{idx}] is missing required fields: {missing}")
            return sample
        required = ("current_all_joint_values", "target_action", "object_pose_env")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"{path} is missing required fields: {missing}")
        return sample

    @staticmethod
    def _resolve_mesh_path(mesh_path_str: str, sample_path: Path) -> Path:
        mesh_path = Path(mesh_path_str).expanduser()
        if mesh_path.is_absolute():
            return mesh_path
        candidates = [
            Path.cwd() / mesh_path,
            sample_path.parent / mesh_path,
            Path(DIR_PATH) / mesh_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    @staticmethod
    def _synthe_object_rigid_props(
        cfg: FingerEyeSyntheLabEnvCfg, sample: dict | None = None
    ) -> sim_utils.RigidBodyPropertiesCfg:
        sample = sample or {}
        return sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=float(
                sample.get(
                    "object_max_depenetration_velocity",
                    getattr(cfg, "synthe_object_max_depenetration_velocity", 0.05),
                )
            ),
        )

    @staticmethod
    def _synthe_object_collision_props(
        cfg: FingerEyeSyntheLabEnvCfg, sample: dict | None = None
    ) -> sim_utils.CollisionPropertiesCfg:
        sample = sample or {}
        return sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=float(sample.get("object_contact_offset", getattr(cfg, "synthe_object_contact_offset", 0.0002))),
            rest_offset=float(sample.get("object_rest_offset", getattr(cfg, "synthe_object_rest_offset", 0.0))),
        )

    @staticmethod
    def _synthe_mesh_collision_props(cfg: FingerEyeSyntheLabEnvCfg, sample: dict | None = None):
        sample = sample or {}
        approximation = str(
            sample.get(
                "object_collision_approximation",
                getattr(cfg, "synthe_mesh_collision_approximation", "convexHull"),
            )
            or "convexHull"
        )
        approximation_key = approximation.replace("_", "").replace("-", "").lower()
        if approximation_key in ("convexhull", "hull"):
            return sim_utils.ConvexHullPropertiesCfg()
        if approximation_key in ("convexdecomposition", "convexdecomp", "decomposition"):
            return sim_utils.ConvexDecompositionPropertiesCfg()
        if approximation_key in ("boundingcube", "box", "cube"):
            return sim_utils.BoundingCubePropertiesCfg()
        if approximation_key in ("boundingsphere", "sphere"):
            return sim_utils.BoundingSpherePropertiesCfg()
        if approximation_key in ("sdf", "sdfmesh"):
            return sim_utils.SDFMeshPropertiesCfg(
                sdf_resolution=int(getattr(cfg, "synthe_mesh_sdf_resolution", 128)),
            )
        if approximation_key in ("trianglemesh", "trimesh", "mesh"):
            return sim_utils.TriangleMeshPropertiesCfg()
        raise ValueError(f"Unsupported synthe_mesh_collision_approximation={approximation!r}")

    def _configure_object_cfg_from_sample(
        self, cfg: FingerEyeSyntheLabEnvCfg, sample: dict, sample_path: Path
    ) -> None:
        object_scale = np.asarray(sample.get("object_scale", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)
        if object_scale.size == 1:
            object_scale = np.repeat(object_scale, 3)
        cfg.object_cfg.spawn.scale = tuple(float(value) for value in object_scale[:3])

        object_mass = sample.get("object_mass")
        if object_mass is not None:
            cfg.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=float(object_mass))

        object_friction = sample.get("object_friction")
        if object_friction is not None:
            cfg.object_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(object_friction),
                dynamic_friction=float(object_friction),
                friction_combine_mode="max",
            )

        object_pose_env = np.asarray(sample["object_pose_env"], dtype=np.float32).reshape(7)
        cfg.object_cfg.init_state.pos = tuple(float(value) for value in object_pose_env[:3])
        cfg.object_cfg.init_state.rot = tuple(float(value) for value in object_pose_env[3:7])

        if not bool(getattr(cfg, "use_sample_mesh", True)):
            return
        mesh_path_str = sample.get("object_collision_mesh_path") or sample.get("object_mesh_path")
        if not mesh_path_str:
            return
        mesh_path = self._resolve_mesh_path(str(mesh_path_str), sample_path)
        if not mesh_path.exists():
            print(f"[WARN] Synthetic object mesh not found, using fallback object: {mesh_path}")
            return
        if sample.get("object_collision_mesh_path"):
            sample = dict(sample)
            sample.setdefault("object_collision_approximation", "convexDecomposition")
        if mesh_path.suffix.lower() in (".usd", ".usda", ".usdc"):
            cfg.object_cfg.spawn.usd_path = str(mesh_path)
            return
        if mesh_path.suffix.lower() not in (".obj", ".stl", ".fbx"):
            print(f"[WARN] Unsupported synthetic object mesh suffix {mesh_path.suffix}; using fallback object.")
            return

        usd_dir_cfg = str(getattr(cfg, "converted_usd_dir", "") or "").strip()
        usd_dir = Path(usd_dir_cfg).expanduser() if usd_dir_cfg else sample_path.parent / "converted_usd"
        if not usd_dir.is_absolute():
            usd_dir = (Path(DIR_PATH) / usd_dir).resolve()
        usd_dir.mkdir(parents=True, exist_ok=True)

        converter_cfg = MeshConverterCfg(
            asset_path=str(mesh_path),
            usd_dir=str(usd_dir),
            force_usd_conversion=bool(getattr(cfg, "force_usd_conversion", False)),
            make_instanceable=False,
            mass_props=cfg.object_cfg.spawn.mass_props,
            rigid_props=self._synthe_object_rigid_props(cfg, sample),
            collision_props=self._synthe_object_collision_props(cfg, sample),
            mesh_collision_props=self._synthe_mesh_collision_props(cfg, sample),
        )
        converter = MeshConverter(converter_cfg)
        cfg.object_cfg.spawn.usd_path = converter.usd_path
        print(f"[INFO] FingerEyeSynthe object USD: {converter.usd_path}")

    def _object_spawn_cfg_for_sample(self, sample: dict, sample_path: Path, env_id: int):
        spawn_cfg = copy.deepcopy(self._synthe_object_spawn_template)
        # Converted/sample USDs may already contain a material prim below /Object.
        # Spawning another preview material at /Object/material can collide in batch mode.
        if hasattr(spawn_cfg, "visual_material"):
            spawn_cfg.visual_material = None
        object_scale = np.asarray(sample.get("object_scale", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)
        if object_scale.size == 1:
            object_scale = np.repeat(object_scale, 3)
        spawn_cfg.scale = tuple(float(value) for value in object_scale[:3])

        object_mass = sample.get("object_mass")
        if object_mass is not None:
            spawn_cfg.mass_props = sim_utils.MassPropertiesCfg(mass=float(object_mass))
        spawn_cfg.rigid_props = self._synthe_object_rigid_props(self.cfg, sample)
        spawn_cfg.collision_props = self._synthe_object_collision_props(self.cfg, sample)

        object_friction = sample.get("object_friction")
        if object_friction is not None:
            spawn_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(object_friction),
                dynamic_friction=float(object_friction),
                friction_combine_mode="max",
            )

        if not bool(getattr(self.cfg, "use_sample_mesh", True)):
            return spawn_cfg
        mesh_path_str = sample.get("object_collision_mesh_path") or sample.get("object_mesh_path")
        if not mesh_path_str:
            return spawn_cfg
        mesh_path = self._resolve_mesh_path(str(mesh_path_str), sample_path)
        if not mesh_path.exists():
            print(f"[WARN] Synthetic object mesh not found, using fallback object: {mesh_path}")
            return spawn_cfg
        if sample.get("object_collision_mesh_path"):
            sample = dict(sample)
            sample.setdefault("object_collision_approximation", "convexDecomposition")
        if mesh_path.suffix.lower() in (".usd", ".usda", ".usdc"):
            spawn_cfg.usd_path = str(mesh_path)
            return spawn_cfg
        if mesh_path.suffix.lower() not in (".obj", ".stl", ".fbx"):
            print(f"[WARN] Unsupported synthetic object mesh suffix {mesh_path.suffix}; using fallback object.")
            return spawn_cfg

        usd_dir_cfg = str(getattr(self.cfg, "converted_usd_dir", "") or "").strip()
        usd_dir = Path(usd_dir_cfg).expanduser() if usd_dir_cfg else sample_path.parent / "converted_usd"
        if not usd_dir.is_absolute():
            usd_dir = (Path(DIR_PATH) / usd_dir).resolve()
        usd_dir = usd_dir / "shared_meshes"
        usd_dir.mkdir(parents=True, exist_ok=True)
        converter_cfg = MeshConverterCfg(
            asset_path=str(mesh_path),
            usd_dir=str(usd_dir),
            force_usd_conversion=bool(getattr(self.cfg, "force_usd_conversion", False)),
            make_instanceable=False,
            mass_props=spawn_cfg.mass_props,
            rigid_props=self._synthe_object_rigid_props(self.cfg, sample),
            collision_props=self._synthe_object_collision_props(self.cfg, sample),
            mesh_collision_props=self._synthe_mesh_collision_props(self.cfg, sample),
        )
        # AssetConverterBase keeps one .asset_hash per directory. Using the
        # same directory for two masses can overwrite the USD referenced by
        # the first environment while the second is being prepared.
        conversion_identity = converter_cfg.to_dict()
        conversion_identity.pop("usd_dir", None)
        cache_key = hashlib.sha256(
            json.dumps(conversion_identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if cache_key in self._synthe_usd_cache:
            spawn_cfg.usd_path = self._synthe_usd_cache[cache_key]
            return spawn_cfg
        converter_cfg.usd_dir = str(usd_dir / cache_key[:20])
        converter = MeshConverter(converter_cfg)
        spawn_cfg.usd_path = converter.usd_path
        self._synthe_usd_cache[cache_key] = converter.usd_path
        return spawn_cfg

    @staticmethod
    def _spawn_object_reference(stage, object_path: str, spawn_cfg, object_pose_env: np.ndarray) -> None:
        if stage.GetPrimAtPath(object_path).IsValid():
            stage.RemovePrim(object_path)
        object_prim = UsdGeom.Xform.Define(stage, object_path).GetPrim()
        object_prim.GetReferences().AddReference(str(spawn_cfg.usd_path))

        pose = np.asarray(object_pose_env, dtype=np.float64).reshape(7)
        scale = tuple(float(value) for value in getattr(spawn_cfg, "scale", (1.0, 1.0, 1.0)))
        xformable = UsdGeom.Xformable(object_prim)
        ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
        translate_op = ops.get("xformOp:translate")
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        orient_op = ops.get("xformOp:orient")
        if orient_op is None:
            orient_op = xformable.AddOrientOp()
        scale_op = ops.get("xformOp:scale")
        if scale_op is None:
            scale_op = xformable.AddScaleOp()

        translate_op.Set(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
        orient_op.Set(Gf.Quatd(float(pose[3]), Gf.Vec3d(float(pose[4]), float(pose[5]), float(pose[6]))))
        scale_op.Set(Gf.Vec3d(*scale))
        # Batch objects are referenced individually to permit different meshes.
        # Apply the same USD overrides as Isaac Lab's spawn_from_usd, which is
        # used by the single-environment path. A reference alone omits them.
        prims = list(Usd.PrimRange(object_prim))
        for cfg, api, modifier, description in (
            (spawn_cfg.rigid_props, UsdPhysics.RigidBodyAPI, schemas.modify_rigid_body_properties, "rigid body"),
            (spawn_cfg.collision_props, UsdPhysics.CollisionAPI, schemas.modify_collision_properties, "collision"),
            (spawn_cfg.mass_props, UsdPhysics.MassAPI, schemas.modify_mass_properties, "mass"),
        ):
            if cfg is None:
                continue
            if not any(prim.HasAPI(api) for prim in prims):
                raise RuntimeError(f"No {description} API found below {object_path}")
            # apply_nested is a side-effect-only decorator; it returns None
            # even when the USD properties were applied successfully.
            modifier(object_path, cfg, stage=stage)

    def _setup_scene(self):
        if not bool(getattr(self, "_synthe_batch_mode", False)):
            super()._setup_scene()
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            self._bind_synthe_robot_physics_material(stage)
            self._bind_synthe_object_visual_material(stage, "/World/envs/env_0/Object")
            return

        self.hand = Articulation(self.cfg.robot_cfg)
        support_cfg = copy.deepcopy(getattr(self.cfg, "support_cfg", None))
        support_spawn_cfg = copy.deepcopy(support_cfg.spawn) if support_cfg is not None else None
        self.support = None
        self.object = None
        independent_ground = bool(getattr(self.cfg, "async_batch_ground", False))
        if not independent_ground:
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            self.setup_gray_ground()
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if independent_ground:
            # The collision filter below includes the entire env subtree in
            # its own group.  A global plane would keep supporting objects
            # after only one environment entered table-free validation.
            for env_id in range(self.num_envs):
                ground_path = f"/World/envs/env_{env_id}/AsyncGround"
                if not stage.GetPrimAtPath(ground_path).IsValid():
                    spawn_ground_plane(prim_path=ground_path, cfg=GroundPlaneCfg())
        self._bind_synthe_robot_physics_material(stage)
        if support_cfg is not None:
            # Shape spawners are decorated with Isaac Lab's clone helper: one
            # source path under env_0 is expanded to all cloned environments.
            support_spawn_cfg.func(
                "/World/envs/env_.*/XHandSupport" if not self.cfg.scene.replicate_physics else "/World/envs/env_0/XHandSupport",
                support_spawn_cfg,
                translation=tuple(float(value) for value in support_cfg.init_state.pos),
                orientation=tuple(float(value) for value in support_cfg.init_state.rot),
            )
            support_cfg.spawn = None
            self.support = RigidObject(support_cfg)
        if getattr(self.cfg, "object_cfg", None) is not None:
            for env_id in range(self.num_envs):
                sample = self._synthe_samples[min(env_id, len(self._synthe_samples) - 1)]
                object_pose_env = np.asarray(sample["object_pose_env"], dtype=np.float32).reshape(7)
                spawn_cfg = self._object_spawn_cfg_for_sample(sample, self._synthe_sample_path, env_id)
                object_path = f"/World/envs/env_{env_id}/Object"
                self._spawn_object_reference(stage, object_path, spawn_cfg, object_pose_env)
                self._bind_synthe_object_visual_material(stage, object_path)
            object_cfg = copy.deepcopy(self.cfg.object_cfg)
            object_cfg.spawn = None
            self.object = RigidObject(object_cfg)

        if bool(getattr(self.cfg, "production_origin_preserving_batch", False)):
            self.scene.filter_collisions(
                global_prim_paths=[] if independent_ground else ["/World/ground"]
            )
            physics_scene = stage.GetPrimAtPath(self.scene.physics_scene_path)
            if physics_scene.GetAttribute("physxScene:invertCollisionGroupFilter").Get() is not True:
                raise RuntimeError("Production collision group filter is not inverted")
            groups = []
            for env_id in range(self.num_envs):
                group_path = f"/World/collisions/group{env_id}"
                group = stage.GetPrimAtPath(group_path)
                includes = [str(path) for path in group.GetRelationship("collection:colliders:includes").GetTargets()]
                allowed = [str(path) for path in group.GetRelationship("physics:filteredGroups").GetTargets()]
                expected_allowed = {group_path}
                if not independent_ground:
                    expected_allowed.add("/World/collisions/global_group")
                if includes != [f"/World/envs/env_{env_id}"] or set(allowed) != expected_allowed:
                    raise RuntimeError(
                        f"Invalid production collision group for environment {env_id}: "
                        f"includes={includes}, allowed={allowed}, expected={sorted(expected_allowed)}"
                    )
                groups.append({
                    "environment": env_id,
                    "includes": includes,
                    "allowed_collision_groups": allowed,
                })
            ground = stage.GetPrimAtPath("/World/collisions/global_group")
            ground_includes = []
            if ground.IsValid():
                ground_includes = [
                    str(path) for path in ground.GetRelationship("collection:colliders:includes").GetTargets()
                ]
            if independent_ground:
                if "/World/ground" in ground_includes:
                    raise RuntimeError("Async batch still has a shared collision ground")
                for env_id in range(self.num_envs):
                    if not stage.GetPrimAtPath(f"/World/envs/env_{env_id}/AsyncGround").IsValid():
                        raise RuntimeError(f"Async ground missing for environment {env_id}")
            elif "/World/ground" not in ground_includes:
                raise RuntimeError("Ground is missing from the production collision group")
            self.production_collision_isolation_audit = {
                "verified": True,
                "inverted_group_filter": True,
                "groups": groups,
            }

        # Batch synthetic setup bypasses FingerEyeLabEnv._setup_scene(), so
        # explicitly install the optional contact sensors here as well.
        self._setup_contact_audit_sensors()

        self.scene.articulations["robot"] = self.hand
        if self.support is not None:
            self.scene.rigid_objects["support"] = self.support
        if self.object is not None:
            self.scene.rigid_objects["object"] = self.object
        for env_id in range(self.num_envs):
            ensure_floor_uv(stage, f"/World/envs/env_{env_id}/Background/floor")

        if self.rand_cfg.enable_all or self.rand_cfg.random_lighting:
            self._setup_env_lights()
        else:
            light_cfg = sim_utils.DomeLightCfg(
                intensity=2000.0,
                color=(0.75, 0.75, 0.75),
                visible_in_primary_ray=self.rand_cfg.visible_in_primary_ray,
            )
            light_cfg.func("/World/Light", light_cfg)
        if self.rand_cfg.enable_all or self.rand_cfg.random_background:
            self._setup_env_backgrounds()
            for env_id in range(self.num_envs):
                self._setup_env_skybox_and_light_for_env(env_id)

        if self.cfg.enable_cameras:
            self._camera_configs = self._camera_configs_from_cfg()
            self._active_cameras = {}
            self._camera_body_names = {}
            self._camera_body_indices = {}
            self._camera_local_pos = {}
            self._camera_local_rot = {}
            self._camera_runtime_local_pos = {}
            self._camera_runtime_local_rot = {}
            for name in self.cfg.camera_name_list:
                if name in self._camera_configs and self._camera_configs[name] is not None:
                    camera_cfg = self._camera_configs[name]
                    print(f"[INFO] Spawning Camera: {name} ({camera_cfg.width}x{camera_cfg.height})")
                    if camera_cfg.spawn is None:
                        self._clone_camera_with_standard_xform_ops(stage, camera_cfg)
                    sensor = TiledCamera(camera_cfg)
                    self.scene.sensors[name] = sensor
                    self._active_cameras[name] = sensor
                    self._register_camera_body_pose_source(name, camera_cfg, stage)

    def _samples_for_env_ids(self, env_ids: torch.Tensor) -> list[dict]:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).detach().cpu().numpy().astype(int).tolist()
        return [self._synthe_samples[min(env_id, len(self._synthe_samples) - 1)] for env_id in ids]

    def _sample_q_full(self, env_ids: torch.Tensor) -> torch.Tensor:
        q_rows = []
        env_joint_names = [str(name) for name in self.hand.joint_names]
        for sample in self._samples_for_env_ids(env_ids):
            q = np.asarray(sample["current_all_joint_values"], dtype=np.float32).reshape(-1)
            sample_joint_names = [str(name) for name in sample.get("joint_names", [])]
            if sample_joint_names:
                if len(sample_joint_names) != len(q):
                    raise ValueError(f"Sample joint_names dim {len(sample_joint_names)} != q dim {len(q)}")
                by_name = dict(zip(sample_joint_names, q, strict=True))
                missing = [name for name in env_joint_names if name not in by_name]
                if missing:
                    raise ValueError("Sample q is missing env joints: " + ", ".join(missing))
                q = np.asarray([by_name[name] for name in env_joint_names], dtype=np.float32)
            q_rows.append(q)
        q_full = np.stack(q_rows)
        if q_full.shape[1] != self.hand.num_joints:
            raise ValueError(f"Sample q dim {q_full.shape[1]} != env hand joints {self.hand.num_joints}")
        return torch.tensor(q_full, device=self.device, dtype=torch.float32)

    def _sample_target_action(self, env_ids: torch.Tensor) -> torch.Tensor:
        action = np.stack(
            [np.asarray(sample["target_action"], dtype=np.float32).reshape(-1) for sample in self._samples_for_env_ids(env_ids)]
        )
        if action.shape[1] != len(self.control_dof_indices):
            raise ValueError(f"Sample action dim {action.shape[1]} != control dofs {len(self.control_dof_indices)}")
        return torch.tensor(action, device=self.device, dtype=torch.float32)

    def get_sample_locked_targets(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        locked_targets = self._sample_q_full(env_ids)
        locked_targets[:, self.control_dof_indices] = self._sample_target_action(env_ids)
        return locked_targets

    def write_sample_state_to_sim(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        q_full = self._sample_q_full(env_ids)
        q_vel = torch.zeros_like(q_full)
        locked_targets = q_full.clone()
        locked_targets[:, self.control_dof_indices] = self._sample_target_action(env_ids)

        samples = self._samples_for_env_ids(env_ids)
        if hasattr(self.hand, "write_root_pose_to_sim") and any("robot_root_pose_env" in sample for sample in samples):
            root_poses = []
            for sample in samples:
                if "robot_root_pose_env" in sample:
                    root_pose = np.asarray(sample["robot_root_pose_env"], dtype=np.float32).reshape(7)
                else:
                    root_pose = self.hand.data.default_root_state[0, :7].detach().cpu().numpy().astype(np.float32)
                    root_pose[:3] -= self.scene.env_origins[0].detach().cpu().numpy().astype(np.float32)
                root_poses.append(root_pose)
            root_pose_env = torch.tensor(np.stack(root_poses), device=self.device, dtype=torch.float32)
            root_pose_w = root_pose_env.clone()
            root_pose_w[:, :3] += self.scene.env_origins[env_ids]
            self.hand.write_root_pose_to_sim(root_pose_w, env_ids=env_ids)
            if hasattr(self.hand, "write_root_velocity_to_sim"):
                self.hand.write_root_velocity_to_sim(
                    torch.zeros((len(env_ids), 6), device=self.device, dtype=torch.float32),
                    env_ids=env_ids,
                )

        self.hand.write_joint_state_to_sim(q_full, q_vel, env_ids=env_ids)
        self.hand.set_joint_position_target(locked_targets, env_ids=env_ids)

        if self.object is not None:
            pose_env = torch.tensor(
                np.stack(
                    [
                        np.asarray(sample["object_pose_env"], dtype=np.float32).reshape(7)
                        for sample in samples
                    ]
                ),
                device=self.device,
                dtype=torch.float32,
            )
            pose_w = pose_env.clone()
            pose_w[:, :3] += self.scene.env_origins[env_ids]
            self.object.write_root_pose_to_sim(pose_w, env_ids)
            zero_object_velocity = torch.zeros((len(env_ids), 6), device=self.device)
            if hasattr(self.object, "write_root_link_velocity_to_sim"):
                self.object.write_root_link_velocity_to_sim(zero_object_velocity, env_ids)
            else:
                self.object.write_root_velocity_to_sim(zero_object_velocity, env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.object is not None:
            zero_object_velocity = torch.zeros((len(env_ids), 6), device=self.device)
            if hasattr(self.object, "write_root_link_velocity_to_sim"):
                self.object.write_root_link_velocity_to_sim(zero_object_velocity, env_ids)
            else:
                self.object.write_root_velocity_to_sim(zero_object_velocity, env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
        self.scene.update(dt=0.0)
        self._compute_intermediate_values(env_ids)
        return locked_targets

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        if bool(getattr(self.cfg, "reset_to_sample_on_reset", True)):
            self.write_sample_state_to_sim(env_ids)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        if self.object is not None:
            initial_z = torch.tensor(self._synthe_initial_object_z, device=self.device, dtype=torch.float32)
            if initial_z.numel() == 1:
                initial_z = initial_z.repeat(self.object_pos.shape[0])
            z_drop = initial_z[: self.object_pos.shape[0]] - self.object_pos[:, 2]
            success = z_drop <= float(getattr(self.cfg, "synthe_drop_tolerance", 0.003))
            obs["task_success"] = success[:, None].float()
            obs["synthe_z_drop"] = z_drop[:, None]
        return obs

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        initial_z = torch.tensor(self._synthe_initial_object_z, device=self.device, dtype=torch.float32)
        if initial_z.numel() == 1:
            initial_z = initial_z.repeat(self.object_pos.shape[0])
        z_drop = initial_z[: self.object_pos.shape[0]] - self.object_pos[:, 2]
        return (z_drop <= float(getattr(self.cfg, "synthe_drop_tolerance", 0.003))).float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out
