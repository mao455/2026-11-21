import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .base_config import DIR_PATH, FingerEyeLabEnvCfg, FingerEyeRandomizationCfg


@configclass
class FingerEyeSyntheLabEnvCfg(FingerEyeLabEnvCfg):
    """Synthetic final-grasp validation task.

    The env is configured from a grasp_sample_*.pkl file at construction time.
    It loads the sample object mesh, object scale/pose, final hand pose, and
    target action into IsaacSim without using coin-standing task semantics.
    """

    task_name = "synthetic_final_grasp"
    sample_pkl = ""
    use_sample_mesh = True
    force_usd_conversion = True
    converted_usd_dir = ""
    reset_to_sample_on_reset = True
    synthe_drop_tolerance = 0.003
    synthe_mesh_sdf_resolution = 128
    synthe_mesh_collision_approximation = "convexHull"
    synthe_object_contact_offset = 0.0002
    synthe_object_rest_offset = 0.0
    synthe_object_max_depenetration_velocity = 0.05

    # Visualization-only override for replay/video inspection.  It is bound
    # as a UsdPreviewSurface after the sample mesh is referenced, so it does
    # not alter the mesh, collision approximation, mass, friction, or contact
    # geometry used by the physics engine.
    object_visual_color = (1.0, 0.85, 0.0)
    object_visual_color_enabled = True
    object_visual_roughness = 0.65

    synthe_robot_static_friction = 8.0
    synthe_robot_dynamic_friction = 8.0
    synthe_object_static_friction = 8.0
    synthe_object_dynamic_friction = 8.0
    synthe_leap_stiffness = 1000.0
    synthe_leap_damping = 120.0
    enable_contact_audit = False

    randomization: FingerEyeRandomizationCfg = FingerEyeRandomizationCfg(
        enable_all=False,
        random_lighting=False,
        random_object_color=False,
        random_background=False,
        random_camera_noise=False,
    )

    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{DIR_PATH}/assets/objects/coin/coin_isaac_sim/xarm_leap_2.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.05),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.82, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=(1.0, 0.0, 0.0, 0.0)),
    )

    def __post_init__(self):
        super().__post_init__()

        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.synthe_robot_static_friction),
            dynamic_friction=float(self.synthe_robot_dynamic_friction),
            friction_combine_mode="max",
        )

        self.robot_cfg = copy.deepcopy(self.robot_cfg)
        self.robot_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.synthe_robot_static_friction),
            dynamic_friction=float(self.synthe_robot_dynamic_friction),
            friction_combine_mode="max",
        )
        robot_body_actuator = self.robot_cfg.actuators["robot_body"]
        robot_body_actuator.stiffness = dict(robot_body_actuator.stiffness)
        robot_body_actuator.stiffness["leap_.*"] = float(self.synthe_leap_stiffness)
        robot_body_actuator.damping = {
            "joint.*": 80.0,
            "leap_.*": float(self.synthe_leap_damping),
        }

        self.object_cfg = copy.deepcopy(self.object_cfg)
        self.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=float(self.synthe_object_max_depenetration_velocity),
        )
        self.object_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=float(self.synthe_object_contact_offset),
            rest_offset=float(self.synthe_object_rest_offset),
        )
        self.object_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.synthe_object_static_friction),
            dynamic_friction=float(self.synthe_object_dynamic_friction),
            friction_combine_mode="max",
        )
