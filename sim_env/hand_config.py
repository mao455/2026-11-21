from isaaclab.utils import configclass
from .scene_config import FingerEyeSyntheLabEnvCfg
from hand_runtime import selected,configure_cameras
from pathlib import Path
@configclass
class FingerEyeXHandSyntheLabEnvCfg(FingerEyeSyntheLabEnvCfg):
    robot_usd=str(Path(__file__).parent.parent/'xhand_collision.usdc')
    robot_urdf=selected()['urdf_path']
    all_actuated_joint_names=selected()['joint_names']
    control_joint_names=[n for n in all_actuated_joint_names if n not in selected()['base_joint_names']]
    xhand_init_joint_values_radian=selected()['initial_q']
    leap_init_joint_values_radian=[0.]*len(control_joint_names)
    xarm_init_joint_values_degree=[]
    action_space=len(control_joint_names)
    enable_cameras=True
    enable_contact_audit=False
    camera_name_list=['third_view']
    recording_only_camera_names=[]
    xhand_camera_configs={}
    xhand_camera_metadata={}
    contact_audit_links={}
    def configure_xhand_cameras(self,finger=None,*,requested_fps=30.,include_third_view=True,width=160,height=120):
        configure_cameras(self,requested_fps,width or 160,height or 120)
