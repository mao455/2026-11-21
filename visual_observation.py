"""Public RGB/encoder allowlist; runtime dimensions come from the selected URDF."""
CAMERAS = {f'{f}_{m}' for f in ('index','thumb') for m in ('front1','front2','back')}

OBS_KEYS = {'time_s','step','actual_q','actual_qvel','initial_overhead','fingereye'}

ROBOT_KEYS = {'urdf_path','joint_names','joint_limits','dt','table_z','action',
              'observation_profile','fingereye_calibration'}

FRAME_KEYS = {'rgb_path','width','height','capture_step','frame_id'}

TOP_KEYS = {'rgb_path','intrinsics','camera_pose_world','width','height','capture_step'}

CALIBRATION_KEYS = set('logical_name finger_module mount_name joint_name parent_link camera_link camera_prim_path_pattern width height nominal_fps hfov_deg vfov_deg dfov_deg fx fy cx cy skew intrinsic_matrix pixel_center_convention renderer_intrinsic_matrix renderer_intrinsic_limitation renderer_hfov_deg renderer_vfov_deg distortion_model distortion_coefficients calibration_status translation rpy extrinsic_source extrinsic_urdf_path optical_axis_convention image_transform enabled color global_shutter_supplier_spec hardware_output_format_supplier_spec dynamic_range_db_supplier_spec focus_mode_supplier_spec nominal_focus_distance_m_supplier_spec depth_of_field_m_supplier_spec synchronization_supplier_spec usb_interface_supplier_spec projection_type fisheye_nominal_width fisheye_nominal_height fisheye_optical_centre_x fisheye_optical_centre_y fisheye_max_fov fisheye_polynomial camera_geometry_version camera_config_path sensor_update_period_s'.split())

