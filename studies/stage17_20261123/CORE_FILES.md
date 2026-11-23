# 最小维护核心与真实绑定

所有路径相对 `FingerEyeLab/experiments/rax_dexbin_tinyobj/`。当前完整工作树是 `fingereye_stage17`，不是有未提交用户工作的 Stage12 原工作区。Stage16 完整交付 `57b5e0660f9287b073a3a6865615f72d9bc0c7c4` 是本轮基线；未发现更新完整交付。代码实施提交 `1c82388e062`。

|正式维护源文件|关键函数/状态|输入|输出|调用者|下游|
|---|---|---|---|---|---|
|run_stage17.py|main|case、光强、冻结控制器/构建选项|真实配置、SHA、回合|命令行|stage17_build_controller.build、stage16_episode|
|stage17_build_controller.py|build|two、最多3抓、色彩模式|单文件冻结控制器及来源|run_stage17|stage16_build_controller.build|
|stage16_build_controller.py|build|法向/排序显式选项|Stage16 数值覆盖层|stage17 builder|stage15_build_controller、stage16_numerics_overlay|
|stage15_build_controller.py|build|接触组、次数、预载|源码有序拼接，不在磁盘覆盖维护源|stage16 builder|run_stage14.build、shape_fit、surface_pair、stage15_control|
|run_stage14.py|build|profile 和接触组|基础单文件源码|stage15 builder|visual_controller、stage9_frames、stage5_control、stage13_recovery、stage9_control|
|stage9_frames.py|S9FrameInbox.select|RGB包、曝光q、当前q和时间|最近合法同步曝光；过期停止|_S9Context|无外部状态|
|stage9_control.py|main、_S9Context、s9_exposure、s9_visual|策略ctx与RGB包|20Hz分派、曝光q视图、去重结果|worker.main|最终policy_main、S9FrameInbox|
|visual_controller.py|color_foreground、color_reliability、initial_hue|uint8 RGB、固定初始hue|H/S前景、可信/未知像素|初始估计/重建/闭合/试抬/保持|OpenCV；不读物体真值|
|visual_controller.py|_R13_ESTIMATE、_S3_ESTIMATE|初始俯视RGB和静态标定|桌面投影轮廓、颜色、不确定高度先验|s5_init|同一颜色规则、ConvexHull|
|visual_controller.py|Robot、HandDescription、RobotVisibility、_S3_VOLUME.project、_s9_frames|静态URDF/标定、曝光实测q、六路RGB|相机FK、掩码、射线、机器人遮挡|s5_init/fit_shape|color_foreground、机器人网格光线查询|
|stage14_shape_fit.py|fit_shape|初始RGB形状先验、当前多视角RGB和曝光q|当前位姿/几何或拒绝报告|最终stage3_live_geometry、Stage3LiveVolume.carve|可靠域采样、least_squares；未知像素不投自由空间票|
|stage14_build_recovery.py|OVERLAY中的s5_init、stage3_live_geometry、_s14_registered_carve|策略状态/新RGB|保存初始形状、当前配准|builder拼接后生效|fit_shape|
|stage14_surface_pair.py|_s14_surface_candidates、_s3_candidates_on_axes、s11_contact_candidates|RGB几何、手模型、先前q|接触候选与IK解|s11_contact_candidates（仅实际抓次≥1启用表面两指候选）|_s12_rgb_contact_regions、optimize、端点和整路径检查|
|visual_controller.py|optimize、_s12_region_ik、_s12_endpoint_gate、_s12_path_gate、SelfAwareGeometry、_S6_BASE_CHECK|候选q/接触法向、RGB包络、完整手网格|可行/首个拒绝规则及冲突见证|候选搜索/s5_move/s5_step|FK、FCL、关节/速度/桌面检查|
|stage16_numerics_overlay.py|optimize覆盖、s12_order_candidates覆盖|同组候选、显式消融配置|IK数值解释和排序|最终候选搜索|原法向权重two4/three7、原排序默认|
|stage5_control.py|s5_init、s5_plan、s5_approach、s5_refine、s5_close、s5_lift、s5_decide|RGB/q状态、已选候选|有门槛的目标q、闭合/试抬/保持报告|最终policy_main|在线碰撞/速度门槛、support快照和决策|
|stage13_recovery.py|s13_legal_start、s5_step覆盖、s13_preflight、s13_local_probe、s13_release_recover|实测q、视觉失败证据、剩余实际抓次数|合法起点、局部改点/完整释放重采样、实际闭合计数|stage15_control|原s5_step、重新配准/候选搜索|
|stage15_control.py|policy_main、s13_evaluate、s13_preload、s13_local_probe覆盖|同一在线状态|最终FSM：计划→接近→闭合→3/16.5/30mm试抬→保持→重观测/重抓/停|stage9_control.main|继承s5函数及恢复函数；最多3次真实闭合，恢复预载默认关|
|visual_controller.py|capture_preload_snapshot、stage3_support_snapshot、stage3_support_pair、stage3_support_decision|两时刻RGB、相机FK、实测运动|carried/grounded/unknown及动作建议|闭合/试抬/保持FSM|同一H/S掩码、运动/轮廓解释；不是正式物理评分|
|stage16_backend.py、stage14_geometry.py、round8_convex_audit.py、../rax_tinyobj_round1/geometry.py|finalize_records、Audit.check|离线每tick实测body/object姿态、authored USD网格|全程相交、手/物桌面距离|后端finish（deferred）|精确FCL/包含测试、evaluator；策略看不到|
|evaluator.py|evaluate、evaluate_strict|离线120Hz记录、初始物体高度|lift30_no_drop_v1和旧strict|后端finish、独立readout|全程几何、末2s实际≥28mm/命令≥29.5mm/净空≥5mm；未改评分|

`stage17_evidence/controllers/*.py`、每回合 `submitted.py` 和 `episode/program/controller.py` 是自动生成的冻结产物，不是第二套维护源。基线实际执行SHA为 `c13c0bce278e3fa3e04308446e52b6307804ae3412e2c66c5cacbc2c35eae2f7`；H/S修正版为 `66b685a974f5823c5f00618ed9c258f1a8911a4c46a14efdea046d75f8032e7c`。初版错误裁剪规则的冻结SHA为 `7e6865a8c6db06d46b0f501ad406a13d4fe25c26c8a51d36fbcbecee76e8d638`，仅作失败证据。

拼接时同名定义的最后绑定至关重要：`stage9_control.main` 在调用时解析最终 `stage15_control.policy_main`；`stage14_build_recovery.OVERLAY` 替换 `stage3_live_geometry` 和 `Stage3LiveVolume.carve` 为 `fit_shape`，正式主线不是早期 visual-hull carve。`stage15_control.s5_lift` 保留 `_S15_LIFT`，`stage13_recovery.s5_step` 保留 `_S13_ORIGINAL_STEP`，这些捕获别名仍有用途。非 fast profile 和恢复关的 legacy fallback 仍调用 `_S13_LEGACY_POLICY`，故不能直接删掉早期FSM。历史冻结程序不重写。

本轮删除了已经被子类无条件覆盖的 `_geometry_Stage3LiveVolume.frames/carve`，让 `_R13_CHECKER.check` 委托到唯一完整网格实现 `_S6_BASE_CHECK`，移除 `stage9_control` 对已经执行曝光配对的 frames 的第二层包装。120个固定q及两种object_check裁决完全相同；704条A、1150条B回放命令逐数相等。几何计算本来已有精确缓存，重复入口合并没有可测的几何提速（约0.102s/60输入），不报虚构收益。

真实调用轨迹取本轮 `lighting/A_L1_legacy/episode`：初始俯视→s5_init→_S3_ESTIMATE；曝光包由_S9Context配对；s5_plan→surface_pair候选→Stage16 optimize→端点/路径门槛→s5_approach；接近阶段fit_shape使用新RGB；s5_close经s13包装记录真实闭合；proof/middle/final分阶段试抬；support规则遇unknown可重观测并额外2mm激励；末保持由ctx继续推进物理；finish才调用离线正式评分。该回合704个20Hz命令、4224个120Hz状态、一次闭合、正式成功，形成上述绑定的正常入口实证。H/S修正版同样从run_stage17自动生成控制器并记录相同SHA。

环境启动、资产定位、相机驱动、批处理和录像不属于策略核心，见 REPRODUCE.md。
