# 最终冻结版本的三指拒绝与实际局部重抓计算链

本页只描述正式回合`00_abc_53930_clean`，提交`9daeff1043917ac40f518b8f80d1103d0fe63bc1`，控制器SHA256`a01b97ab2e9ddd083ec852978945a7ec77ebcbc484b469aa060231bb781a7c1e`。本回合完整性通过，真实执行3抓、2→2→2，最终失败；局部恢复选择2次、动作执行2次、救回0次。它属于80个正式回合，不是录像复跑，也不借用历史验收结果。

## 三指候选：第一个失败条件

正常入口为`run.py --controller frozen_v4/submitted.py --phase heldout`。`stage5_control.py:s11_contact_candidates`对两指与三指共用落点、逐指位置/法向、限位、端点和路径检查。初次auto调用有两指可行候选；36个三指候选均先在`per_finger_position`失败，因此选择范围为`only_one_group_feasible`。不能把它写成两组均可行后的优劣比较。

代表三指候选`contacts3-e9bc18e7209b451f3920`的逐指位置误差为0.474841, 2.907383, 3.443091 mm；阈值为每指严格小于0.8 mm。拇指通过，食指和中指失败。这就是该候选的第一个失败条件。表面区域界限通过、关节限位通过；法向检查也失败。

该候选的独立端点诊断还发现`left_hand_index_camera_component3_link`与`left_hand_middle_camera_component1_link`精确网格重叠，间隙−2.675811 mm、阈值0。这个端点见证发生在更早IK拒绝的候选上，不是“第一处失败”。该候选未进入完整路径验证、未下发三指闭合。完整落点、q、逐层门控及端点见证见[链条JSON](final_version_recovery_chain.json)。

## 从公开失败判断到真实第二抓

1. 第一次真实闭合起于物理步1548，步1554收到下发确认。`s5_decide`在步3222给出`retry / in_progress_grounded_RGB_veto`，依据普通RGB中的落地与重复抬升图像风险；在线未使用物体位姿或仿真接触。
2. `s13_preload`保持姿态观察24步后，在步3246取得新证据；新鲜标志通过，但视觉置信度0，不能支持双侧接触预载恢复，因此该分支仅被考虑，未选中、未执行。
3. `s13_release_recover → stage3_recover_release`沿共享释放流程降低并开指。释放重采样被登记为有条件后备，随后在步3870调用`s13_local_probe`。新图来自frame967、capture_step3868，由`stage3_live_geometry`构成公开RGB稀疏表面；不是沿用初始包络。
4. `s13_legal_start`保留实测q，另生成限位内命令起点；最大投影1.5180023e−7 rad，并验证实测到命令起点的连接。原始负边界残差及合法q均保存在事件中。
5. `s13_local_fit`固定腕部旋转，在局部平移和活动指关节上优化。该次采用高度−2 mm、横向0；实际几何新颖度2.570651 mm。新的局部移动和预闭合路径分别8个节点，均通过`s13_preflight`。
6. `s13_local_reposition`实际执行48物理步，于步3918记`executed`。随后首条`s5_close`命令确实下发并产生6步物理记录，步3924由`s13_recovery.py:s5_step`确认第2次实际抓取。对应候选`contacts2-05f1b64416800cc1d1c3`，计划SHA256`05f1b64416800cc1d1c31255be76e81238d4f126a8c2302843128b1d02e0b317`。

第2次仍失败，步4896再次`retry / observed_failure`。第二次局部恢复在步5016通过8+8节点检查，采用高度+2 mm，几何新颖度1.422105 mm；步5064完成移动并开始第3次实际闭合，步5070确认。步6042因实际3抓预算用完而`stop_limit / physical_attempt_limit`，没有第4次闭合。

| 事件 | 恢复1 | 恢复2 |
|---|---:|---:|
| 公开失败判断步 | 3222 | 4896 |
| 局部候选选择与检查步 | 3870 | 5016 |
| 局部移动实际完成步 | 3918 | 5064 |
| 下一抓闭合起始步 | 3918 | 5064 |
| 下发确认步 | 3924 | 5070 |
| 新落点高度调整mm | −2 | +2 |
| 实际几何新颖度mm | 2.570651 | 1.422105 |
| 最终救回 | 否 | 否 |

两次局部恢复均未完成完整退让与全局重采样，故`release_resample executed=0`。有条件后备登记2次，该后备实际启用0次；登记不等于实际选择或执行。常规闭合内的`s5_preload_adjust`也不等于`s13_preload_recovery`，后者实际执行0次。

## 下发关节与物理证据

精确命令保存在[链条JSON](final_version_recovery_chain.json)的`actual_closure_commands`：`physical_start_step`3918及5064，`steps=6`，`phase=s5_close`，18维`target_q`原样保留。`raw_physics_snapshots`保存闭合前、首步和首条6步后的实测q/qvel及目标。`acknowledged_local_plans`保存各自的计划q、落点、候选ID和SHA；不能用延续自首次接近的旧`binding`代替局部计划。该数据层限制另见[绑定审计](../evidence/audit/local_binding_provenance_caveat_v4.json)。

![失败观察与两次局部恢复使用的新RGB](../LOCAL_EVIDENCE.md)

各行frame805、811、967、1254，六列为实际FingerEye槽位。原始uint8数组、逐路SHA256和曝光步号保存在`audit/final_chain_rgb/`及链条JSON；这里的图片用于离线核对，不作为成功标签。

## 结果和时间界限

物理评分为失败，`invalid_episode_geometry`首见步1776；末尾2秒最低物体抬升−0.214 mm、命令抬升0.352 mm，均未达主门槛。这里展示的是局部恢复真实执行证据，不是恢复成功。仿真位姿与几何只在离线评分使用。

实际物理时间50.35 s，端到端墙钟174.096842 s。策略侧非RPC墙钟44.234286 s，包含首次规划、进程调度与IPC空隙；命令间已记录子集37.585800 s，不能称为纯CPU耗时。独立分段墙钟严格相加为后端ready→finish的147.096590 s；物理和墙钟残差见[首回执QA](../evidence/audit/first_formal_receipt_qa_v4.json)，14项检查全部通过。

本页仅分析正式队列的第一条回执；完整40对象结论须以全体80回合的正式报告为准。
