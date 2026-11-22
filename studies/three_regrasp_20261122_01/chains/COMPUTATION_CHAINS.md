# 三指拒绝与实际第二抓的可复核计算链

本附录使用已经完成的同初态入口预检，均不计入正式 40 物体 × 2 条件分母。旧版提交为 `7fa3bd37752a38d5b2dc65da6735804c78057a09`；候选预检提交为 `a1bbe71ae86f79ba5f5f818352a75dc3eeb50460`。下列是原始物理运行的视频，不是录像复跑。最终 80 回合另以冻结清单所载提交为准。

## 1. 方块三指：逐指 IK 达标，端点自碰先拒绝

输入是旧版 `baseline_runs/cube_three/episode/agent_trace.jsonl` 中记录的首个三指候选，以及该回合 `public_responses.jsonl` 中真实公开 RGB 和本体初态。审计 [baseline_cube_first_rejection.json](../evidence/audit/baseline_cube_first_rejection.json) 保存输入文件哈希、完整 q、三个表面落点、法向、逐层检查与带符号几何证据。复核脚本 `stage13_rejection_replay.py::main` 只读取这些数据，重新运行等价守卫，**不重新求 IK，不启动物理，不下发动作**。

计算链是 `visual_controller.py::_s12_rgb_contact_regions` 的 RGB 包络表面假设 → `_s12_region_ik` 区域 IK → 固定落点 IK → `_s12_endpoint_gate` 的共同检查 → 仅通过后才进入 `_s12_complete_candidate` / `_s12_path_gate` 的接近、闭合和抬升路径检查。表面假设不等于已观察到的真实接触面；此处没有使用物体真值供在线规划。

| 顺序 | 量 | 记录值 | 门槛 | 判定 |
|---|---|---|---|---|
| 1 | 逐指位置误差：拇指／食指／中指 | 0.04109／0.20909／0.21058 mm | 各自 < 0.8 mm | 通过 |
| 2 | 逐指法向误差 | 0.31245／0.10085／0.16353 rad | 各自 < 0.5 rad | 通过 |
| 3 | 关节限位 | 无越限 | 原限位 | 通过 |
| 4 | 桌面净空，食指相机 component3 | 2.99968 mm | ≥ 0.65 mm | 通过 |
| 5 | `left_hand_index_camera_component3_link` 对 `left_hand_middle_link2` | **−0.10556 mm** | 不允许网格重叠，0 mm | **第一个失败条件** |
| 6 | 中指相机 component1 对 RGB 包络 | −6.02615 mm | 0 mm | 另一个拒绝 |
| 7 | 中指相机 component3 对 RGB 包络 | −3.59163 mm | 0 mm | 另一个拒绝 |
| 8 | 中指 link2 对 RGB 包络 | −4.57084 mm | 0 mm | 另一个拒绝 |

负值是 FCL 三角网格接触穿透证据。端点失败名为 `endpoint_clearance_collision`，所以没有把该候选推进整段接近或抬升路径，也没有物理闭合。不能将“末端误差达标”写为“可执行三指”。离线真几何反事实显示 RGB 包络对 component3 存在保守误挡，但 component1 与 link2 的真物体碰撞、以及机器人自身的食指—中指碰撞仍存在；因此不放宽原碰撞门槛。

候选修改同时在区域 IK 和固定落点 IK 的残差中保留壳体及指间净空项，并搜索腕姿、指间高度差与接触布局。最终仍使用同一套端点、接近、闭合和抬升检查，不能把仅改进优化残差计作执行成功。同初态固定三指的方块和圆柱预检均为 0 次实际闭合。

## 2. 三棱锥：公开失败到真实第二抓

候选运行是 `development_a1bbe71/pyramid_auto3/episode`，控制器 SHA256 为 `a691c3fc3f5307def9b63ec47b2b01e6e940746df81e00a0fba625ef4dd81067`，配置明确为 `fast + auto + max_attempts=3`。该诊断物体为 `abc_781914`，不属于正式 40 物体。单一连续 episode 共 9,072 个 120 Hz 物理步，75.6 s 物理动作时间，入口端到端墙钟 309.30 s；实际 3 次两指闭合，0 次三指，最终失败。

紧凑原始链 [regrasp_execution_chain.json](../evidence/audit/regrasp_execution_chain.json) 保存全部 23 个 `s13_recovery` 事件、原始 trace 行号、3 条实际闭合命令，以及各闭合前一帧至下发后 6 个物理步的目标 q、实测 q、速度和离线物体位姿。它核对了命令确实推进物理，而非仅有规划或日志标签。

| 物理步 | 文件／函数 | 中间证据与动作 |
|---|---|---|
| 1320 → 1326 | `stage5_control.py::s5_close`；`stage13_recovery.py::s5_step` | 首次 `s5_close` 真正下发并推进 6 步后，`grasp_dispatched` 才将实际抓次记为 1。 |
| 2160 | `s13_evaluate` → `s5_decide` | 普通 RGB 的 `grounded_evidence=true`、`carrying_evidence=false`，给出 `observed_failure` / `retry`。在线没有使用私有评测成功值。 |
| 2184 | `s13_preload` → `s13_preload_proposal` | 新曝光有效，但可信配准置信度 0，预载目标已到 4.5 mm 上限；有界预载恢复被考虑但不适用，没有再试抬，也不增加抓次。 |
| 2280、2328 | `s13_release_recover` → `stage3_recover` → `s13_local_probe` → `s13_legal_start` | 已实际部分开指后，用新 RGB 建图；两次局部动作被选中，但实测到合法命令种子的连接在 fraction=0 就与 RGB 包络冲突。原始记录将拒绝事件标为 `branch=planning`；这是局部恢复中的规划守卫拒绝，不能误算为局部动作执行或抓次。 |
| 2664 | 原释放 FSM → `s13_release_recover` | 完成完整释放与短退，实际退让 7.99772 mm ≥ 6 mm；双指投影开度约 5.86／5.80 mm，新图与静止门槛通过，`release_resample` 记执行 1 次。 |
| 2664 | `s13_legal_start` | 原实测 q 保留，索引 16 为 −2.52844e−10 rad；独立命令种子设为 1e−7 rad，最大投影 1.00253e−7 rad，小于 1e−5 拒绝阈值。原实测至种子的 0、0.25、0.5、0.75、1 插值均过几何检查，继续规划。 |
| 2664 → 4776 | `stage5_control.py::s5_plan` → `s11_contact_candidates` → 共同候选守卫与评分 | 重新采样 `previous_orthogonal`，auto 同时考虑两指和三指，只有通过共同可行性检查的候选进入相同负载评分；三指拒绝，选择可执行两指。新命令计划 SHA256 为 `09068a6c4e00590966b8c1b03e65f30e5e522d05e27e9b6ece24e3a23f997453`。 |
| **4776 → 4782** | `s5_close` → 包装后的 `s5_step` | **第二次闭合实际下发**，返回后的实测 q 与物理记录俱在，抓次从 1 变 2。目标腕部前三轴为 `[0.0216946588, −0.0513402903, 0.1842578515]` m，腕姿后三轴为 `[−2.4600514901, 0.9753326161, 2.3349873304]` rad。 |
| 5616 → 6120 | 同一失败与释放流程 | 第二抓仍公开确认失败；预载仍不适用，完整释放重采样再执行一次。 |
| 8232 → 8238 | `s5_close` → `s5_step` | 第三次闭合实际推进物理，抓次变 3，仍两指。 |
| 9072 | `s13_evaluate` → `s5_decide` | `physical_attempt_limit`，预算结束，未救回，私有离线 `lift30_no_drop_v1` 也失败。 |

相同预闭合阶段，以实际 q 的 FK 指垫落点和离线物体位姿计算：1→2 和 2→3 均为 **2→2**；相对物体腕姿分别改变 1.56953、1.57007 rad（约 89.93°、89.96°），共同手指平均落点分别移动 31.317、31.665 mm，均超过预定 30° 或 5 mm 的大幅手势门槛。它证明同一 episode 里的真实重抓和大幅手势变化，**没有证明换指组，也没有证明救回**。

## 3. 核对与复跑

从实验目录运行下列只读重放可重新得到三指拒绝见证（输出放新文件，避免覆盖原审计）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python stage13_rejection_replay.py \
 --episode ../../outputs/three_regrasp_20261122_01/baseline_runs/cube_three/episode \
 --output /tmp/cube_three_rejection_reproduced.json
```

物理重放应在记录的相应提交和独立新 session 中使用原始 `launch.json` 所载命令；三棱锥命令等价为：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python run.py \
 --session /tmp/fingereye_pyramid_a1bbe71_rerun \
 --case-file stage12_cases/pyramid_diagnostic.json \
 --mode full --profile fast --fingers auto --max-attempts 3
```

物理复跑属于新运行，必须重新核对抓次、指组与结局，不能把原运行的统计直接贴到复跑视频，也不增加正式分母。[development_video_verification.json](../evidence/audit/development_video_verification.json) 对 4 个预检案例的外部与 FingerEye 两路共 8 个现有文件保存 SHA256、30 fps 元信息及全片解码验证；最终交付应选 4–6 段代表片，并补齐同物体干净／加噪的冻结版本对照。

## 4. 实际选指与实际承载接触的边界

[development_contact_diagnostic.json](../evidence/audit/development_contact_diagnostic.json) 另从私有 `all_hand_object_contacts.jsonl` 离线检查实际接触：方块、圆柱的固定两指与 auto 回合都有拇指／食指的 component3 相机接触面正法向力，峰值约 0.294 N、0.320 N，没有中指接触。三棱锥的 1,512 个接触报告样本均为空。因此三棱锥上的三次闭合是确实下发和执行的物理尝试，不能进一步声称三次都建立了实际承载接触。近距离 pair 报告、正法向力与非正间距分别保存在诊断中，避免把接近物体的 link 数误报为承载手指数。这些真值仅用于离线审计，不进入策略选择。
