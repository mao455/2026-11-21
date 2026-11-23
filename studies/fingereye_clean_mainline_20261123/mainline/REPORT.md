# FingerEyeLab 正式主线清理与发布报告

正式维护入口是 [`strategy.py::main(ctx)`](strategy.py)，由 [`../run_stage17.py`](../run_stage17.py) 直接提交，默认路径与 `--controller mainline/strategy.py` 相同。原交付 `9ce1883ce509757bbc3a2ea072dfe33e378be08b` 的 H/S v2 已实跑控制器是本轮提取起点；不再从多代模块拼接控制器。

本轮实际执行源码提交为 [`cb875498f8142ffb6fc97d1d75c68f61098cc888`](https://github.com/ztr583/grasp_small_obj/commit/cb875498f8142ffb6fc97d1d75c68f61098cc888)，策略 SHA256 为 `20016ea28e2d164bddbf957c5ce49d4a6a6b08785f5d6849aa345e5df81a8edd`。原控制器 SHA256 为 `66b685a974f5823c5f00618ed9c258f1a8911a4c46a14efdea046d75f8032e7c`。正式评分 `evaluator.py` 未改，SHA256 为 `31ddc54031124d2f561a512edb96e0adce2da94c1cae174692e982034a93b22b`。

|验证|实际结果|实际闭合|物理时间|墙钟时间|
|---|---|---:|---:|---:|
|A 首抓，显式 `--controller`|原正式评分成功，完整性通过|1|34.7 s / 4164 步|100.449732 s|
|B 重抓，默认控制器路径|首抓失败后第二抓成功，完整性通过|2|78.3 s / 9396 步|227.217500 s|
|A/B 同记录提取前后|2104 条动作精确相同，10307 个语义事件及 270 条非计时 note 相同|使用既有记录|不作为新物理试验|不声称提速|
|最终交付树检查|103 项测试、13 个子测试通过|—|—|7.48 s|

A/B 最后连续 2 s 的最低实际抬升分别为 31.270942 / 32.346504 mm，几何均通过。旧 strict 评分分别为失败/成功，原正式 `lift30_no_drop_v1` 均成功。在线最终动作均是 `stop_unknown`，策略保留可能已抓住的物体；离线正式成功不反写为在线已确认抓持。完整读数和原件 SHA 见 [physical/readout.json](evidence/physical/readout.json)。

实际从本维护分支移除 **501 个已跟踪文件、16,521,604 字节（16.52 MB，约 15.76 MiB）**。逐文件的路径、SHA、理由和归档提交见 [deletions.json](evidence/deletions.json)，不是只隐藏目录。全部移除项可从 `9ce1883…` 恢复；未改原用户工作树，未删私有资产、基础设施依赖、唯一原始物理/RGB/接触记录。私有发布目标为 `ztr583/grasp_small_obj` 的 `fingereye-clean-mainline`；公开目标为 `mao455/2026-11-21` 的 `main`，选集目录 `studies/fingereye_clean_mainline_20261123`。固定远端提交、读回结果和发布回执列在发布后的 `DELIVERY.md`；公开 manifest 给出对应源提交。

## 入口核对与维护边界

原正常入口 `run_stage17.py` 的 fast 路径会依次经 Stage17/16/15 builder 和 Stage14 builder，将视觉、FSM、恢复、曝光适配、形状配准、表面候选、预载包装及数值诊断拼接到同一 Python 模块。同名后定义覆盖与 `_S13_ORIGINAL_STEP`、`_S14_INIT`、`_S15_LIFT` 等定义时捕获共同决定行为，不能仅保留文本中的最后一个函数。

本轮按真实绑定提取为 3260 行单脚本，函数名不再重复定义：初始化主体为 `initialize_state`，下发主体为 `dispatch_checked_step`，固定目标 IK 为 `solve_target_ik`，视觉响应为 `compare_visual_response`，候选排序主体为 `rank_contact_candidates`。保留必要的曝光、计时与恢复包装；这些包装显式调用同文件中的唯一主体，不覆盖另一份算法源。未启用的旧状态机、三指搜索分支、legacy 亮度分割和数值排序消融已经从正式策略中移除。保留 `s16_ik_terms` 作为原来的数值记录，未改变求解目标或排序。

实际绑定及包装捕获见 [runtime_bindings.json](evidence/runtime_bindings.json)，原始定义来源/提取操作见 [extraction.json](evidence/extraction.json)。正式维护的算法文件只有 `mainline/strategy.py` 和现有共享依赖 `general_grasp_contacts.py`；评分只维护 `evaluator.py`。每回合 `episode/program/controller.py` 是自动保存的字节相同冻结副本，不编辑它；生成的 backend 源只负责现有仿真适配。旧冻结控制器及 builder 已从当前维护分支移除，历史按固定 Git 提交复原。

配置沿用 H/S v2、固定两指、最多三次实际闭合、恢复预载关闭、历史任务裕量排序和原评分。关闭的是“失败后的恢复预载”分支；常规 servo 闭合仍含 `s5_preload_adjust/hold` 相位，不能把这些相位数算成多次实际闭合。实际闭合由首次下发闭合命令的 `grasp_dispatched` 事件计数。`stage9_config.json` 沿用原字节，仅提供运行时频率等基础配置；有效 `S11_CONFIG` 来自实际提交脚本。

## 最小核心文件与完整调用链

下表路径相对本实验目录；所有 `strategy.py` 行均指 `mainline/strategy.py`。输入只含允许的 RGB、实测关节状态和静态标定。环境、数据库、相机和渲染列在末尾依赖说明，不混作策略算法。

|链路 / 文件|关键函数|输入|输出|谁调用它 / 它调用谁|
|---|---|---|---|---|
|RGB 与曝光关节状态：`mainline/strategy.py`|`main`, `_S9Context`, `S9FrameInbox.select`, `s9_exposure`|初始俯视、六路 RGB 路径、frame_id/capture_step、曝光 q/qvel、当前 q/qvel、静态 robot_description|最新合法曝光；视觉用曝光 q，控制用当前 q|`worker.py` 调 `main` → `policy_main`；FSM 的 observe/step 走 `_S9Context` → inbox；视觉走 `s9_exposure`|
|分割：同文件|`color_reliability`, `color_foreground`, `initial_hue`, `color_projection_known`|uint8 RGB、初始色相、投影像素|H/S 前景、可靠域、未知域、固定色相|初始估计、近手体素、`fit_shape`、闭合响应、试抬都调用同一规则；使用 cv2 HSV / nearest domain sampling|
|初始几何：同文件|`initialize_state`, `_S3_ESTIMATE`, `_R13_ESTIMATE`, `Robot.fk`|初始 RGB、静态俯视相机模型、URDF|桌面足迹、轴/中心、明确不确定的高度先验、碰撞包络|`s5_init` → 初始化与初始 shape prior；估计只用 RGB/桌面标定，不读取对象真值|
|首次精化：同文件|`_S3_VOLUME.frames/carve`, `s5_refine`, `RobotVisibility.depth`|近手 RGB、曝光 q、静态鱼眼标定、机器人静态网格|带未知域的多视图体素边界、可见性和有效性报告|`s5_refine` 使用初始化时创建的 **基础 `_S3_VOLUME`**；调用公共分割、投影、FK 和机器人遮挡估计|
|重抓几何配准：同文件|`Stage3LiveVolume.frames/carve`, `stage3_live_geometry`, `fit_shape`, `_s9_frames`|初始 RGB 形状先验、当前六路可靠掩膜与曝光 q|当前配准位姿/高度尺度、各视角解释率、歧义/拒绝报告|恢复与局部探测调用；`carve` 直接使用 `fit_shape`；不会沿用已被覆盖的旧稀疏 hull carve|
|抓点：同文件|`s5_plan`, `s11_contact_candidates`, `axis_contact_candidates`, `_s14_surface_candidates`, `_s12_rgb_contact_regions`|RGB 几何、静态指定指垫、已尝试位置、当前实测合法关节种子|两指候选接触点/法向、候选身份及排除理由|FSM → plan → candidate；首抓用轴极值，实际首抓后可用当前 RGB hull 表面；调用 IK、任务排序和完整门槛|
|IK：同文件|`Robot.tips`, `solve_target_ik`, `optimize`, `solve_close_ik`, `close_ik`, `s13_local_fit`|静态 FK、候选点/法向、当前/先验 q、关节限位|目标 q、每指位置/法向残差、solver 信息|候选/闭合/局部改点调用；least_squares 只产生建议，随后必须通过门槛|
|任务排序：同文件 + `general_grasp_contacts.py`|`s11_contact_quality`, `s12_task_quality`, `s12_order_candidates`; `task_wrench_margin`|候选几何、静态条件质量/摩擦/力边界与 pad 半径|条件任务裕量、运动/净空项、历史排序理由|候选生成后调用共享任务裕量；依赖 `_unit`, `_tangents`, linprog 等；这是条件模型，不是实测接触力|
|候选与路径门槛：`mainline/strategy.py`|`_s12_endpoint_gate`, `_s12_path_gate`, `SelfAwareGeometry.check/diagnose`, `_S6_BASE_CHECK`, `s13_legal_start`, `s13_preflight`|候选/路径 q、RGB 包络、全手静态网格、关节与速率界限|有顺序的裁决、首拒规则、冲突部件和数值见证|候选完整绑定/恢复预检调用；check 的缓存和基础检查共享同一完整碰撞实现；不把私有真实物体几何输入策略|
|下发与实际抓次：同文件|`s5_move`, `s5_step`, `dispatch_checked_step`|目标 q、步数、phase、当前观测和剩余预算|20 Hz 实际命令、新观测、闭合计数/guard note|所有 FSM 动作调用；每次先检查几何、关节、命令速率；闭合另检查实测速率；`_S9Context` 分成 6 个物理步|
|闭合：同文件|`s5_close`, `capture_preload_snapshot`, `compare_visual_response`, `compare_preload`|六路 RGB、曝光 q、实测关节跟踪差和速度、候选接触点|有界 squeeze、视觉/电机响应、闭合后状态|FSM → close → close IK → checked dispatch；电机跟踪差仅是代理，不能独立认定接触力|
|试抬与在线判定：同文件|`s5_lift`, `lift_and_hold`, `early_proof`, `stage3_support_snapshot/pair/decision`, `s5_decide`, `s13_evaluate`|曝光配对 RGB、关节运动、试抬/保持历史、剩余预算|continue / retry / reobserve / stop_unknown / stop_supported 等与理由|FSM 依次 proof 3 mm、middle 16.5 mm、final 30 mm；未知可在原有边界内增加一次 2 mm 激励；不读取真实离桌高度|
|释放与重抓 FSM：同文件|`policy_main`, `s13_preload`, `s13_preload_proposal`, `s13_release_recover`, `release_and_resample`, `stage3_recover/recover_release`, `local_pose_probe`|在线失败证据、实际闭合数、最新 RGB 和关节状态|预载拒绝、释放/局部改点或完整重采样、下一次实际闭合|`policy_main` 管理最多三次实际闭合；恢复预载只记录新证据并以 disabled 理由拒绝；局部失败继续完整释放重采样|
|正式判分：`evaluator.py`|`evaluate`, `evaluate_strict`|**离线私有** 120 Hz 实际轨迹、初始真实高度、完整 authored USD 几何检查|原正式分数、逐项门槛、历史 strict 并列结果|原 host/backend 在结束后评分；`stage15_readout.read` 用原始记录独立重算并核对 host 分数、完整性及控制器 SHA；策略不能调用这些私有输入|

## 同记录与新的实际轨迹

[replay_comparison.json](evidence/replay_comparison.json) 分别记录 A 的 694 条命令/695 次观测/3668 个语义事件/72 条非计时 note，B 的 1410 条命令/1413 次观测/6639 个事件/198 条 note。target_q 的最大误差为 0，steps/phase 全部相同；各返回掩膜、初估、几何/门槛完整返回值与决策 note 的规范化 SHA 完全相同。计时字段（包括剩余墙钟预算数值）和 `s9_io_audit` 计时摘要排除，字段清单明确记录；仍比较对应 action/reason，不声称墙钟预算数值相等。未记录字段、未执行分支和其他物体不在等价结论内；不据这两次回放声称提速。

新的 B 是贯穿全链路的实际轨迹，原件为 `FingerEyeLab/outputs/clean_mainline/recovery_B/episode`，详见 [trace_B.json](evidence/trace_B.json)：

|步 / 物理时间|实际动作与调用关系|
|---|---|
|0 / 0 s|`main` → 曝光 inbox → `policy_main` → `s5_init/initialize_state` → 俯视 H/S 初估与静态 FK → `s5_plan` → 接触候选/IK/任务排序/端点与路径门槛 → 下发 approach；近手 `s5_refine` 调基础体素 carve|
|1530 / 12.75 s|`s5_close` 首次实际下发，`s5_step` 记录 `grasp_dispatched(actual_grasps=1)`；servo 调共享分割/响应与 close IK|
|3450 / 28.75 s|`s13_evaluate` → `s5_decide` 返回 grounded / `retry`；未满足在线抓持证据，进入恢复|
|3450–4242 / 28.75–35.35 s|恢复预载关闭，仍获取新证据并记录拒绝；`s13_release_recover` → `stage3_recover`；4194 步实际到合法种子的连接被几何门槛拒绝，4242 步局部改点又因 fresh RGB map 拒绝，未下发该局部抓取|
|4578 / 38.15 s|完整释放重采样分支执行；`Stage3LiveVolume/fit_shape` 对当前可靠多视图重新配准，再经候选/IK/完整门槛|
|6624 / 55.2 s|第二次实际闭合下发，`actual_grasps=2`，分支 `release_resample`|
|8736 / 72.8 s|final RGB 仍未知，执行既有受保护的额外 2 mm 激励和保持；没有新增一次闭合|
|9396 / 78.3 s|在线最终 `stop_unknown`；结束后原 `evaluate` 对最后连续 2 s 判成功；独立读回同分数、完整性和脚本 SHA|

A 的 694 条新实跑命令与历史 A 全部精确相同，物理时间相同。B 新实跑为 1566 条命令，历史为 1410 条；初始俯视分别有 870 / 1310 个通道相差最多 1 uint8 灰阶，初始几何估计相同。B 首个命令差异在索引 823 的 `s5_orient`，最大 q 差 `2.141256238985534e-08`；后续增加了闭合调整/保持、再观察和最终激励。B 总计多 936 物理步（7.8 s）：lift +300、hold +360、reobserve +240、preload_adjust +24、preload_hold +12。两次均成功、两次实际闭合；新渲染/新物理输入并不完全相同，不能把这些差异归因为提取改写。证据见 [fresh_comparison.json](evidence/fresh_comparison.json)。

## 门槛、20_safe 与已有难例

端点依次检查公共接触区、每指位置误差 `<0.8 mm`、法向误差 `<0.5 rad`、关节界限、全手净空/碰撞和近物体姿态变化（精化 `≤0.12 rad`）。RGB 包络净空门槛为 0.65 mm，真实三角网格重叠不放行；路径保留原 close/probe 的对象重叠豁免，关节、自碰、桌面和速率检查仍在。路径候选速率门槛为 `<0.15 m/s`、`<1.5 rad/s`，实际下发更保守地用 `≤0.14`、`≤1.4`；这些检查对象和阈值不同，未强行合并。本轮未更改门槛或评分。

原正式评分要求：120 Hz 全部记录连续；全回合 authored USD 几何有效；末连续 240 步/2 s 最低实际抬升 `≥28 mm`、桌面净空 `≥5 mm`、完成命令抬升 `≥29.5 mm`。传统 strict 分数单独并列，不替换正式门槛。像素/包络的不确定性、仿真原生接触和正式几何是三类证据，不互相冒充。

**20_safe 记为三指接触抓取成功。** 这是已有三指诊断回合，原正式和 strict 评分都成功：3816 步、31.8 s、1 次实际闭合，末窗最低抬升 30.046345 mm、净空 29.764688 mm。第三处正载荷位于**中指靠指根的掌侧 component1 相机壳面**，不是 component3 指定垫或指背；掌侧/指侧外壳接触不作为失败。[实际接触点三视图](../stage17_evidence/figures/middle_three_views.png) 已亲自打开核看；图使用记录姿态和完整 URDF 网格，红点为实际原生接触，指背图中的红点是遮蔽侧投影，不能误读为指背受力。

component1 固定在 middle_link2，包覆掌侧并连接侧缘/背侧；front1/front2 镜头朝掌侧，back 镜头在背侧。component3 指定区在掌侧约 x=11.75 mm、z=38 mm。20_safe 的记录 step3576 / 29.808333 s 接触点在 middle_link2 局部 `(14.393430, −5.259074, 15.857269) mm`，世界 `(17.933778, 3.416512, 74.717432) mm`；壳面法向约 `(0.984949, −0.172757, 0.005433)`，指向掌侧 +x、略偏侧面 −y。URDF SHA 为 `c4f906a3ca73426367de4015eb42098c80ed5a7abe0d6aea7f13807c3920cdb8`；USD/URDF authored 四部件顶点核对相等不代表 PhysX cooked 凸块相等。

末 2 s 的 40 个 20 Hz 样本：thumb/index 指定垫各 40/40 正载荷，中指 component1 壳 40/40、约 2.466–2.588 mN；中指 component3 指定垫 0/40。**三指接触参与：是；三指定指垫承载：未通过；此例指背接触：无证据。** 40 帧不是 40 次独立实验；未以壳体力大小推断第三指承担多少竖直重量或是否不可替代。实际点和力见 [contact_capture](../stage16_evidence/closeout/contact_capture/actual_contacts_summary.json) 与 [hold_native_contacts.csv](../stage16_evidence/closeout/contact_capture/hold_native_contacts.csv)。

**D 记为已有正式失败，不在本轮放宽规则。** D0/D20.4 首个坏状态为 step1719 / 14.333333 s、`s5_preload_adjust`，评分 authored USD 网格与内缩 0.5 mm 对象的相交发生在 index component3；同刻 PhysX native 分离 +43.558 µm、力 0 N。手最低 7.456462 mm、对象最低 −0.282629 mm，未先触犯桌面 −0.5 mm 界限；首个卡点是 authored 评分几何与 PhysX 碰撞形状给出不同结果。最终冻结 D 首坏 step1737 / 14.483333 s、native 分离 +85.673 µm、力 0 N，且未抬起。15/15 同刻原始未加速网格复算一致，见 [geometry_recompute.csv](../stage16_evidence/closeout/gates_recovery/geometry_recompute.csv) 与 [first_gates.csv](../stage16_evidence/closeout/gates_recovery/first_gates.csv)。这不证明在线视觉误拒，也不证明可安全删掉碰撞规则。cooked 形状差异与真实无遮挡侧视证据仍为后续研究阻塞；本轮主线交付不依赖解决它。

20/34 正常三指未下发候选及 B/C 的历史拒绝、连续帧、姿态证据仍按固定 Stage17 公共归档和本地唯一结构化原件追溯。没有新增证据支持删除壳体碰撞检查。录像既有外部视角存在掌背遮挡，不能独立证明毫米净空/全程无滑落；几何和受力结论取同刻记录，不补画假侧视。

## 实际清理与可追溯位置

|实际移除范围|原因 / 保留位置|
|---|---|
|Stage14–17 builder、重复 visual/FSM/恢复/数值 overlay 源，以及旧 run_stage14/15/16 等策略入口|正式入口已经直接使用提取脚本；新入口无这些导入；历史完整复跑使用 `9ce1883…` 工作树|
|Stage17 的六种控制器副本、Stage14–16 evidence 中的冻结 Python 控制器与历史 audit 脚本、`versions/`|同一逻辑不继续维护多份；冻结版本用固定提交恢复，运行生成物只在各回合保存|
|过期噪声/三指搜索/数值调参消融脚本及其专用测试；旧 Stage6–13 用例目录和未引用配置|不属于固定 H/S v2 正式算法；原配置和实验记录仍可从固定 Git 恢复|
|`publication/` 中间公开导出、旧发布 allowlist、重复报告、旧 replay 摘要和 readout_cache|公开以新主线选集为唯一当前交付；历史公开提交 `1a0c46…` 可追溯；唯一原始物理/RGB/接触记录未删|
|忽略的 Python/pytest 缓存|可再生，不是证据；精确目录记录在删除清单|

先检查保守静态导入/显式源文件引用闭包，再检查源码字符串中的动态导入。`stage9_recording.py` 是生成源码中的导入，已恢复并保留。相机同步原先导入重复 `visual_controller.Robot/NAMES`，在先确认这两个定义 AST 完全相等后，接到 `mainline.strategy`；后续首抓/重抓从该接线实跑。共享环境/渲染/注册表与实际 fingerprint 所需基础设施保留，不把历史编号当作可删依据。原用户工作树的 status/index/暂存与未暂存 diff 哈希在运行冻结和交付核验中比较。

## 复跑、依赖、光照与发布状态

[README.md](README.md) 给出已执行 A/B 命令和同记录回放命令。本轮 A 显式 `--controller`，B 省略该参数；两份 session 配置中的 `controller_source` 都为 `/home/CNS2025331827/docs/fingereye_stage17/FingerEyeLab/experiments/rax_dexbin_tinyobj/mainline/strategy.py`，沙箱 `program/controller.py` SHA 与上列正式源相同。物理运行期间未改变受保护源码；运行后只继续移除已证明未使用的历史用例/配置与重复文档，正式策略、入口、评价器和全部存活 Python 基础设施字节保持执行提交一致，见发布前源核验。

实验室依赖：IsaacLab Python `/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python`；`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1`；资产根 `/home/CNS2025331827/docs/fingereye_stage16/FingerEyeLab`。沿用资产数据库/对象注册表、120 Hz 环境、20 Hz 控制、30 fps RGB、曝光配对、bubblewrap 沙箱、现有资产数据库/文件注册表与日志设施；不迁移或公开资产。运行模式 minimal，未额外编码录像，保留全部 RGB 曝光/120 Hz 轨迹和原生接触。原始大日志留实验室。

H/S v2 既有实际光照证据沿用，不把本轮两次正常光复跑冒充新的光照套件。原 Stage17 同物体/初态/种子对照：灯强 0.4，原版 1/2、H/S v2 2/2；1.0 与 1.6 均为两版 2/2，唯一 F→S 是暗光 B；总计原版 5/6、H/S v2 6/6，正常光 B 原版更快。数字 V 回放与真实改灯是不同试验，详见保留的 [lighting_pairs.csv](../stage17_evidence/lighting_pairs.csv)、历史固定报告和原始记录索引；不声称统计泛化成功。可靠色像素依 H/S：S>65（无色相）/60（有固定色相）、色相环距<12；V>8 且非近白全通道裁剪，其他像素未知而不投自由空间票。初始、精化、重抓和试抬使用同一规则。

发布复用现有 `publish_github.py --only research --preserve-unlisted --config mainline/publish.json` 的显式 allowlist，增加精确限定旧 Stage17 目录的清除能力；本地 bare-repo 测试证明其他 studies 保留。私有仓库直接普通推送完整维护分支，不用公开选集快照替代完整仓库，不 force push。公开只发获准核心源码、运行说明、结构化最小证据和接触标注图；不推私有资产、RGB/录像或大日志。发布后分别从远端新取仓库核对提交、文件 SHA 与已移除路径，回执见 `DELIVERY.md`。

本轮状态：正式入口提取、同记录裁决核对、两个正常入口闭环、主线清理与测试均**实际通过**；20_safe **三指接触抓取成功**，指定三垫承载**未通过**；D 是**已有失败**，几何差异定位停在上述具体阻塞。发布状态以实际远端读回回执为准，不用本地 commit 代替成功发布。下一轮是否研究 D 或其他单例，留待本主线交付完成后决定。
