# FingerEyeLab 跨手抓取核心代码

当前公开版本与 cross-embodiment 第一轮的实际执行实现逐字节对齐。正式策略入口是 `mainline/strategy.py::main(ctx)`，由 `run_stage17.py` 加载；XHand 与新三指手使用同一策略，通过 `--robot-urdf` 选择手。

本次只同步 8 个关键 Python 文件与本 README。现有 `requirements.txt` 和 `.gitignore` 保留。URDF/STL/USD、静态标定侧车、物体资产、RGB/录像、运行日志和实验报告不在本次公开范围内。

## 当前源码与调用链

对应完整工作树源码提交：`082cb44dfae83e4f7d16d8d75d5dd4dc9a959ce5`；其中运行实现来自代码提交 `ac44cacc7e8fd47a116ea95e4d02ccb816ca47a3`，后续提交只记录证据和更正文档。

下表“原路径”均相对完整框架的 `FingerEyeLab/experiments/rax_dexbin_tinyobj/`。公开文件仅按表中位置复制，Python 源码未改写。

| 公开文件 | 原路径 | 作用 |
|---|---|---|
| [run_stage17.py](run_stage17.py) | `run_stage17.py` | 正式入口、URDF 选择、公共参数和冻结校验 |
| [cross_hand_runtime.py](cross_hand_runtime.py) | `cross_hand_runtime.py` | 只读资产到派生手模型、命名映射、指链/接触区域、相机及 PhysX 适配 |
| [mainline/strategy.py](mainline/strategy.py) | `mainline/strategy.py` | RGB 感知、抓取排序、动态 IK、避碰、闭合/试抬/保持状态机 |
| [mainline/hand_description.py](mainline/hand_description.py) | `mainline/hand_description.py` | 共享关节树、限位、网格变换、FK、指垫和碰撞拓扑 |
| [mainline/joint_path.py](mainline/joint_path.py) | `mainline/joint_path.py` | 使用实际关节组和限位的确定性 RRT-Connect 路径搜索 |
| [mainline/general_grasp_contacts.py](mainline/general_grasp_contacts.py) | `general_grasp_contacts.py` | 共用任务力学裕量评分 |
| [runtime/worker.py](runtime/worker.py) | `worker.py` | `main(ctx)`、`observe()`、`step()` 的观测/RPC 接口 |
| [runtime/evaluator.py](runtime/evaluator.py) | `evaluator.py` | 原物理评分和保存回合的独立重评 |

调用链为 `run_stage17 → 完整环境的 episode/沙箱 host → worker.Context → strategy.main(ctx)`。`robot_description` 声明关节名称、控制顺序、限位、指链、指垫和可用相机；策略按名字索引并按声明顺序输出 `target_q`。六轴掌座通过公共运行接口表达，当前 XHand 为 18 维、新手为 17 维。输入校验和仿真后端消费同一描述，完整 host 实现留在运行环境中。

共用 H/S v2、两点夹持、最多 3 次实际闭合、恢复预载关闭。物理 120 Hz、控制 20 Hz、RGB 30 fps。相机使用模型清单及曝光时的实测关节状态，候选指对来自活动关节树。优化按实际关节数与限位构建，姿态正则按活动关节数归一化。

在线输入限于初始俯视 RGB、手上 RGB、实测关节状态和静态手部/相机标定。物体真值、物体 mesh 和原生接触只用于环境物理、离线诊断及评分，不反馈给策略。

策略 SHA256：`855d66722acf946af0881e5929d26009d8833be8993a09c33de4568a3320a03b`。
原评分 SHA256：`31ddc54031124d2f561a512edb96e0adce2da94c1cae174692e982034a93b22b`。

## 已验证结果与边界

2026-11-24 的开发回归中，同一策略在同一规则方块、同一初态上让两只手实际抓起并通过原 `lift30_no_drop_v1` 评分：

| 手 | 实际闭合/恢复 | 最后连续 2 秒最小抬升 | 最小离桌净空 | 墙钟耗时 |
|---|---|---|---|---|
| 新三指手 i41 | 1 / 0 | 31.939 mm | 31.445 mm | 513.474 s |
| XHand i42 | 1 / 0 | 32.024 mm | 31.928 mm | 116.543 s |

随后冻结代码、参数、评分及资产，完成 3 个未参与本轮开发的方块/圆柱初态、共 6 回合成对验证，换手仅切换 URDF 和独立输出目录：

| 案例 | XHand | 新三指手 |
|---|---|---|
| 偏置/旋转方块 | 正式成功 | 闭合前规划拒绝，0 次闭合 |
| 中心圆柱 | 正式成功 | 位置/法向未同时通过，0 次闭合 |
| 偏置圆柱 | 正式成功 | 接触几何通过后路径自碰撞，0 次闭合 |

XHand 为 **3/3**，新手为 **0/3**；没有依据这些结果继续调参或替换失败案例。新手在开发任务上的物理链已完成，对上述未参与开发案例的泛化尚未通过。规划拒绝不能证明该手物理上无法抓取。完整环境 58 项主套件及 6 项路由检查通过；这些检查不替代实际抓取证据。

原正式评分要求最后连续 240 个 120 Hz 物理步（2 秒）、实际抬升至少 28 mm、最低点离桌至少 5 mm、掌座命令抬升至少 29.5 mm，并保持全程几何有效及物理记录连续。没有因换手放宽评分。两手开发成功回合的在线最终状态仍为 `stop_unknown`，成功结论来自独立物理评分；历史严格接触力指标未通过。

目前只验证 `--mode minimal`。它仍记录 RGB、物理和决策日志；新手的 `full` 录像仍存在固定六路名称错误。第三种手未验证。

## 标定与运行条件

手的关节树、限位、机械安装和网格来自只读 URDF/STL。完整运行环境按原 URDF SHA256 自动加载 `hand_calibrations/<sha256>.json`，选择手的用户接口仍是一个 URDF 路径。

XHand 接触/光学静态配置沿用基线。新手相机机械挂点来自 CAD/URDF，机械到光学帧由 CAD 约定推导；鱼眼光学参数补自现有工作区 `fingereye_camera_cfg.py` 的统一仿真配置，并核对了渲染与投影参数及曝光 FK 一致性。新手候选接触区域来自末节网格静态几何。这些不构成硬件光学、软指垫材料或驱动响应的实测标定。

本仓库是核心源码选集。直接执行这里的 `run_stage17.py` 不会补齐完整仿真环境；它仍依赖同版完整框架中的 `stage15_runtime`、`stage14_resources`、`stage16_episode`、观测/动作校验、对象注册表、相机模块、IsaacLab/Isaac Sim 和 bubblewrap。`cross_hand_runtime.py` 也依赖完整框架的 `source/fingereye` 及相机配置。`requirements.txt` 是既有 IsaacLab 环境的补充依赖，不包含模拟器本身。

以下命令在已具备这些依赖及对应静态标定的**当前完整框架**中运行，加载本公开策略。完整 host 向沙箱提供的 `hand_description.py`、`joint_path.py`、`general_grasp_contacts.py` 必须与本仓库源码一致；仅有旧版 XHand 环境和新策略文件不足以跨手运行。`runtime/` 中两个文件在完整框架中分别对应实验目录根部的 `worker.py` 与 `evaluator.py`。

```bash
export FULL_CHECKOUT=/path/to/current/full-framework
export PUBLIC_CHECKOUT=/path/to/2026-11-21
cd "$FULL_CHECKOUT/FingerEyeLab/experiments/rax_dexbin_tinyobj"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# 使用现有 IsaacLab Python；case 为同一个固定开发方块任务。
python run_stage17.py --controller "$PUBLIC_CHECKOUT/mainline/strategy.py" \
  --case-file /path/to/dev_box.json --robot-urdf /path/to/new_hand.urdf \
  --session /path/to/new_session --mode minimal --evaluation-mode deferred --phase explore

python run_stage17.py --controller "$PUBLIC_CHECKOUT/mainline/strategy.py" \
  --case-file /path/to/dev_box.json --robot-urdf /path/to/xhand.urdf \
  --session /path/to/xhand_session --mode minimal --evaluation-mode deferred --phase explore
```

开发 case 内容为 `{"object_id":"r14_box","dx":0,"dy":0,"yaw":0,"mass_kg":0.007,"friction":0.82,"noise_seed":17001,"rgb_noise_std":0}`。两次运行顺序执行，输出 session 必须全新。公开选集未携带实验冻结清单，因此上述命令用于 `explore`；正式 heldout 复跑还需要原完整冻结文件与其全部匹配的环境/资产。

本次没有另跑一轮物理仿真；上表对应原始已保存回合。发布时核对 8 份 Python 与当前执行源码字节相同，并对公开副本做语法、导入、路径及原记录重评检查。后续对外发布继续在这个公开小仓库维护关键源码与 README。
