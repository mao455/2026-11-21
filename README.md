# FingerEye shared grasp — minimal handoff

在已安装 Isaac Sim 5.1 / Isaac Lab 2.3 的 Linux Python 环境中运行。使用该环境的 Python，保留其 torch；安装 `pip install -r requirements.txt`，系统安装 `bubblewrap`（Ubuntu: `sudo apt install bubblewrap`）。需要支持 RTX 的 NVIDIA 驱动/GPU。无需原机器路径或旧输出。

进入本目录，只用一个入口：

```bash
python run_grasp.py --robot-urdf robots/wuji/hand.urdf
```

也可传 `/path/to/hand.urdf`。默认提供 28 mm 盒体、初态及质量/摩擦，输出自动写入 `outputs/<时间戳>/`。`--output` 指定一个尚不存在的目录；`--case-file` / `--objects` 可替换场景和物体清单。两个仓库本目录中的运行源码和配置相同。

URDF 与资源约定：

- 支持树状 URDF、fixed/revolute/continuous/prismatic 关节、collision mesh 或 box/sphere/cylinder；网格使用米和 URDF scale，可用相对路径或本地可解析的 `package://`。URDF 配套网格必须可读。已有六轴、无几何的浮动掌链映射为公共掌接口；没有则自动添加一次。闭链、mimic/耦合传动不在本版支持范围。
- 包含 Wuji 左手（原 `wuji_left_w_fingereye_6dof_floating_joint.urdf` 的等价相对资源版本）和 XHand。原三指手也可传供应方 URDF 与全部相邻网格；其单文件超过 GitHub 100 MB 限制，未重复打包。无需登记新手哈希、修改代码或选择策略。
- 固定 `*_fingereye_link` 是虚拟 FingerEye 壳体挂点。每处按 `calibration/shared_simulation_modules_v1.json` 装 front1/front2/back；未知装配默认 finger 图表，URDF link 可用 `fingereye_module="thumb"` 表达另一种机械图表。已有装配按固定零件集合复用镜头静态配置，**不匹配整份 URDF 哈希**。其他相机可提供显式固定 optical frame（`*_optical_link`）与 profile 后缀，使用 OpenGL -Z 光轴、+Y 向上；可选同名 `.fingereye.json` 表达不同镜头静态参数，不是策略登记。
- `calibration/` 是共享仿真名义配置，不冒充硬件实测标定。渲染与策略使用同一运行 URDF 光学 link、同一镜头内参，安装位姿经曝光时实测 q 的 FK 计算。修改 joint origin、关节限位、网格尺度/指节长度会传入 FK、候选、IK 和仿真转换；修改相机挂点同时影响两边。可选 authored collider 缓存仅精确输入复用，修改后自动重建，不需补哈希。

策略只读初始俯视 RGB、手上 RGB、实测 q/qvel 和静态手模型；真值与原生接触只供私有评分/诊断。保持 120 Hz 物理、20 Hz 控制、30 Hz 相机、上方抓取、最多三次实际闭合、抓间保留现场、原 evaluator 与两秒保持。逐指闭合反馈、有限预载、两三指选择沿用 D111。

新增例子：把 OBJ/STL 放在 `examples/`（或自己的资源目录），在 `examples/objects.json` 增加 `object_id`、相对 asset、可选 sha256，在 `examples/box.json` 复制并修改 object_id、dx/dy/yaw 和物理参数。运行 `python run_grasp.py --robot-urdf robots/wuji/hand.urdf --case-file examples/my_case.json`。几何/初态只进入仿真，策略仍从 RGB 估计。

运行文件：

| 文件 | 用途 |
| --- | --- |
| `run_grasp.py` | 唯一用户入口、默认例子、自动输出 |
| `strategy.py` | 唯一共享策略：观测、规划、执行、闭合、预载、试抬与恢复 |
| `hand_runtime.py`, `hand_description.py` | URDF 规范化、运动链/FK、静态表面、相机装配和仿真驱动 |
| `contact_regions.py`, `contact_intent.py`, `kinematic_candidates.py`, `surface_repair.py`, `joint_path.py` | 有限接触面、候选、可达性、同面修正及整手路径 |
| `motor_feedback.py`, `grasp_mechanics.py` | 逐指跟踪亏差、力学评分与两三指比较 |
| `host.py`, `sandbox.py`, `worker.py`, `resources.py` | 子进程协议、观测隔离、源码复制、运行资源租约 |
| `sim_backend.py`, `simulator.py`, `sim_env/*.py` | 固定物理循环、Isaac 适配和场景/相机环境，无源码动态拼接 |
| `observation_guard.py`, `visual_observation.py`, `initial_rgb_geometry.py`, `initial_image.py` | RGB/编码器边界、初始观测 |
| `objects.py`, `lighting.py`, `fingereye_camera_cfg.py` | 私有物体清单、光照、共享镜头参数 |
| `evaluator.py`, `geometry.py`, `convex_geometry.py`, `geometry_acceleration.py`, `gravity.py`, `diagnostics.py`, `contact_diagnostics.py`, `self_contacts.py`, `recording.py` | 原评分器、严格几何、重力和隔离的物理记录 |
| `config/`, `calibration/`, `robots/`, `examples/`, `xhand_collision.*`, `selfcollisions.json` | 配置、可复用静态描述、必要资源、默认例子 |

D111 的 8/10 是历史成绩，不是本次发布的十例成绩。本版只报告 `validation/acceptance.json` 中这次轻量检查和单盒体冒烟。停止楔体攻关与十例复跑；成功率后续由接手者调。

本次单例：Wuji box_28 正常完成，2 指、1 次闭合；末尾实际抬升至少 30.11 mm、保持 2 s，但第 6312 步闭合时 `left_finger3_link4` 侵入物体，原 evaluator 判失败（`invalid_episode_geometry`）。不将抬升算合法成功，也未继续调参复跑。静态镜头核验最大位置差 2.79e-8 m、角差 2.15e-7 rad。
