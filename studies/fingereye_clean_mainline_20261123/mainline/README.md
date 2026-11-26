# FingerEyeLab 共享抓取入口（D108）

D108 从 D107 执行 `612409f998e`、证据 `53e2b925560` 继续。首次迁移冻结于 `71e08c1d91a`；看过首次结果后，共享修复版本冻结于 `ff3189380dd`，四例与原两手六例使用同一份代码。结果见 `D108_REPORT.txt` 与 `evidence/d108/`；D107 清理与回归见 `D107_REPORT.txt`。

本轮原有五个首抓成功全部保留。Wuji 首次、修后均为 0/4、四例零闭合，尚未抓通；修后解决了初始装配误拒与观测停滞，仍停在可执行抓姿搜索。运行示例可用于扩展和复验，不表示示例已经抓取成功。

正常入口为 `run_stage17.py --robot-urdf ... --controller mainline/strategy.py`。运行器按 URDF 内容哈希选择 `hand_calibrations/<sha256>.json`，生成物理与规划共用的模型。换手只改 URDF 参数；共享策略根据手的运动能力和观测几何选姿、联合 IK、检查路径、接近闭合、有限预载与试抬。确认空抓后从现场退让、重新观察并选姿，每回合最多三次实际闭合，抓间不重置物体。

Wuji 使用 `for_feixin` 的 `wuji_left_w_fingereye_6dof_floating_joint.urdf`。已有 3P+3R 虚拟掌链归一化为公共掌座接口，不另加自由度。源 XYZ 旋转链与公共 ZYX 坐标通过相等旋转矩阵映射，不能仅重命名角度；实体掌/指连杆和手指限位保持一致。

按用户授权，Wuji 的三个固定 `*_fingereye_link` 壳体挂点各安装 front1/front2/back 虚拟镜头。静态 sidecar 引用 `shared_simulation_modules_v1.json`，复用项目既有镜头 profile 与名义壳体到光学帧变换，渲染、曝光 FK 和策略投影共用它。这是**共享仿真配置**，不代表 Wuji 实机光学标定。未配置的未知手仍显式停止，不能把壳体原点当镜头中心。

在线只使用约定的初始俯视 RGB、手上 RGB、曝光时及当前实测 q/qvel、URDF 与静态描述。物体真值、原生接触和物体 mesh 只在环境及离线审计中使用。物理/控制/RGB 为 120/20/30 Hz，成功沿用原 evaluator 的合法抓起并保持标准。

完整四例顺序复跑（新输出目录）：

```bash
cd /home/CNS2025331827/docs/fingereye_d106_recovery
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
E=FingerEyeLab/experiments/rax_dexbin_tinyobj
"$PY" "$E/mainline/evidence/d108/run_panel.py" \
  --out "$PWD/FingerEyeLab/outputs/d108/wuji_reproduce_001"
```

四个物体和初态沿用 D107 首次迁移前登记，D108 仅替换用户指定的带浮动链源 URDF 与授权的统一相机配置。`run_panel.py` 保存每例完整 argv、源码/配置哈希及预算回执，并逐例调用正常入口。最初四例结果与看过结果后的任何修后复验分开保存。

原两手六例回归：

```bash
"$PY" "$E/mainline/evidence/d105/run_trials.py" \
  --root "$PWD" --out "$PWD/FingerEyeLab/outputs/d108/regression_reproduce_001" \
  --hands xhand new
```

`new` 仅是历史批处理中的原三指手 URDF 标签，不是策略分支。D106 完整基线位于 `FingerEyeLab/outputs/d106_baseline/frozen_v11/`，最小输入位于 `mainline/fixtures/d106/`。D107 已保住 XHand 三例和原三指手盒体、圆柱共五个首抓成功；原三指手楔体仍失败。

给师长的单例命令、资产位置、清单与初态扩展方法见 [TEACHER_CASE.md](TEACHER_CASE.md)。新运行均放项目现有 `outputs` 下。公开仓库是明确文件选集；完整 IsaacLab 环境与手/物体资产由实验室工程提供。
