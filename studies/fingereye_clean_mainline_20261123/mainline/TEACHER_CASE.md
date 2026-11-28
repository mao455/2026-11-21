# 扩展物体与初态：现成 Wuji case

在完整 FingerEyeLab 工程、IsaacLab Python 环境中运行。所有输出放进项目 `FingerEyeLab/outputs/`，每次使用新目录。

```bash
cd /home/CNS2025331827/docs/fingereye_d106_recovery
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
E=FingerEyeLab/experiments/rax_dexbin_tinyobj
HAND=/home/CNS2025331827/docs/for_feixin/models/wuji_left/xarm7_wuji_left_description/wuji_left_w_fingereye_6dof_floating_joint.urdf
"$PY" "$E/run_stage17.py" \
  --session "$PWD/FingerEyeLab/outputs/teacher/cylinder_001" \
  --robot-urdf "$HAND" --controller "$E/mainline/strategy.py" \
  --registry-manifest "$E/mainline/fixtures/wuji_first/objects.json" \
  --case-file "$E/mainline/fixtures/wuji_first/cylinder_28.json" \
  --evaluation-mode inline --mode minimal --fingers auto --max-attempts 3 \
  --no-preload --phase explore --profile fast --color-mode hs --light-multiplier 1.0
```

`--no-preload` 关闭运行器的历史预载分支；共享策略内部仍执行有界预载和试抬。每回合最多三次实际闭合，抓间保留现场物体状态。物理/控制/手上 RGB 频率为 120/20/30 Hz；物体真值和原生接触仅在环境与离线评分中使用。

新增物体：

1. 将以米为单位、封闭且尺度正确的物体网格放在 `FingerEyeLab/assets/objects/<自己的集合>/`。四个现成示例位于 `assets/objects/d107_transfer/`。沿用示例的坐标约定：物体底面在局部 z=0，桌上初始放置后不悬空也不穿桌。
2. 复制 `mainline/fixtures/wuji_first/objects.json`，为新物体添加唯一 `object_id`、相对 **FingerEyeLab** 的 `asset` 路径和该文件的 SHA256（`sha256sum 文件`）。该清单供环境装载资产，不提供给在线策略。
3. 复制 `cylinder_28.json`，修改 `object_id` 和初态 `dx`、`dy`（米）、`yaw`（弧度），按需要修改质量、摩擦和观测噪声。保留一份固定清单便于比较；改变初态应另建 case。
4. 用上述命令指定新 `--registry-manifest`、`--case-file` 和新 `--session`。无需修改策略。

换手只改 `--robot-urdf`。程序按 URDF 内容哈希加载 `hand_calibrations/<sha256>.json`。Wuji 的 sidecar 引用 `shared_simulation_modules_v1.json`：在三个源 URDF 固定 FingerEye 壳体挂点上各安装 front1/front2/back 三个虚拟镜头，复用已有名义模块变换和镜头参数；渲染、曝光 FK 与策略投影共用这份配置。这是**共享仿真配置**，不代表 Wuji 实机镜头标定。未知手需要提供对应静态描述；不能自动把壳体原点当成光学中心。

完整四例顺序复跑：

```bash
"$PY" "$E/mainline/evidence/d109/run_panel.py" \
  --out "$PWD/FingerEyeLab/outputs/teacher/wuji_panel_001"
```

独立评分：

```bash
"$PY" "$E/mainline/evidence/d105/compile_results.py" \
  --runs "$PWD/FingerEyeLab/outputs/teacher/wuji_panel_001" \
  --output "$PWD/FingerEyeLab/outputs/teacher/wuji_panel_001_results.json"
```

报告须同时记录首抓、实际闭合次数、三抓内结果、真正救回的抓次和失败阶段。达到三抓上限是允许的最大预算，并不保证规划器一定能找到三次可执行闭合。`episode/result.json` 是原 evaluator 结果，`agent_trace.jsonl` 保存策略阶段与拒绝原因；未经物理执行的候选不计为实际抓次。

D110 历史冻结对照的命令见对应 Git 版本；本轮使用下面的 D111 命令，不把当前源码冒充 D110 冻结执行。已固定的四个 Wuji 物体和初态继续沿用，单例命令如上。

D111 清理后的稳定入口与十例命令：`bash "$E/mainline/evidence/d111/REPRODUCE.txt"`。必要用例和归档 RGB 几何已放在 `mainline/fixtures/`；历史 `outputs/d106_baseline`、`d107`–`d110` 和一次性维护工作树已删除，正常入口不依赖它们。执行与评分使用当前源文件，报告中的 D110 只作冻结对照。
