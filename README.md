# FingerEyeLab：共享跨手抓取核心源码

共用入口 `run_stage17.py` → `mainline/strategy.py::main(ctx)`，保留 `observe()` / `step()`。同一策略按 URDF、自动加载的静态标定和 RGB 几何选择指组、接触区域、掌姿及路径；`auto`、`two`、`three` 共用执行链。掌轴距向下方向不超过 45°，物理/控制/RGB 为 120/20/30 Hz，每场景最多三次实际闭合，规划拒绝不耗抓次。

**D104 是已见开发场景的固定源码回归，尚未通过两手共同物理验收。** 本仓库 10 个 Python 文件逐字节对应完整工程实跑提交 `2d67ab9d7574f157d54758f333fa7319be3a1a6a`，策略 SHA256 `2e4aaa72a31e6ad3ca56cf800f4a3724617ca2a3d951dba0179ec4c5651eb1ba`。六例运行期间源码不变；这不是未见案例验证。

| 手 / 三个场景 | 首抓正式成功 | 最多三抓正式成功 | 一次未闭合 | 实际闭合总数 |
|---|---:|---:|---:|---:|
| 新三指手 | 0/3 | 0/3 | 1/3 | 2 |
| XHand | 0/3 | 0/3 | 2/3 | 1 |

场景为长方体、圆柱、楔体，成对固定 RGB 几何、初态、任务与预算。新手长方体/圆柱、XHand 圆柱分别实际闭合一次，最大实测掌角 12.12°/20.53°/13.23°，末尾最小抬升 32.43/31.98/32.03 mm 并保持 2 s，但预载时末节本体侵入，原 evaluator 均判失败。其它三例在 2/3/2 轮搜索后因 CPU 预算保留停止，没有完整路径。三个有闭合回合最终持物状态不明，未授权释放，未实际试满三次；没有选定有限指垫共同承载的证明。

继承 D99–D102 的候选局部恢复、曝光实测 q 投影、移除整手最低点对物顶的误拒，以及同面分离的三指候选。D104 只在名义增量闭合路径被拒后，用 URDF 闲置关节避让，保持选中指/掌座节点与时长；XHand 圆柱两次避让确实执行，越过原 ring 相机碰桌阻塞。真实限位、自碰、桌/物碰撞及原 evaluator 未放宽。本组没有触发实际接近失败换候选、无效 RGB 故障恢复或确认空抓后的连续重抓，不能称这些分支已实跑验证。

尚存问题：每指 0.8 mm 终点门槛未考虑闭合可补行程；初值/粗排仍主要 upright+yaw，首区域可能耗掉指组预算；最终承载 LP 仅排序，任务不满足的候选仍可能执行。接触指整块本体的物体检查豁免及固定追加 1.05 mm 预载，尚不能保证指定接触区先触或避免过压。同场景历史第一轮 XHand 基线正式成功 2/3、零闭合 0/3，当前退化保留。

关节树、顺序、限位/effort、网格、指链和机械安装来自统一手描述及所选 URDF。相机渲染/投影共用仿真光学配置，非硬件标定；摩擦 0.3、每指 1 N、区域半径 1.2 mm 是显式假设。新手区域是几何提案，材料分区、软垫顺应性、驱动响应、硬件光学仍需独立标定。闭合 0.35 mm、总上限 4.5 mm、缺额阈值 0.45 等仍有旧手开发经验。在线只用 RGB、实测 q/qvel 和静态标定；真值、物体 mesh、原生接触及 evaluator 仅用于环境与终态离线核验。源 URDF/STL 只读。

公开选集仅含 10 个关键 Python、短 README 和原依赖元数据；**不能独立启动**，需要同版本完整 FingerEyeLab、IsaacLab/IsaacSim 及 Shapely 2.1.2。资产、标定、日志及动作轨迹不上传。

以下在完整宿主上顺序执行，两条命令共用入口、策略、任务、模式及评测，仅 URDF 选择手，session 区分新输出。固定 RGB 缓存来自已见场景，首次闭合前固定规划几何；实时手上 RGB 仍采集校验，闭合后恢复实时几何。

```bash
cd /home/CNS2025331827/docs/fingereye_d104_unused_path/FingerEyeLab/experiments/rax_dexbin_tinyobj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
E=/home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/experiments/rax_dexbin_tinyobj/round2_evidence
COMMON=(--controller mainline/strategy.py --fingers auto --max-attempts 3
  --case-file "$E/d98_panel/fixtures/box_center/case.fixed_rgb.json"
  --evaluation-mode inline --mode minimal --no-preload --phase explore
  --profile fast --color-mode hs --light-multiplier 1.0)
"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf '/home/CNS2025331827/docs/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM/urdf/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM.urdf' \
  --session /home/CNS2025331827/docs/d104_readme_new_fresh_001
"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf \
  --session /home/CNS2025331827/docs/d104_readme_xhand_fresh_001
```

正式复跑请用完整工程 `$E/d104_unused_path/run_panel.py` 的 `verify` 及 `run --all --output-root <尚不存在的目录>`，统一限制 600 s worker CPU、1800 s 宿主、24000 物理步并保存清理/身份回执。完整逐回合结果、规划累计墙钟、物理动作时间、候选与实际发送、承载审计和复跑步骤均在 `$E/d104_unused_path/analysis/{REPORT.md,REPRODUCE.txt}`；材料 pin SHA256 `ad50e65f5de703baad3427db059e4f174697d05c3af5120de2528fc7c5eb178e`。尚未完成三指指定垫承载、新案例或固定 two/three 对照验收。
