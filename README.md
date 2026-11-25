# FingerEyeLab：共享抓取策略核心源码

同一入口 `run_stage17.py` → `mainline/strategy.py::main(ctx)`，保留 `observe()` / `step()`。通过 URDF 选择手，运行器提供具名关节、控制顺序、限位、指链、接触区域与相机描述；`auto`、`two`、`three` 共用优化和执行链，模式只限制允许的指组，两指和三指从首次规划参与比较。任务要求大体从上方竖直抓取，当前共享掌姿范围为距向下方向 45°；最终看真实离桌、连续保持及原物理评分。

**第二轮仍在开发，尚未冻结，也未完成两手共同验收。** 本源码对应完整工程已实跑 D88 `40e0d9dac70`，策略 SHA256 为 `9c323536ff9242bcad5a7d1e98167f17ee5fb720047eaa9cdbd70d264fb6a2fd`。D85 引入全手采样穿透代价，保留原精确碰撞与路径检查；D86 将优化中 FCL 接触返回上限改为 32。D87 按既有位置/角度容差把表面法向残差系数改为 `1500*.0008/(2*sin(.5/2))≈2.425`。D88 只在原首次/细化关节求解后增加固定 q、固定当前接触面与指面区域的坐标优化，每次最多 20 次求值，只接受原坐标相关代价严格下降；不增加关节求解，也未改碰撞、路径或 evaluator。原始关节求解和坐标优化的状态、optimality 分别按变量范围保存。D88 两手仍为零完整路径、零实际闭合、零物体抬升，原 evaluator 均失败；本体碰撞尚未解决。

D87 与 D88 使用同一份固定 RGB、30 mm 开发方块及可比预算，实际结果如下：

| 轮次 / 手 | 实际两指/三指 IK 次数 | 完整路径 | 实际闭合 | 原 evaluator | 规划墙钟 / CPU | 抓取动作时间 |
|---|---:|---:|---:|---|---:|---:|
| D87 / 新三指手 | 24 / 10 | 0 | 0 | 失败 | 208.339 / 208.315 s | 0 s |
| D87 / XHand | 44 / 16 | 0 | 0 | 失败 | 259.774 / 259.423 s | 0 s |
| D88 / 新三指手 | 24 / 10 | 0 | 0 | 失败 | 216.018 / 215.962 s | 0 s |
| D88 / XHand | 42 / 16 | 0 | 0 | 失败 | 262.901 / 262.524 s | 0 s |

D88 新手最近的 AC 接触区域 13/8 候选，最大位置误差由 D87 的 0.871 mm 降至 0.551 mm，位置/法向检查已通过，但 `Empty_Link4/11` 仍与物体包络相交；XHand 最近两指降至 0.194 mm，也因 `left_hand_pinky_link2` 相交被拒。两手分别有 2/1 个保留两指终点通过位置/法向检查，三指仍无通过者；几何改善没有转化成实际抓取。两手各 1530 项运行指纹前后、跨手一致，无残留进程。宿主耗时为新手 415.935 s、XHand 348.559 s，坐标求解不计作新增原始关节 IK。

零闭合没有记作成功，也没有打开原生接触记录。历史开发回合曾出现抬升保持；这不替代当前策略验证，亦无三块经独立材料标定的指垫持续承载证据。接触区域、实际承载与原 evaluator 成功分别核验，规划或低几何代价不能代替物理结果。

物理 120 Hz、控制 20 Hz、RGB 30 fps，每回合最多三次实际闭合。在线只用初始一次俯视 RGB、实时手上 RGB、实测 q/qvel 和静态标定；投影使用曝光时 q。物体真值、mesh、原生接触力及 evaluator 只用于环境与离线核验。原 URDF/STL 只读。树、限位、网格与机械安装来自资产；静态侧车按 URDF 哈希自动加载。相机光学来自完整工程统一仿真鱼眼配置，渲染与投影参数核对，不冒充硬件标定。候选接触面来自静态标定及末节碰撞网格；网格面和 1.2 mm 有限圆盘不能单独区分材料指垫与一体相机壳。摩擦 0.3、每指 1 N、未标定区域的 1.2 mm 半径是显式共享假设；材料分区、顺应性与驱动响应仍需独立标定。闭合 0.35 mm 进给、4.5 mm 总上限及 0.45 跟踪缺额判据仍继承旧手开发经验。

公开仓库仅保留 10 个关键 Python 文件和简短 README，与完整工程所述已测提交逐字节核对。资产、标定、日志、RGB/视频和完整证据留在完整工程；此源码选集需要同版本宿主工程、IsaacLab/IsaacSim 及依赖，不能独立启动。接触区域构建使用 `Shapely 2.1.2`。完整证据位于 `round2_evidence/iterations.json`、`d87_audit/D87_PAIRED_CLOSEST_TERMINAL_REPORT.json`、`d88_audit/D88_PAIRED_CLOSEST_TERMINAL_REPORT.json` 及各轮审计目录；D88 报告 SHA256 为 `86909b3fce25716b1a5646b00383e55bdc1a3248f7f9da840a7d8d9f67213abe`。

以下在完整工程切到上述已测提交后复跑开发案例。两条命令共用任务、入口、策略、模式与评测，只切换 URDF；`session` 仅区分输出，须使用新目录。固定 RGB 缓存来自 D73 已保存的初始俯视图及首次有效手上曝光，明确不是当前曝光；首次实际闭合前固定其规划几何，实时图像仍采集和校验，闭合后恢复使用实时几何。普通无缓存运行继续更新实时几何。这不是冻结评测。

```bash
cd /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/experiments/rax_dexbin_tinyobj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
COMMON=(--controller mainline/strategy.py --fingers auto --max-attempts 3
  --case-file round2_evidence/d73_review/cube30_public_cache/dev_cube30_fixed_rgb.json
  --mode minimal --evaluation-mode inline --phase explore)

"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf '/home/CNS2025331827/docs/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM/urdf/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM.urdf' \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_d88_new_001

"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_d88_xhand_001
```

正式开发配对另由完整工程 `d88_review/RUN_PAIR.py` 固定 1800 s 宿主与 600 s 策略 CPU 预算并保存收据；旧配对目录与收据保留，重跑需新输出名。两手物理验收后才能冻结源码、参数和规则，再预登记新物体/初态，比较旧策略、新 auto、固定 two/three，保留失败、超时、零闭合、重抓和 XHand 退化。
