# FingerEyeLab：共享跨手抓取核心源码

共用入口为 `run_stage17.py` → `mainline/strategy.py::main(ctx)`，保留 `observe()` / `step()`。运行器按 URDF 和静态标定提供具名关节、限位、指链、接触区域与相机。`auto`、`two`、`three` 共用优化和执行链；两指与三指从首次规划参与比较。任务要求大体竖直，当前共享掌轴范围为距向下方向45°。

**开发快照，尚未冻结，也未完成两手共同验收。** 这10个Python文件逐字节对应完整工程已实跑 D96 提交 `027393ef771f5e3d67e7bb6ef370b6ea725b76fa`；策略 SHA256 为 `e0958583694a8e33e955aa60c9832ec9991ca9062974966e362763cd74fb7fb5`。未包含后续未实跑的记录器修正。

D96沿用同一份RGB几何、30 mm开发方块和任务，仅替换URDF及对应静态标定。新增一个共享接近分支：保持可行终点的手指姿态，检查高位运动及下降的完整路径；保留原有开手分支、关节限位、跨指/物体/桌面碰撞检查及原评测。

| 手 | 通过完整规划路径 | 实际闭合 | 实际抬升与保持 |
|---|---:|---:|---|
| 新三指手 | 1 | 0 | 未实现 |
| XHand | 0 | 0 | 未实现 |

新手实际发出了接近命令，但在下降前因实测关节状态持续不稳定、无法满足稳定观测要求而停止；终点和名义路径通过不能证明物理执行成功。XHand仍未找到完整可执行路径。两手本轮零闭合、零抬升，没有三块指定指垫承载证据。历史成功不替代当前共享竖直策略验证。

物理120 Hz、控制20 Hz、RGB30 fps，每回合最多三次实际闭合。在线只使用初始俯视RGB、实时手上RGB、实测q/qvel和静态标定；图像投影使用曝光时q。物体真值、mesh、原生接触力及evaluator仅用于环境和离线核验。URDF/STL只读。

关节树、限位、网格和机械安装来自资产，静态侧车按URDF哈希加载。光学来自统一仿真鱼眼配置，尚非硬件标定。摩擦0.3、每指1 N及未标定区域1.2 mm半径是显式假设；材料分区、软垫顺应性、驱动响应仍需独立标定。几何接触圆盘本身不能证明材料指垫或排除一体相机壳。闭合0.35 mm进给、4.5 mm总上限、0.45跟踪缺额判据仍来自旧手开发经验。

公开仓库仅保留10个关键Python、本README及现有依赖说明。资产、标定、日志、RGB/视频和完整证据留在完整工程。这份源码选集需要同版本完整宿主工程、IsaacLab/IsaacSim和依赖，**不能独立启动**；接触区域构建另需Shapely 2.1.2。

在上述D96完整工程提交复跑。以下两次运行共用入口、任务、策略、模式和评测，仅URDF选择手，session用于区分新的输出目录。固定缓存来自D73初始俯视图及首次有效手上曝光，首次实际闭合前固定规划几何；实时RGB仍采集校验，闭合后恢复实时几何。缓存不代表当前曝光，以下仍是开发回合。

```bash
cd /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/experiments/rax_dexbin_tinyobj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
COMMON=(--controller mainline/strategy.py --fingers auto --max-attempts 3
  --case-file round2_evidence/d73_review/cube30_public_cache/dev_cube30_fixed_rgb.json
  --mode minimal --evaluation-mode inline --phase explore)
"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf '/home/CNS2025331827/docs/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM/urdf/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM.urdf' \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_d96_new_001
"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_d96_xhand_001
```

完整配对记录由 `round2_evidence/d96_review/RUN_PAIR.py` 固定1800 s宿主、600 s策略CPU预算并保存收据，旧输出目录不覆盖。证据位于完整工程 `round2_evidence/iterations.json`、`d96_review/` 和 `d96_audit/`。两手物理验收后再冻结并预登记新案例，比较旧策略、新auto及固定two/three，保留失败、超时、零闭合和XHand退化。
