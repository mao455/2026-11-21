# FingerEyeLab：共享跨手抓取核心源码

同一入口 `run_stage17.py` → `mainline/strategy.py::main(ctx)`，保留 `observe()` / `step()`。运行器按 URDF 和静态标定提供具名关节、限位、指链、接触区域与相机；`auto`、`two`、`three` 共用优化与执行链，两指和三指从首次规划比较。任务要求大体竖直，当前共享掌轴范围为距向下方向 45°。

**仍在开发，未冻结，未完成两手共同验收。** 本源码对应完整工程已实跑 D90 `dfdbbab8710`，策略 SHA256 `d90494e018b09270c6382ce20707b959c79e341af3a51d2c3c6cc96308b84c6d`。D90 在同一批最多六个接触区域组合、同一保存 q 上，把既有整手桌面净空代价加入接触与采样本体穿透排序；保留原桌面/任务类别、候选、预算和精确碰撞门槛。

固定同一份 RGB、30 mm 开发方块和任务，仅换手及静态标定，顺序实跑：

| 手 | 实际两指 / 三指关节 IK | 完整路径 / 实际闭合 | 原 evaluator | 规划墙钟 / CPU | 抓取动作时间 |
|---|---:|---:|---|---:|---:|
| 新三指手 | 33 / 11 | 0 / 0 | 失败 | 264.322 / 264.095 s | 0 s |
| XHand | 42 / 16 | 0 / 0 | 失败 | 265.524 / 264.950 s | 0 s |

新手最近两指 AB 区域18/18的位置误差0.064 mm、法向0.438/0.414 rad，仍因末节 `Empty_Link4/7` 与 RGB 包络相交约2.25/2.52 mm被拒绝。最近三指 ABC4/26/7的位置误差4.081 mm、法向0.272/1.055/1.046 rad，尚未进入路径。XHand也没有可执行路径。两手宿主时间459.811/349.828 s，1565项运行指纹前后、跨手一致，无残留进程。两手零抬升、零闭合，原生接触未打开；历史成功不能替代当前共享竖直策略验证，也没有三块指定指垫持续承载证据。

物理120 Hz、控制20 Hz、RGB30 fps，每回合最多三次实际闭合。在线仅初始俯视RGB、实时手上RGB、实测q/qvel及静态标定；投影使用曝光时q。物体真值、mesh、原生接触力及evaluator仅环境/离线核验。URDF/STL只读。树、限位、网格、机械安装来自资产；静态侧车按URDF哈希加载。光学来自统一仿真鱼眼配置，未冒充硬件标定。摩擦0.3、每指1 N及未标定区域1.2 mm半径是共享显式假设；材料分区、软垫顺应性、驱动响应仍需独立标定。几何圆盘不能单独证明材料指垫或排除一体相机壳。闭合0.35 mm进给、4.5 mm总上限、0.45跟踪缺额判据仍是旧手开发经验。

公开只同步10个关键Python与本README，和完整工程所述已测版本逐字节核对。资产、标定、日志、RGB/视频和完整证据留完整工程。本源码选集需要同版本宿主工程、IsaacLab/IsaacSim及依赖，不能独立启动；区域构建依赖Shapely 2.1.2。

在完整工程上述提交复跑，以下共用入口、任务、策略、模式和评测，仅URDF决定换手，session区分新输出目录。固定缓存来自D73初始俯视图及首次有效手上曝光，首次实际闭合前固定规划几何；实时RGB仍采集校验，闭合后恢复实时几何。缓存明确不代表当前曝光。这是开发回合。

```bash
cd /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/experiments/rax_dexbin_tinyobj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
COMMON=(--controller mainline/strategy.py --fingers auto --max-attempts 3
  --case-file round2_evidence/d73_review/cube30_public_cache/dev_cube30_fixed_rgb.json
  --mode minimal --evaluation-mode inline --phase explore)
"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf '/home/CNS2025331827/docs/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM/urdf/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM.urdf' \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_d90_new_001
"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_d90_xhand_001
```

正式开发配对由 `round2_evidence/d90_review/RUN_PAIR.py` 固定1800 s宿主/600 s策略CPU预算并保存收据，旧目录不覆盖。完整证据：`round2_evidence/iterations.json` 和 `d90_audit/D90_PAIRED_CLOSEST_TERMINAL_REPORT.json`（SHA256 `4912b53bd5301e6d2b6c30540b92d6069639c6ec1f693a95d9a6b9e9e7b4ec93`）。两手物理验收后才冻结，再预登记新案例并比较旧策略、新auto、固定two/three，保留失败、超时、零闭合与XHand退化。
