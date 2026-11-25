# FingerEyeLab：共享抓取策略核心源码

目标是保持同一入口 `run_stage17.py` → `mainline/strategy.py::main(ctx)` 和 `observe()` / `step()`，通过 URDF 选择手。运行器生成统一的 `robot_description`；策略按具名关节、声明的控制顺序、限位、指链、接触面和相机执行。`auto`、`two`、`three` 共用优化与执行链，模式仅限制允许的指组；两指和三指从首次规划参与比较。

**第二轮仍在开发，尚未冻结，也未完成两手共同验收。** 当前采用大体竖直抓取原则（共享 45° 掌姿范围是任务解释），以实际抬升、保持、指定接触面承载和原物理评分验收。最近实际运行的生产改动为 `3b2b9aab45f`：用既有全手自碰撞/物体残差排序原来的 IK 初值池，保持初值数量、优化目标、门槛和执行链。策略 SHA256 为 `9a520d3f273f72bd1a608c771505026074fbcf80f583ed8df8324aafe19e31e5`。

D74 与 D76 在同一份固定 RGB 开发方块上均为两手零完整路径、零实际闭合，原 evaluator 均判失败。D76 新手 AB/AC/BC/ABC 均进入首次规划，10 次区域搜索；最近的 BC 位置误差为 0.620/0.380 mm，法向误差为 1.130/0.156 rad，末节仍与 RGB 物体包络相交。XHand 两指/三指各 10 组，27 次区域搜索，最近两指最大位置误差 1.549 mm，且有物体相交。规划墙钟新手 200.536 s、XHand 574.959 s；抓取动作时间均为 0。两手实际安装的 RGB 几何相同，685 个运行源码指纹前后及跨手一致。没有把离线 IK 探针或零闭合回合记作成功。

历史 D69 新手已有大体竖直的两指抬升、保持，但选定的两块有限 1.2 mm 指垫圆盘没有持续承载，由邻近几何接触；不能据此声称指定指垫成功。当前没有三块指定指垫承载成功，也没有第二轮冻结或新案例成绩。物体确实离桌、原 evaluator 成功、选定指垫承载是分别核验的结果。

第一轮 `ac44cacc7e8` 曾让两手抓起开发方块并保持；随后三对冻结案例为 XHand 3/3、新手 0/3。第一轮新手成功姿态距竖直向下约 80.7°，不满足当前竖直原则；XHand 历史姿态约 39.6°。历史结果不代表第二轮大体竖直、自动选指或三指承载已验证。可达、IK、几何数值或规划通过均不能代替真实抓取证据。

物理 120 Hz、控制 20 Hz、RGB 30 fps，每回合最多三次实际闭合，规划拒绝不消耗抓次。在线仅使用初始一次俯视 RGB、实时手上 RGB、曝光时与当前实测 q/qvel、静态标定；物体真值、mesh、原生接触力和 evaluator 只用于环境及离线核验。源 URDF/STL 只读。树、限位、安装变换和候选面来自资产；1.2 mm 半径、摩擦 0.3、每指 1 N 是显式假设，不能冒充软垫或驱动实测。相机机械安装来自资产，光学来自完整工程统一仿真配置；材料、顺应性、驱动力和硬件光学仍需标定。闭合 0.35 mm 进给、4.5 mm 总上限及 0.45 关节跟踪缺额判据继承旧手开发经验，尚非跨手驱动或承载标定；当前审计明确区分关节缺额、实际指垫法向行程与物理承载。

公开目标为 [mao455/2026-11-21](https://github.com/mao455/2026-11-21)，仅同步 10 个关键 Python 文件及本 README，逐字节核对完整工程对应源码。资产、标定、日志、RGB/视频和评测证据留在完整工程；**此源码选集不能独立启动 IsaacLab**，需要同版本完整工程及其宿主依赖。接触区域构建还使用 `Shapely 2.1.2`。完整证据、源码哈希与逐回合结果位于完整工程 `round2_evidence/STATUS.txt`、`iterations.json`、`d76_audit/PAIRED_READOUT.txt` 和各轮审计目录。

以下在完整工程复跑已参与开发的 30 mm 方块、中心初态。公共 RGB/几何缓存来自 D73 的首次有效手上 RGB 曝光及初始俯视 RGB，用于隔离手模型和规划差异，明确不冒充当前曝光。显式提供此缓存时，首次实际闭合前的规划几何固定；实时手上 RGB 仍采集、校验并用于运动及抬升证据，实际闭合后的恢复重新使用实时几何。未提供缓存的普通运行继续更新实时几何。这不是新的冻结评测。顺序运行，每次指定全新输出目录；共享任务、策略、模式与评测，仅切换 URDF，session 只区分输出。

```bash
cd /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/experiments/rax_dexbin_tinyobj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
COMMON=(--controller mainline/strategy.py --fingers auto --max-attempts 3
  --case-file round2_evidence/d73_review/cube30_public_cache/dev_cube30_fixed_rgb.json
  --mode minimal --evaluation-mode inline --phase explore)

"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf '/home/CNS2025331827/docs/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM/urdf/urdf_整手总装-1113-带相机-修正零位与限位.SLDASM.urdf' \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_new_001

"$PY" run_stage17.py "${COMMON[@]}" \
  --robot-urdf /home/CNS2025331827/docs/fingereye_cross_round2/FingerEyeLab/assets/xhand3camFE_left/planning/xhand3camFE_left_floating.urdf \
  --session /home/CNS2025331827/docs/fingereye_cross_round2/runs/readme_xhand_001
```

下一阶段须先让两手通过同一份策略的物理验收，再冻结源码、参数和评测；预登记新物体/初态，在相同任务与可比计算预算下比较旧策略、新 auto、固定 two/three，完整保留失败、零闭合、超时、重抓和 XHand 退化。
