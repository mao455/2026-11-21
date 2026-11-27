# FingerEyeLab 共享抓取入口（D110）

冻结执行版 `058c570113b`：XHand 3/3、原三指手 2/3、Wuji 3/4，八例均首抓成功；原六个成功全部保住，新增 Wuji 盒体和五边柱。原三指楔体0次闭合；Wuji楔体1次闭合失败，恢复后规划无解。第二、第三抓没有实际救回。逐例成绩、失败边界与配对证据见 [D110_REPORT.txt](D110_REPORT.txt) 和 [FINAL_SUMMARY.json](evidence/d110/FINAL_SUMMARY.json)。

正常入口为 `run_stage17.py --robot-urdf ... --controller mainline/strategy.py`。运行器按 URDF 内容哈希选择 `hand_calibrations/<sha256>.json`，生成物理与规划共用的模型。换手只改 URDF 参数；共享策略根据手的运动能力和观测几何选姿、联合 IK、检查路径、接近闭合、有限预载与试抬。确认空抓后从现场退让、重新观察并选姿，每回合最多三次实际闭合，抓间不重置物体。

Wuji 使用 `for_feixin` 的 `wuji_left_w_fingereye_6dof_floating_joint.urdf`。已有 3P+3R 虚拟掌链归一化为公共掌座接口，不另加自由度。源 XYZ 旋转链与公共 ZYX 坐标通过相等旋转矩阵映射，不能仅重命名角度；实体掌/指连杆和手指限位保持一致。

按用户授权，Wuji 的三个固定 `*_fingereye_link` 壳体挂点各安装 front1/front2/back 虚拟镜头。静态 sidecar 引用 `shared_simulation_modules_v1.json`，复用项目既有镜头 profile 与名义壳体到光学帧变换，渲染、曝光 FK 和策略投影共用它。这是**共享仿真配置**，不代表 Wuji 实机光学标定。未配置的未知手仍显式停止，不能把壳体原点当镜头中心。

在线只使用约定的初始俯视 RGB、手上 RGB、曝光时及当前实测 q/qvel、URDF 与静态描述。物体真值、原生接触和物体 mesh 只在环境及离线审计中使用。物理/控制/RGB 为 120/20/30 Hz，成功沿用原 evaluator 的合法抓起并保持标准。

同一冻结版原十例完整顺序复跑（包含三只手，需完整实验室资产）：

```bash
bash /home/CNS2025331827/docs/fingereye_d106_recovery/FingerEyeLab/experiments/rax_dexbin_tinyobj/mainline/evidence/d110/REPRODUCE.txt
```

文件内保存十条实际正常入口命令，使用已冻结case、原始URDF和自动静态描述。默认输出位于 `FingerEyeLab/outputs/d110/reproduce/`；再次运行请改为新的输出目录。脚本采用文本后缀以兼容现有发布选集，内容可直接交给bash。`new`仅是批处理对原三指手的历史标签。

本轮固定装配外表面候选、面内接触意图、第三指收益比较、整手避让和逐指有限探闭均为共享实现。限位、真实自碰、桌面碰撞和非法穿透继续检查，没有统一增加碰撞余量。观测改变接触面或法向会触发从实测q出发的局部修正；保持不变时沿用已检查路径。全部成功仍为两指，三指改进尚无本批物理胜出证据。

正式十例和诊断回放分开；D108首次迁移与D109基线不被本轮成绩覆盖。原始物理日志/RGB保留在实验室 `FingerEyeLab/outputs/d110/finite_retry_v53_*`。结构化证据中的绝对路径指向当时实际会话；重跑命令已改回主工作树。

给师长的单例命令、资产位置、清单与初态扩展方法见 [TEACHER_CASE.md](TEACHER_CASE.md)。新运行均放项目现有 `outputs` 下。公开仓库是明确文件选集；完整 IsaacLab 环境与手/物体资产由实验室工程提供。
