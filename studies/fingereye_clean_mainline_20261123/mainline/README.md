# FingerEyeLab 共享抓取入口（D111）

冻结执行版 `b0f59424c83` 已完成十例物理复验：XHand 3/3、原三指手 2/3、Wuji 3/4，原八个首抓全部保住。两个楔体各实际闭合两次，仍失败；三手零闭合均为 0，第二/第三抓救回均为 0。逐例成绩、因果配对、剩余停止位置和清理结果见 [D111_REPORT.txt](D111_REPORT.txt)。

正常入口为 `run_stage17.py --robot-urdf ... --controller mainline/strategy.py`。运行器按 URDF 内容哈希选择 `hand_calibrations/<sha256>.json`，生成物理与规划共用的模型。换手只改 URDF 参数；共享策略根据手的运动能力和观测几何选姿、联合 IK、检查路径、接近闭合、有限预载与试抬。确认空抓后从现场退让、重新观察并选姿，每回合最多三次实际闭合，抓间不重置物体。

Wuji 使用 `for_feixin` 的 `wuji_left_w_fingereye_6dof_floating_joint.urdf`。已有 3P+3R 虚拟掌链归一化为公共掌座接口，不另加自由度。源 XYZ 旋转链与公共 ZYX 坐标通过相等旋转矩阵映射，不能仅重命名角度；实体掌/指连杆和手指限位保持一致。

按用户授权，Wuji 的三个固定 `*_fingereye_link` 壳体挂点各安装 front1/front2/back 虚拟镜头。静态 sidecar 引用 `shared_simulation_modules_v1.json`，复用项目既有镜头 profile 与名义壳体到光学帧变换，渲染、曝光 FK 和策略投影共用它。这是**共享仿真配置**，不代表 Wuji 实机光学标定。未配置的未知手仍显式停止，不能把壳体原点当镜头中心。

在线只使用约定的初始俯视 RGB、手上 RGB、曝光时及当前实测 q/qvel、URDF 与静态描述。物体真值、原生接触和物体 mesh 只在环境及离线审计中使用。物理/控制/RGB 为 120/20/30 Hz，成功沿用原 evaluator 的合法抓起并保持标准。

十例复跑命令：

```bash
bash mainline/evidence/d111/REPRODUCE.txt
```

所有运行输出放 `FingerEyeLab/outputs/d111/` 下的新会话目录。用例、物体清单、初始 RGB 几何和静态光学描述位于稳定输入路径；必要输入的逐文件哈希见 `evidence/d111/STABLE_INPUTS.json` 与 `INPUT_READBACK.json`。D106–D110 原始输出和嵌套工作树已退役删除；历史证据里的绝对会话路径用于审计，不是当前运行依赖。历史回合需要在对应 Git 引用上重建，不能直接读取已退役的输出。

给师长的单例命令、资产位置和初态扩展方法见 [TEACHER_CASE.md](TEACHER_CASE.md)。公开仓库为明确文件选集；完整 IsaacLab 环境和资产由实验室工程提供。
