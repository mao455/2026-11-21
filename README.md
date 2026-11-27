# FingerEyeLab 共享抓取主线（D109）

[共享策略](studies/fingereye_clean_mainline_20261123/mainline/strategy.py) · [D109 实跑报告](studies/fingereye_clean_mainline_20261123/mainline/D109_REPORT.txt) · [运行命令](studies/fingereye_clean_mainline_20261123/mainline/README.md) · [新增物体示例](studies/fingereye_clean_mainline_20261123/mainline/TEACHER_CASE.md) · [源码与证据清单](studies/fingereye_clean_mainline_20261123/PUBLICATION_MANIFEST.json)

D109：Wuji 圆柱首抓成功，四例共 1/4；XHand 3/3、原三指手 2/3 首抓成功，保住原五个成功回合。第二、第三抓均无实际救回。逐例闭合次数、失败阶段与配对证据见 D109 报告。

三只手使用同一正常 `--robot-urdf` 入口、共享策略与静态描述接口。在线只使用约定 RGB、实测 q/qvel 和静态手模型；物理/控制/RGB 为 120/20/30 Hz，每回合最多三次实际闭合，抓间不重置物体。成功按原 evaluator 的合法抓起并保持标准判定。

本轮从 URDF 固定装配提取入射方向首先能碰到的外表面，联合规划整只手和未参与夹持的手指，按关节速度执行完整路径。圆柱的缺图闭合限制与旧手窄通道回退均有单项物理对照；原始失败、修后成绩和诊断分别保存。详见 D109 报告与结构化证据。

Wuji 使用共享名义仿真镜头配置，渲染、曝光 FK 与策略投影共用；这不是实机光学标定。D108 首次迁移 0/4 保留在 D108 报告中，不以本轮成绩替换。

这是明确清单发布的算法、接口、参数和紧凑证据选集。完整 IsaacLab 环境、私有手/物体资产、原始 RGB、录像和大日志保留在实验室完整工程中。
