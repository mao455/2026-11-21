# FingerEyeLab 共享抓取主线（D108）

[共享策略](studies/fingereye_clean_mainline_20261123/mainline/strategy.py) · [D108 实跑报告](studies/fingereye_clean_mainline_20261123/mainline/D108_REPORT.txt) · [运行命令](studies/fingereye_clean_mainline_20261123/mainline/README.md) · [新增物体示例](studies/fingereye_clean_mainline_20261123/mainline/TEACHER_CASE.md) · [源码与证据清单](studies/fingereye_clean_mainline_20261123/PUBLICATION_MANIFEST.json)

三只手使用同一正常 `--robot-urdf` 入口、共享策略与静态描述接口。在线只使用约定 RGB、实测 q/qvel 和静态手模型；物理/控制/RGB 为 120/20/30 Hz，每回合最多三次实际闭合，抓间不重置物体。成功按原 evaluator 的合法抓起并保持标准判定。

D108 实跑：XHand 3/3 首抓成功，原三指手 2/3 首抓成功，原有五个成功回合的全部指令与 D107 相同。Wuji 首次及修后均为 0/4、四例零闭合；修复了装配误拒与初始观测停滞，仍未找到可执行抓姿。原三指手楔体在重抓接近时触发实测关节不稳定保护。本轮没有第二、第三抓救回。

本轮接入 Wuji 左手的既有六轴虚拟掌链，在三个固定 FingerEye 壳体挂点使用统一的名义仿真镜头配置。该配置不是实机光学标定。首次迁移与看过结果后的修后复验分开报告；不能用后续成绩替换首次成绩。详见 D108 报告及结构化证据。

这是明确清单发布的算法、接口、参数和紧凑证据选集。完整 IsaacLab 环境、私有手/物体资产、原始 RGB、录像和大日志仍在实验室完整工程中。源码历史基线 D107 执行 `612409f998e`、证据 `53e2b925560`；旧 H/S v2 与其他 studies 是历史阶段，不代表当前共享手迁移成绩。
