# Stage13：三指候选、重抓与40对象配对评测

主配置为 **fast + auto + 最多3次实际抓取**，正式源提交 `9daeff1043917ac40f518b8f80d1103d0fe63bc1`，基线 `7fa3bd37752a38d5b2dc65da6735804c78057a09`。40个不同几何对象各跑原视觉与 FingerEye 高斯噪声 σ=3 灰度级一次，共80条正式回执。120 Hz物理、20 Hz控制、30 fps RGB；主评分 `lift30_no_drop_v1`。

| 条件 | 首抓成功 | 最终成功 | 第二/第三新增 | 平均实际抓次 | 实际两指/三指 |
|---|---:|---:|---:|---:|---:|
| 原视觉 | 20/40 | 20/40 | 0/0 | 1.525 | 61/0 |
| 加噪 | 21/40 | 21/40 | 0/0 | 1.550 | 62/0 |

三指物理执行为0，重抓救回为0，本结果应称为两指抓取基线。唯一结局不一致为 `abc_189313` 原视觉启动崩溃、加噪首抓成功，不能把 +1/40 解释为噪声收益。两侧控制器完整结束的38个对象结局相同；另一个对象两侧均超时。全部3条设施异常均保留分母。

- [完整评测报告](EVALUATION.md)
- [80回合CSV](results/episodes.csv)、[80回合紧凑JSON](results/episodes_compact.json)、[源文件绑定](results/SOURCE_BINDING.json)、[总汇](results/SUMMARY.json)
- [真实转换表](results/transitions.csv)、[恢复动作逐周期](recovery/choices.csv)、[选指与下发](selection/actual_dispatches.csv)、[分阶段时间](timing/stages.csv)
- [冻结40对象](freeze/objects40.json)、[原始冻结清单](freeze/freeze.json)、[原始80回合计划](freeze/plan.json)、[实际命令](freeze/commands80.json)
- [最终三指拒绝与实际第二/第三抓计算链](chains/FINAL_VERSION_CHAINS.md)、[历史方块拒绝复现](chains/COMPUTATION_CHAINS.md)
- [复核与复跑说明](REPRODUCE.md)、[视频登记与本地证据范围](LOCAL_EVIDENCE.md)、[QA摘要](qa/summary.json)

公开研究仓库仅提供可读的报告、算法与结构化结果，不要求独立运行。原始RGB、录像和逐步日志留本地。源报告及完整本地证据已独立核验后才清理冗余帧；本发布副本只转换链接并提取紧凑字段，不重新评测或改写任何正式结局。
