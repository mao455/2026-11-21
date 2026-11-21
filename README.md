# FingerEyeLab 算法与研究分析选集

仓库创建日期：2026-11-21。之后每次发布继续更新此仓库，不每日新建。

这里保留部分算法代码、Stage9–12 文字分析、结构化实验数据和结果表。不包含完整后端、机器人资产及所有依赖，不保证独立运行。完整运行代码见 [grasp_small_obj / mao0629](https://github.com/ztr583/grasp_small_obj/tree/mao0629)。

当前推荐仍为 V9、两指、最多三次实际抓取。策略只使用普通 RGB、本体状态和静态标定。仿真真值、接触和评分仅用于离线分析。

主成功口径现为完成约 30 mm 抬升并连续保持 2 s 未掉落：实际物体抬升至少 28 mm（2 mm 跟踪容差）、命令抬升至少 29.5 mm、保持段桌面净空至少 5 mm，且全程几何检查通过。物体速度、转动及接触力保留作诊断和旧严格口径对照，不再单独否定主成功。宿主执行完整性检查和原运动守卫继续生效。

## 阅读顺序

- [30 mm 成功口径](reports/SUCCESS30_RESULTS.txt)：判据、边界测试与已有轨迹重新裁决。
- [Stage12 结果](reports/STAGE12_RESULTS.txt)：统一承力排序与三指候选拒绝证据。
- [Stage12 算法](reports/STAGE12_ALGORITHM.txt)：候选到实际下发的调用链。
- [Stage11 结果](reports/STAGE11_RESULTS.txt)：闭合、恢复、选指与并发。
- [Stage11 算法](reports/STAGE11_ALGORITHM.txt)：真实状态机及跨文件轨迹。
- [Stage10 结果](reports/STAGE10_RESULTS.txt)：反馈重抓及失败证据。
- [Stage9 结果](reports/STAGE9_RESULTS.txt)：时序基线与配对结果。
- algorithms/：当前共享算法实现及配置。
- data/：结构化汇总、计时、并发及轨迹数据；configurations/：预定初态。

Stage12 冻结的 8 例已有物理轨迹，按新口径离线重裁决为 5/8，旧严格口径为 3/8。变化来自两例方块：抬到约 30 mm 后未掉落，旧口径只因物体原生速度未过门。这次重裁决不代表重新执行这 8 例；三指仍无实际执行成功。原报告及原始裁决保留，新旧结果的逐例分母与证据见 30 mm 报告和 data/stage12/success30_evidence.json。

原报告中的本地 outputs 路径用于定位原始证据，相关 RGB、录像、逐物理步日志和大数据不在本仓库。分析选集不等于全部原始数据备份。MANIFEST.json 记录各文件的原路径、字节数、SHA256、研究源码版本及双仓库快照号；原文件中的机器路径和实验时戳按原样保留。

## 双仓库更新

在保留完整本地研究资料的工作区执行：

```bash
python FingerEyeLab/experiments/rax_dexbin_tinyobj/publish_github.py --push
```

省略 --push 只在临时目录准备两个提交并核验。新增算法或分析文件须登记到 github_publish.json 的明确文件清单。两个仓库使用正常推送；若其中失败，命令非零退出并写入发布回执，修复后重跑同一命令完成同步，不强制覆盖远端。
