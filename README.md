# FingerEyeLab 算法与研究分析选集

仓库创建日期：2026-11-21。之后每次发布继续更新此仓库，不每日新建。

这里保留部分算法代码、Stage9–11 文字分析、结构化实验数据和结果表。不包含完整后端、机器人资产及所有依赖，不保证独立运行。完整运行代码见 [grasp_small_obj / mao0629](https://github.com/ztr583/grasp_small_obj/tree/mao0629)。

当前推荐仍为 V9、两指、最多三次实际抓取。仿真真值、接触和严格评分仅用于离线分析，不是策略输入。

## 阅读顺序

- [Stage11 结果](reports/STAGE11_RESULTS.txt)：闭合、恢复、选指与并发。
- [Stage11 算法](reports/STAGE11_ALGORITHM.txt)：真实状态机及跨文件轨迹。
- [Stage10 结果](reports/STAGE10_RESULTS.txt)：反馈重抓及失败证据。
- [Stage9 结果](reports/STAGE9_RESULTS.txt)：时序基线与配对结果。
- algorithms/：当前共享算法实现及配置。
- data/：实际已有的汇总、计时、并发及轨迹数据，未重新生成实验结论。

原报告中的本地 outputs 路径用于定位原始证据，相关 RGB、录像、逐物理步日志和大数据不在本仓库。分析选集不等于全部原始数据备份。MANIFEST.json 记录各文件的原路径、字节数、SHA256、研究源码版本及双仓库快照号；原文件中的机器路径和实验时戳按原样保留。

## 双仓库更新

在保留完整本地研究资料的工作区执行：

```bash
python FingerEyeLab/experiments/rax_dexbin_tinyobj/publish_github.py --push
```

省略 --push 只在临时目录准备两个提交并核验。新增算法或分析文件须登记到 github_publish.json 的明确文件清单。两个仓库使用正常推送；若其中失败，命令非零退出并写入发布回执，修复后重跑同一命令完成同步，不强制覆盖远端。
