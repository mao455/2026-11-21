# 复核与复跑

本目录属于 `/tmp/fingereye-success30-push-cae50/code/FingerEyeLab`，分支 `mao0629`；不使用 `bc+rl_RSS`。基线为 `7fa3bd37752a38d5b2dc65da6735804c78057a09`，策略实现为 `a1bbe71ae86f79ba5f5f818352a75dc3eeb50460`。**最终40物体及80回合使用 `frozen_v4`，准确提交见 `frozen_v4/freeze.json` 的 `source_commit`**。分析、报告后续修订不改变正式结果记录的源提交。

所有命令使用 `/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python`。Isaac、驱动、依赖、CPU亲和及本地机器人/网格来源见 `frozen_v4/environment.json`、`audit/environment_data_audit.json` 和每回合 `launch.json`。普通运行采用 minimal，仍保留策略所需六路 FingerEye RGB；录像单独 full 复跑。

## 冻结与几何更正

`frozen_v4/freeze.json` 记录策略、运行时文件、40对象注册、80回合计划及预登记初态的哈希。`overhead_inputs_v4/manifest.json` 保存两条件共用的预采普通初始俯视图，来自抓取前的几何资格检查。`audit/protocol_geometry_amendment.json` 记录几何去重更正，`audit/geometry_tessellation_crosscheck.json` 保留解析基元与表面对照证据。替代物体须通过同一稳定高度、接触区域与夹取空间资格；不按抓取结局选入。移动工作树后应显式迁移绝对路径并重新登记协议，不能修改已有冻结清单后仍声称同一冻结。

正式物理回合统一使用 fast + auto + 最多3次实际抓取；120 Hz物理、20 Hz控制、30 fps RGB。σ=3灰度级的高斯噪声只加于六路 FingerEye；原视觉σ=0，两组保留原路径其余设置。每个物体一个预登记初态/种子。闭合实际下发后才计抓次，规划拒绝不计；原位预载复验另占统一物理预算。

## 历史V2（排除于最终分母）

`frozen_v2` 对应提交 `117244a8197c431e265d56477da3f84cc3ab814f`，其共享俯视图保留在 `overhead_inputs/`。更正前的 `abc_152578` 与 `abc_702588` 是同一圆柱几何，仅126/127侧面离散不同，故不能按两个物体计数。旧队列归档为 `protocol_geometry_pilot_117244a/`，不计入最终80回合、不拼接其余39对象结果；V3先行队列同样整体排除；最终队列在 `formal_runs/` 从冻结V4重新运行全部80回合。

更正前的复核脚本、录像计划和本文原版完整保留在 `reproduction/history_v2/`，仅供核对历史，不能作为当前入口执行。其中原始绝对路径代表归档前位置；原始 `launch.json` 与轨迹保留。V3先行队列的脚本快照保留在 `reproduction/history_v3/`，最终冻结前即停止并整体排除；不能拼接到V4。更早 `frozen/`、V9两指三抓及入口开发对照同样不进入最终分母。

旧队列曾使用最多两路合同，并在启动异常后实际回退单路；其启动失败及恢复回执审计见 `audit/formal80_serial_fallback.json`、`audit/formal80_recovered_receipts.json`。这些属于V2历史，不能借用到V4的时间或成功率。`reproduction/resume_batch_serial.py` 保留为旧合同兼容包装器，最终V4使用原入口直接显式 `--concurrency 1`。

## 复核最终结果

```bash
cd /tmp/fingereye-success30-push-cae50/code/FingerEyeLab
PYTHON=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
STUDY=outputs/three_regrasp_20261122_01
EXPERIMENT=experiments/rax_dexbin_tinyobj
"$PYTHON" "$EXPERIMENT/stage13_readout.py" --root "$STUDY/formal_runs" --output "$STUDY/formal_readout"
"$PYTHON" "$STUDY/reproduction/timing_readout.py" --batch "$STUDY/formal_runs" --output "$STUDY/timing_readout"
"$PYTHON" "$STUDY/reproduction/selection_audit.py" --batch "$STUDY/formal_runs" --output "$STUDY/selection_readout"
"$PYTHON" "$STUDY/reproduction/contact_audit.py" --batch "$STUDY/formal_runs" --output "$STUDY/contact_readout"
"$PYTHON" "$STUDY/reproduction/final_validation.py" --freeze frozen_v4
"$PYTHON" "$STUDY/reproduction/object_catalog.py" --freeze frozen_v4
"$PYTHON" "$STUDY/reproduction/result_figures.py" --freeze frozen_v4
"$PYTHON" "$EXPERIMENT/stage13_report.py" --root "$STUDY" --freeze frozen_v4
```

`result_figures.py` 只读取冻结V4计划的80条逐例记录，并要求结果源提交匹配。全部80回执返回后，`reproduction/finish_delivery.py` 依次完成读出、分段时间、选指/接触审计、验证、独立录像复跑、40物体目录图、结果图和报告；它不启动或重试正式队列，不删除数据。录像复跑仍会实际占用GPU，应在正式队列结束后运行。

交付流水线在生成V4分析前将同名旧分析移入 `analysis_history/`，随后核对逐例集合恰为冻结80条及其源提交；空事件/接触CSV也重新写出，避免残留旧队列行。录像仅在源视频及合成文件哈希完全一致时复用，配对复跑另绑定正式 `compact.json`、回执与配置哈希。最终录像清单只列选定4–6段，其他清单版本留在 `videos/history/`。

`final_validation.py` 校验冻结几何资格文件哈希、40个唯一注册对象、全部稳定摆放/接触区域门槛和零抓取资格检查，并逐例重新计算物理判据与实际动作次数。普通闭合以闭合标记匹配到的 `s5_close` 实际物理步计次，预载复验以连续 `s13_preload_recovery` 实际物理步计一次；命令后回执缺失不抹去已经执行的尝试，而是单列回执不可得。`--allow-partial` 和 `--limit` 只用于运行中检查，绝不能产生 `final_complete=true`。

原始数据清理后的保留范围与逐文件哈希见清理清单；读取已归档文件须按该清单恢复。逐例 `result.json`、实际 `steps.jsonl`、动作与反馈 `agent_trace.jsonl`、`compact.json`、启动配置和评测接触记录是判据与动作证据。文件时间戳不能替代精确墙钟。

清理脚本默认只生成清单。实际清理要求最终80回合原始证据QA通过，并绑定报告、逐例CSV、转换表、冻结文件、录像清单及 `audit/final80_validation.json` 的当前哈希；不以基础设施失败时不可获得的输入冒充已验证。归档前的绝对路径按原 `launch.json --session` 解析到所有归档副本，保护报告引用。旧输出的所有权与引用检查在任何删除前完成；被新报告引用的唯一原始证据即使生成校验过的gzip，也保留原文件。

## 独立复跑一个正式案例

应使用新 session；已有正式目录禁止覆盖或按结局重试。以下复跑不增加正式分母：

```bash
cd /tmp/fingereye-success30-push-cae50/code/FingerEyeLab
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python experiments/rax_dexbin_tinyobj/run.py \
  --session /tmp/fingereye-review-v4-abc311740-clean \
  --case-file experiments/rax_dexbin_tinyobj/stage13_cases/01_abc_311740_clean.json \
  --mode full \
  --controller outputs/three_regrasp_20261122_01/frozen_v4/submitted.py --phase heldout
```

加噪对照改用 `01_abc_311740_noise.json` 和新的 session。逐回合实际命令在 `formal_runs/<name>/launch.json`；预登记80命令与几何规则见 `frozen_v4/plan.json`。基线/开发命令分别见 `baseline_plan.json`、`development_plan.json` 及运行记录；须在对应提交的独立工作树中执行，不能用最终版本替代旧版结果。

## 正式队列与断点续跑

下面命令仅用于已登记冻结V4的正式队列；若 `formal_runs/contract.json` 属于旧版本，应保留原目录并先完成历史归档，禁止覆盖合同或混入旧结果。每个回执落盘后不按结果重试；基础设施失败照实留在分母。

```bash
cd /tmp/fingereye-success30-push-cae50/code/FingerEyeLab
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python experiments/rax_dexbin_tinyobj/stage13_batch.py \
  --root outputs/three_regrasp_20261122_01/formal_runs \
  --plan outputs/three_regrasp_20261122_01/frozen_v4/plan.json \
  --controller outputs/three_regrasp_20261122_01/frozen_v4/submitted.py \
  --concurrency 1 --recording-mode minimal --resume
```

## 录像

`videos/index.html` 与 `videos/manifest.json` 标明原始验收录像和独立复跑、源提交、抓次、手型、结局、完整30 fps解码及哈希。录像复跑须与对应最终正式回合的实际抓次、指组序列及结局分别核对；不能补进正式分母。`reproduction/video_plan_v4.json` 由交付流水线从冻结V4的80回合计划生成配对配置，运行根目录为 `video_replays_v4/`，避免复用历史V2的 `video_replays/` 回执。额外三指或救回录像使用 `extra_video_replay_v4/`。

## 显式选择冻结版本

分析脚本不自动猜测最终队列。`final_validation.py`、`object_catalog.py`、`result_figures.py`、`finish_delivery.py` 与清理apply使用 `--freeze frozen_v4`，或设置 `FINGEREYE_FINAL_FREEZE=frozen_v4`。历史快照只供复核，不能作为最终入口。交付流水线按所选目录的 `protocol_tag` 自动使用 `video_replays_v4/`、`extra_video_replay_v4/` 及相应计划文件，并要求录像启动时HEAD仍与正式冻结源提交一致。

完成冻结后，等待正式80回合并交付的入口：

```bash
cd /tmp/fingereye-success30-push-cae50/code/FingerEyeLab
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python outputs/three_regrasp_20261122_01/reproduction/finish_delivery.py --wait --freeze frozen_v4
```

以下是冻结前的俯视图登记命令，应由资格检查和注册维护者执行；它会更新共享注册及case，只允许创建新的输出目录：

```bash
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python outputs/three_regrasp_20261122_01/reproduction/freeze_overhead_inputs.py --directory overhead_inputs_v4
```

## V4策略守卫与物理验收版本

V4候选把预载恢复阶段 `s13_preload_recovery` 接入 `stage5_control.py::s5_move_segment` 的既有实测速度守卫，保留原阈值。这是策略/动作守卫版本变化；`a1bbe71`的两指/三指开发回合及三棱锥连续三抓录像只证明旧提交，不能替代新版本验收。新验收记录单列在 `guard_acceptance_readout/`，预期四例为 `cube_two`、`cube_three`、`cube_auto` 和 `pyramid_auto3`；报告逐例标记实际源提交、真实抓次、指组、预载复验和控制器完成状态。该小规模验收不增加正式80回合分母。

RGB配对、精确几何重复、形状差异三轮预跑均整队排除。第三轮的家族5%规则、近球拟合门槛、五候选来源和统一缩放见 `audit/protocol_diversity_amendment.json`、`audit/diversity_v3_analysis.json`、`audit/diversity_replacement_proposal.json`。最终40对象以 `frozen_v4/plan.json` 与实际零抓取稳定摆放资格为准；静态候选预检不冒充物理验收。
