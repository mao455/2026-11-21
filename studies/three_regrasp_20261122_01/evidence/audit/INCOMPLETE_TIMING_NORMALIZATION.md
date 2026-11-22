已准备独立离线规范化脚本 `reproduction/normalize_incomplete_timing.py`。当前仅在真实前 5 例的临时输出上验证；未修改等待中的 `finish_delivery.py`，未重启 finalizer，未改正式 episode、receipt、compact、冻结文件、HEAD 或运行时。

脚本按逐例 `timing_complete=false` 选择不完整回合，不按 `batch_wall_timing_complete` 选择；后者可能只是调度器回执被恢复，物理与策略计时仍完整。被选回合的发表字段 `policy_compute_s` 改为 JSON `null`／CSV 空值，原值保存在 `incomplete_recorded_between_dispatch_accumulator_s`，旁边 `policy_compute_time_scope` 明确说明其只累加已有命令间记录，可能为空，不代表完整计算时间或 CPU 时间。若嵌套 `policy_timing_audit.recorded_between_dispatch_compute_s` 也存在，同样置空并保留原累加器和说明。

规范化同步逐例 JSON、`SUMMARY.json.results` 和 `episodes.csv`。全部成功判定、物理时间及其下界、首抓与重抓、实际指组、恢复次数、转换、完整性和现有组统计保持不变。完整回合逐例 JSON 保持原始字节。元数据审计放在 readout 目录之外，避免被现有 `validate_readout` 当作额外 episode。

真实前 5 例的 15 项检查通过，见 [incomplete_timing_normalization_first5_v4.json](incomplete_timing_normalization_first5_v4.json)。包含复制模式、带外部原始快照的显式 in-place 模式、幂等、逐例／CSV／SUMMARY 同步，以及正式源文件未改的哈希检查。唯一变化的例为 `02_abc_76395_clean`：`policy_compute_s` 从 `0` 变成 `null`，原 `0` 保留；43.5 秒物理下界、三次真实两指闭合和基础设施超时失败均保持不变。测试输出为 `/tmp/fingereye_timing_normalization_first5_v4_m3h_gdvq/normalized_readout`，测试生成器为 [incomplete_timing_normalization_check_v4.py](incomplete_timing_normalization_check_v4.py)。

安全接入采用 root 选定的发表副本方案：先等待当前 finalizer 进入终态并确认进程退出，完成其 80 例分析与录像流程；之后复制 `formal_readout` 到全新的 `publication_readout` 并规范化。保留原始 `formal_readout`、正式 compacts 和全部视频引用哈希。不要在此后再次运行 `stage13_readout.py` 覆盖发表副本。以下命令都在 `FingerEyeLab` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python outputs/three_regrasp_20261122_01/reproduction/normalize_incomplete_timing.py --readout outputs/three_regrasp_20261122_01/formal_readout --output outputs/three_regrasp_20261122_01/publication_readout --audit-output outputs/three_regrasp_20261122_01/audit/publication_timing_normalization_v4.json
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python experiments/rax_dexbin_tinyobj/stage13_report.py --root outputs/three_regrasp_20261122_01 --freeze frozen_v4 --readout outputs/three_regrasp_20261122_01/publication_readout
```

`publication_readout` 必须尚不存在；脚本拒绝覆盖已有目录。报告生成器目前将数据链接硬编码为 `formal_readout/`，因此 root 在上述渲染后将 Markdown 中的数据链接及对应目录说明指向 `publication_readout/`。规范化的 `SUMMARY` 所有非 `results` 字段保持不变，因此现有统计图与汇总数值无须改变。正式原始验证和选择／接触／视频审计仍绑定未改的 raw compacts；发表副本的字段变更有单独规范化审计。若今后选择同步 raw compacts，必须重做所有 compact 哈希审计及受影响视频引用，当前方案不需要同步。

超时计算链已由 root 固定到 `audit/timeout_02_abc76395_batch_end_compact_v4.json`，SHA256 为 `1eeb3e5338012d5ffe82894477571d4263af8979e2b2d90eb3283f76ef7862af`；当前 `timeout_chain_builder_v4.py` 和 [timeout_compact_snapshot_binding_v4.json](timeout_compact_snapshot_binding_v4.json) 保持 root 修改后的状态。最终 canonical compact 补充元数据不影响这条快照链。`timeout_chain_source_*`、`timeout_chain_builder_original_v4.py` 与 `timeout_chain_before_source_snapshot_v4.json` 是本子任务在获知 root 已完成快照前准备的额外副本，不是权威引用，可在最终去重清理时删除；不得误删 root 的 `timeout_02_abc76395_batch_end_compact_v4.json`、链条 JSON、builder 或绑定审计。
