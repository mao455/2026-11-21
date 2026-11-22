`02_abc_76395_clean` 的正式版本为 `9daeff1043917ac40f518b8f80d1103d0fe63bc1`，控制器 SHA256 为 `a01b97ab2e9ddd083ec852978945a7ec77ebcbc484b469aa060231bb781a7c1e`。配置沿用冻结 V4：fast、auto、最多三次实际抓取、20 Hz 控制、30 fps FingerEye、minimal。该例在 1500 秒预算后返回 `124 / wall_timeout`，实际入口到清理结束墙钟为 **1505.057078 秒**。控制器没有完成，完整性无效；本例计入失败及基础设施超时分母，不能当作已完成的物理失败或成功轨迹。

三次闭合均实际执行，并由 trace 中的目标命令、`steps.jsonl` 的真实 `s5_close` 物理步以及公开 observation 回执交叉确认：

| 尝试 | 闭合开始步 | 公开确认步 | 实际指组 | 恢复分支 |
|---|---:|---:|---|---|
| 1 | 1344 | 1350 | 两指 | initial |
| 2 | 3216 | 3222 | 两指 | local_pose |
| 3 | 4362 | 4368 | 两指 | local_pose |

两次恢复分别从公开失败判定步 2616、4194 开始；预载分支因缺少可信的新双侧响应或可用预载余量而未执行。局部姿态分支均通过 8 节点移动检查及 8 节点闭合检查，随后真实下发第二、第三次闭合。局部恢复统计为考虑 2、选择 2、执行 2、救回 0。`release_resample` 仅有两条条件后备记录；没有完成全局退让重采样，实际执行计 0。两条真实手型转换均为 2→2，腕姿变化和共同手指落点变化均未达到冻结的大幅手势门槛。没有三指执行，也没有已认证救回。

不同计时来源必须分别解释：

| 证据 | 数值 | 可支持的结论 |
|---|---:|---|
| 连续持久化 `steps.jsonl` | 5220 步，**43.5 秒** | 可逐步核验的物理时间下界 |
| 最后公开 observation 回执 | step 5286，**44.05 秒** | 后端已回执的状态时间；尾部缺少对应连续物理记录 |
| 已发出 step 请求累计 | 882 条、5292 步，**44.1 秒** | 已请求的标称物理时长；最后请求未见回执，不能称实际完成时长 |
| 调度器观测墙钟 | **1505.057078 秒** | 实际超时及清理墙钟，可列入基础设施耗时表 |

各记录流的尾部不同是事实；其原因未被证实，不归因为缓冲丢失。完整 `rpc_timing.json`、`wall_profile_backend.json`、结束主结果和 `s9_io_audit` 均缺失。因此，完整分段后端墙钟、策略非 RPC 墙钟与策略计算时间为 **NA**；原始 compact 的 `policy_compute_s=0` 是空累加器，不能解释为零计算耗时。该例从完整回合及完整分阶段耗时分布中排除；成功率、实际抓次、实际指组和恢复动作分母均保留。持久化首抓/释放退让/再抓物理分段只作为已观察下界，独立报告。

超时前的延迟审计证明，同一个后端 step 请求内两次曝光曾相隔 **35.515257 秒**，而最近 20 次窄图像采集函数平均约 **0.04055 秒**。这只定位到后端请求内部，不独占归因于物理、渲染、评测或调度。更早也有同类慢段，因此其他会话 GPU 进程的并发观测不足以证明因果。未干预其他会话进程。

独立前 5 例检查全部通过，检查了 41,226 张 FingerEye 文件；`partial_mode=true`、`final_complete=false`，绝不提前宣称 80 例完成。两组已有配对的初始 RGB、标定和初始物理状态均一致。该检查只证明已返回记录的可得证据一致，不补齐超时缺失文件。

计算链绑定不可变的[回合结束compact快照](../evidence/audit/timeout_02_abc76395_batch_end_compact_v4.json)，而非最终汇总会补充元数据的活动compact；原始回执和物理记录保持原样。[快照绑定审计](../evidence/audit/timeout_compact_snapshot_binding_v4.json)记录哈希与17项链条复核结果。

可复核文件：

- [完整超时计算链及命令/关节数据](../evidence/audit/timeout_chain_02_abc76395_v4.json)
- [前 5 例部分验证](../evidence/audit/final80_partial_first5_v4.json) 与 [图像验证汇总](../evidence/audit/final80_partial_first5_v4_frames.json)
- [超时前延迟快照](../evidence/audit/latency_diagnostic_02_abc76395_v4.json)
- [只读审计生成器](../evidence/audit/timeout_chain_builder_v4.py)

复核命令（在 `FingerEyeLab` 执行，不启动仿真）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python outputs/three_regrasp_20261122_01/reproduction/final_validation.py --freeze frozen_v4 --allow-partial --limit 5 --output outputs/three_regrasp_20261122_01/audit/final80_partial_first5_v4.json
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python outputs/three_regrasp_20261122_01/audit/timeout_chain_builder_v4.py
```
