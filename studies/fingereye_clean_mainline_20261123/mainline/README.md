# FingerEyeLab 正式主线

唯一维护的策略源是 `mainline/strategy.py`，入口是 `main(ctx)`。`run_stage17.py` 默认直接提交此文件，`run.py` 和 `run_published.py` 转入同一入口；也可显式指定 `--controller mainline/strategy.py`。不再拼接或生成算法源码。每回合 `episode/program/controller.py` 是沙箱保存的字节相同冻结副本；`stage9_executed_backend.py` 等是环境适配器生成物，不是另一个策略源。

配置固定为已实跑的 H/S v2、两指、最多三次实际闭合、恢复预载关闭、历史任务排序和原正式评分。物理 120 Hz、控制 20 Hz、RGB 30 fps。逐像素高斯噪声在 A/B 用例中为零。初始俯视 RGB、六路 FingerEye RGB、曝光与当前实测关节状态、静态手模型及标定构成策略输入；对象真值、原生接触和私有判分只进入离线核查。

从 `FingerEyeLab/experiments/rax_dexbin_tinyobj` 执行（会话目录必须是新的）：

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python run_stage17.py \
  --controller mainline/strategy.py \
  --asset-root /home/CNS2025331827/docs/fingereye_stage16/FingerEyeLab \
  --case-file stage17_evidence/cases/A.json \
  --session ../../outputs/clean_mainline/first_A \
  --mode minimal --light-multiplier 1 --evaluation-mode deferred
```

重抓代表例把 `A.json` 换成 `B.json`，会话改为 `../../outputs/clean_mainline/recovery_B`。可省略 `--controller`，默认源完全相同。`--mode full` 增加原有录像；本轮新回合采用 minimal，保留全部物理、RGB、决策和接触记录。

算法的唯一项目内导入依赖是 `general_grasp_contacts.py`（本策略实际导入 `task_wrench_margin`，用于条件任务力旋量裕量计算）；NumPy、SciPy、OpenCV、trimesh、Embree、python-fcl 等见现有 `requirements-research.txt` 和实验室 IsaacLab 环境。正常入口复用现有资产数据库/对象注册表、IsaacLab、USD/URDF、渲染相机、bubblewrap 沙箱、日志和资源锁；未迁移私有资产。资产根必须包含 assets/scripts/source。公开代码选集本身不包含启动物理实验所需的私有资产。

同记录验证：从历史固定提交恢复原控制器到实验室临时目录，再分别回放；不需要仿真。

```bash
git show 9ce1883ce509757bbc3a2ea072dfe33e378be08b:FingerEyeLab/experiments/rax_dexbin_tinyobj/stage17_evidence/controllers/hs_v2.py > /tmp/hs_v2_reference.py
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python mainline/tools/replay.py \
  --episode ../../outputs/stage17/lighting_v2/A_L1_hs_v2/episode \
  --controller /tmp/hs_v2_reference.py --output /tmp/A_before.json
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python mainline/tools/replay.py \
  --episode ../../outputs/stage17/lighting_v2/A_L1_hs_v2/episode \
  --controller mainline/strategy.py --output /tmp/A_after.json
```

对 B 使用 `B_L1_hs_v2`。比较返回值的规范化 SHA（只排除计时字段）与全部 target_q/steps/phase。`evidence/replay_comparison.json` 记录本轮实际结果及逐类计数；原始 RGB 和大日志只留实验室。完整链路、门槛、删除清单、真实运行轨迹和远端核验见 `REPORT.md` 与 `evidence/`。

历史 Stage17 源与所有移除项可用 `git show 9ce1883ce509757bbc3a2ea072dfe33e378be08b:<path>` 恢复；公开历史固定在 https://github.com/mao455/2026-11-21/tree/1a0c46bb3115cd785d0c83194970a5257aa02db6/studies/stage17_20261123 。旧入口需在该完整历史工作树运行。本维护分支仅支持本文件列出的正式入口。

本维护分支的发布使用显式新清单（旧 `github_publish.json` 已按历史归档移除）：

```bash
python publish_github.py --source-root /home/CNS2025331827/docs/fingereye_stage17 \
  --config mainline/publish.json --only research --preserve-unlisted \
  --work-root /tmp/fingereye-clean-publish --receipt mainline/evidence/publish_receipt.json --push
```

私有完整树用 `git push origin HEAD:refs/heads/fingereye-clean-mainline`，避免用公开选集替代完整代码树。公开发布只清除旧 Stage17 选集目录；固定历史提交、其他 studies 和原用户工作树保留。
