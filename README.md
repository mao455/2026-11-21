# FingerEyeLab 当前策略

唯一维护策略：`mainline/strategy.py::main(ctx)`。算法依赖：`mainline/general_grasp_contacts.py`；正式评分：`runtime/evaluator.py`。
H/S v2、固定两指、最多三次实际闭合、恢复预载关闭。120 Hz 物理、20 Hz 控制、30 fps RGB。
输入只含初始俯视与六路 FingerEye RGB、实测关节状态、静态标定。

这里仅发布核心代码，不含私有资产、IsaacLab 环境、原始记录和实验产物。
完整可运行版本在 https://github.com/ztr583/grasp_small_obj/tree/fingereye-clean-mainline 。
在完整仓库的 `FingerEyeLab/` 下，使用 IsaacLab Python 环境及 bubblewrap：

```bash
python experiments/rax_dexbin_tinyobj/run.py \
  --controller /absolute/path/2026-11-21/mainline/strategy.py \
  --session outputs/first_grasp \
  --case-json '{"object_id":"abc_311740","dx":-0.0008,"dy":0.0012,"yaw":-0.2,"mass_kg":0.007,"friction":0.82,"noise_seed":16111,"rgb_noise_std":0}'
```

完整仓库默认控制器与此策略内容相同；运行时记录加载路径、SHA256 和配置。
退出码只表示流程执行情况；正式成功读取 `episode/result.json` 中的 `success` 和执行完整性。
当前树移除了全部 stage/round 实验、证据、报告和发布工具。历史可从 Git 提交 `acf176266127d5b9e298429722240e2f8835e12e` 及其父提交追溯。
