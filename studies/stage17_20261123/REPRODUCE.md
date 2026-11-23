# 复跑

完整实验树：`/home/CNS2025331827/docs/fingereye_stage17`。原始用户树 `/home/CNS2025331827/docs/grasp_small_obj` 的未提交内容未纳入、未更改。需要本地有授权的手URDF/USD、对象资产和相机标定；公开选集不包含这些资产。Python为 `miniconda3/envs/isaaclab/bin/python`，RTX4090D，IsaacLab。`stage15_runtime`显式解析私有资产根，策略在bubblewrap内只读公开RGB/q/静态手标定。

```bash
cd /home/CNS2025331827/docs/fingereye_stage17/FingerEyeLab/experiments/rax_dexbin_tinyobj
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
$PY run_stage17.py --session ../../outputs/stage17/my_new_hs --case-file stage17_evidence/cases/A.json --asset-root /home/CNS2025331827/docs/fingereye_stage16/FingerEyeLab --color-mode hs --light-multiplier 0.4 --evaluation-mode deferred --mode full
$PY run_stage17.py --session ../../outputs/stage17/my_new_baseline --case-file stage17_evidence/cases/A.json --asset-root /home/CNS2025331827/docs/fingereye_stage16/FingerEyeLab --controller stage17_evidence/controllers/stage16_frozen.py --color-mode legacy --light-multiplier 0.4 --evaluation-mode deferred --mode full
$PY stage17_light_batch.py
$PY stage17_light_batch_v2.py
$PY stage17_v_replay.py --episode ../../outputs/stage17/lighting/A_L1_legacy/episode --controller stage17_evidence/controllers/hs_v2.py --output /tmp/new_v_replay
$PY stage17_replay.py --episode ../../outputs/stage17/lighting/A_L1_legacy/episode --controller stage17_evidence/controllers/clean_legacy.py --output /tmp/new_exact_replay.json
$PY stage17_evidence/tools/checks.py
$PY stage17_evidence/tools/offline_review.py
$PY stage17_evidence/tools/check_middle_usd.py
$PY stage17_evidence/tools/v_floor.py
$PY stage17_evidence/tools/collect_lighting.py
$PY stage17_evidence/tools/summarize_perception.py
```

session是不可覆盖的回合目录；重跑必须换新名字。`--mode minimal`保留全部RGB及物理记录但不编码mp4；本轮配对批次使用minimal，历史代表例复用已有30fps录像，逐帧抽取位置在video_frames.json。`--mode full`由正常后端录制六路拼图和external；mp4编码不参与策略。额外侧视图是记录姿态的离线CAD重绘，不是新增相机输入或新的闭环分母。

真实改光在私有场景初始化后枚举UsdLux.LightAPI，原色和exposure不变，仅intensity乘0.4/1/1.6，并落盘light_intervention.json；没有RGB后处理。回放才做数字V变换：OpenCV uint8 HSV，H/S固定，V乘系数，rint、clip和RGB反量化，逐图报告均值/裁剪/量化误差。两个实验不能混称。

物理120Hz/控制20Hz/相机30fps，正常步长6/曝光间隔4物理tick；最多3次真实闭合。逐像素高斯噪声0 DN。运行期间不得修改或新增顶层py/json文件，否则原完整性门槛会将回合判无效。本轮开发中发生的无效回合保留并另建immutable_rerun，未改评分救回无效记录。

公开选集中的`code/`可在具备这些依赖的Python中独立构建相同控制器（无需启动仿真）：

```bash
cd studies/stage17_20261123/code
$PY -c "from stage17_build_controller import build; print(build('/tmp/new_stage17_controller.py',color_mode='hs',fingers='two',attempts=3,preload=False))"
sha256sum /tmp/new_stage17_controller.py
```

应得到66b685a974f5823c5f00618ed9c258f1a8911a4c46a14efdea046d75f8032e7c。完整闭环仍需实验室仓库的环境、驱动与私有资产，公开选集不是无资产独立仿真发行包。
