# 本地证据与发布范围

本包按既有双仓库清单公开报告、算法与结构化结果；原始 RGB、录像、逐步轨迹、缓存和大清理清单留本地，不上传 Git。正文中指向本页的链接表示引用了留在本地的证据，不表示这些原文件已随仓库发布。

正式源提交：`9daeff1043917ac40f518b8f80d1103d0fe63bc1`；基线：`7fa3bd37752a38d5b2dc65da6735804c78057a09`。

本地研究根目录：`/tmp/fingereye-success30-push-cae50/code/FingerEyeLab/outputs/three_regrasp_20261122_01`。`freeze/`中的原始登记文件按字节保留，因此包含历史绝对路径；迁移后应显式重建路径与输入，不应改写冻结后仍声称同一协议。

[视频清单](videos/manifest.json)记录六段代表录像、源提交、复跑标记、动作与结局核对和哈希。录像本体保留在本地 `videos/`。所有录像复跑不增加正式80回合分母。

`results/episodes_compact.json`覆盖80条结果；它精简了重复的候选树和逐步数组，[SOURCE_BINDING](results/SOURCE_BINDING.json)逐例记录完整源JSON的SHA256。原始的80逐例JSON、完整动作/图像日志和逐例launch均留本地。

下表记录发布报告直接引用的留本地文件；文件未存在或是历史外部路径时哈希留空。

| 本地相对路径或历史绝对路径 | SHA256 |
|---|---|
| `audit/final_chain_rgb.png` | `9628a597f26b68689c69168d995cc6b21a8f52a3cf1901a3ec211ee09af182aa` |
| `audit/recovery_choices_formal80_v4/audit/recovery_choices_formal80_v4/03_abc_145502_clean.json` | 未在此源目录单独核验 |
| `audit/recovery_choices_formal80_v4/audit/recovery_choices_formal80_v4/PUBLICATION_BINDING.json` | 未在此源目录单独核验 |
| `audit/recovery_choices_formal80_v4/audit/recovery_choices_formal80_v4/SUMMARY.json` | 未在此源目录单独核验 |
| `audit/recovery_choices_formal80_v4/audit/recovery_choices_formal80_v4/publication_binding.csv` | 未在此源目录单独核验 |
| `audit/recovery_choices_formal80_v4/audit/recovery_choices_formal80_v4/recovery_choices.csv` | 未在此源目录单独核验 |
| `audit/recovery_choices_formal80_v4/formal_runs/03_abc_145502_clean/episode/agent_trace.jsonl` | 未在此源目录单独核验 |
| `formal_runs/03_abc_145502_clean/episode/agent_trace.jsonl` | `fc360250653915dab2d3823e45862b23d4777ec4a86640325e0e59f2acf2c834` |
| `formal_runs/resources.jsonl` | `16f06cefad068aeb38dc55d0c62b41d64794c95b7c90b515474d7601432360d8` |
| `videos/diagnostic_pyramid_regrasp.mp4` | `38100f523178da63e65ff1819523de59610c48d47c5cd09ee3c0f16fd5eb14cb` |
| `videos/final_pair_clean.mp4` | `39c32e8d8989c70d6cdc4e27277058cefc884215e44ab7c9d51a88236f939201` |
| `videos/final_pair_noise.mp4` | `3ece55b613490156f961f3b0d978dbf9d727929426fb3bedf0445527d25d9237` |
| `videos/final_recovery_or_three.mp4` | `4454ea5fc5ac93b46ac9fc1b20cac8f62844bc18590087b91bf3388be7730915` |
| `videos/index.html` | `d7edae0ca0795f1addf4c527df8bc524e6af358936398feac71ab52b5adedb8d` |
| `videos/precheck_cube_auto_failure.mp4` | `045bac85c5c4a7f4ab6fbe51f4e76fb75880532157fc77a50c08833f2437cba3` |
| `videos/precheck_cube_two_success.mp4` | `3a4a7c511ff1a6a356dfbd967edbb3e482d03f7bf74237d630890ccfc0a34728` |
