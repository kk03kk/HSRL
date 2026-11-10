# SERVER —— NUS 服务器盘点报告

```yaml
status: confirmed        # 2026-11-10 19:55 由 local 复核确认（内容未改动，仅状态）
owner: server
updated: 2026-11-10 19:21 +0800
base_commit: 224cb49
reviewed_by: local @ 711f604
```

> 本文件只由服务器 Agent 写。盘点在 NUS 登录节点完成，未运行 torch，未启动/取消任何作业，
> 未删除/移动任何产物。所有路径相对 `~/projects/HSAI`。

## 0. 结论摘要

- 代码：`HEAD=224cb49a819896ad4e1b5dcca3be3da3c6256ed3`，已包含 D 阶段提交 `d10e352`；工作树 **dirty**，原因是运行中的联赛工作副本（未提交，见 §2）。
- 当前有 **1 个正式训练作业在跑**：`1102669 hsai-league-36h`（2×H100-47，已运行 14h20m）；另有 `1097487` no-shell CPU allocation，均**不要动**。
- 模型：`artifacts/models/` 只剩 V1.1 当前模型；final/30M/240M/v9786 已归档到 `records/archive/models-2026-11-09/`，SHA 完整可查（§3）。
- 排行榜：`artifacts/models/leaderboard-v1/` 只剩 V1.1 重置桩；`reports/20261109-live-v1.json` 与 `manifest-2026-11-09.json` 已归档，且锚定的是**旧引擎 `c7b5f27d6f40d1d2`**，不能与当前 `c4c71014e3d41760` 混用（§4）。
- 战团库：服务器上**没有**本地那份 39 cohort 的 `library-v1`；当前活跃库是 4 源 `league-eot-v1.1-20261109`（154 MB），旧 bot 库 1.9 GB 已归档且按 MANIFEST 标记 0 可用（§4）。
- 资源：`artifacts` 4.6 GB，`records` 19 GB，仓库 30 GB；`/home` 余量 16 TB（92% used，共享存储）。
- 旧项目：`~/projects` 下只有 `HSAI`，未发现 `AutoTavern` / `HSBG` / `hearthstone-rl` 独立副本（§8-Q7）。

## 1. 计费/资源现状

```text
$ du -sh artifacts records
4.6G    artifacts
19G     records
$ du -sh ~/projects/HSAI
30G     /home/h/hengzhi/projects/HSAI
$ df -h ~
Filesystem                              Size  Used Avail Use% Mounted on
cfs.comp.nus.edu.sg:/mnt/storpool/home  179T 163T   16T  92% /home
```

| 目录 | 大小 | 说明 |
|---|---:|---|
| `artifacts/exports` | 889 MB | 2 个旧传输包（`nus-1080772-closeout.tar.gz` 586 MB、`HSAI-1085131-research.tar.gz` 345 MB），附 `.sha256`，**清理候选，未删** |
| `artifacts/runs` | 3.0 GB | 联赛 36h 训练 2.0 GB + 源码库/评测/旧跑批 |
| `artifacts/local` | 318 MB | `v1.1-model.tgz` 276 MB、`v1.1-inference.tgz` 56 MB（与 V1.1 权重重复，**清理候选，未删**） |
| `artifacts/models` | 289 MB | V1.1 model+trainer + leaderboard 重置桩 |
| `artifacts/libraries` | 154 MB | 当前活跃的 10 个 model-EOT/联赛库 |
| `records/archive` | 19 GB | 旧模型 17 GB、旧库 1.9 GB、旧面板/文档归档 |
| `records/research` | 1.1 MB | 含 `autonomous-2026-11-08/TRAINING_REVIEW.md`（10,924 B，**确实在服务器**） |
| `records/long-runs` | 2.3 MB | 长跑摘要 |

> 存储配额未在登录节点暴露可读的 `quota` 结果；以上用 `df` 共享文件系统余量作近似。

## 2. 代码版本

```text
$ git rev-parse HEAD
224cb49a819896ad4e1b5dcca3be3da3c6256ed3
$ git log --oneline -5
224cb49 sync(local): refresh manifest after the reorganisation commit
fb572a9 Repo: three-end sync model, local inventory, docs rewrite and archive reorganisation
d10e352 Live V1.1: deterministic supervisor, delivery reconciliation, placement settle, auto-requeue
e4d4dfc Sync current HSAI server workspace and V1.1 action rules
65c714e Bind the long run to its own inputs(): fix the 9-second 1093247 failure
```

- 本轮 `git pull --rebase` 从 `e4d4dfc` 快进到 `224cb49`，共 3 个新提交（`d10e352`、`fb572a9`、`224cb49`）。
- **D 阶段已包含**：`d10e352` 新增 `client/hsrl_agent/live/supervisor.py`、`trace.py`，并改 `controller.py` / `execution_status.py` / `frame_gate.py` / `runtime.py` 等。
- 工作树 dirty：`1` 个已跟踪修改 + `10` 个未跟踪工作副本（加上本轮生成的 `docs/sync/manifests/server.json`）：

```text
 M hsbg/az/train/personal.py          # 允许 model_eot_v1.1 / model_eot_league 驱动
?? config/eval-league-3.json
?? config/eval-league-4.json
?? config/league-36h.json
?? deploy/slurm/elo_3_v1_1_league.sbatch
?? deploy/slurm/elo_4_v1_1_league.sbatch
?? deploy/slurm/league_36h.sbatch
?? deploy/slurm/league_combine.sbatch
?? deploy/slurm/league_source.sbatch
?? workflows/combine_league_library.py
?? workflows/eval_three_models_elo.py
```

> 这些是**正在运行的 36h 联赛作业的脚本与配置**。作业每 30 分钟启动新的 Python 进程，改文件会影响下一段，
> 因此在作业结束前**冻结**；本轮只提交 `docs/sync`，不提交这些工作副本。

## 3. 模型与权重

`artifacts/models/` 当前只有两个目录：`v1.1/` 与 `leaderboard-v1/`。
其余模型（120 个 checkpoint）已归档，路径见 `records/archive/models-2026-11-09/MANIFEST.json`。

| 模型 | 文件 | 字节 | SHA256 | 状态 |
|---|---|---:|---|---|
| V1.1 model | `artifacts/models/v1.1/model.pt` | 67,864,188 | `68a8a58743617bef004fef556bd263a4f4939e22b5f5f70352dfa697abc0848e` | 当前唯一上架 |
| V1.1 trainer | `artifacts/models/v1.1/trainer.pt` | 254,312,937 | `dfcb59ce785d71a215f756a134996f579a610d267270a79856d76e9458ede556` | 当前完整 Trainer |
| final 9691 | `records/archive/models-2026-11-09/models/elo-8/models/m1095539-v9691/model.pt` | 67,864,188 | `68a8a58743617bef004fef556bd263a4f4939e22b5f5f70352dfa697abc0848e` | 与 V1.1 同文件 |
| 30M 7641 | `.../elo-8/models/m1095539-v7641/model.pt` | 67,864,188 | `52bcddbb7ea0a3e7c3c322ac4247b92e0f88371d57427dc30e9fb6b227977642` | 已归档 |
| 240M 9595 | `.../elo-8/models/m1095539-v9595/model.pt` | 67,864,188 | `caf0fe568f0ef4d2a61b033e4e28e3097b0aaa4d155bb06b93a68b934010c6d4` | 已归档 |
| v9786 | `.../elo-8/models/m1085131-v9786/model.pt` | 67,864,188 | `988fdc7fedb8fc8d794b5db00914e58a69754900c3a3d7bc55e2745d608bd2a3` | 已归档 |
| 7357 | `.../elo-8/models/m1080772-v7357/model.pt` | 67,864,188 | `60594db00a627884d47afd4a326639ab9b1fa7c48eb88618399051d971dcff82` | 已归档 |
| 10004 | `.../elo-8/models/m1085131-v10004/model.pt` | 67,864,188 | `32ec7a5c249fd8591ccda4d40100cbb76664f2f20ffe1ebda6011ea02f428a4a` | 已归档 |

当前 36h 联赛训练产物（`artifacts/runs/league-v1.1-36h-20261109/`）：

| 模型 | 字节 | SHA256 |
|---|---:|---|
| milestone-867747936 | 67,864,188 | `010fc6fd7f50d052ce7c804b987214740e26a746f3bca20024189d7e19b0fcd6` |
| milestone-976909004 | 67,864,188 | `56c09ec886b1aa336e1b56aae87dd01ddc44348e0e462dccd3f9a03fd23f36da` |
| milestone-1033972345 | 67,864,188 | `03c00d0eb2e45f668d12da847cc66b178ffd483c8d6467f2c69e903569404308` |
| 最新快照 step_011801（生成于 19:16） | 67,864,188 | `3613e7923363b384ed6340be30fd1c83f4ce0646f06cfc70395f78b179abbba4` |

## 4. 战团库 / 数据集 / 评测产物

### 4.1 战团库

| 资产 | 路径 | 大小 | 结论 |
|---|---|---:|---|
| 当前活跃库（10 份） | `artifacts/libraries/*` | 154 MB / 3,419 files | 5 个联赛/校准库 + 5 个 smoke/validation 库 |
| 主训练库 | `artifacts/libraries/league-eot-v1.1-20261109` | 43 MB | `driver=model_eot_league`，4 源（V1.1/7357/9786/10004），train/dev/test = 8,872 / 7,600 / 7,690 frames |
| 旧 bot 库（5 份） | `records/archive/libraries-2026-11-09/` | 1.9 GB / 8,703 files | MANIFEST 标记 `usable_libraries_after_archive: 0`；不删 |
| 本地 39 cohort `library-v1` | — | — | **服务器不存在**；真源在本地，见 §6 |

主库校验：`validation.json` 记录 train/dev/test 均 `bad_files=0`、`illegal_actions=0`、`truncated_games=0`、`mirror_rejected=0`。
该库没有 `index.csv`，不是本地 `library-v1` 的 39 cohort 结构。

### 4.2 评测产物

| 文件 | 当前路径 | SHA256 / 状态 |
|---|---|---|
| V1.1 重置榜 | `artifacts/models/leaderboard-v1/leaderboard.json` | `f6ee1b86eace12d888e7c9e0854b18eb2a8960e6719378d954ebe84b035399a3` |
| 清单 | `artifacts/models/leaderboard-v1/manifest.json` | `72b6b426ced9f8fc498deee061020e89f3e18b2647b1f82e3cb57540544767bb` |
| 重置报告 | `artifacts/models/leaderboard-v1/reports/20261109-v1.1-reset.json` | `317fd1ca475c78faaa119f5d3e78f192dfecdacd380883f3735e8dfbc6be6dff` |
| 11 模型 live-v1 报告 | **已归档** `records/archive/models-2026-11-09/models/leaderboard-v1/reports/20261109-live-v1.json` | `5b7387180b927be6ac7815a410284072fe9814a2220e4107e15dcfd376320b49` |
| 2026-11-09 榜单清单 | **已归档** `records/archive/models-2026-11-09/models/leaderboard-v1/manifest-2026-11-09.json` | `d3b2ee71d676f1948aaa1e2108e1282aabbd54411c21ffb743a6403cb328e060` |

> **重要锚点警告**：归档 live-v1 报告的 `protocol_sha256=46e7013dea6638aa7f8bd55d39d61c0debc6e3f7d8e7a91b62ca0110479f4698`，
> 其内嵌 engine `source_hash=c7b5f27d6f40d1d2`；归档 manifest 同样是 `c7b5f27d6f40d1d2`。
> 当前引擎是 `c4c71014e3d41760`（2026-11-09 审计修复后）。**旧 live-v1 结果不能作为当前引擎结论，必须在新引擎上重跑后才可混用。**

当前引擎下有效的正式评测：

- 8 模型 pairwise Elo（`1099277`）：`artifacts/runs/elo-8-1099277/report-corrected.json`，1,792 局，0 illegal / 0 truncated / 0 engine_bad，协议 `837231e0d2f6e583549b095581e60e2986ac899511872b6e0798091b4dc85e94`，引擎 `c4c71014e3d41760`。
- 4 模型联赛 Elo（`1107182`）：`artifacts/runs/elo4-v1.1-league-long-20261110/report-final.json`，3,072 局，协议 `6bf5cae30094a8e021567b475740ca60f6c295da9e66844f1f8515b74e1f2faa`，引擎 `c4c71014e3d41760`。结果：milestone-1033972345 第一（1036.16），V1.1 第四（970.10）。

## 5. 作业与产物

当前在跑（**不要取消/不要改代码**）：

| 作业 | 状态 | 开始 | 说明 |
|---|---|---|---|
| `1102669 hsai-league-36h` | RUNNING 14h20m | 2026-11-10 05:01:50 | `xgpi20`，2×H100-47，32 CPU，96 GB，时限 37h；72 段 × 30 min |
| `1097487 no-shell` | RUNNING 1d19h | 2026-11-08 23:42 | 当前 opencode 会话所在 CPU allocation；不要取消 |

正式结果/主产物：

| 作业 | 状态 | 输出目录 | 正式结果 |
|---|---|---|---|
| `1095539 hsai-p50-forced` | COMPLETED 11:04 | `artifacts/runs/p50-forced-tier-long-20261108-135823/` | 是；checkpoint 已归档 |
| `1098876/1098877 hsai-lb-1109-a/b` | COMPLETED | `artifacts/models/leaderboard-v1/eval/`（已归档） | 旧引擎 live-v1，**不再有效** |
| `1099277 hsai-elo-8` | COMPLETED 1:00 | `artifacts/runs/elo-8-1099277/` | 是；1,792 局 |
| `1102644–1102647 hsai-league-src` | COMPLETED | `artifacts/runs/league-src-{v1.1,7357,9786,10004}-20261109/` | 联赛库源 |
| `1102668 hsai-league-combine` | COMPLETED 4:27 | `artifacts/libraries/league-eot-v1.1-20261109/` | 联赛主库 |
| `1102669 hsai-league-36h` | RUNNING | `artifacts/runs/league-v1.1-36h-20261109/` | 进行中 |
| `1107182 hsai-elo4-league` | COMPLETED 2:17 | `artifacts/runs/elo4-v1.1-league-long-20261110/` | 4 模型联赛 Elo |

可清理候选（**只记录，未删**）：`artifacts/exports/` 889 MB、`artifacts/local/v1.1-*.tgz` 333 MB、
`records/archive/v1.1-panel-*` 85 MB、旧 smoke/failed 目录（`elo-8-smoke-*`、`elo3-*`、`elo4-smoke-*`、`v1.1-panel-*`）。
任何删除需用户授权，并在目录留 `cleanup-<date>.txt`。

## 6. 与本地 manifest 的差异

两份 manifest：本地 `head=fb572a9`（dirty=false），服务器 `head=224cb49`（dirty=true）。
本地 manifest 生成于 `224cb49` 之前，所以 **HEAD 差异是本地清单过期，不是服务器回退**；
本地 `pull --rebase` 后应变成 `224cb49`，再重跑一次 `--side local`。

| 项 | 本地 | 服务器 | 判断 |
|---|---|---|---|
| `config/live/autoplay-255790.json` | `b8ea2eb1…` / 32,991 B | `3b0a4b76…` / 31,314 B | **CRLF/LF 工作树差异**，Git blob 同源；建议比 `git hash-object` |
| `data/bg_cards.json` | `f01b1fa5…` / 1,318,166 B | `a066afce…` / 1,267,818 B | 同上（CRLF） |
| `pyproject.toml` / `requirements.txt` | 713 / 71 B | 688 / 66 B | 同上（CRLF） |
| `artifacts/models/v1.1/*` | `68a8a587…` | `68a8a587…` | **一致** |
| `artifacts/models/library-v1/*` | 有（39 cohort） | **缺失** | 真实 drift；见 D-012 |
| `artifacts/models/leaderboard-v1/manifest-2026-11-09.json` | 缺失 | 缺失（已归档） | 两边当前路径都没有；归档件是旧引擎 |
| `engine/` 树 | 152 files / 4.8 MB | 852 files / 1.77 GB | 服务器含 `target/` 构建产物；源码以 Git 为准 |
| `artifacts/models` | 205 files / 9.6 GB | 9 files / 322 MB | 本地保留多份历史；服务器已归档到 `records/archive` |
| `artifacts/libraries` | 196 files / 360 MB | 3,419 files / 137 MB | 结构和用途不同，非同一资产 |
| `artifacts/{datasets,imports,archives}` | 有 | 无 | 本地独有；服务器不需要 |
| `artifacts/runs` | 1.4 GB | 5.0 GB（du 3.0 GB） | 各自运行证据；不要求相等 |
| `exports/`、`records/` | 脚本未统计 | 脚本未统计 | `tools/sync_manifest.py` 盲区；见 D-014 |

## 7. 我建议的三端分工调整

1. **保持现状的写权**：训练/评测/建库/`SERVER.md` 在 NUS；实机/`library-v1`/`LOCAL.md` 在本地；源码/文档在 GitHub。
2. **服务器不再声称拥有 `library-v1`**：训练用 `artifacts/libraries/league-eot-v1.1-20261109`，实机用本地 39 cohort 库；两套库不混用。
3. **旧 live-v1 结论降级**：在 `c4c71014` 上重跑 live-v1 之前，所有引用旧 `46e7013d…` 的排名只作历史证据。
4. **文档路径修正**：`PROJECT_STATE` §3/§4 引用的 `artifacts/models/leaderboard-v1/reports/20261109-live-v1.json` 与 `manifest-2026-11-09.json` 当前不存在，应改为归档路径或标注 `archived / old-engine`。
5. **清单脚本补盲**：建议把 `artifacts/exports` 与 `records/` 纳入统计；跨端比较改用 Git blob SHA，避免 CRLF 假差异。
6. **36h 作业期间冻结**：不 pull 训练代码、不改 `hsbg/`、`workflows/`、`library`、`.venv`；新实验用新目录/新作业号。
7. **清理只做候选**：`artifacts/exports`、`artifacts/local/*.tgz`、旧面板目录先记录，等用户授权后按 D-009 留清单再删。

## 8. 必答问题（对应 `LOCAL.md` §5）

### Q1. `git rev-parse HEAD` / 是否干净 / 是否已含 D 阶段改动？

- HEAD：`224cb49a819896ad4e1b5dcca3be3da3c6256ed3`，`main`，**dirty**（11 条：1 修改 + 10 未跟踪 + 本清单）。
- 已含 D 阶段：`d10e352 Live V1.1: deterministic supervisor, delivery reconciliation, placement settle, auto-requeue` 在历史中；新增 `client/hsrl_agent/live/supervisor.py`、`trace.py`。
- 证据：`git log --oneline -5`、`git status --porcelain=v1`、`git ls-tree -r --name-only HEAD client/hsrl_agent/live`。

### Q2. `artifacts/models/` 各模型 `model.pt` SHA256？

- 当前目录只有 V1.1；其余在 `records/archive/models-2026-11-09/models/elo-8/`。实测 `sha256sum`：
  - V1.1：`68a8a58743617bef004fef556bd263a4f4939e22b5f5f70352dfa697abc0848e`
  - final 9691：同 `68a8a587…`
  - 30M 7641：`52bcddbb7ea0a3e7c3c322ac4247b92e0f88371d57427dc30e9fb6b227977642`
  - 240M 9595：`caf0fe568f0ef4d2a61b033e4e28e3097b0aaa4d155bb06b93a68b934010c6d4`
  - v9786：`988fdc7fedb8fc8d794b5db00914e58a69754900c3a3d7bc55e2745d608bd2a3`
  - 7357：`60594db00a627884d47afd4a326639ab9b1fa7c48eb88618399051d971dcff82`
  - 10004：`32ec7a5c249fd8591ccda4d40100cbb76664f2f20ffe1ebda6011ea02f428a4a`
- 完整表格见 §3；所有归档件与 `MANIFEST.json` 记录一致。

### Q3. `artifacts/models/leaderboard-v1/` 是否存在？报告与 manifest 是否在？

- 目录存在，但当前只有重置桩：`leaderboard.json`、`manifest.json`、`reports/20261109-v1.1-reset.json`（SHA 见 §4.2）。
- `reports/20261109-live-v1.json`：**当前路径不存在**；归档在 `records/archive/models-2026-11-09/models/leaderboard-v1/reports/20261109-live-v1.json`。
- `manifest-2026-11-09.json`：**当前路径不存在**；归档在 `records/archive/models-2026-11-09/models/leaderboard-v1/manifest-2026-11-09.json`。
- 两者的协议/引擎锚点为 `46e7013d…` / `c7b5f27d6f40d1d2`，**旧引擎**；当前引擎 `c4c71014e3d41760` 下无效，需重跑。

### Q4. 战团库有几份、多大、cohort 数是否等于本地 39？

- 活跃库：10 份，154 MB，3,419 files；主训练库 `league-eot-v1.1-20261109` 为 4 源（V1.1/7357/9786/10004），train/dev/test = 8,872 / 7,600 / 7,690 frames。
- 旧 bot 库：5 份，1.9 GB，8,703 files，归档于 `records/archive/libraries-2026-11-09/`，MANIFEST 标记可用库为 0。
- **不等于本地 39 cohort**：服务器没有 `library-v1`，也没有 `index.csv`；39 cohort 库是本地实机资产。
- 证据：`du -sh artifacts/libraries records/archive/libraries-2026-11-09`、`manifest.json`、`validation.json`。

### Q5. `engine/` 源码指纹是否仍为 `c4c71014e3d41760`？是否重编译 37.0.2？

- 是。运行时探测：`PYTHONPATH=engine/python python -c "import hsbg_engine; ..."` 输出
  `game_version=37.0.2.256589`、`source_hash=c4c71014e3d41760`、`server_rules_version=37.0.2`。
- 构建：`engine/python/hsbg_engine/_core.abi3.so` 与 `engine/target/release/lib_core.so` 均为 **2026-11-09 08:55**，即审计修复后重编译。
- 版本锁：`config/engine-upstream-lock.json` 记录上游 `1c2086a997afde46d68c6c5992cab194ddcfce88`，`checked_on=2026-11-09`；卡表/词表 SHA 与 V1.1 manifest 一致。
- 注意：直接 `import hsbg_engine` 在登录节点未设置 `PYTHONPATH` 时会失败；激活 `.venv` 或用 `PYTHONPATH=engine/python` 即可。

### Q6. 已完成/进行中/可清理的 Slurm 作业与产物目录？

- 进行中：`1102669 hsai-league-36h`（RUNNING，14h20m，36h 分段训练）、`1097487 no-shell`（RUNNING，会话 allocation）。
- 正式完成：`1095539`、`1098876/1098877`、`1099277`、`1102644–1102647`、`1102668`、`1107182`（详见 §5）。
- 可清理候选：`artifacts/exports` 889 MB、`artifacts/local/v1.1-*.tgz` 333 MB、旧 smoke/failed 面板目录 ~100 MB；
  **本轮未删任何东西**。
- 证据：`squeue -u $USER`、`sacct -S 2026-11-01 -X -o JobID,JobName%32,State,Elapsed,AllocTRES%40`、`ls -la artifacts/runs/`。

### Q7. 是否残留旧项目路径或重复代码副本？

- 否。`~/projects` 下只有 `HSAI`；`find ~ -maxdepth 3` 未发现独立的 `AutoTavern` / `HSBG` / `hearthstone-rl` 目录。
- `~/.cache`、`~/.cargo`、`~/.rustup` 是工具缓存，不是项目副本；`records/archive`、`records/source-docs` 是有意保留的证据。
- 只有 1 个 Git worktree：`/home/h/hengzhi/projects/HSAI 224cb49 [main]`。

### Q8. `artifacts/`、`records/` 总占用与磁盘余量？

- `artifacts` 4.6 GB；`records` 19 GB；仓库总计 30 GB（含 `.venv`、`engine/target`）。
- `/home`：179 TB 总，163 TB 已用，**16 TB 可用**，92% used；inode 使用 2%。
- 证据：`du -sh artifacts records ~/projects/HSAI`、`df -h ~`、`df -i ~`。

## 9. 当前 36h 联赛作业（补充观察，不改动）

- 启动：`2026-11-10 05:01:50`，已完成 segment 28/72，segment 29 进行中（18:54 启动）。
- 最新：`version=11811`，`cum_decisions=1,103,012,914`，`history_len=2182`；最新快照 `step_011811`。
- 健康：逐 iter `illegal_actions=0`、`truncated_games=0`、`invalid_games=0`、`dropped_decisions=0`、`search_failures=0`、`restarts=0`；`forced_end_turns` 每 iter 1–2（记录，不当作工程错误）。
- 训练栈：V1.1 起点 + 4 源 model-EOT 联赛库，7 个严格 EOT 快照席，`sims=0`，`pipeline=1`，`move_guard=1`。
- 预计按 72×30 min 于 `2026-11-11 17:01` 左右结束，早于 37h 时限；下一里程碑约在 1.13B 决策。
- 已完成的 `1107182` 4 模型评测显示：最新里程碑（1.034B）显著优于 V1.1 与更早里程碑，但该评测只含 4 个模型；正式全榜仍需新协议。

## 10. v2 清单复核（2026-11-10）

- HEAD：`ceb1959`；dirty：**是**。原因是运行中的联赛冻结副本：`hsbg/az/train/personal.py`（1 个已跟踪修改）+ 10 个未跟踪文件；未改动、未提交。
- `sync_manifest.py --side server`：pins=15；**缺失项=无未解释缺失**。按 `scope` 允许的单边 pin：
  `artifacts/models/library-v1/*`（local）、`artifacts/models/v1.1/trainer.pt`（server）、
  `artifacts/libraries/league-eot-v1.1-20261109/manifest.json`（server）。
- `sync_compare.py` 结果：**`DRIFT`**（`compare_exit=1`）。产出：
  `docs/sync/manifests/compare-20261110.json`。
- 硬漂移（原样抄 compare 输出）：

```text
- pin config/live/autoplay-255790.json differs (b8ea2eb13e415940ddbb17d90a79a97b8ab9ed031e63d08e2dd779615ac830fc vs 3b0a4b769116888dbec24110f946dc84acef9bbbb56d8066e8aac08f851e508e; compared by worktree-sha256/worktree-sha256)
- pin data/bg_cards.json differs (f01b1fa509020323b3fdc64369ea3afb53e218bffd2c154463ef21ffd8d00768 vs a066afce04b6bcf6788495eb2eb7d6b02dc6fb173dc5759d79e244a4e7f0656a; compared by worktree-sha256/worktree-sha256)
- pin pyproject.toml differs (b7a12f24abfaa04d418b585cf5daea4a5d4989db58d805338e6edce118b40615 vs 62ff035b77dd7b80f8a9df846c8a30b6079ffa6870b1d8824554b80e9f73cc62; compared by worktree-sha256/worktree-sha256)
- pin requirements.txt differs (b1913b1a75e9cfd7ac1f8db1279db545c674e51f26f79506f7ab81b6a031cd7d vs ba296bf316e4587a87916099e6e36e5ac048c425baea01a6dbe1c44f9f99282d; compared by worktree-sha256/worktree-sha256)
```

- 复核判断：这 4 条是 **CRLF 假差异**，不是真实漂移。两份 manifest 里的 `git_blob` 完全一致：
  `autoplay=508b89ce1512…`、`bg_cards=124870badd63…`、`pyproject=9e9ac27e1e33…`、`requirements=30b8653fc42b…`。
- 原因在 `sync_compare.py` 的全局 `both_blob` 判断：它要求 **所有** 两侧同名 pin 都有 `git_blob`；
  但 `library-v1`（local）、`v1.1/trainer.pt`（server）、`league-eot`（server）按 `scope` 本就单边缺失，
  于是 `both_blob=False`，四个 CRLF pin 退化为 worktree-sha256 比较。建议本地改为**逐 pin** 判断
  “两侧都有 blob 就用 blob”，而不是全局 `all(...)`。
- 另一处观察：`sync_compare.py` 当前只遍历 `artifacts`，未输出 manifest v2 里新增的 `records` 段。
  本次量级直接取 manifest：server `records/archive` 22.4 GB / 9,854 files（含归档模型 17 GB + 旧库 1.9 GB）、
  `records/research` 1.6 MB / 23 files、`records/long-runs` 6.2 MB / 11 files；local `records/archive` 57 MB、
  `records/evidence` 5.3 GB、`records/source-docs` 17 MB。属于单边资产，不参与判定。
- artifacts 量级（server vs local）：models `322 MB vs 9,633 MB`、runs `5,016 MB vs 1,432 MB`、
  libraries `137 MB vs 360 MB`、exports `931 MB vs 0 MB`。均为 side-specific INFO。
- 36h 作业：`1102669 RUNNING 14:57:16 hsai-league-36h`；另有 `1097487 RUNNING 1-20:18:34 no-shell`。
  作业健康：最新 `version=11891`、`cum_decisions=1,114,645,875`，逐 iter `illegal=0` / `truncated=0` / `restarts=0`。
- 按约定，本次 compare 非 `consistent`，**未追加 D-018**；等本地修好 `both_blob` 逻辑后重跑，再决定是否记为通过。

## 11. 2026-11 全量回归（handoff 2026-11-10 P0）

执行：`1109107 hsai-regression`，节点 `xcnz5`，`normal`，8 CPU / 64 GB，2026-11-10 22:15:58 → 22:18:09（2m11s）。
输出：`artifacts/runs/server-regression-20261110-01/`（`pytest.log`、`cargo.log`、`verify.log`、`verify-validation.json`）。

```bash
python -m pytest tests/ -q \
  --ignore=tests/test_live_supervisor.py \
  --ignore=tests/test_live_execution_trace.py \
  --ignore=tests/test_live_resource_migration.py \
  --ignore=tests/test_live_simulator_check.py
cargo test --release --manifest-path engine/Cargo.toml -p hsbg-engine
python tools/verify_workspace.py
```

| 检查 | 结果 |
|---|---|
| pytest（排除 4 个 live 测试） | **FAIL**，9 failed / 0 errors |
| `cargo test --release -p hsbg-engine` | **PASS**，44 个 test binary，942 passed / 0 failed / 1 ignored |
| `verify_workspace.py` | **FAIL**，仅 `python` gate 失败；`sources` / `cards` / `engine-source` / `rust` 全 PASS |

pytest 失败清单（verify_workspace 的 python gate 相同）：

```text
FAILED tests/test_az_ppo.py::test_engine_mcts_selfplay_ppo_smoke
FAILED tests/test_curriculum_long.py::test_greedy_guard_preserves_buy_and_resets_per_game
FAILED tests/test_move_guard.py::test_policy_guard_off_is_bitwise_legacy
FAILED tests/test_move_guard.py::test_policy_guard_on_no_move_mask_is_bitwise_equal_to_off
FAILED tests/test_move_guard.py::test_policy_guard_inverse_and_repeat
FAILED tests/test_move_guard.py::test_policy_guard_cap_and_forced_fallback
FAILED tests/test_move_guard.py::test_policy_guard_multi_game_isolation
FAILED tests/test_move_guard.py::test_lockstep_meta_has_guard_stats
FAILED tests/test_parallel_collect.py::test_parallel_matches_serial_bot
```

失败原因：9 项全部触发同一条生产校验
`ValueError: 强制动作规则：max=80, move_guard=1, move_cap=5, move_repeat=2, forbid_cancel=1`。
这些测试仍用旧口径（`max_actions_per_turn=40`、`move_cap=1/2/3/99` 等），而 `PPOSelfPlayPolicy`
现在硬性要求 `max=80 / guard=1 / cap=5 / repeat=2 / forbid_cancel=1`。`tests/` 与 `hsbg/az/train/ppo.py`
在工作树里没有任何改动，因此这是**测试落后于代码**，不是生产路径或冻结副本导致的。

### 交接问题回答

1. **当前扩展 `build_info().source_hash`**：`c4c71014e3d41760`，与 `config/engine-upstream-lock.json`
   和源码指纹一致；`build_is_current=True`；`db_info()=(3983, "37.0.2.256589", "37.0.2")`，
   `N_CARDS=3984`。
2. **Windows 构建机**：服务器只有 Linux 计算节点，**没有 Windows 构建机**；Windows 轮子只能由本地
   按 S3 部署构建（MSVC）产出。
3. **pytest 耗时/已知失败**：`verify_workspace` 的 python gate 约 **49.5 s**，整次回归 **2m11s**；
   服务器侧已知失败 **9 个**（全为上述动作规则测试口径），不是 22 个。本机的 22 个可能包含本地
   live/环境相关失败；服务器排除 4 个 live 测试后为 9。

### handoff P0/P1 状态

- P0-1 拉取：已完成，HEAD `402480b`，已拿到 `docs/ARCHITECTURE.md`、`AGENTS.md` 更新和 `docs/analysis/` 三份报告。
- P0-2 全量回归：已执行并记录（本节的 RUN 1109107）；Python 失败原因已定位为测试口径落后。
- P0-3 引擎发版记录：已新建 `docs/sync/ENGINE_RELEASES.md`，首行当前 `c4c71014e3d41760`。
- P1-4 实机日志：尚未收到 `artifacts/imports/live-*`；到达后按约定校验规则，发现写回本文件，不进 Git。
- P1-5 文档归位：`docs/analysis/` 已在仓库；模拟器/训练类文档归服务器维护。
- 规则：按 `docs/analysis/ENGINE_RULES_FINAL_DECISION.md`，**不为对齐 HSBRSIM 改规则**。

### 清单与工作树

- `sync_manifest.py --side server` v2：HEAD `402480b`，pins=15，dirty=True（仅运行中联赛冻结副本）。
- `sync_compare.py`（本地已修 `both_blob`）：`result: consistent (with documented exceptions)`，exit=0；
  15 pins 全部 `OK` 或 `OK(scope)`；trees 全部 OK；artifacts 仅 INFO 量级差异。
  唯一 soft 提示是本地 manifest 仍指向 `c416d40`，本地 pull 后重跑即可消失。
- `git status`：除 1 个已跟踪修改 `hsbg/az/train/personal.py` + 10 个未跟踪联赛文件外干净；
  这些是 `1102669` 的冻结工作副本，未提交、未改动。

### 11.1 测试口径修复与绿色复跑（2026-11-10 22:33）

修复提交：`9cba1de test: align action-rule tests with V1.1 enforcement`。

改动：

- `tests/test_move_guard.py`：`_make_policy` 的 `max_actions_per_turn` 由 40 改为 80；
  删除已不可达的“关闭守卫逐位等价”用例，改为断言策略拒绝 `move_guard=False`；
  no-move mask 用无守卫 stub 基线验证；inverse/repeat/cap/multi-game 统一使用生产口径
  `cap=5 / repeat=2`；lockstep meta 断言 `move_cap=5`，并验证 lockstep 也不能关闭守卫。
- `tests/test_parallel_collect.py`：`max_actions_per_turn` 40 → 80。
- `tests/test_curriculum_long.py`：`PPOGreedyPolicy` 的 `move_cap` 由 1 改为 5；按生产口径断言
  `[197,197,8,203,203,8]`、`reset()` 后恢复换位；关闭守卫被拒绝。
- `tests/test_az_ppo.py`：MCTS `search_fn` 现在断言被 `sims=0` 规则拒绝；smoke 改跑无搜索自对弈，
  断言 `search_calls=0`、`has_pi=0`。

复跑：`1109179 hsai-regression2`，节点 `xcnf17`，2026-11-10 22:31:53 → 22:33:42（1m49s）。
输出：`artifacts/runs/server-regression-20261110-02/`。

| 检查 | 结果 |
|---|---|
| pytest（排除 4 个 live 测试） | **PASS**，0 failed / 0 errors |
| `cargo test --release -p hsbg-engine` | **PASS**，942 passed / 0 failed / 1 ignored |
| `verify_workspace.py` | **PASS** 全部 gate（sources / cards / engine-source / rust / python） |

结论：9 个动作规则测试口径已修复；服务器全量回归全绿。
