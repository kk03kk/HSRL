# LOCAL —— 本地 Windows 工作区报告

```yaml
status: draft
owner: local
updated: 2026-11-10 18:10
base_commit: e4d4dfc (+57 未提交条目)
```

## 1. 我是谁 / 我在做什么

本地 Windows（`D:\HSAI`）负责：代码开发、37.0.2 模拟器维护、实机桥接 `client/hsrl_agent/live/`、
真实酒馆战棋验证、本地小规模冒烟与证据留存。不做长训与大评测。

## 2. 本地实测状态（2026-11-10）

| 项 | 值 |
|---|---|
| 代码 HEAD | `e4d4dfc3c69d30bc10802c7a193861f448b0be52`，`main`，**dirty（57 条）** |
| 远程 | `https://github.com/kk03kk/HSRL.git` |
| V1.1 权重 | `artifacts/models/v1.1/model.pt` = `68a8a58743617bef004fef556bd263a4f4939e22b5f5f70352dfa697abc0848e` |
| 引擎 | `import hsbg_engine` OK，`N_CARDS=3984`；源码指纹 `c4c71014e3d41760` |
| 卡表 | `data/bg_cards.json`、`data/card_zhcn.json`、`data/p2/cards_37.json` 在位 |
| 实机 profile | `config/live/autoplay-255790.json`（reviewed，2560×1600，dpi 144，build 255790） |
| 战团库 | `artifacts/models/library-v1`：39 cohort，`index.csv` + `SHA256SUMS.txt` |
| 环境 | Python 3.13.5；torch 2.13.0 **cpu**（无 CUDA）；numpy 2.4.0；rapidocr 3.10.0；onnxruntime 1.31.1 |
| 依赖告警 | `numba 0.61.0` 需要 `numpy<2.2`，与 numpy 2.4.0 冲突（仓库未使用 numba） |
| 实机验证 | 2026-11-09 夜 7 局 + 2026-11-10 D 阶段 6 次 canary（含 2 局完整、1 局冠军） |

## 3. 本地独有（GitHub 与服务器都不应有）

- `artifacts/runs/live-*`：每局的 `decisions*.jsonl[.gz]`、`actions.jsonl`、`session-archives/`（Power.log 字节前缀）、截图、`result.json`、`health.json`。
- `artifacts/local/`：离线回放、审计脚本产出、D 阶段 canary 摘要。
- `records/evidence/`、`records/migration/`、`records/source-snapshots/`。

## 4. 本地缺、应该在服务器（请服务器 Agent 核实）

| 路径 | 本地 | 说明 |
|---|---|---|
| `artifacts/models/leaderboard-v1/reports/20261109-live-v1.json` | ❌ | 11 模型 live-v1 评测（NUS `1098876/1098877`） |
| `artifacts/models/leaderboard-v1/manifest-2026-11-09.json` | ❌ | 同上 |
| `records/research/autonomous-2026-11-08/TRAINING_REVIEW.md` | ❌ | 训练复盘（服务器侧） |
| `artifacts/runs/p50-forced-tier-long-20261108-135823/` | ❌ | p50 长跑输出 |
| 训练 Trainer / 30M–240M 快照（`52bcddbb…`、`caf0fe56…`、`68a8a587…`） | 仅 final 一份 | 其它快照应在服务器 |

## 5. 需要服务器回答的问题

1. `git rev-parse HEAD` 是多少？是否干净？是否等于 `e4d4dfc` 或包含本轮的 D 阶段提交？
2. `artifacts/models/` 下有哪些模型目录？各自的 `model.pt` SHA256（至少 final / 30M / 240M / v9786）？
3. `artifacts/models/leaderboard-v1/` 是否存在？内容清单与 `protocol_sha256`？
4. 战团库（`library-v1` 或其它）有几份、多大、是否与本地 39 cohort 吻合？
5. `engine/` 的源码指纹是否仍是 `c4c71014e3d41760`？是否已重新编译并验证 37.0.2 卡表？
6. 训练/评测 Slurm 作业与产物目录：哪些已完成、哪些是正式结果、哪些可清理？
7. 服务器上是否还残留旧项目路径（`~/projects/hearthstone-rl`、`AutoTavern` 等）或重复的代码副本？
8. 服务器是否有多余的 `artifacts/`/`records/` 副本？总占用多少、磁盘余量多少？

## 6. 我的承诺

- 不改服务器；不推 GitHub（等你和用户确认）。
- 本地已完成可逆整理：历史配置/旧工具配置/传输包/旧快照/论文 移入 `records/archive|source-docs`，根目录只剩 6 个文件；`.gitignore` 增加 `*.tgz` 与 `soclaas*`。
- 本地 `tools/sync_manifest.py` 已可复现生成清单；服务器请用**同一个脚本**生成 `manifests/server.json`。
