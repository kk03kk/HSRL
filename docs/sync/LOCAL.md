# LOCAL —— 本地 Windows 工作区报告

```yaml
status: confirmed
owner: local
updated: 2026-11-10 22:10
base_commit: 见本文件所在提交
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

## 7. 对服务器盘点的复核（2026-11-10 19:55，本地）

服务器提交 `711f604`（`SERVER.md` + `DECISIONS.md` D-010…D-015 + `manifests/server.json`）已拉取并逐条核对。

**确认成立**（本地复核无异议）：

1. 服务器 HEAD `224cb49` 已包含 D 阶段 `d10e352`；dirty 仅来自运行中联赛的 11 个冻结工作副本（D-013）。
2. V1.1 `model.pt` 两边同 SHA `68a8a587…`；服务器另有 `trainer.pt` `dfcb59ce…`（本地没有，续训以服务器为准）。
3. 归档模型 SHA 与本地记录一致（30M `52bcddbb…`、240M `caf0fe56…`、v9786 `988fdc7f…`、7357 `60594db0…`、10004 `32ec7a5c…`）。
4. **旧 live-v1 是旧引擎**（`c7b5f27d`）：本地 `README` / `PROJECT_STATE` 已按 D-011 降级为"历史证据"。
5. 战团库按用途分开（D-012）：训练用服务器 `league-eot-v1.1-20261109`，实机用本地 `library-v1`（39 cohort）。
6. 8 模型 Elo `1099277` 与 4 模型联赛 Elo `1107182` 是当前引擎下的有效结果；本地文档已更新。
7. 旧项目：服务器 `~/projects` 下没有 `AutoTavern`/`HSBG`/`hearthstone-rl` 副本；`records/archive` 19 GB 为归档。
8. 资源：`artifacts` 4.6 GB、`records` 19 GB、仓库 30 GB、`/home` 16 TB 可用。

**接受并已在本地实施**（D-014）：`tools/sync_manifest.py` 升到 v2 —— pins 同时给 worktree SHA256 与
**Git blob**；统计加入 `artifacts/exports` 与 `records/`；排除 `target/`、`__pycache__` 等构建目录；
新增 `tools/sync_compare.py` 输出 drift 报告。本地 v2 清单已提交（`manifests/local.json`）。

**需要服务器补一步**：用 v2 脚本重跑一次

```bash
python tools/sync_manifest.py --side server --out docs/sync/manifests/server.json
python tools/sync_compare.py docs/sync/manifests/local.json docs/sync/manifests/server.json
```

v1 清单没有 `git_blob`，所以四个 CRLF 文件（`autoplay-255790.json`、`bg_cards.json`、
`pyproject.toml`、`requirements.txt`）目前只能按 worktree SHA 比较、会显示假差异；重跑后应全部 `OK`。

## 8. 架构调整后的本地范围（2026-11-10）

- 本地**只做**：client/ 实机桥接、config/live/、实机证据、docs/sync/LOCAL.md、docs/live/**、docs/analysis/**。
- 本地**不再做**：改 engine//hsbg//workflows//deploy//data//	ools/、本地编译 Rust、跑训练栈测试、规则审计。
- 自检：python client/tools/check_ownership.py --strict（越界改动会列出来，不会自动回滚）。
- 运行时只需要：client/** + hsbg/env/{obs_std,action_space_std}.py + hsbg/az/train/common.py + hsbg_engine 轮子 + numpy/torch/PIL/rapidocr/onnx。
- 已知：本地 hsbg_engine 仍是 2026-11-06 构建（c7b5f27d），按 S3 由服务器发版后用部署构建对齐。

## 9. 统一对局记录 v2（2026-11-11，本地）

模拟器与实机现在写**同一套**记录：`hsai.game-trace/v2`，规范与审计见
[`docs/analysis/TRACE_UNIFIED_20261111.md`](../analysis/TRACE_UNIFIED_20261111.md)。

- 代码：`client/hsrl_agent/trace_schema.py`（规范，纯 stdlib）、`client/hsrl_agent/live/trace.py`
  （单写线程）、`client/tools/sim_decisions.py`（模拟器生产者）、`client/tools/trace_audit.py`（离线审计）。
- 流分离：`decisions` / `actions`(execution) / `waits` / `events` / `decisions-full.gz` /
  `observations.npy` / `decisions-resolved` / `audit.json`。
- 旧的实机日志 92% 是等待轮询（753 行里 666 行）。按新规则重放同一局：**87 条决策、69 条有效动作**，
  T1 从 50 条降到 3 条（buy/play/end）。
- 模拟器整局验收通过：`league-latest-12311` + 引擎 `c4c71014e3d41760`，seed 700001，
  第 7 名（T11 淘汰，对局 17 回合），**131 条决策 / 131 条执行 / 131 条输入向量，audit complete=true，
  0 丢事件 0 写错误**。产物：`artifacts/local/sim-trace-20261111-001/`。
- **需要服务器端决定**（本轮只记录、未改规则）：模拟器里 `Freeze` 是开关（T9/T10 各 10 次原地打摆），
  且需要目标的法术 `Play` 有 68% 是空操作；实机桥接有"已全冻不给 Freeze"的保护，模拟器没有。
  MoveGuard 只挡换位，挡不住这两类空转。
- 实机侧只做了代码修复与离线测试（71 条 `test_live_*`+`test_client_contracts` 全绿），
  下一局真实游戏按新格式落盘后跑 `trace_audit.py check <match-dir>` 验收 `complete=true`。
- **面板对齐 + 规则对齐复跑（2026-11-11 13:50，第三次）**：7 席=**引擎 bot**（`game.bot_action`，无对手库）、
  采样前清 **Cancel(7)**（长训 `forbid-cancel=1`）、商店全冻时**不提供 Freeze**（实机 `actions.py:77` 同规则）。
  同模型同 seed：**第 1 名 / 15 回合 / 215 决策 / 868 个 bot 动作 / `complete=true` / 12.4 s**；
  `freeze` 60→9、`freeze_masked` 11、`cancel_masked` 40。产物 `artifacts/local/sim-trace-20261111-003/`。
- **更正上一版结论**：此前说的"法术 `Play` 有 68% 空操作"是**记录缺陷**，不是规则 bug。
  引擎是**两步施法**（`Play` → `pending=Target` → `Choose(k)` 选目标），37/37 次"无变化"的 play
  后面都紧跟一条 `choose` 且属性变化发生在那一步。记录层已修：`delta` 增加 `pending_kind`，
  两步动作用 `parent_decision_id` 串链（本局 44 条），`Target` 的 options 是实体 id 不是卡牌 id。
- 对照：7 席自对弈时同 seed 是第 7 名 → **名次对面板的敏感度远大于对模型的敏感度**。

