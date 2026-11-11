# HSAI 当前状态

更新：2026-11-11 10:05 +0800
阶段：**37.0.2 模拟器已冻结；36h 联赛训练进行中；唯一排行榜 `leaderboard-v2` 已建立（12311 第一）；等待实机 E 阶段授权。**

## 1. 运行位置与工作方式

- **三端分工**（详见 `docs/SYNC_MODEL.md`）：
  - 本地 Windows `D:\HSAI`：代码开发、模拟器维护、**实机桥接与真实对局验证**、证据留存；
  - GitHub `kk03kk/HSRL`（`main`）：唯一代码/配置/文档真源，不含大文件；
  - NUS `~/projects/HSAI`：训练、评测、建库、数据生成，服务器侧 Agent 独立工作。
- 本地活动工作区就是 `D:\HSAI`（2026-11-10 起恢复为双端同时活动；此前"不再使用本地"的说法已作废）。
- 用户只给大方向；两端 Agent 各自自主规划、执行、验证、归档，通过 GitHub 的 `docs/sync/` 交换结论。

## 2. 当前作业（服务器 2026-11-10 19:21 盘点）

| 作业 | 状态 | 说明 |
|---|---|---|
| `1102669 hsai-league-36h` | **RUNNING**（2×H100-47，05:01 起，预计 11-11 16:49 结束） | 72 段 × 30 min；09:49 时 segment **59/72**，`version=13806`、累计 `1.395B` 决策；`illegal/truncated/dropped=0`。**不要取消、不要改训练代码/库/依赖** |
| `1097487` | RUNNING | no-shell CPU allocation；服务器 Agent 会话所在；不要取消 |
| `1109300 hsai-gauntlet12311` | COMPLETED | 12311 vs 4 个 Elo 模型 ×128 局（512 局），协议 `e4182351…`；4 对全胜，305–207（59.57%） |
| `1099277 hsai-elo-8` | COMPLETED | 旧 8 模型 Elo，1,792 局，协议 `837231e0…`；已归档，不作当前榜 |
| `1107182 hsai-elo4-league` | COMPLETED | 4 模型 Elo，3,072 局，协议 `6bf5cae3…`；作为 `leaderboard-v2` 的旧模型锚点 |
| `1095539 hsai-p50-forced` | COMPLETED | 强制升本长训，`810,501,828` 累计决策 |
| `1098876/1098877` | COMPLETED | 11 模型 live-v1/dev 512 局；**旧引擎 `c7b5f27d`，不再作为当前结论** |

## 3. 最近结论

> **引擎口径警告（2026-11-10 服务器盘点确认）**：11 模型 live-v1 与 2026-11-09 榜单的
> `engine source_hash` 都是 **`c7b5f27d6f40d1d2`**；当前引擎是审计修复后的 **`c4c71014e3d41760`**。
> 下面所有 live-v1/p50 数字只能当**历史证据**，当前引擎棋力必须用新协议重跑（见 D-011）。

- `1095539` 从 7357 续训，新增 `251,628,096` 决策；`stop_reason=time_budget`，请求的 `958.9M` 未达到。
- p50 终评：final `2.234` vs 7357 `2.340`，Δ `-0.105`，CI 跨 0；win `+14.06pp`，top4 `-2.34pp`（旧引擎）。
- live-v1/dev 512（旧引擎）：没有任何模型显著优于 7357；`v9786` 点估计最好（`3.799`），差异不显著。
- 当前 run 的 live-v1 最佳是 30M `v7641`（`3.930`）；final `v9691` 为 `3.965`；60M `v7936` 最差（`4.557`）（旧引擎）。
- 240M `v9595` 在 p50 96 局最好（`2.031`，Δ `-0.510`），但 live-v1 为 `4.107`，排 8/11（旧引擎）。
- 240M `v9595` 会自主升本：T2 `93/93`、T5 `86/96`、T6 `43/45`、T8 `84/86`；T12 `0/48`，T14 `3/32`。它不是“不升本”，而是**前中期升得快、T12 停止**。
- 60M 强制阶段把 T12 tier6 率推到 `74.3%`，但 T12 战团 P50 只有 `920`，live-v1 退化到 `4.557`：**高本没有转化为核心战团强度**。
- **当前唯一榜（2026-11-11）**：`artifacts/models/leaderboard-v2/leaderboard.json`，5 个模型；
  `league-latest-12311` 以 **1067.68**（gauntlet-anchored，95% CI [1036.97, 1099.46]）排第一。
  它对旧第一 `milestone-1033972345` 直接 128 局为 69–59（53.91%，CI 跨 0.5），因此“第一”是点估计。
- 旧 8 模型 Elo `1099277`（1,792 局）与旧 4 模型 Elo `1107182`（3,072 局）保留为证据/锚点，不再作当前榜；
  旧榜文件归档在 `records/archive/leaderboards-20261111/`。

详细数据（**服务器侧路径，本地未同步，待确认**，见 `docs/INVENTORY_LOCAL.md` §5）：

- `records/research/autonomous-2026-11-08/TRAINING_REVIEW.md` §8–9（server-only / 待确认）
- `artifacts/models/leaderboard-v2/leaderboard.json`（当前唯一榜；旧榜已归档）
- `artifacts/runs/p50-forced-tier-long-20261108-135823/`（server-only / 待确认）

## 4. 当前有效资产

- 模拟器：`engine/`，酒馆 `37.0.2.256589`，原生动作 `245`，观测 `3227 + 124 cards`；审计后源码指纹 `c4c71014e3d41760`。
- 训练栈：`hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- 主续训起点：`7357` 完整 Trainer（累计 `558,873,732`、Adam `545,990`）。
- 当前模型（服务器实测）：`artifacts/models/v1.1/model.pt` `68a8a587…`（67,864,188 B，本地同 SHA）+ `trainer.pt` `dfcb59ce…`（254,312,937 B，**仅服务器有**）。
- 归档模型（`records/archive/models-2026-11-09/models/elo-8/`，120 个 checkpoint）：final 9691 `68a8a587…`、30M 7641 `52bcddbb…`、240M 9595 `caf0fe56…`、v9786 `988fdc7f…`、7357 `60594db0…`、10004 `32ec7a5c…`。
- 战团库（**按用途分开，D-012**）：
  - 服务器训练/评测真源 `artifacts/libraries/league-eot-v1.1-20261109`（43 MB，4 源，train/dev/test = 8,872 / 7,600 / 7,690 frames）；
  - 本地实机真源 `artifacts/models/library-v1`（39 cohort，3.4 GB，含 `index.csv` + `SHA256SUMS.txt`）；
  - 旧 bot 库 1.9 GB 已归档到 `records/archive/libraries-2026-11-09/`（可用数 0）。
- 评测产物：
  - **当前唯一榜**：`artifacts/models/leaderboard-v2/leaderboard.json`（5 模型；12311 1067.68 第一）。
  - 旧模型锚点：`artifacts/runs/elo4-v1.1-league-long-20261110/report-final.json`（3,072 局，协议 `6bf5cae3…`）。
  - 12311 gauntlet：`artifacts/runs/gauntlet-12311-128-20261110/report.json`（512 局，协议 `e4182351…`）。
  - 旧 8 模型报告：`artifacts/runs/elo-8-1099277/report-corrected.json`（1,792 局，协议 `837231e0…`，已归档为历史证据）。
  - 旧引擎历史：11 模型 live-v1 报告与 `manifest-2026-11-09.json` 已归档到
    `records/archive/models-2026-11-09/models/leaderboard-v1/`（引擎 `c7b5f27d`，**不可与当前引擎混用**）。
- 文档归档：`records/archive/`（按 `<topic>-<YYYYMMDD>/` 组织；此前文档里写的
  `records/archive/docs-2026-11-09/` 并不存在，已修正）。
- 三端规范：`docs/SYNC_MODEL.md`；本地盘点 `docs/INVENTORY_LOCAL.md`；
  跨端协作区 `docs/sync/`（`LOCAL.md` 与 `SERVER.md` 均已盘点，`manifests/` 见 D-014/D-017）。
- 2026-11-09 模拟器审计：37.0.2 数据/池/机制逐项核对；修复 `Jar o' Gems` 每战斗触发次数错误；新增回归测试；Rust 942 passed / 0 failed / 1 ignored；`verify_workspace` 全绿（Python 385 passed / 14 skipped）。报告见 `docs/SIMULATOR_AUDIT_2026-11-09.md`。
- 2026-11-09 Elo 准备 → **已执行**：8 模型 `1099277` 跑完 1,792 局（0 非法/0 截断/0 引擎异常）。
- 2026-11-09 实机审计：上一局 61/61 已发送动作有服务端回执、无内部异常；修复悬停期无关对手状态取消点击、训练 Move 规则不一致、5 选一未标定布局、同步 JPEG 阻塞和缺少逐决策记录。报告见 `docs/LIVE_V1_1_AUDIT.md`。
- 2026-11-10 D 阶段：新增确定性监督器（异常分级/失败预算/心跳）、批处理不再一票否决、未提交对账与已提交未采纳对账、名次稳定化（`placement_source`）、结算横幅续局、等待分类细化、归档去重索引；`tests/test_live_supervisor.py` 22 例全绿。实机 canary 已验证：完整一局第 2 名（终局名次与结果屏一致）、两次赛后续局、unsent 对账两次生效、批处理 S3 分类。细节见 `docs/LIVE_V1_1_AUDIT.md` §7。


## 5. 下一阶段

三条线并行，**都需要用户明确授权**：

### 5.1 实机 E 阶段（本地，最高优先级）
D 阶段已完成并停止。E 阶段按 `docs/LIVE_V1_1_AUDIT.md` §7 的建议分两步：
1. **E-0**：1 局 canary，确认 `retired_stale_inflight` 与 `rejected_reconciliation` 在真实对局中的行为（目前只有单测）。
2. **E-1**：3 局连续验收，口径：S0=0、每局 S1≤2、名次字段正确、续局无需人工、后期动作速率达标。

### 5.2 服务器：联赛训练与评测（进行中，**不要打扰**）
- `1102669 hsai-league-36h` 正在跑（05:01 起，预计 11-11 16:49 结束；09:49 时 segment 59/72，`version=13806`）；期间冻结训练代码/库/依赖，新实验用新目录、新作业号。
- 当前唯一榜 `leaderboard-v2` 已建立：12311 1067.68 第一；旧模型分数作为冻结锚点。后续新模型按 `docs/LEADERBOARD.md` 的 gauntlet 128×N 规则算分入榜。
- 待办：36h 结束后保存 final/评估 13806；需要时用**当前引擎**重跑 live-v1（旧 `46e7013d…` 是 `c7b5f27d` 引擎，不可混用）。

### 5.3 三端核对（本地 + 服务器）
1. 本地盘点与目录整理完成（`docs/INVENTORY_LOCAL.md`）。
2. **服务器盘点已完成**（`docs/sync/SERVER.md` + `manifests/server.json`，commit `711f604`）：作业、模型 SHA、库、评测产物、旧项目残留（无）、磁盘（16 TB 可用）全部有证据。
3. 待服务器做：用 v2 `tools/sync_manifest.py` 重跑一次清单（Git blob pins + 排除 build 目录），随后 `tools/sync_compare.py` 应输出一致。

## 6. 硬约束

- 登录节点只有 **1 GiB** 虚拟内存；重任务只能在 Slurm 计算节点。
- 所有 NUS 作业由代理提交、排队、监控和处理；不把结果传来传去。
- 作业内不要包 `srun`；保持 Slurm 隔离；GPU 以真实 Driver UUID 为准。
- 新作业使用新输出目录；不覆盖历史证据。
- 工程错误、非法动作、截断、引擎异常、注入失败严格拒绝；正常棋力波动不自动止损。
- 不删除历史证据，不修改游戏目录，不 `reset --hard`/`git clean`/全仓格式化。
- 不把 p50、live-v1、pairwise Elo 的绝对分数混榜。
