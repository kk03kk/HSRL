# HSAI 当前状态

更新：2026-11-10 18:20 +0800
阶段：**37.0.2 模拟器已冻结；V1.1 实机 D 阶段工程改造完成（含 6 次 canary），等待用户授权进入 E 阶段连续验收。**

## 1. 运行位置与工作方式

- **三端分工**（详见 `docs/SYNC_MODEL.md`）：
  - 本地 Windows `D:\HSAI`：代码开发、模拟器维护、**实机桥接与真实对局验证**、证据留存；
  - GitHub `kk03kk/HSRL`（`main`）：唯一代码/配置/文档真源，不含大文件；
  - NUS `~/projects/HSAI`：训练、评测、建库、数据生成，服务器侧 Agent 独立工作。
- 本地活动工作区就是 `D:\HSAI`（2026-11-10 起恢复为双端同时活动；此前"不再使用本地"的说法已作废）。
- 用户只给大方向；两端 Agent 各自自主规划、执行、验证、归档，通过 GitHub 的 `docs/sync/` 交换结论。

## 2. 当前作业

| 作业 | 状态 | 说明 |
|---|---|---|
| `1098924` | CANCELLED | A/B 4 臂；用户要求停止，未启动、无结果 |
| `1097487` | RUNNING | no-shell CPU allocation；当前 opencode 会话所在；不要擅自取消 |
| `1095539` | COMPLETED | 强制升本长训，`810,501,828` 累计决策 |
| `1098876/1098877` | COMPLETED | 11 模型 live-v1/dev 512 局评测 |

当前没有正式训练或评测作业在跑。

## 3. 最近结论

- `1095539` 从 7357 续训，新增 `251,628,096` 决策；`stop_reason=time_budget`，请求的 `958.9M` 未达到。
- p50 终评：final `2.234` vs 7357 `2.340`，Δ `-0.105`，CI 跨 0；win `+14.06pp`，top4 `-2.34pp`。
- live-v1/dev 512：**没有任何模型显著优于 7357**；`v9786` 点估计最好（`3.799`），但差异不显著。
- 当前 run 的 live-v1 最佳是 30M `v7641`（`3.930`）；final `v9691` 为 `3.965`；60M `v7936` 最差（`4.557`）。
- 240M `v9595` 在 p50 96 局最好（`2.031`，Δ `-0.510`），但 live-v1 为 `4.107`，排 8/11，显著差于 7357。
- 240M `v9595` 会自主升本：T2 `93/93`、T5 `86/96`、T6 `43/45`、T8 `84/86`；T12 `0/48`，T14 `3/32`。它不是“不升本”，而是**前中期升得快、T12 停止**。
- 60M 强制阶段把 T12 tier6 率推到 `74.3%`，但 T12 战团 P50 只有 `920`，live-v1 退化到 `4.557`：**高本没有转化为核心战团强度**。

详细数据（**服务器侧路径，本地未同步，待确认**，见 `docs/INVENTORY_LOCAL.md` §5）：

- `records/research/autonomous-2026-11-08/TRAINING_REVIEW.md` §8–9（server-only / 待确认）
- `artifacts/models/leaderboard-v1/reports/20261109-live-v1.json`（server-only / 待确认）
- `artifacts/runs/p50-forced-tier-long-20261108-135823/`（server-only / 待确认）

## 4. 当前有效资产

- 模拟器：`engine/`，酒馆 `37.0.2.256589`，原生动作 `245`，观测 `3227 + 124 cards`；审计后源码指纹 `c4c71014e3d41760`。
- 训练栈：`hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- 主续训起点：`7357` 完整 Trainer（累计 `558,873,732`、Adam `545,990`）。
- 当前长训：`1095539` final `68a8a587...`；240M `caf0fe56...`；30M `52bcddbb...`。
- live-v1 点估计最佳：`v9786` `988fdc7f...`。
- 评测 manifest：`artifacts/models/leaderboard-v1/manifest-2026-11-09.json`（**server-only / 待确认**）。
- 文档归档：`records/archive/`（按 `<topic>-<YYYYMMDD>/` 组织；此前文档里写的
  `records/archive/docs-2026-11-09/` 并不存在，已修正）。
- 三端规范：`docs/SYNC_MODEL.md`；本地盘点 `docs/INVENTORY_LOCAL.md`；
  跨端协作区 `docs/sync/`（`LOCAL.md` 已写，`SERVER.md` 待服务器 Agent 填写）。
- 2026-11-09 模拟器审计：37.0.2 数据/池/机制逐项核对；修复 `Jar o' Gems` 每战斗触发次数错误；新增回归测试；Rust 942 passed / 0 failed / 1 ignored；`verify_workspace` 全绿（Python 385 passed / 14 skipped）。报告见 `docs/SIMULATOR_AUDIT_2026-11-09.md`。
- 2026-11-09 Elo 准备：8 个模型已 staging/审计（8/8 forward + card table 一致），28 对 × 64 局配置完成；16 局单对冒烟 0 非法动作/0 截断/0 引擎异常；正式协议 SHA `837231e0...`，等待启动确认。
- 旧战团库 `artifacts/libraries/*` 已整体移动到 `records/archive/libraries-2026-11-09/`，不再使用。
- 注意：修复改变源码指纹，旧 bot 库/评测 manifest 不能与新引擎混用；下一轮先逐帧复验旧库并发布新 manifest。
- 2026-11-09 实机审计：上一局 61/61 已发送动作有服务端回执、无内部异常；修复悬停期无关对手状态取消点击、训练 Move 规则不一致、5 选一未标定布局、同步 JPEG 阻塞和缺少逐决策记录。报告见 `docs/LIVE_V1_1_AUDIT.md`。
- 2026-11-10 D 阶段：新增确定性监督器（异常分级/失败预算/心跳）、批处理不再一票否决、未提交对账与已提交未采纳对账、名次稳定化（`placement_source`）、结算横幅续局、等待分类细化、归档去重索引；`tests/test_live_supervisor.py` 22 例全绿。实机 canary 已验证：完整一局第 2 名（终局名次与结果屏一致）、两次赛后续局、unsent 对账两次生效、批处理 S3 分类。细节见 `docs/LIVE_V1_1_AUDIT.md` §7。


## 5. 下一阶段

三条线并行，**都需要用户明确授权**：

### 5.1 实机 E 阶段（本地，最高优先级）
D 阶段已完成并停止。E 阶段按 `docs/LIVE_V1_1_AUDIT.md` §7 的建议分两步：
1. **E-0**：1 局 canary，确认 `retired_stale_inflight` 与 `rejected_reconciliation` 在真实对局中的行为（目前只有单测）。
2. **E-1**：3 局连续验收，口径：S0=0、每局 S1≤2、名次字段正确、续局无需人工、后期动作速率达标。

### 5.2 pairwise Elo（服务器）
用户已认可 2 model + 6 bot 方案；runner 已实现、16 局冒烟通过，**正式 1,792 局等待启动确认**。
0. 确认 `config/eval-elo-8.json`、8 个模型 SHA、engine `c4c71014e3d41760`、协议 SHA `837231e0...`。
1. `sbatch deploy/slurm/elo_8.sbatch`（1 × H100-47、8h）。
2. 输出 pairwise Elo、bootstrap CI、两两胜率矩阵、名次分布和门禁报告。

### 5.3 三端核对（本地 + 服务器，本轮新增）
1. 本地已完成盘点与目录整理（`docs/INVENTORY_LOCAL.md`）。
2. 服务器 Agent 按 `docs/sync/SERVER.md` 盘点并填写 `SERVER.md` + `manifests/server.json`。
3. 双方在 `docs/sync/DECISIONS.md` 冻结规范，再决定迁移/清理动作。

## 6. 硬约束

- 登录节点只有 **1 GiB** 虚拟内存；重任务只能在 Slurm 计算节点。
- 所有 NUS 作业由代理提交、排队、监控和处理；不把结果传来传去。
- 作业内不要包 `srun`；保持 Slurm 隔离；GPU 以真实 Driver UUID 为准。
- 新作业使用新输出目录；不覆盖历史证据。
- 工程错误、非法动作、截断、引擎异常、注入失败严格拒绝；正常棋力波动不自动止损。
- 不删除历史证据，不修改游戏目录，不 `reset --hard`/`git clean`/全仓格式化。
- 不把 p50、live-v1、pairwise Elo 的绝对分数混榜。
