# HSAI 当前状态

更新：2026-11-09 09:25 +0800  
阶段：**37.0.2 模拟器已冻结；旧战团库已全部归档；8 模型 pairwise Elo 配置已准备，等待启动确认。**

## 1. 运行位置与工作方式

- 活动工作区：NUS `~/projects/HSAI`。
- 代理直接运行在 NUS 登录节点；所有开发、测试、评测、训练、Slurm 提交与监控都在服务器完成。
- 不再使用本地 `D:\HSAI` 作为活动工作区，不需要把结果传回本地；历史本地资料只作为归档证据。
- 用户只给大方向；代理自主规划、执行、测试、提交、监控和归档。

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

详细数据：

- `records/research/autonomous-2026-11-08/TRAINING_REVIEW.md` §8–9
- `artifacts/models/leaderboard-v1/reports/20261109-live-v1.json`
- `artifacts/runs/p50-forced-tier-long-20261108-135823/`

## 4. 当前有效资产

- 模拟器：`engine/`，酒馆 `37.0.2.256589`，原生动作 `245`，观测 `3227 + 124 cards`；审计后源码指纹 `c4c71014e3d41760`。
- 训练栈：`hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- 主续训起点：`7357` 完整 Trainer（累计 `558,873,732`、Adam `545,990`）。
- 当前长训：`1095539` final `68a8a587...`；240M `caf0fe56...`；30M `52bcddbb...`。
- live-v1 点估计最佳：`v9786` `988fdc7f...`。
- 评测 manifest：`artifacts/models/leaderboard-v1/manifest-2026-11-09.json`。
- 文档归档：`records/archive/docs-2026-11-09/`。
- 2026-11-09 模拟器审计：37.0.2 数据/池/机制逐项核对；修复 `Jar o' Gems` 每战斗触发次数错误；新增回归测试；Rust 942 passed / 0 failed / 1 ignored；`verify_workspace` 全绿（Python 385 passed / 14 skipped）。报告见 `docs/SIMULATOR_AUDIT_2026-11-09.md`。
- 2026-11-09 Elo 准备：8 个模型已 staging/审计（8/8 forward + card table 一致），28 对 × 64 局配置完成；16 局单对冒烟 0 非法动作/0 截断/0 引擎异常；正式协议 SHA `837231e0...`，等待启动确认。
- 旧战团库 `artifacts/libraries/*` 已整体移动到 `records/archive/libraries-2026-11-09/`，不再使用。
- 注意：修复改变源码指纹，旧 bot 库/评测 manifest 不能与新引擎混用；下一轮先逐帧复验旧库并发布新 manifest。


## 5. 下一阶段

用户已认可 2 model + 6 bot 的 pairwise Elo 方案，但要求**先完成文档整理，暂不急着评测**。

当前状态：方案已记录在 `docs/ELO_EVAL.md`；runner 尚未实现，尚未提交任何 Elo 评测。

下一步（等待用户启动确认）：

0. 确认 `config/eval-elo-8.json`、8 个模型 SHA、engine `c4c71014e3d41760` 和协议 SHA `837231e0...`。
1. 启动 `sbatch deploy/slurm/elo_8.sbatch`（1 × H100-47、8h、1,792 局）。
2. 审计 8 份 `model.pt` 的 SHA、架构、卡表、前向。
3. 先跑 16 局冒烟。
4. 再跑正式 28 对 × 64 局 = 1,792 局。
5. 输出 pairwise Elo、bootstrap CI、两两胜率矩阵、名次分布和门禁报告。

## 6. 硬约束

- 登录节点只有 **1 GiB** 虚拟内存；重任务只能在 Slurm 计算节点。
- 所有 NUS 作业由代理提交、排队、监控和处理；不把结果传来传去。
- 作业内不要包 `srun`；保持 Slurm 隔离；GPU 以真实 Driver UUID 为准。
- 新作业使用新输出目录；不覆盖历史证据。
- 工程错误、非法动作、截断、引擎异常、注入失败严格拒绝；正常棋力波动不自动止损。
- 不删除历史证据，不修改游戏目录，不 `reset --hard`/`git clean`/全仓格式化。
- 不把 p50、live-v1、pairwise Elo 的绝对分数混榜。
