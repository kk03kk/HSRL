# HSAI 当前状态

更新：2026-11-11 22:20 +0800
阶段：**36h 联赛已完成并通过全链路审计；E1 校准 r1 有效（final 点估计第一，但与 13806 不显著）；
r2 复现已完成并与 r1 一致；按用户规则不做挑战赛。下一轮长训未启动，等待用户确认。**

## 1. 运行位置与工作方式

- **三端分工**（详见 `docs/SYNC_MODEL.md`）：
  - 本地 Windows `D:\HSAI`：实机桥接与真实对局验证、实机证据、Windows 轮子部署构建；
  - GitHub `kk03kk/HSRL`（`main`）：唯一代码/配置/文档真源，不含大文件；
  - NUS `~/projects/HSAI`：模拟器、训练、评测、建库、数据、权重与服务器侧资产。
- 本地活动工作区就是 `D:\HSAI`；两端通过 `docs/sync/` 交换结论。
- 用户只给大方向；两端 Agent 各自自主规划、执行、验证、归档。

## 2. 当前作业（2026-11-11 22:20 盘点）

| 作业 | 状态 | 说明 |
|---|---|---|
| `1102669 hsai-league-36h` | **COMPLETED** | 72/72 段；`2026-11-11T16:46:33+08:00` 结束；elapsed `1-11:44:43`；final `v14858`、累计 `1,552,463,499`；`illegal/truncated/dropped=0` |
| `1115858 hsai-e1-four`（r1） | COMPLETED | E1 四模型 round-robin，2,688 局，valid/eligible；报告 `artifacts/runs/e1-four-20261111/report/report.json` |
| `1116285 hsai-e1-four`（r2） | COMPLETED | 同 config、冻结代码复现；6 对与 r1 逐对完全一致 |
| `1116737 hsai-e1-challenge` | FAILED | 448 局完成（final 243–205），但 2 模型 BT bootstrap 报错未写 report；已修 fallback，未再正式重跑 |
| `1117087 hsai-e1-challenge` | CANCELLED | 用户要求停止重复评测，已取消 |

## 3. 36h 联赛复盘（已完成）

- 配置：`config/league-36h.json`；512 局/轮、8 actor 进程、`pipeline=1`（lag=1）、
  `sims=0`、1 学习席 + 7 严格镜像席、engine `c4c71014e3d41760`。
- 稳定性：5,316 条 history 全部 `illegal/truncated/dropped/invalid=0`、worker errors=0、
  无非有限损失；4 次 KL 早停，0 次零 optimizer-step。
- 学习曲线：policy loss ~0.0025–0.003，value loss 1.22→0.89，entropy 1.44→1.41，
  approx_kl ~0.006，clip_frac ~0.067；anchor_kl 0.022→0.227；LR 第 10000 轮后固定 1.667e-6。
- 资源：trainer 在 cuda:0、8 个 actor 在 cuda:1，两张 H100-47 都使用；
  decisions/s 从 ~5.6k 升到 ~7.2k（对局变长），wall ~20–21 s/iteration。
- **审计发现的两个主要训练问题**（详见 `docs/AUDIT_2026-11-11.md`）：
  1. p75 + 单一来源席导致 7 个镜像席在 turn 8+ 拿到同一张棋盘；有效唯一棋盘约 260 个，
     不是 8,872 个。已新增 `p75band`（turn 8–16 恢复到 ~52 帧/turn）。
  2. Freeze 动作被刷满引擎 10 次上限；final 的 Freeze 占决策 22%，
     实测 `forbid_unfreeze` 规则不损失胜率、减少 16% 决策。训练入口已支持，默认关闭。

## 4. E1 校准（当前唯一正式评测口径）

正式报告：`artifacts/runs/e1-four-20261111-r2/report/report.json`（r2；与 r1 六对逐对一致）
正式榜：`artifacts/models/e1-four-20261111/leaderboard.json`
协议：`0fbf28d2f82c08062291ef6526275507dfeffa1f5db85159ac7531437f3f723d`
（6 对 × 448 局，ordered56，shared 种子 seed-block bootstrap，含 code SHA）

| 排名 | 模型 | Elo | 95% CI | 排名 95% | 平均名次 | 冠军率 |
|---:|---|---:|---|---:|---:|---:|
| 1 | `league-final-14858` | **1086.53** | [1062.49, 1111.01] | [1, 2] | **1.589** | 51.34% |
| 2 | `league-latest-13806` | 1068.45 | [1045.78, 1091.47] | [1, 3] | 1.644 | 49.48% |
| 3 | `league-latest-12311` | 1055.92 | [1032.16, 1079.91] | [2, 3] | 1.681 | 47.17% |
| 4 | `V1.1` | 1000.00 | [1000, 1000] | [4, 4] | 1.876 | 36.68% |

- **final 是点估计第一**；对 13806 为 240–208（53.57%，CI 跨 0.5），不是显著超越；
  对 12311 为 243–205（54.24%）；对 V1.1 为 275–173（61.38%）。
- 13806 对 12311 为 237–211（52.90%）；三者是统计上接近的领先集团，V1.1 明显最后。
- `league-latest-12311` 的旧 v2 分数 1067.68 与 E1 的 1055.92 不能混榜（协议不同）。
- r2 已完成 6 对，与 r1 **逐对完全一致**；正式榜用 r2 报告构建；
  **冠军 = `league-final-14858`**（点估计第一；按用户规则不做挑战赛）。

## 5. 当前有效资产

- 模拟器：`engine/`，酒馆 `37.0.2.256589`，源码指纹 `c4c71014e3d41760`，动作 245，
  观测 3227 + 124 cards。
- 训练栈：`hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- 36h final：`artifacts/models/league-final-20261111-step014858/`
  - model `ba7e8bbc…`（67,864,188 B），trainer `c5262f26…`（254,312,937 B），
    `checkpoint.json` + `MODEL_CARD.md` + manifest 完整。
- 候选：`league-latest-13806`（`77d3fe54…`）、`league-latest-12311`（`5069aa62…`）、
  `league-latest-14556`（`14d7c4d5…`，未入 E1）。
- 下一阶段 fork：`artifacts/models/stage2-proposal-20261111/`（从 final 完整分叉；
  LR 视野 25000、anchor=final；`workflows/fork_stage.py` 生成）。
- 战团库：`artifacts/libraries/league-eot-v1.1-20261109`（43 MB，train/dev/test =
  8,872 / 7,600 / 7,690 frames）。
- 审计产物：`docs/AUDIT_2026-11-11.md`、`artifacts/runs/audit-behavior-20261111-02/`、
  `artifacts/runs/audit-freeze-20261111-01/`、`artifacts/runs/audit-unfreeze-20261111-01/`、
  `artifacts/runs/audit-scaling-20261111-01/`、`artifacts/runs/audit-regression-20261111-01/`。

## 6. 审计修复摘要

详见 `docs/AUDIT_2026-11-11.md`。已实施并通过回归：

- 镜像：`p75band` 模式 + 桶内排序缓存；训练入口支持。
- Freeze：`forbid-unfreeze` 训练/评测规则（默认 0；A/B 不损胜率，-16% 决策）。
- PPO：零 step 不推进 LR/version；预构建 advantages 不再重算；nonfinite/空 mask 显式报错。
- 进程池：`weights_loaded` 泄漏修复；reload 版本/标准化一致性。
- 评测：Elo→sigmoid 尺度、shared seed-block bootstrap、`_fit_bt` 失败显式报错、
  code SHA、ordered56 座位校验、challenge 校验、rank CI 冠军规则。
- 观测：新增字段级 gold test；本地实机的 obs/action 差异已写入 `docs/sync/DECISIONS.md` D-026。
- 回归：`1115905` `verify_workspace` 全绿（Rust 942 passed / Python 全绿）。

## 7. 下一阶段

### 7.1 下一轮长训（未启动，需用户确认）

推荐方案（`config/league-stage2.json` + `deploy/slurm/league_stage2.sbatch`）：

1. 从 `artifacts/models/stage2-proposal-20261111/` 开始（final 完整 fork；
   LR 2e-6→1.667e-6/25000 iter；anchor=final）。
2. `mirror-sample=p75band`；`forbid-unfreeze=1`；`seed=320000000`；8 processes；
   512 局/轮；30 min × 72 段；2×H100-47（trainer cuda:0 / actors cuda:1）。
3. 每 4 段 dev 32 局、36/72 段 test 32 局；监控 mirror 多样性、freeze 比例、
   anchor_kl、entropy、decisions/s、illegal/truncated。
4. 先用短程（≤2h）确认 p75band + forbid-unfreeze + stage fork 的实际吞吐/稳定性，
   再决定是否进入正式长训。

### 7.2 实机（等待用户授权）

- 当前 live pin 仍是 V1.1；**服务器已交接 final 14858 的 live 包与本地步骤**
  （`docs/sync/SERVER_TO_LOCAL_20261111.md`），等待本地 E-0 canary。
- 本地实机 obs/action 投影存在 D-026 列出的近似/差异，需要本地用真实日志确认；
  在修复或明确接受前，不建议把新模型直接上 E 阶段连续验收。

### 7.3 跨端

- `docs/sync/DECISIONS.md` D-026：实机观测/动作差异，待本地取证。
- E1/协议/审计文档为服务器写；本地只读。GitHub 是同步中心，未 commit/push（按规范等待用户）。

## 8. 硬约束

- **E1 评测只跑一轮**（6 对 × 448 局，4 模型排名）；报告 valid 后停止。不做第二轮复现、
  不做冠军挑战赛；只有 invalid/工程错误才重跑并先请示。

- 登录节点约 1 GiB 虚拟内存；重任务只在 Slurm 计算节点。
- 作业内不要包 `srun`；GPU 以原始 Driver UUID/Torch 实际计算为准。
- 新作业使用新输出目录；不覆盖历史证据。
- 工程错误、非法动作、截断、引擎异常、注入失败严格拒绝；正常棋力波动不自动止损。
- 不删除历史证据；不修改游戏目录；不 `reset --hard`/`git clean`/全仓格式化。
- 不同 `protocol_sha256` 不能混榜；p50/live-v1/pairwise Elo 不能混分。
