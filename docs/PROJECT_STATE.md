# HSAI 当前状态

## 0. 最新（2026-11-13 17:15 +0800）

- **自主开发期总账见 [`docs/analysis/AUTONOMY_SUMMARY_20261113.md`](analysis/AUTONOMY_SUMMARY_20261113.md)**
  （195 提交 / 120 作业 / 56 GPU-h / 26 个实验臂；资产、进步、负结果、未解问题都在里面）。
- **专家先验已落地**：[`docs/analysis/EXPERT_PRIORS_20261113.md`](analysis/EXPERT_PRIORS_20261113.md)
  + [`docs/EXPERT_QA.md`](../docs/EXPERT_QA.md)。核心测量结论：
  champion 升本曲线 **T2/T5 与高手标准一致、T8 之后停住**（T10 4.44 vs 真人 5.52），
  **每回合金币 p50 = 0**（从不存钱，升五本要 9 金），而 **0/33 个流派能在 4 本成型**
  （15 个最高核心在 6 本、16 个在 5 本）⇒ 结构上拿不到真核心。
- **H10 / L1 两个长臂已跑完并判定为负**（D-045）：`comp-target` 把覆盖率做上去
  （0.292→0.579）的代价是等级 4.44→4.07、板面 −11%、bot 场名次显著变差；
  `terminal-strength-coef` 也没能把板面做厚（板面类奖励第四次证否）。
- **主线已转向"课程而非奖励"**：`M0/M1`（`config/league-entry-m1.json` / `-m0.json`）——
  入场收窄到 **T8 单回合 + p75**、`comp-target`/`tier-target` 全关、
  M1 额外只注入**观测等级 ≥5** 的真人帧（新代码 `--entry-min-tier`，5 条单测 + 6k 库注入门禁全绿）。
  两个作业 `1132106`/`1132107` 正在排队（a100-40，5 段 × 20 min）。
- 新增主指标 **实际战斗胜率**（`combat_by_turn`）：champion T10 0.51 / T11 0.21，
  T9 之前 0.78–0.87 ⇒ 悬崖在 T10–T11（`ACCEPTANCE_BASELINE.md` §8）。
- 运维新规：**评测前必须 pin 快照**（`keep-snapshots 3` 会轮转掉），见 D-044。

---

更新：2026-11-13 01:20 +0800
阶段：**单 Agent 双环境架构已上线（本文 §1）；E1 冠军 `league-final-14858`（点估计第一）。
本轮定论：`tier-target` 过程奖励被打穿（tier 超额达成、板面 −33%、覆盖 −70%），根因是
**名次奖励已饱和**（`mean_rank 1.177`）。改为"先修环境"：B3 用 `human-board-sample auto`
加强对手、纯名次奖励；H1 检验阵容覆盖奖励能否抵消 tier 捷径。详见
[`docs/analysis/REWARD_GAMING_20261113.md`](analysis/REWARD_GAMING_20261113.md)、
[`docs/analysis/ACCEPTANCE_BASELINE.md`](analysis/ACCEPTANCE_BASELINE.md)。**

## 1. 运行位置与工作方式

- **单 Agent，两个执行环境**（详见 `docs/ARCHITECTURE.md`）：
  - 本地 Windows `D:\HSAI`：实机桥接与真实对局验证、实机证据、运维层 `ops/`、全量源码副本；
  - NUS `~/projects/HSAI`：模拟器、训练、评测、建库、数据、权重与服务器侧资产；
  - GitHub `kk03kk/HSRL`：**异地归档**，里程碑推送，不再是同步总线。
- 两端用 **git over SSH** 同步（`python ops/hsai_ops.py push|pull`）；服务器仓库设
  `receive.denyCurrentBranch=updateInstead`，有未提交改动时推送会被**拒绝**。
- 大文件（权重/库/回放）走 `ssh cat` 流式传输 + SHA256，实测 17.7 MB/s，不进 Git。
- 用户只给目标；Agent 自主规划、执行、验证、归档。开工先跑 `python ops/hsai_ops.py state`。

## 2. 当前作业

**2026-11-13 01:20 实测**：

| 作业 | 状态 | 说明 |
|---|---|---|
| `1124485 entry-h1-20261113` | RUNNING | G2 配方 + `comp-target 0.5`（阵容覆盖奖励首次启用）；6 段 × 20 min，约 03:10 结束 |
| `1124567 entry-b3-20261113` | RUNNING | 混合课程 + `human-board-sample auto`（T≤8 p50 / T9–12 p75 / T13+ p90）+ **纯名次奖励**；同样 6 段 |

已结束（2026-11-12 夜～13 凌晨）：

| 作业 | 状态 | 结论 |
|---|---|---|
| `1122988` E3 | COMPLETED | 入场局 +10~24%，整局更差 → 病灶是 tier tempo |
| `1123685` G2 | COMPLETED（1:59:13, step_015066） | tier 超额达成但板面 −33%、覆盖 −70% → **tier-target 被打穿** |
| `1123899` G4 | FAILED（续训守卫，代码改动后未续） | `mean_rank 1.833`，比 G2 更差 |
| `1123401` F1 / `1123634` G1 / `1123686` G3 | 撤销 | F1 三项全退；G1 改为 G2；G3 tempo 无变化（入场局里 T7 检查不触发） |

同步节奏说明（实测）：**每个 late-entry 作业各占 1 张 `h100-47`**；QOS `normal` 的按型号
配额是 `h100-47=2 / a100-40=2 / a100-80=1 / h100-96=1 / nv=8`，所以同时最多两个训练作业。
`a100-80` 池里有 aarch64 节点（`deploy/slurm/common.sh` 会拒绝，`ExitCode 2:0`）；
`nv` 池实测是 **TITAN V**（`xgpd0`），跑评测太慢，别用。

2026-11-11 22:20 盘点（历史）：

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
- 观测：新增字段级 gold test；本地实机的 obs/action 差异已写入 `docs/DECISIONS.md` D-026。
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

### 7.2 实机（待 D-026 取证）

- live pin 已是 `league-final-14858`（`config/live/model.json`，SHA `ba7e8bbc…`），
  但 `real_game_accepted: false`。
- 2026-11-12 夜间两局终局有效（第 6 名 / 第 5 名，12 回合），**未见相对 12311 的优势**
  （12311 那批是 2/2/3/3/4）。记录见 `docs/analysis/LIVE_NIGHT_RUN_20261112.md`。
- 与 D-026 一致：实机观测是 `public-approx-v1` 有损投影，训练用的是完整引擎观测。
  在取证并修复或明确接受之前，不建议把新模型直接上 E 阶段连续验收。

### 7.3 架构（2026-11-12 完成）

- 单 Agent 双环境管理已落地：`ops/hsai_ops.py` 是唯一远端入口，GitHub 降级为异地归档，
  `docs/sync/` 散文协议已退休（归档于 `records/archive/sync-protocol-20261112/`）。
- 两端 HEAD 一致、工作区干净；服务器 12 个未提交文件已审阅、验证并提交（`304cb0b`）。
- 验证证据（2026-11-12 14:14）：
  - 三方同步：本地 / NUS / GitHub 全部 `5f028c9`，两端 `dirty=0`；
  - `ops/hsai_ops.py doctor` 七点全通过；
  - 本地 `tools/verify_workspace.py --skip-rust` 全绿；服务器 `ops/check_docs.py` 干净、
    `tests/test_mirror_band.py tests/test_ops_guard.py` 30 passed；
  - 漂移守卫实测：服务器工作区脏 → 推送被拒；干净 → 推送被接受并更新工作区；
  - 模型传输实测：67.9 MB / 6.8 s，243 MB / 13.7 s（17.7 MB/s），逐文件 SHA256 一致；
  - 新 clone 一致性：`git archive HEAD` 743 文件 / 23.9 MB，其中 `ops/check_docs.py` 全绿。
- `docs/DECISIONS.md` D-026 仍然开放：实机观测/动作差异，待本地取证。

## 7.4 跨端遗留

- D-026：实机观测/动作投影差异，待本地用真实日志取证。
- 实机 pin 已更新为 `league-final-14858`，但 `real_game_accepted: false`；
  两局终局有效（第 6 / 第 5 名），尚未显示相对于 12311 的优势。

## 8. 硬约束

- **E1 评测只跑一轮**（6 对 × 448 局，4 模型排名）；报告 valid 后停止。不做第二轮复现、
  不做冠军挑战赛；只有 invalid/工程错误才重跑并先请示。

- 登录节点实测 **4 核 / 16 GB**、多人共享（旧文档的"约 1 GiB"是错的）；重任务只在
  Slurm 计算节点，`ops/hsai_ops.py` 的守卫默认拦截。
- 作业内不要包 `srun`；GPU 以原始 Driver UUID/Torch 实际计算为准。
- 新作业使用新输出目录；不覆盖历史证据。
- 工程错误、非法动作、截断、引擎异常、注入失败严格拒绝；正常棋力波动不自动止损。
- 不删除历史证据；不修改游戏目录；不 `reset --hard`/`git clean`/全仓格式化。
- 不同 `protocol_sha256` 不能混榜；p50/live-v1/pairwise Elo 不能混分。
