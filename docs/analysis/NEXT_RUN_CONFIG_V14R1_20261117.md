# 下一轮训练 · 完整配置审核单（v1.4-r1）

状态：**待用户审核，未提交**。生成时间 2026-11-17。
参数真值来源：`deploy/slurm/stage_arena.sbatch`（脚本内硬编码 + `${HSAI_ARENA_*}` 默认值）。
标注约定：**★ = 本轮用户要求的改动**；**△ = 相对 r3 训练时的实际行为变化**（代码修复所致，非用户要求）。

---

## A. 作业与资源

| 项 | 值 | 来源 |
|---|---|---|
| 脚本 | `deploy/slurm/stage_arena.sbatch` | 固定 |
| 分区 / GPU | `gpu` / `h100-47` **2 片 MIG**（各 46.4 GiB） | 脚本 |
| CPU / 内存 / 墙钟 | 32 核 / 96 GB / **3:00:00** | 脚本 |
| 段数 / 单段时长 | `HSAI_ARENA_SEGMENTS=1` / `HSAI_ARENA_MINUTES=165` | 默认 |
| GPU 落位 | MIG UUID 重定向，learner=cuda:0，8 个采集进程=cuda:1 | 脚本 + D-063 |
| 代码版本 | 服务器 HEAD（本轮为 `3ad5a43` 或之后） | 脚本写进 `run.env` |

## B. 起点权重与续训方式

| 项 | 值 |
|---|---|
| `HSAI_ARENA_CKPT` | **`artifacts/models/arena-v1.3-r3/model.pt`**（sha256 `1808f178…c206472`） |
| 续训语义 | `--checkpoint` 只继承**权重**，**重建 Adam + 调度器**（刻意如此，为拿到论文的 6e-06，见 D-048） |
| 不带 `--trainer-state` | 带了会用 state 里的 `base_lrs` 覆盖 `--lr`（会变成 1.667e-06） |
| 输出目录（建议） | `artifacts/runs/nus-arena-v1.4-r1-20261117`（必须不存在） |

> ⚠ 脚本里 `HSAI_ARENA_CKPT` 的**默认值仍是 `arena-v1.2/model.pt`**。本轮必须显式覆盖，否则会从 v1.2 起训。

## C. 场地：战团库（对手战力与规则）

**训练用库**：`artifacts/libraries/arena-cal-v14-snap`
- 1 个文件，**16,658 帧**，`fidelity=full`（完整席位快照：战团 + 手牌 + 商店 + 金币 + 本数）
- 快照来自真实游戏（Firestone 回放重建），按回合分层

**评测用库（不参与训练）**：`arena-cal-v14hold-snap`，2,355 帧（与训练库帧签名交集 0）

**注入方式**：`--mirror-seats 7 --mirror-p 1 --mirror-sample p50band --mirror-strict 0 --mirror-max-turn 0`
即：**每回合**给 7 个对手席位各注入一条同回合快照；注入后该席位 `ended_turn=true`（不会再行动，只作为战斗对手）。

**逐回合目标战力（总攻+总血，p50band = 该回合 45%–55% 分位带）**

| 回合 | 1–2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 帧数 | 951 | 993 | 1356 | 1721 | 1544 | 1493 | 1139 | 790 | 643 |
| **p50** | 5 | 16 | 25 | 40 | 65 | 99 | 159 | 346 | **900** |
| band 下–上 | 5–5 | 16–16 | 24–25 | 39–40 | 64–65 | 98–100 | 157–161 | 343–349 | 890–907 |

| 回合 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 帧数 | 610 | 560 | 538 | 554 | 500 | 498 | 462 | 457 | 450 | 448 |
| **p50** | 1588 | **3000** | 4509 | 5965 | 7012 | 7928 | 8969 | 9933 | 10943 | 11943 |
| band | 1574–1608 | 2961–3019 | 4465–4563 | 5908–6007 | 6934–7062 | 7860–7986 | 8872–9057 | 9812–10007 | 10842–11057 | 11835–12058 |

> 这张表就是"这一轮要打多强的对手"。它与上一轮（r3 训练时用的 v14-snap）**完全相同**——本轮**不改场地强度**。

## D. PPO 超参（逐项）

| 项 | 值 | 说明 |
|---|---|---|
| 学习率 | `6e-06` | 论文 Table 7 |
| lr 退火 | `--max-updates 300`、`--lr-min-ratio 0.8333` | 300 轮线性到 5e-06 |
| 有效 batch | `games 512 × epochs 1`；minibatch `1024` × grad-accum `8` = **8192** | |
| clip | `0.2` | |
| target-KL | `0.05`（1.5× 早停） | |
| γ / λ | `1.0` / `0.995` | |
| entropy | `0.002` | |
| grad-clip | **`1.0`**（PPOConfig 默认 0.5，脚本显式覆盖） | |
| value coef / loss | `0.5` / `rank`（CE，V(s)=Σp·u） | 代码默认 |
| KL anchor | β `0.1`，前 `100` 轮，tail `0.02`，`--reset-anchor 1` | |
| sims | `0` | 冻结契约 |
| AMP | `--amp bf16 --leaf-amp bf16 --anchor-amp 1` | |
| 采集 | `--lockstep 1 --processes 8 --workers 1 --threads 1 --pipeline 1 --leaf-fused 1 --records-dtype fp32` | |
| 快照 | `--snapshot-every 5 --keep-snapshots 3` | |
| 教师 CE 归一化 | `labeled`（代码默认，与历史一致） | △ 现在会打印 `teacher_label_share` |
| 健康门 | `--health-gate 1`（代码默认） | △ 新增 |

## E. 奖励（本轮唯一奖励 = 终局名次效用）

| 项 | 值 |
|---|---|
| `--rank-utility-table` | **`1.00, 0.35, -1.00, -1.15, -1.30, -1.45, -1.60, -1.75`**（名次 1→8） |
| `--win-bonus` | `0`（已并入上表） |
| 升本节奏奖励 | **关**：`--tier-target-coef 0.0`（table / gate 也都为空，避免陈旧值） |
| 其它 shaping | `--comp-target-coef 0 --shaping-coef 0 --terminal-strength-coef 0` |
| 过程价值头 | 关 |

## F. 教师 BC（★ 用户改动 2）

| 项 | r3 那轮 | **本轮** |
|---|---|---|
| `HSAI_ARENA_FORCED_TIER` | `2:2,5:3,7:4,8:5,10:6` | ★ **`2:2,5:3,7:4,8:5`**（去掉 `10:6`） |
| 语义 | 到点且 Upgrade 合法（金币够 + 未满级 + 不在待选）时，每回合最多写一条 `teacher_action=3` | 同左，**但最后一个里程碑变成 T8→5 本** |
| 强制升本的回合 | 无（`prob=0`） | 无 |
| `--forced-tier-prob` | `0` | `0`（不变） |
| `--forced-tier-carry` | `1` | `1`（里程碑错过后再顺延 1 回合） |
| `--forced-teacher-coef` | `0.1` | `0.1`（不变） |
| `--forced-teacher-until` | `1B` | `1B`（全程不放手） |
| 提前升级 | 记 `achieved_by=self`，不再写标签 | 同左 |

**效果**：本轮教师**不会**再在 T10 推模型升 6 本；T8 之后不再有任何升本标签。模型要自己决定什么时候上 6。

> 脚本注释里写的是"12 -> 6"，实际值是 `10:6` —— 注释与代码不一致（见 J）。

## G. 动作契约 / 规则限制（★ 用户改动 1：不 ban 种族）

| 项 | 值 |
|---|---|
| `max_actions_per_turn` | **100**（原 80） |
| 换位上限 `move_cap` / `move_repeat_limit` | **2** / 2 |
| Freeze 上限 | **1**（每回合只能按一次冻结键） |
| `forbid_cancel` | 1（全局禁用 Cancel） |
| `forbid_unfreeze` | 1 |
| `ban_trinkets` | 1（禁用 `BG36_MagicItem_414` 加重护手） |
| `no_cancel_choose` | 0 |
| **`lobby-exclude`** | ★ **空（不 ban 种族，10 个类型抽 5 个，与真实一致）** |
| `sims` | 0 |

> 改动 1 **不需要改任何东西**：脚本默认就是 `${HSAI_ARENA_LOBBY_EXCLUDE:-}`（空）。只要提交时不设这个环境变量即可。

## H. 模型结构（与 r3 完全一致，不新增任何结构开关）

| 项 | 值 |
|---|---|
| arch | `entity`（Entity Transformer） |
| d_model / layers / heads / dim_ff | 384 / 8 / 8 / 1536 |
| card_dim / dropout | 48 / 0.1 |
| token 数 | 49 |
| 参数量 | **16,436,034**（与 r3 相同） |
| `--use-z` / `--use-attr` / `--mag-query` | **0 / 0 / 0**（三项借鉴结构本轮继续关闭） |
| `--move-head-split` | 0 |

## I. 相对 r3 训练时的**全部**差异（3 项用户决定 + 3 项代码修复）

| # | 项 | r3 训练时 | 本轮 | 来源 |
|---|---|---|---|---|
| 1 | 动作上限 | 80 / cap5 / freeze2 | **100 / cap2 / freeze1** | ★ 用户 D-070 |
| 2 | 教师升本计划 | `2:2,5:3,7:4,8:5,10:6` | **`2:2,5:3,7:4,8:5`** | ★ 用户（本次） |
| 3 | 种族 ban | 无 | **无**（保持 10 抽 5） | ★ 用户（本次） |
| 4 | 加重护手封禁 | **实际从未生效**（`player_json['pending']` 判定恒 False，r3 仍选中 496 次） | △ **现在真的生效**（修复 `5a67150`） | 代码修复 |
| 5 | 健康门 | 不存在（非法动作/截断不会停机） | △ 硬门开启：非法动作/截断/非有限损失 → **整作业停** | 代码修复 D-069 |
| 6 | 教师 CE 观测 | 不可见 | △ 打印 `teacher_label_share`（分母口径可审计） | 代码修复 D-069 |

## J. 审核中发现的 4 处不一致（建议修，未改）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | `stage_arena.sbatch:59` | `HSAI_ARENA_CKPT` 默认仍是 `arena-v1.2/model.pt` | 忘了覆盖会**从 v1.2 起训** |
| 2 | `stage_arena.sbatch:56` vs 头部注释 | 实际 `MINUTES` 默认 **165**，注释写 150 | 只影响文档 |
| 3 | `stage_arena.sbatch:80` | 注释写"12 -> 6"，实际值是 `10:6` | 误导审核 |
| 4 | `hsbg/az/train/forced_tier.py` 默认 vs 脚本 | 代码默认计划 `2:2,5:3,6:4,8:5`，脚本传 `2:2,5:3,7:4,8:5` | 不传参数时行为不同 |

## K. 提交串（**等你点头**才执行）

```bash
cd ~/projects/HSAI
export HSAI_ARENA_OUT=artifacts/runs/nus-arena-v1.4-r1-20261117
export HSAI_ARENA_CKPT=artifacts/models/arena-v1.3-r3/model.pt
export HSAI_ARENA_FORCED_TIER=2:2,5:3,7:4,8:5      # ★ 去掉 10:6
# 不设 HSAI_ARENA_LOBBY_EXCLUDE  => 10 种族全开（★ 不 ban）
# 其余全部走脚本默认：SNAPS=v14-snap / MINUTES=165 / SEGMENTS=1 /
#   GAMES=512 / PROCS=8 / LR=6e-06 / WIN_BONUS=0 / RANK_UTILITY=1.00,0.35,-1.00,… /
#   FREEZE_CAP=1 / BAN_TRINKETS=1 / TARGET_KL=0.05 / ENTROPY=0.002
sbatch --parsable --export=ALL deploy/slurm/stage_arena.sbatch
```

## L. 本轮明确**不会**做的事

- 不改场地强度（库仍是 v14-snap，p50band）
- 不改奖励表、不改 λ/γ/entropy/lr/anchor
- 不开任何借鉴结构（z / attr / mag_query）
- 不 ban 种族、不做单变量消融
- 只跑 1 段（165 分钟，3 小时墙钟内），一次性作业
