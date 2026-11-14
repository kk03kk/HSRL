# 训练配方（冻结版）

状态：**当前生效并冻结**（2026-11-14 用户确认）。入口脚本：[`deploy/slurm/stage_arena.sbatch`](../deploy/slurm/stage_arena.sbatch)。

> **冻结规则**：本轮之后**不每轮改配方**。只有发现**具体问题**（工程错误 / 指标异常 / 明确的机制缺陷）才由 Agent 提出修改建议，**用户决定**是否采纳。变更必须同时更新本文件与 `DECISIONS.md`。

---

## 1. 参照物：AutoTavern 论文 Table 7

论文：Build, Don't Play（AutoTavern），[arXiv:2610.33845](https://arxiv.org/html/2610.33845v1)，§6 Model and training。

论文原文要点：
- PPO，**每轮约 100 万决策**，奖励是**终局名次效用**；
- **一个 stage = 300 轮 ≈ 3.1 亿决策（57 万局）**，每个 stage 从上一个 stage 继续，并带 **KL anchor**；
- 网络 entity transformer，d=384 / 8 层 / 8 头 / 约 18M 参数；
- 算力：8×H200 + 128 CPU 核。

Table 7 超参（本仓库对齐记录见 `records/source-docs/hsbg/docs/MULTISTAGE_TRAINING.md` §12）：

```
lr 6e-06 → 5e-06        有效 minibatch 8192（微批 1024 × 梯度累积 8）
clip 0.2                target-KL 0.05
lambda 0.995            entropy-coef 0.002
grad-clip 1.0           KL anchor β=0.1，仅前 100 轮
sims 0（不做搜索）
```

---

## 2. 完整参数表

### 2.1 作业层

| 项 | 值 | 依据 |
|---|---|---|
| 脚本 | `deploy/slurm/stage_arena.sbatch` | — |
| 分区 / 期限 | `gpu` / `03:00:00` | 本站 `gpu` 分区上限 3 小时；长训须由用户另行批准（`gpu-long` 可达 3 天） |
| GPU | `--gres=gpu:h100-47:2` | 与冠军同型号；**注意 `h100-47` 是 MIG 切片，见 §6** |
| 排除节点 | `xgpi15,xgpi17` | 历史故障节点 |
| CPU / 内存 | 32 / 96G | 8 采集进程 + learner |
| 训练分钟数 | `HSAI_ARENA_MINUTES=165` | 作业上限 180 分钟，留 15 分钟给加载与收尾存档 |
| 段数 | 1（不做段内续训） | 分段续训会让 anchor 的 `i` 跳到优化步数，β=0.1 的保护期无法在第二段重现（代码事实） |
| 输出目录 | `artifacts/runs/nus-arena-v3-tier-<日期>-<序号>` | 新作业必须新目录，已存在则拒绝 |

### 2.2 起点与网络

| 项 | 值 | 依据 |
|---|---|---|
| 权重 | 当前冠军 `model.pt` | 每轮从冠军分叉 |
| 优化器 | **新建 Adam**（不带 `--trainer-state`） | 带 trainer-state 会用 state 里的 `base_lrs` 覆盖 `--lr`，论文的 6e-06 会失效 |
| 网络 | `arch=entity`，d=384 / 8 层 / 8 头，16,436,034 参数 | 与冠军同构，保证可严格加载 |

> **与论文的已知差异 #1**：论文每个 stage 从上一 stage 继续（含训练状态）；我们只继承权重、重建优化器。

### 2.3 对手场（本项目唯一被验证有效的难度杠杆）

| 项 | 值 | 依据 |
|---|---|---|
| 快照库 | `artifacts/libraries/arena-cal-v3-snap` | 1,498 帧，覆盖 turn 1–18；建库见 `tools/build_arena_library.py` + `tools/build_arena_snapshot.py` 同源作业 `deploy/slurm/build_arena_snap.sbatch` |
| 注入席位 | `--mirror-seats 7 --mirror-p 1` | 7 个非 learner 席位全部注入 |
| 采样 | `--mirror-sample p50band` | 分位带 0.45–0.55 内均匀取样：同一难度目标 + 恢复对手多样性 |
| 严格模式 | `--mirror-strict 0` | 缺帧时回退引擎 bot（库只有 seat 0，靠跨席回退） |
| 回合上限 | `--mirror-max-turn 0` | 不限 |
| 人类战团 / 入场课程 | 全关（`--human-board-p 0 --entry-p 0`） | 本轮不启用 |

**校准系数（`build_arena_library.py` 的 `DEFAULT_MULT`）**：前 8 回合 0.85 / 第 9 回合 1.00 / 10–18 回合 1.05–1.15。含义：前期对手略弱于模型（鼓励升本），第 9 回合势均力敌，后期只比模型强一点点且缓慢上升，**绝不出现一个回合跳好几倍**。

> 每次换冠军后，**必须基于新冠军的战力曲线重新建库**，这是迭代流程的一部分。

### 2.4 PPO

| 参数 | 值 | 来源 |
|---|---|---|
| games / processes / workers / epochs | 512 / 8 / 1 / 1 | 一轮 ≈ 12.4 万决策（论文一轮 ≈ 100 万，我们是其 1/8） |
| minibatch × grad-accum | 1024 × 8 = **有效批 8,192** | 论文 Table 7 |
| lr / lr-min-ratio | **6e-06** / 0.8333（→ 5e-06） | 论文 Table 7 |
| max-updates | 4,650 | ≈ 论文一个 stage（300 轮 × 每轮约 15.5 步）的学习率退火长度 |
| clip / target-KL | 0.2 / 0.05 | 论文 |
| gamma / lambda | 1.0 / 0.995 | γ=1 因为奖励只在终局、一局数百步 |
| entropy / pi / seq / max-grad-norm | 0.002 / 0 / 0 / 1.0 | 论文 |
| 采样精度 | amp bf16、leaf bf16、records fp32、anchor-amp 1 | 冠军同款 |
| pipeline | `--pipeline 1`（collect k+1 与 update k 并行）、`--pipeline-guard abort`、`--pipeline-zero-step-fallback 1` | 双卡分卡后才有收益 |
| sims / seed | 0 / 611071401 | sims=0 = 不做搜索；seed 固定保证可比 |

### 2.5 KL anchor

| 参数 | 值 | 来源 |
|---|---|---|
| `--anchor-beta` | 0.1 | 论文 Table 7 |
| `--anchor-iterations` | **100** | 论文 Table 7（一个 stage 的前 100/300 = 33%） |
| `--anchor-beta-tail` | 0.02 | 本仓库沿用值（论文未给 tail） |
| `--reset-anchor` | 1 | 分叉时把 anchor 重建为当前策略 |

### 2.6 奖励

奖励 = **终局名次效用**（唯一的值函数目标）+ **一个小的升本节奏项**。

| 参数 | 值 |
|---|---|
| `--tier-target-coef` | 0.25 |
| `--tier-bonus`（命中） | 0.15 → 实际 +0.0375 |
| `--tier-penalty`（落空） | 0.10 → 实际 −0.025 |
| 目标表 | `2:2,5:3,7:4,8:5` = 第 2/5/7/8 回合**起点各查一次**，要求达到 tier 2/3/4/5 |
| 板面门控 | `--tier-gate-k 0.75`，参考值 `2:1,5:30,7:100,8:150`（Σ攻+血）；门控不通过则**不发也不扣** |
| 其余塑形 | 全 0（comp-target / shaping / terminal-strength / retained） |

实测量级（512 局一轮）：检查 ~1,600 次、命中 ~240（15%）、每轮净 **−25 ≈ 每局 −0.049**，即名次全距（2.0）的 2.4%。单局区间 [−0.10, +0.15]。

> **与论文的已知差异 #2**：论文只有终局名次效用，没有塑形项。用户 2026-11-14 决定保留该项做实验。
> **门控是这条奖励的安全阀**：2026-11-13 实测过"只看 tier 的奖励会被'把金币全砸升级、不买随从'打穿"（板面 −33%）。

### 2.7 冻结契约（改它需要用户明确授权）

`--max-actions-per-turn 80`、`--move-guard 1 --move-cap 5 --move-repeat-limit 2`、`--forbid-cancel 1`、`--forbid-unfreeze 1`、`--sims 0`。

### 2.8 快照与恢复

`--snapshot-every 5 --keep-snapshots 3`：约每 5 轮（≈1.5 分钟）存一份 `model.pt + trainer.pt`，只留 3 份。作业被墙时可以从最近快照恢复。

---

## 3. 规模口径（不要混用"轮"这个字）

| 单位 | 论文 | 我们 |
|---|---|---|
| 一轮 | ≈100 万决策 | ≈12.4 万决策（512 局） |
| 一个 stage | 300 轮 = 3.1 亿决策 | — |
| 一次 3 小时作业 | — | ≈620 轮 ≈ **7,700 万决策** ≈ 论文一个 stage 的 1/4 |

**所以一次 3 小时作业不是"训练完成"，而是"论文尺度下的一个 stage 的一部分"。** 要"变强"需要多轮累积。

---

## 4. 吞吐基线（用于判断作业是否健康）

| 场景 | 采集速率 | 每轮墙钟 | 备注 |
|---|---|---|---|
| 单卡 47GB（learner 与采集共用） | 5,900 决策/秒 | ~55 s | v2 实测 |
| **双卡分卡（本配方）** | **8,400–9,100 决策/秒** | **~16 s**（collect 15 s + ppo 12 s 重叠） | 2026-11-14 smoke/正式作业实测 |
| 论文字面 | — | — | 8×H200 + 128 核，不可比 |

每轮 PPO 优化步数应为 **15–16**（= 有效批 8,192 生效）。若日志里出现 ~124 步，说明 `--grad-accum` 没生效。

---

## 5. 健康红线（触发即停，保留现场）

| 指标 | 红线 |
|---|---|
| `illegal_actions` / `invalid_games` / `truncated_games` | 必须 **0** |
| 非有限损失、引擎异常、注入失败率骤降 | 立即停 |
| `kl` | 连续两轮 > 0.03 |
| 吞吐 | 掉到 1,500 决策/秒以下 |
| `pipeline guard abort` | 立即停 |
| 每轮 PPO 步数 | 不在 15–16（说明累积配置失效） |

正常棋力波动**不是**红线，只记录。

---

## 6. 本站 GPU 事实（2026-11-14 实测，踩过坑）

- `h100-47` 不是整卡，是 **MIG 3g.47gb 切片（46.4 GiB）**。
- 一次分配 2 张切片时，两张可能落在**同一张物理卡**上 → `torch.cuda.device_count()==1` → `--gpus 1` 越界崩溃。
- 也可能分属两张物理卡 → `device_count()==2`，此时才是真双卡。
- 脚本因此内置**双卡硬门禁**：`device_count < 2` 就在建目录前 `exit 3`，由外部重投换节点（约 1 分钟一轮）。
- 账号 QOS（`normal`）按用户上限：`h100-47=2`、`h100-96=1`、`a100-80=1`、`a100-40=2`、`nv=8`、`gres/gpu=8`。
  **"两张整卡 H100"这个账号拿不到**（整卡上限 1 张）。
- `nv` 类型是 TITAN V 11.8 GiB，太弱，不用于训练。

---

## 7. 与冠军那套配方的差异（为什么会不同）

冠军（`league-36h`，2026-11-11）用的是 `personal.py` + `config/league-36h.json`，实际生效 `lr≈1.667e-06`（被 trainer state 的 `base_lrs` 覆盖）、有效批 1,024、anchor 250 轮。本配方按论文 Table 7 重新对齐，**因此与冠军那条不是同一套超参**，两者不能按"同配方"直接比吞吐。
