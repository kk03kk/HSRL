# HSAI 当前状态

状态：**当前生效**（2026-11-14 19:50 +0800 重写）。
只写"现在是什么"，**不写历史沿革**；历史在 [`analysis/`](analysis/) 与 [`DECISIONS.md`](DECISIONS.md)。

> 服务器侧实时状态（作业、GPU、HEAD）一律用 `python ops/hsai_ops.py state` 取，**不抄本文件**。

---

## 1. 一句话

在**校准战团场**上按**论文 Table 7 配方**做分阶段 PPO 训练，每一轮用统一协议评测新模型，胜过当前冠军就换库继续，不胜就归因。当前冠军 `league-final-20261111-step014858`，第一轮新配方训练正在跑（作业 1140935）。

---

## 2. 目标与研发模式

见 [`CHARTER.md`](CHARTER.md)。要点：
- 用户定方向，Agent 执行/分析/验证；**不自主规划多天路线**；**任何 GPU 作业需用户批准**。
- 一轮 = 训练 → 评测 → 与冠军对比 → 有进步换库继续 / 没进步归因。
- 配方已冻结，不每轮改动。

---

## 3. 当前冠军

| 项 | 值 |
|---|---|
| 版本 | `artifacts/models/league-final-20261111-step014858/`（`model.pt` 67.8 MB + `trainer.pt` 254 MB） |
| 来源 | 36h 联赛作业 `artifacts/runs/league-v1.1-36h-20261109/`（`personal.py` + `config/league-36h.json`，7.63 亿决策 / 5,315 轮） |
| 网关成绩 | 赢家场 768 局 `mean_rank` **5.299**，区间 [5.19, 5.40] |
| 校准场成绩（512 局） | `mean_rank` **2.650**，前四 83.2%，吃鸡 38.3%，非法动作 0 |
| 逐回合 M1/M2 | 见 [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) §7 |
| 实机 | `real_game_accepted: false`；两局终局有效（第 6 / 第 5 名），无可比优势 |

---

## 4. 正在运行

| 项 | 值 |
|---|---|
| 作业 | **1140935**（`deploy/slurm/stage_arena.sbatch`），节点 xgpi13，双卡 MIG 分属两张物理卡 |
| 输出 | `artifacts/runs/nus-arena-v3-tier-20261114-1/` |
| 配方 | 论文 Table 7 + 升本奖励 + 冠军权重分叉，见 [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) |
| 规模 | 165 分钟训练，预计约 620 轮 / 7,700 万决策 / 22:28 结束 |
| 实测健康 | 8,400–9,100 决策/秒，每轮 ~16 s，PPO 15–16 步/轮，非法动作 0，注入 3.6 万/轮，升本奖励净 −25/轮 |

---

## 5. 当前有效资产

| 类别 | 路径 | 说明 |
|---|---|---|
| 引擎 | `engine/`（`hsbg_engine` 37.0.2.256589，指纹 `c4c71014e3d41760`） | 245 原生动作；观测 3227 floats + 124 cards |
| 训练器 | `workflows/train_entity_sync.py` | entity 16.4M；`--trainer-state` 才是真续训（没有 `--resume`） |
| 训练入口 | `deploy/slurm/stage_arena.sbatch` | 含双卡硬门禁 |
| 建库入口 | `deploy/slurm/build_arena_snap.sbatch` | 校准库 + 快照库同源生成（纯 CPU） |
| 评测入口 | `deploy/slurm/eval_human_gap_snap.sbatch` + `workflows/eval_human_gap.py` | 产出 M1/M2 与名次面板 |
| 战团库（当前） | `artifacts/libraries/arena-cal-v3-snap/` | 1,498 帧，turn 1–18，基于冠军 `gap-calsnap2` 面板校准 |
| 战团库（历史） | `arena-cal-v1{,-snap}`、`arena-cal-v2{,-snap}` | 保留可复现，不再用于训练 |
| 真人库 | `data/real_library/firestone-bank-6k/train-m6k.jsonl` | 12,689 条战团，覆盖 turn 1–18（99% 是第一名战团） |
| 人类战力曲线 | `config/human_curve_m6k_20261114.json` | 逐回合分位 |

---

## 6. 已经确认的事实（不要重复试验）

1. **对手场是唯一被证实有效的难度杠杆**（D-052 / D-054）。奖励家族的多轮实验没有超过对手场的作用。
2. **名次奖励在模型所处板面区间不敏感**（D-052）—— 所以训练前必须先测信号，不能只看配置系数。
3. **"配置的 reward coef" ≠ "实际付出的量级"**（D-053）—— 必须从 collect stats 反算真实量级。
4. **只加 tier 奖励会被打穿**：模型会把金币全砸升级、不买随从（板面 −33%）。所以升本奖励必须带**板面门控**。
5. **comp-target（只奖励凑阵容）被否**：模型买完核心卡就不玩了。**retained_growth 被否**（用户否决）。
6. **战斗归因必须用引擎的 `won_last/lost_last`**（D-055）—— "没掉血" ≠ "赢了"。
7. **分段续训会破坏 anchor 保护期**：`i = trainer.updates`（优化步数）会让第二段的 β=0.1 直接失效。
8. **本站 `h100-47` 是 MIG 切片**；两张切片可能同物理卡 → `device_count==1` → 必须用双卡门禁重投。
9. **`--trainer-state` 会覆盖 `--lr`**（用 state 里的 `base_lrs`），想要论文的 6e-06 就不能带它。

---

## 7. 下一步（本轮训练结束后）

1. **评测**（需用户批准，1 卡约 11 分钟）：`arena-cal-v3-snap` 场上 512 局，得 M1/M2 + 名次面板。
2. **同场重测冠军**：v3 场是新场地，冠军基线必须在同一场地重测，否则不可比。
3. **判胜**（见 [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) §5）：
   - 有进步 → 新模型成为候选冠军 → **基于它的战力曲线重建战团库**（`build_arena_library.py`）→ 下一轮训练。
   - 无进步 → 归因（工程 / 场地 / 奖励 / 容量 / 样本量），提出建议 → 用户决定。
4. **实机**：仍未通过（`real_game_accepted: false`），需 D-026 取证后再谈。

---

## 8. 硬约束

见 [`../AGENTS.md`](../AGENTS.md)：远端脚本纯 ASCII、登录节点不跑重活、不改 `CUDA_VISIBLE_DEVICES`、新作业新目录、失败保现场、不删资产、密钥不进对话。
