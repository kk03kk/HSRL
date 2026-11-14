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

## 3. 当前冠军 / 候选冠军

### 3.1 现任冠军（上一轮基准，仍是最稳的一版）

| 项 | 值 |
|---|---|
| 版本 | `artifacts/models/league-final-20261111-step014858/`（`model.pt` 67.8 MB + `trainer.pt` 254 MB） |
| 来源 | 36h 联赛作业 `artifacts/runs/league-v1.1-36h-20261109/`（`personal.py` + `config/league-36h.json`，7.63 亿决策 / 5,315 轮） |
| 赢家场成绩 | 768 局 `mean_rank` **5.299**，区间 [5.19, 5.40] |
| 校准场成绩（512 局） | v2 场 2.650 / v3 场 **2.615** / v3hold 场 **2.918** |
| 实机 | `real_game_accepted: false`；两局终局有效（第 6 / 第 5 名），无可比优势 |

### 3.2 候选冠军（2026-11-14 第一轮，**双场验证通过，等用户确认换位**）

| 项 | 值 |
|---|---|
| 版本 | `artifacts/models/arena-v3-tier-20261114/`（硬链接到 run，不占双份磁盘） |
| 来源 | `artifacts/runs/nus-arena-v3-tier-20261114-1/`（作业 1140935，303 轮 / 3,900 万决策 / 164.6 分钟） |
| 与冠军的差异 | 对手场换成 `arena-cal-v3-snap` + 超参按论文 Table 7 + 新增升本奖励（三处一起变，**不是单变量 A/B**） |
| **训练场评测**（512 局） | `mean_rank` 2.615 → **2.352**；吃鸡 41.6% → **48.6%**；配对 CI [+0.064, +0.463] 排除 0 |
| **held-out 评测**（512 局） | `mean_rank` 2.918 → **2.637**；吃鸡 41.0% → **44.3%**；配对 CI [+0.061, +0.512] 排除 0 |
| 判胜 | **IMPROVED**（`tools/compare_panels.py`，两个场地一致，效果量 −0.26 / −0.28） |
| 注意 | 模型**主动降级换板面**：T8–T18 中位板面 +1%~+16%，但平均 tier **−0.47~−0.74**；升本奖励没有把 tier 拉上去，而名次反而更好 |

---

## 4. 正在运行

**没有作业在跑**（2026-11-14 23:25 起）。第一轮训练与两个场地的评测都已完成，队列为空。

上一次运行的记录（保留作对照）：作业 1140935，节点 xgpi13，双卡 MIG 分属两张物理卡，
输出 `artifacts/runs/nus-arena-v3-tier-20261114-1/`，303 轮 / 302 次更新 / 164.6 分钟，
`illegal_actions=0 / invalid_games=0 / truncated_games=0` 全程零。

评测产物：
- `artifacts/runs/gap-v3-vs-champion-20261114/panel.json`（训练场 v3-snap，1,498 帧）
- `artifacts/runs/gap-v3hold-vs-champion-20261114/panel.json`（held-out v3hold-snap，455 帧）

---

## 5. 当前有效资产

| 类别 | 路径 | 说明 |
|---|---|---|
| 引擎 | `engine/`（`hsbg_engine` 37.0.2.256589，指纹 `c4c71014e3d41760`） | 245 原生动作；观测 3227 floats + 124 cards |
| 训练器 | `workflows/train_entity_sync.py` | entity 16.4M；`--trainer-state` 才是真续训（没有 `--resume`） |
| 训练入口 | `deploy/slurm/stage_arena.sbatch` | 含双卡硬门禁 |
| 建库入口 | `deploy/slurm/build_arena_snap.sbatch` | 校准库 + 快照库同源生成（纯 CPU） |
| 评测入口 | `deploy/slurm/eval_human_gap_snap.sbatch` + `workflows/eval_human_gap.py` | 产出 M1/M2 与名次面板 |
| 战团库（训练用） | `artifacts/libraries/arena-cal-v3-snap/` | 1,498 帧，turn 1–18，基于冠军 `gap-calsnap2` 面板校准 |
| 战团库（**held-out 考试用**） | `artifacts/libraries/arena-cal-v3hold-snap/` | 455 帧，turn 1–18，源自 `eval-m6k.jsonl`（与训练池不同的 1,349 局），**只用于判胜，不用于训练** |
| 战团库（历史） | `arena-cal-v1{,-snap}`、`arena-cal-v2{,-snap}` | 保留可复现，不再用于训练 |
| 判胜工具 | `tools/compare_panels.py` | 逐回合 M1/M2 对比 + 配对 bootstrap；`--m1-min-n` 默认 100 |
| 发布工具 | `tools/publish_model_dir.py` | run → 规范模型目录（硬链接，不占双份磁盘） |
| 真人库 | `data/real_library/firestone-bank-6k/train-m6k.jsonl`（研究用）/ `eval-m6k.jsonl`（held-out） | 12,689 / 1,349 条战团（99% 是第一名战团） |
| 人类战力曲线 | `config/human_curve_m6k_20261114.json` | 逐回合分位 |

---

## 6. 已经确认的事实（不要重复试验）

1. **对手场是唯一被证实有效的难度杠杆**（D-052 / D-054）。奖励家族的多轮实验没有超过对手场的作用。
2. **名次奖励在模型所处板面区间不敏感**（D-052）—— 所以训练前必须先测信号，不能只看配置系数。
3. **"配置的 reward coef" ≠ "实际付出的量级"**（D-053）—— 必须从 collect stats 反算真实量级。
4. **只加 tier 奖励会被打穿**：模型会把金币全砸升级、不买随从（板面 −33%）。所以升本奖励必须带**板面门控**。
5. **comp-target（只奖励凑阵容）被否**：模型买完核心卡就不玩了。**retained_growth 被否**（用户否决）。
6. **战斗归因必须用引擎的 `won_last/lost_last`**（D-055）—— "没掉血" ≠ "赢了"。
7. ~~**分段续训会破坏 anchor 保护期**~~ —— **这是 2026-11-14 的误判，已撤回**：`trainer.updates` 是累计**轮数**（`[iter 302] ... updates=302`），anchor 保护期按累计轮数走，续训不会跳过。分段续训是安全的。
8. **本站 `h100-47` 是 MIG 切片**；两张切片可能同物理卡 → `device_count==1` → 必须用双卡门禁重投。
9. **`--trainer-state` 会覆盖 `--lr`**（用 state 里的 `base_lrs`），想要论文的 6e-06 就不能带它。

---

## 7. 下一步（等用户决定）

第一轮已经走完「训练 → 双场评测 → 判胜」全流程，结论是 **IMPROVED**。下一步按 CHARTER §2.2 是：

1. **确认换位**：把 `arena-v3-tier-20261114` 定为下一轮的起点与比较基准（候选已双场验证）。
2. **重建战团库**：用新模型的战力曲线重新校准（`tools/build_arena_library.py --panel <新panel> --panel-model arena_v3`）→ `arena-cal-v4-snap`，
   同时**新建一份 held-out 考试场**（换 split 或换数据源），保证下一轮还能做交叉验证。
3. **下一轮只改一个变量**（本轮同时改了对手场 + 超参 + 奖励，无法归因）：优先候选
   - (a) `--max-updates 300` 修退火口径（K4，纯参数、零风险）
   - (b) 升本奖励去留（本轮实测它没把 tier 拉上去，而名次反而变好，见 `NEGATIVE_RESULTS.md`）
   - (c) 流水线永久转串行（K1，影响吞吐不影响质量）
4. **实机**：仍未通过（`real_game_accepted: false`），需 D-026 取证后再谈。

---

## 8. 硬约束

见 [`../AGENTS.md`](../AGENTS.md)：远端脚本纯 ASCII、登录节点不跑重活、不改 `CUDA_VISIBLE_DEVICES`、新作业新目录、失败保现场、不删资产、密钥不进对话。
