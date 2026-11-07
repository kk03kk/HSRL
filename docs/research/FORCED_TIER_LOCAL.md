# 强制升本与解除强制的本地验证（2026-11-07）

本轮研发的本地证据。**CPU + `--opponent bot`（无镜像库注入）**，只证明机制与数据通路，
不作为棋力结论；正式结论必须在 NUS 用 dev 库评测。

## 1. 引擎规则可行性（200 局扫描，`tools/probe_forced_tier.py` 同口径）

| 目标 | 准时 | 顺延（carry=1） | 未达 | 原因 |
|---|---:|---:|---:|---|
| T2 → 2 本 | 197/200（98%） | 3 | 0 | 个别英雄/开局待选差异 |
| T5 → 3 本 | 200/200 | 0 | 0 | 7 金 vs 4 金 |
| T6 → 4 本 | 139/200（70%） | 57（→T7） | 3 | **T6 强制小饰品吃掉 0–3 金**（引擎机制） |
| T8 → 5 本 | — | — | — | 该扫描的 learner 不买随从，多数局在 T7–T8 前阵亡；到达 T8 的都能升 |

结论：计划可执行；唯一真实冲突是 T6 小饰品，用 `carry=1` 顺延到 T7 并如实记 `deferred`。
**不修改难度表、不补金币、不跳过**。

## 2. 端到端 PPO 短跑（真实 16.4M 模型，7357 完整 trainer）

环境：CPU、`--opponent bot`、`--games 4`、`--iterations 2`、`minibatch 64`、
`forced-tier-plan 2:2,5:3,6:4,8:5`、`carry 1`、`lr-min-ratio 1.0`、`anchor-beta 0.01`、
`reset-anchor 1`。

| 项 | 结果 |
|---|---|
| 更新 | 7357 → 7359（2 轮），无异常、无数值错误 |
| 非法动作 / 截断 | 0 / 0 |
| T2→2 本 | `on_time` 4/4 |
| T5→3 本 | `on_time` 4/4 |
| T6→4 本 | `on_time` 3/4 + `deferred` 1（小饰品冲突） |
| T8→5 本 | `on_time` 4/4（第 2 轮 4/4，第 1 轮 3 on_time + 1 deferred） |
| `blocked` | 每目标 0–4 次（待选阶段那一瞬，属于预期） |
| history | `forced_tier` 段带 `plan/carry/prob/teacher/forced_actions/targets` |

结论：**强制升本在真实模型上按计划执行，且不污染 PPO**（forced 记录 `old_logprob=0`、
`forced=True`，被策略损失排除；价值/回报链保留）。

## 3. lockstep + 教师标签（prob=0、teacher 开）

`--forced-tier-schedule 0:0.0 --forced-teacher-until 10M --forced-teacher-coef 0.05 --lockstep 1`：

- `teach=13.0699 teach_n=2`：教师标签通路打通，CE 只在有标签的样本上算，且**被强制的样本
  已被排除**；
- 每回合最多一条标签（已实现），不会把一整个回合的十几条决策重复标注。

## 4. 探针：free vs forced（真实 7357 网络，16 局 × 两臂，`--opponent bot`）

`records/research/long1085131/forced-tier-probe.json`

| 回合 | free 等级 P50 | free ≥5 本 | free 身材 P50 | forced 等级 P50 | forced ≥5 本 | forced 身材 P50 |
|---:|---:|---:|---:|---:|---:|---:|
| T5 | 2 | 0% | 47 | **3** | 0% | 37 |
| T6 | 2 | 0% | 79 | **4** | 0% | 59 |
| T8 | 3 | 0% | 193 | **5** | **80%** | 153 |
| T10 | 3 | 0% | 385 | **5** | **100%** | 433 |
| T12 | 3 | 0% | 654 | **5** | **100%** | 687 |
| T14 | 3 | 0% | 880 | **5** | **100%** | 1083 |
| T17 | 3 | 0% | 1243 | **5** | **100%** | 1530 |

计划执行率（16 局）：

| 目标 | self | on_time | deferred | missed |
|---|---:|---:|---:|---:|
| T2→2 本 | 0% | **100%** | 0% | **0%** |
| T5→3 本 | 0% | **100%** | 0% | **0%** |
| T6→4 本 | 0% | 87.5% | 12.5% | **0%** |
| T8→5 本 | 0% | 93.8% | 6.3% | **0%** |

**结论**：free 臂在 tier 3 停到 T17（0% 五本），forced 臂 T8 起稳定 5 本、**0 未达**。
机制确实把模型放到了高本状态上，这正是本轮要的"状态分布改变"。

**必须说明的局限**：两臂都是对引擎 bot（mean_rank 1.5、top4 100%），bot 太弱，**没有压力
去长身材**，所以两臂的身材差只有 +5%~+40%，**不能用它评估"高本经营是否学会"**。
高本成长必须在 NUS 用 `train-start` 镜像库（T12=1135）评测；这也是为什么长训起点选
`train-start` 而不是 bot。

## 5. 一个必须写下来的定量限制（决定下一轮的备用方案）

初始策略在目标回合给 Upgrade 的概率约 **1e-6**（对应 CE ≈ 13 nat）。而 Adam 在
`lr=1.75e-6` 下每个优化步对单个 logit 的位移约 `lr`，6000 个 iteration × ≈96 步 ≈ 1 nat
量级——**教师 CE 无法在 2.6 亿决策内把 13 nat 的差距补平**。

因此解除强制后的自主性不能只靠 CE，而依赖两条通路：

1. **稀有 on-policy 样本**：模型偶尔（~1e-6 × 4 个截止回合 × 512 局/轮）真的自己按了 Upgrade，
   在高本价值已经学到的前提下，这些样本的优势为正，PPO 会给 Upgrade 正向梯度并自我强化；
2. 教师 CE 提供一个小的持续推力（coef 0.05）。

**风险与触发条件**：若长训最后 100M（完全自主段）里
`upgrade_taken_turns/upgrade_available_turns` 在 T8–T12 仍 < 0.05，说明这条路走不通，
下一轮应改用**定向机制**（对动作头 bias 单独用大 lr / 反事实 DPO，仓库已有
`hsbg/az/train/dpo_upgrade.py` / `UpgradeSamples` / 训练入口 `dpo_upgrade_probe.py`），
而不是继续加教师系数。

## 6. 复现命令（本地）

```powershell
# 单测
& .venv\Scripts\python.exe -m pytest tests/test_forced_tier.py -q

# 200 局可行性扫描（纯引擎，秒级）
& .venv\Scripts\python.exe records\research\long1085131\probe_forced_tier3.py

# 端到端短跑（4 局 × 2 轮，CPU 约 5 分钟）
& .venv\Scripts\python.exe workflows/train_entity_sync.py `
  --checkpoint artifacts/models/nus-1080772/step-007357-360m/model.pt `
  --trainer-state artifacts/models/nus-1080772/step-007357-360m/trainer.pt `
  --out artifacts/local/forced-tier-smoke/serial --device cpu --amp off --leaf-amp off --anchor-amp 0 `
  --games 4 --iterations 2 --processes 1 --workers 1 --threads 1 --opponent bot --behavior net `
  --fast-act 1 --minibatch 64 --epochs 1 --lr 2e-6 --lr-min-ratio 1.0 --max-updates 20000 `
  --target-kl 0.03 --entropy-coef 0.008 --anchor-beta 0.01 --reset-anchor 1 `
  --max-actions-per-turn 80 --move-guard 1 --move-cap 5 --move-repeat-limit 2 `
  --human-board-p 0 --eval-every-decisions 0 --pipeline 0 --lockstep 0 `
  --forced-tier-plan 2:2,5:3,6:4,8:5 --forced-tier-carry 1 `
  --forced-tier-schedule 0:1.0 --forced-tier-prob 1.0 --forced-teacher-until 0
```
