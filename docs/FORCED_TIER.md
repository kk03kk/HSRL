# 强制升本课程（Forced Tier-Up Curriculum）

> 状态：已实现并通过单测（`tests/test_forced_tier.py`，37 项）与本地端到端短跑；默认关闭
> （空计划 = 逐位等价旧路径）。本轮训练用它作为核心机制，见
> `docs/research/NEXT_TRAINING_PLAN_1085131.md` 与 `config/p50-forced-tier.json`。

## 1. 为什么需要"强制"

复盘（`docs/research/LONG1085131_REVIEW.md` §5）证明模型在 tier 3 停住**不是缺激励，
而是当前难度下的最优反应**：

- 升本价每回合 −1，T10 只要 5 金、T13 只要 2 金，T10 有 95/96 局在回合初买得起；
- 但它**从不跨回合留钱**（回合末金币 P50=0、P90=1，从未 ≥8），升本的期权每回合被自己花掉；
- 后期（T12 起）它在对手分布下界之外、实战 0–6% 胜率，升本的收益在"活不到"的回合兑现。

在这种自洽的局部最优里，任何"奖励升本"的做法都会与真实回报打架（仓库 A/B 已验证
`tier_target_coef=1` 无行为变化、coef=10 只推动 tier 不提升强度）。所以要**先把它直接
放到高本状态上采样**，再逐步把控制权还回去。

## 2. 机制

### 2.1 计划与判定

```
--forced-tier-plan "2:2,5:3,6:4,8:5"      # 回合 → 目标酒馆等级（T9+ 自由，六本不强制）
--forced-tier-carry 1                      # 目标回合没做到时允许顺延的回合数
```

在 learner 的**每一条决策**上判定（不是只在回合初）：

1. 找当前适用目标：`turn >= 截止回合`、`turn <= 截止回合 + carry`、`tier < 目标`、且尚未
   过期（`missed`）；
2. **唯一合法性判据是引擎给的动作掩码**中 Upgrade 那一位（`mask[3]`）——它同时编码
   "金币够 + 未到等级上限 + 不在待选阶段"。本模块**不读金币、不补金币、不改引擎状态**；
3. 合法 → 把本步动作强制为 `Upgrade`；不合法 → **不改动作**，记一次 `blocked`（每回合最多
   一次），并继续等待本回合内 upgrade 变得合法的时刻；
4. 目标达成（模型自己升的也算）→ 按 `turn` 与截止回合的关系分类为
   `self / on_time / deferred / missed`。

为什么逐决策而不是回合初：T6 有**强制的小饰品选择**（待选阶段只有 Choose 合法），
回合初那一刻 `mask[3]=0`；逐决策判定会在待选结束、金币仍够的**第一时刻**补上升本。

### 2.2 与 PPO 的关系（不伪造策略梯度）

被强制的决策按**与"回合动作上限 / 换位守卫"完全相同的口径**处理：

```python
rec["forced"]      = True
rec["old_logprob"] = 0.0        # 不是行为分布采样
rec["action"]      = 3          # 实际执行的 Upgrade
```

`build_batch_from_records` 里 `valid = ~forced`，所以：

- **策略损失 / 辅助损失 / 教师 CE 全部排除**这些样本（不会把"脚本替我按的键"当成策略选择）；
- **价值与回报链保留**（末局名次仍是真实结果），critic 仍然学到"到了 5 本会怎样"。

因此强制升本只会改变**状态分布**（让高本状态进入训练数据），不会直接改写策略梯度。

### 2.3 逐步解除（weaning）

```
--forced-tier-schedule "0:1.0,90000000:0.6,180000000:0.3,260000000:0.0"
--forced-teacher-until 260000000
--forced-teacher-coef  0.05
```

| 累计 decisions | 强制概率 | 教师标签 | 含义 |
|---|---:|---|---|
| 0–90M | 1.0 | 关（强制已覆盖） | 保证每局都到 5 本，学会高本经营 |
| 90–180M | 0.6 | 开 | 40% 回合由模型自己决定；没升的回合给 DAgger 标签 |
| 180–260M | 0.3 | 开 | 70% 自主，标签继续纠偏 |
| ≥260M | 0.0 | 关 | **完全自主**，长训后段本身就是"解除规则"的验证窗口 |

教师标签写进 `record["teacher_action"]=3`，由 `teacher_bc_coef`（本轮取
`max(--teacher-bc-coef, --forced-teacher-coef)`）产生 CE 梯度；只在**目标未达成、Upgrade
此刻合法、本回合还没标注过**时写一条（每回合最多一条，避免一整个回合重复同一标签）。
被强制的那一步不写标签（已经发生了，没有"教"的必要）。

概率门控与采样用 policy 自己的 RNG，重放可复现；`prob=0` 且无教师时等价于"只观察"。

### 2.4 记录与统计

- 逐局 record 只在命中计划时多三个字段：`forced_tier_target / _deadline / _deferred`
  （默认关闭时 record 集合不变）；
- collect meta（`history.jsonl` 的 `collect.forced_tier` 与顶层 `forced_tier`）：

```json
{"plan":"2:2,5:3,6:4,8:5","carry":1,"prob":1.0,"teacher":false,
 "forced_actions":16,"teacher_labels":0,"blocked":6,"prob_skips":0,
 "targets":{"2":{"self":0,"on_time":4,"deferred":0,"missed":0,"forced_actions":4,
                 "teacher_labels":0,"blocked":0}, ...}}
```

- `self` = 目标回合之前模型自己就升到了（说明策略本身会升，不需要强制）；
- `on_time` = 恰好在目标回合升到；`deferred` = 顺延回合内补到；`missed` = 超期未达；
- `blocked` = 目标适用但 Upgrade 此刻不合法（金币不足 / 待选阶段）的次数。

**不会静默跳过**：每个目标都有明确归类，`missed` 会如实出现在 history 与报告里。

## 3. 可行性实测（本地，真实 7357 网络）

引擎规则（`engine/.../config.rs`、`flow.rs`）决定了两处真实约束，已在
`tools/probe_forced_tier.py` 与 200 局引擎扫描中量化：

| 目标 | 准时率（carry=0） | 顺延（carry=1） | 原因 |
|---|---:|---:|---|
| T2 → 2 本 | 98%（197/200） | +3 | 2 局因 T0/T1 待选/极端英雄差异拿不到 4 金 |
| T5 → 3 本 | 100% | 0 | 7 金 vs 4 金，稳定 |
| T6 → 4 本 | 70%（139/200） | +28.5%（57 局顺延到 T7） | **T6 强制小饰品吃掉 0–3 金**，7 金升本因此可能被挤掉 |
| T8 → 5 本 | 24%（49/200） | 0 | 该扫描里的 learner 不买随从、T7–T8 就阵亡；**到达 T8 的局都能升** |

结论与处理：

- **T6 的真实冲突来自小饰品**（引擎机制，不能也不该绕过）。`carry=1` 让 98.5% 的局仍在
  T7 补到 4 本，并如实记为 `deferred`；这就是"目标回合在正常经济下不总可实现 → 设计顺延
  规则并记录"的处理方式，而不是改金币或跳过。
- **T8 的低准时率是探针的假象**（不买随从的 learner 活不到 T8）。真实短跑里 4/4 局都
  在 T8 准时升到 5 本（见 §4）。
- 因此默认计划保持用户给的 `2:2,5:3,6:4,8:5`，`carry=1`，不改成"更宽松"的表；如果正式
  长训的 `deferred` 比例超过 30%，再考虑把 4 本目标挪到 T7（配置一行改动，不需要改代码）。

## 4. 本地验证（`records/research/long1085131/`）

1. **状态机单测**：37 项，覆盖解析/概率计划/on_time/self/deferred/missed/blocked/概率门控/
   按局隔离/折叠/PPO 接线（forced 记录 `logprob=0`、`forced=True`、被策略损失排除）。
2. **端到端 CPU 短跑**（4 局 × 2 iter，真实 16.4M 模型，`--opponent bot`）：
   `2:2/5:3/6:4/8:5` 全部 `on_time` 4/4（T4 有一局 `deferred`，与上面的小饰品分析一致），
   非法动作 0、截断 0、PPO 正常更新（7357→7359），`history.jsonl` 的 `forced_tier` 段
   带每目标计数。
3. **探针**（24 局 × 两臂，free vs forced）：见 `records/research/long1085131/forced-tier-probe.json`。

## 5. 相关文件

| 文件 | 作用 |
|---|---|
| `hsbg/az/train/forced_tier.py` | 计划解析、状态机、概率计划、统计 |
| `hsbg/az/train/ppo.py` | `CollectConfig` 字段、`PPOSelfPlayPolicy._forced_decide/_finish` 接线 |
| `workflows/train_entity_sync.py` | CLI、按 decisions 求值、history 记录 |
| `hsbg/az/train/evaluate.py` / `personal.py` | 评测期"规则内"探针（正式 Dev/Test 必须为空） |
| `config/p50-forced-tier.json` | 本轮正式配方（含解除时间表） |
| `workflows/run_forced_tier_long.py` | NUS 长训入口（训练场阶梯 + 解除强制的对齐） |
| `deploy/slurm/p50_forced.sbatch` | Slurm 提交脚本 |
| `tools/probe_forced_tier.py` | 本地可行性/效果探针 |
