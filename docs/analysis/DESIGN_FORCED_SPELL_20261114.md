# 设计：`forced_spell` —— 把"买法术"从零频动作变成被采样动作

> 状态：**设计定稿，未实现**（实现预计 1–1.5 h + 冒烟 10 min）。
> 为什么不是奖励项：见 `RUNS_20261113.md` §5.79 —— 动作条件奖励对"永不被采样的动作"
> 结构性无效（冒烟 21 轮 `reward_shaping_mean` 恒 0.0）。
> 范式来源：`hsbg/az/train/forced_tier.py`（强制升本课程，已上线且测试齐全）。

## 1. 目标（可判定的成功标准）

**先登记判据，再花卡**：

| # | 判据 | 现在的值 | 达标 |
|---|---|---|---|
| 1 | 采集期真的发生了"买法术" | **0 次/局** | `spell_buy_actions > 0` 且 `forced_spell_actions > 0` |
| 2 | 面板侧法术使用率 | **0 / 0（买/打）** | `buy_spell` **> 0.5/回合** 或 `play_spell` > 1.0/回合 |
| 3 | 机制：留存增重（预登记判据②） | +40（真人赢家 +690） | **≥ +150** |
| 4 | 名次（预登记判据①，**768 局**） | 5.372 vs champion 5.299 | 配对 Δ 的 95% CI 上界 **< 0** |

判据 1/2 是"机制是否被打开"，3/4 是"打开了有没有用"。**只有 1、2 都过才看 3、4。**

## 2. 动作与合法性

- 动作下标：商店法术槽 = `A_BUY + N_SHOP - 1 = 15`（`reward_shaping.SPELL_BUY_ACTION`，
  已存在）。`N_SHOP = 8`。
- **合法性只看引擎给的掩码** `mask[15]`：引擎每回合刷新都会填充该槽（`recruit.rs:325`），
  掩码在每次刷新时合法（`actions.rs:254`）。**不读金币、不改引擎状态、不补金币**。
- 不合法就不强制，并记一次 `blocked`（与 `forced_tier` 同口径）。

## 3. 强制策略（每局状态机）

```
每局: remaining = K (默认 3)            # 每局最多强制买 K 次法术
每次 learner 决策:
    if remaining == 0: 不强制
    if turn < turn_min (默认 6): 不强制   # T6 之前买法术通常是浪费，先让它学会经营
    if not mask[15]: blocked += 1; 不强制
    if rand() >= force_prob(decisions): prob_skips += 1; 不强制
    否则: force = True; remaining -= 1
```

- `force_prob` 用 `forced_tier.parse_prob_schedule` 的分段常数表（按累计决策数），
  计划形如 `"0:1.0,20M:0.5,60M:0.0"`：前期**必须**真的发生，后期把控制权交还策略。
- 每局配额 K 与 `turn_min` 走 env（`HSAI_FORCED_SPELL_PER_GAME` / `HSAI_FORCED_SPELL_TURN_MIN`）。

## 4. ML 纪律（照抄 `forced_tier` 的三条，一条都不能少）

1. **被强制的决策不进策略梯度**：`record["forced"]=True`、`logprob=0`。
   PPO 的策略/辅助损失排除这些样本，**只保留价值链与回报链**。
   理由：不能把"脚本替我按的键"当成策略自己的选择，否则会伪造 on-policy ratio。
2. **可选 teacher 标签**：未强制但 `mask[15]` 合法时写 `record["teacher_action"]=15`，
   走既有的 DAgger CE 通路，让策略**自己**学会按（这才是长期起作用的机制）。
   teacher 默认关，先只做强制。
3. **关闭时逐位一致**：空计划 ⇒ 不构造 tracker、不新增 record 字段、不读引擎。
   （这是本仓库的硬约定，必须有一条测试钉住。）

## 5. 落点（具体到函数）

| 文件 | 改动 |
|---|---|
| `hsbg/az/train/forced_spell.py` | **新增**：`ForcedSpellTracker`（`reset()` / `decide(turn, mask, decisions)` → 带 `.force` 的判定对象 / `stats()` / `game_meta()`），纯逻辑、零引擎依赖 |
| `hsbg/az/train/ppo.py` | 在 `__init__` 构造 `self._spell`（默认 `None`）；在 `decide()` / `decide_batch()` 里采样后、`_finish()` 前取判定；`_finish()` 里把 `action, logprob = int(UPGRADE_ACTION), 0.0` 泛化成 `int(forced.action)`（`UPGRADE_ACTION` 只是 tier 的默认值） |
| `hsbg/az/train/ppo.py` `CollectConfig` | 加 `forced_spell_plan` / `forced_spell_per_game` / `forced_spell_turn_min` / `forced_spell_prob` / `forced_spell_teacher` |
| `hsbg/az/train/{parallel_collect,process_pool}.py` | 逐局 meta 带 `forced_spell_game`，顶层 `aggregate_forced_spell`（照 `aggregate_forced_tier` 抄） |
| `hsbg/az/train/personal.py` | `OPTIONS` 加 `forced-spell-*`（否则 `validate_config` 会拒） |
| `workflows/train_entity_sync.py` | argparse + 启动打印 `[forced-spell] 开启：...` |
| `deploy/slurm/spell_l5.sbatch` | 参数化，`spell_coef=0` 时**不传**这两个新 flag（未打补丁的代码也能跑） |

**注意**：`hsbg/az/train/*` 与 `workflows/train_entity_sync.py` 都属 digest 文件 ⇒
只能在"没有 RUNNING/PENDING 训练作业"时推送（守卫见 D-051）。

## 6. 测试清单（先写测试再接线）

1. 计划解析：空/`off`/非法值 fail-fast；`per_game`/`turn_min` 边界。
2. **只在合法时强制**：`mask[15]=False` ⇒ `force=False` 且 `blocked+1`。
3. **每局配额**：K=3 ⇒ 第 4 次不再强制；`reset()` 后配额复原。
4. **回合下限**：`turn_min=6` ⇒ T5 不强制。
5. **概率门控**：`force_prob=0` ⇒ 一次都不强制且 `prob_skips>0`。
6. **记账与聚合**：逐局 meta → `aggregate_forced_spell` 求和正确（多进程 512 倍虚高那个坑照 `forced_tier` 处理）。
7. **关闭逐位一致**：空计划 ⇒ record 字段集合、`CollectConfig` 行为与旧路径一致。
8. **接线**：`OPTIONS → argparse → CollectConfig → 采集钩子` 四层贯通（照抄 `test_spell_bootstrap_wiring.py`）。

## 7. 冒烟（10 min，8 局 / 2 进程）必看三行

```
[forced-spell] 开启：per_game=3 turn_min=6 prob=1.0      # 补丁真的生效
history: forced_spell_actions > 0                        # 强制真的发生了
history: spell_buy_actions > 0                           # 动作真的进到了轨迹里
```

**任何一条为 0 就不要发 150 min 的正式段**（这正是 §5.79 学到的：先证可达，再谈量级）。

## 8. 已知风险与缓解

| 风险 | 缓解 |
|---|---|
| 强制动作让 PPO 的 on-policy 假设失真 | 被强制样本 `logprob=0` 且 `forced=True`，策略/aux 损失排除；价值仍从中学习 |
| 买法术挤掉关键回合的经济（尤其 T6 前） | `turn_min=6` + 每局最多 3 次 + 概率计划逐步归零 |
| 改了 `ppo.py` 热路径，静默跑歪 | 关闭时逐位一致的测试 + 冒烟三行 + 先跑 8 局再跑 150 min |
| digest 改动打断在跑臂 | 只在无 RUNNING/PENDING 训练作业时推送；本次实验不与其它臂共用 OUT |
