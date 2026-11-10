# HSBRSIM × HSAI 代码级差异对比报告

生成时间：2026-11-10 21:10 +0800
对比对象：
- **HSAI**（当前工作区）`D:\HSAI`，`main` @ `d3a4c45`（工作树干净）
- **HSBRSIM**（Gallo13th）`D:\HSBRSIM` @ `328d0e0959055a6e80e25f92aad972981259b10c`

性质：**只做客观差异记录**，不评价设计、不提修复或优化建议。

---

## 0. 结论摘要

1. 两个仓库是**同一上游的两条独立分叉**：都声明为 AutoTavern / TavernGym 引擎的 vendored fork
   （HSAI：`engine/crates/hsbg-engine/src/lib.rs:3-6`；HSBRSIM：`rust/crates/hsbg-engine/src/lib.rs:2-5`）。
2. **观测层与动作层几乎完全相同**：`hsbg/env/obs_std.py` 只有版本文案差异，`action_space_std.py` 无差异，
   引擎动作常量（`N_SHOP=8 … A_DARK_GIFT`）逐项相同。
3. **模拟器规则层有 12 处实质差异**，集中在：禁用卡/禁用饰品机制、动作掩码空集逃生、
   买不起饰品单的清理时机、回合末选择结算、手牌配对（link/queue_marks）完整性、
   内容脚本越界防御、"Jar o' Gems" 触发语义。
4. **训练侧差异是"两套实验配置"的差异**：`CollectConfig` 74 vs 96 个字段、`PPOConfig` 71 vs 80；
   默认值 `max_actions_per_turn` 80 vs 40、`move_guard` True vs False；
   HSBRSIM 多出 ceiling head、replication BC、game-weight 奖励权重；HSAI 多出 forced-tier、mirror-strict、forced-teacher。
5. **评估侧**：HSAI 的 `evaluate.py` 多 mirror/economy/exposure/forced-tier 采集；HSBRSIM 的 `metrics.py` 多
   weighted-eval（ESS、long-strong 子集）。
6. **卡表数据语义相同**：两边引擎 `data/cards.json` 归一化后哈希一致（`51e11b0f18a87068…`），
   只是序列化字节不同（原始 SHA256 不同）；`data/p2/cards_37.json` 两边**完全相同**（`c7e6c159…`）。

---

## 1. 方法与范围

| 手段 | 说明 |
|---|---|
| 目录级 diff | `git diff --no-index --stat/--name-status`：`D:\HSAI\engine` ↔ `D:\HSBRSIM\rust`；`D:\HSAI\hsbg` ↔ `D:\HSBRSIM\hsbg` |
| 语义比对 | 用 `ast` 解析两边 `ppo.py`，比较 `CollectConfig` / `PPOConfig` 字段名、默认值、关键函数签名 |
| 数据比对 | 引擎 `cards.json` 归一化（排序键 + 紧凑序列化）后再哈希；逐卡逐字段比对 |
| 排除 | `__pycache__`、`*.pyc`、`.git`、构建产物 `target/` |
| 人工阅读 | 引擎 `src/*.rs` 全量 diff；训练/评估关键模块按函数逐段读 |

**差异规模**

| 范围 | 结果 |
|---|---|
| `engine/` ↔ `rust/` | 34 个文件变化（26 改、7 仅 HSBRSIM、1 仅 HSAI），+543 / −441 行 |
| `hsbg/` | 41 个文件改、6 个仅 HSBRSIM、74 个仅 HSAI |
| `hsbg/env/` | 仅 `obs_std.py` 的注释/版本文案；`action_space_std.py` 完全一致 |
| 引擎测试 | 各 41 个 `.rs`：HSAI 独有 `server_rules_audit.rs`；HSBRSIM 独有 `empty_mask.rs`、`trinket_ban.rs` |

---

## 2. 模拟器 / 游戏规则差异（按重要性排序）

### R1 · 线上禁用卡（LIVE_BANNED_CARDS）是否存在

| | |
|---|---|
| HSAI | `engine/crates/hsbg-engine/src/config.rs:11` 定义 `pub const LIVE_BANNED_CARDS`（4 张：`BG36_MagicItem_412`、`BG36_MagicItem_412t2`、`BG32_MagicItem_300`、`BGS_121`）；`config.rs:73` 的 `excluded_cards` 默认值 = 该列表（`config.rs:257`）；并在 `flow.rs:169/189/195/207/757` 对商店、饰品、暗月礼物池与 `trinket_eligible` 过滤 |
| HSBRSIM | 无 `LIVE_BANNED_CARDS`（`config.rs` 该常量不存在）；`config.rs:79/282` 的 `excluded_cards` 默认 **空**，注释写明 "Empty by default (Flourishing Frostling is out of the pool in the 36.6.3 data)"；`flow.rs` 中**没有** `excluded_cards` 过滤（`trinket_eligible` 只查 `has_trinket_id` 与 `offer_rule`，`flow.rs:759` 附近） |
| 差异 | 同一份规则配置下，HSAI 会从池子里剔除 4 个 id；HSBRSIM 只有显式传 `excluded_cards` 才剔除。卡池候选集合因此不同。 |

### R2 · 禁用饰品与 A/B 经济参数（仅 HSBRSIM）

| | |
|---|---|
| HSAI | `RulesConfig` 无这些字段 |
| HSBRSIM | `config.rs:38` `tier_upgrade_discount`、`config.rs:43` `banned_trinkets`、`config.rs:46` `tier_upgrade_discount_scope`、`config.rs:49` `gold_bonus_by_turn`；`config.rs:248` `gold_bonus_at()`；`flow.rs:382` 金币刷新时加 `gold_bonus`；`flow.rs:675` 饰品候选按 `banned_trinkets` 过滤；`recruit.rs:1663-1666` 升本价按 discount/scope 折减 |
| 差异 | HSBRSIM 的规则面比 HSAI 多 4 个可配置项（额度、范围、逐回合金币、禁用饰品 id 列表），默认值均为"不改行为"（0/空/`"all"`）。 |

### R3 · 动作掩码"全 False"时的逃生动作

| | |
|---|---|
| HSAI | `actions.rs:229-231`：`m[A_CHOOSE+i] = option_legal(...)`；`m[A_CANCEL] = top.cancellable();` —— 没有"无合法选项时放行 Cancel"的分支 |
| HSBRSIM | `actions.rs:219-231`：先累计 `any |= ok`，然后 `m[A_CANCEL] = top.cancellable() \|\| !any;`（注释说明付费饰品单买不起时避免空掩码） |
| 差异 | 当挂起单所有选项都不可点且 `cancellable()==false` 时，HSBRSIM 的掩码至少有一个合法位（Cancel），HSAI 的掩码可能全 False。 |

### R4 · "买不起的付费饰品单"在什么时候被清掉

| | |
|---|---|
| HSAI | 三处主动清理：`actions.rs:154`（**每次 `apply()` 成功后**遍历所有座位）、`flow.rs:534`（回合结束时）、`recruit.rs:2523`（**每次 `choose()` 结算完后**重新检查）；函数可见性 `pub(crate)`（`recruit.rs:2325`） |
| HSBRSIM | 只在两处：`recruit.rs:2457`（`take_trinket` 之后）与 `recruit.rs:2322`（`cancel()` 内部）；函数改为私有 `fn`（`recruit.rs:2357`）；`apply()` 与 `choose()` 里**没有**调用 |
| 差异 | 同一局面下，HSAI 会在更多时机把"买不起"的单从菜单里移除；HSBRSIM 只在"拿了一个饰品后"或"主动取消时"才移除。 |

### R5 · 回合末挂起选择的结算方式

| | |
|---|---|
| HSAI | `flow.rs:795` 调用 `resolve_end_turn_choices()`（定义 `flow.rs:814-840`）：每轮先 `drop_unaffordable_trinket_offers`，取合法项随机选；**无合法项时**：可取消则取消，否则返回 `Err("unresolvable end-of-turn choice")`；超过 256 轮返回 `Err`。测试 `flow.rs:1272/1281` 断言"保留 pending、不静默清空" |
| HSBRSIM | `flow.rs:800-810` 内联循环：无合法项或 `guard>256` 时 **`pending.clear()` 并 break**（静默清空）；不返回错误 |
| 差异 | 同一个"回合末菜单无解"的局面：HSAI 视为工程错误并保留现场；HSBRSIM 丢弃菜单继续。 |

### R6 · `end_turn_silent` 对挂起菜单的处理

| | |
|---|---|
| HSAI | `flow.rs:844-849`：`if ... \|\| !pl.pending.is_empty() { return; }` —— 有挂起菜单就不结束回合 |
| HSBRSIM | 同样位置：删掉 `pending.is_empty()` 条件，改为 `pl.pending.clear();` 后继续结束回合（`flow.rs:849`） |
| 差异 | 定时/静默结束回合时，HSAI 会跳过有挂起菜单的座位；HSBRSIM 会清空菜单并让它结束。 |

### R7 · 手牌配对（link）与"等待队列"完整性机制（仅 HSAI）

| | |
|---|---|
| HSAI | `state.rs:1050 queue_marks()`、`state.rs:1082 unlink_removed_hand_card()`、`state.rs:1089 prune_removed_hand_links()`；`engine.rs:226` 每次 `play/apply` 后 `prune_removed_hand_links()`；`engine.rs:3243-3248` 发第二张牌后校验 `link` 目标仍在手牌/队列；`content/timewarp.rs:380`、`experimental.rs:86` 入队时同步 `queue_marks`；`invariants.rs:59-70` 新增"dangling hand link"不变量检查 |
| HSBRSIM | 上述字段与函数**全部不存在**（`state.rs:1064-1098` 删除 `unlink_removed_hand_card` / `prune_removed_hand_links`；`drop_temporary_cards`、`discard_effects`、`choose()` 里的 `queue_marks` 维护也删除） |
| 差异 | HSAI 维护一张与手牌实体 id 同步的 `QueueMark` 表并做不变量校验；HSBRSIM 无这套结构与校验。 |

### R8 · 内容脚本的越界/阵营防御（仅 HSBRSIM）

| | |
|---|---|
| HSAI | `recruit.rs:631 sell_minion()` 直接 `remove(idx)`；`content/buddies_a.rs`、`content/heroes_active.rs`、`content/spells_p2.rs` 取目标后直接按 `me.side` 索引 |
| HSBRSIM | `recruit.rs:631-645`：`if idx >= self.boards[side].len()` → 打印 `[BUG] sell_minion out of range …` + 回溯并 `return`；`buddies_a.rs:528`、`heroes_active.rs:98-101`、`spells_p2.rs:382-386` 增加 `side != me.side \|\| i >= len` 检查，并按实际 `side` 调用 `sell_minion` |
| 差异 | 相同非法脚本输入下，HSAI 可能 panic（`Vec::remove` 越界）或误卖对侧；HSBRSIM 直接跳过。 |

### R9 · "Jar o' Gems"（`BG30_MagicItem_546`）触发语义

| | |
|---|---|
| HSAI | `content/trinkets_p8.rs:275-297`：监听 `ev::ATTACK_DONE`，条件包含 `c.combat_data(me.side, me.card) >= 1` 早退，并 `add_combat_data(…,1)` —— 即**每场战斗只触发一次** |
| HSBRSIM | `content/trinkets_p8.rs:274-292`：监听 `ev::ATTACKS`（每次攻击事件），无 `combat_data` 计数 —— 注释写 "after **every 2** friendly attacks" |
| 差异 | 同一场战斗内该饰品的触发次数与触发时机不同；监听的事件类型不同（`ATTACK_DONE` vs `ATTACKS`）。 |

### R10 · 内置 bot 的选择与回退

| | |
|---|---|
| HSAI | `bot.rs:39-49`：先算 `pick_option` 的 `Choose`，再查 `legal_mask`；不合法则取第一个合法动作，否则返回 `preferred` |
| HSBRSIM | `bot.rs:41`：直接 `return Action::Choose(pick_option(...))`，不做掩码校验；回退逻辑移到 `play_turn`（`bot.rs:208-215`）：优先 `EndTurn`，否则若 `A_CANCEL` 合法则 `Cancel`，否则 `Choose(0)` |
| 差异 | bot 在"首选选项非法"时的行为不同（HSAI 选第一个合法位；HSBRSIM 先试 Cancel）。 |

### R11 · 大厅种族池注释

| | |
|---|---|
| HSAI | `flow.rs:114/162` 无这两段注释 |
| HSBRSIM | `flow.rs:114-117/162-166` 增加两条注释（NAGA 按配置进入大厅、`BG32_820` 双种族不开放 NAGA）；**逻辑代码无差异**（diff 只有注释行） |
| 差异 | 仅注释，无行为差异（记录在案以免误判）。 |

### R12 · 引擎 lib.rs 的 fork 说明与版本常量注释

| | |
|---|---|
| HSAI | `lib.rs:3-6` 写明 "snapshot injection, audited fixes and dated live bans are explicit fork differences"；`GAME_VERSION` 注释称"保持上游值" |
| HSBRSIM | `lib.rs:2-5` 写明 "(AutoTavern 1.1.0, BG 37.0.2, MIT) … Public API and behaviour are otherwise identical to upstream"；`GAME_VERSION` 注释改为"保持上游 37.0.2 的值" |
| 差异 | 仅文档注释；`GAME_VERSION="37.0.2.256589"` 两边相同。 |

---

## 3. 状态、观测与动作空间

### O1 · 观测布局

| | |
|---|---|
| HSAI | `hsbg/env/obs_std.py:24-30`（注释：数据基线 37.0.2.256589）、`obs_std.py:191-205`（`N_CARDS_REFERENCE=3984`、`N_CARDS_NOTE`） |
| HSBRSIM | 同文件同位置，仅把"37.0.2.256589"写成"37.0.0.255790" |
| 差异 | **代码常量一致**，只有版本文案不同。注：HSBRSIM 该注释与其自身引擎 `cards.json` 的 `meta.client_version = 37.0.2.256589` 不一致（其文档内部口径）。 |

### O2 · 动作空间

| | |
|---|---|
| HSAI | `engine/crates/hsbg-engine/src/actions.rs`：`N_SHOP=8, N_BOARD=7, N_HAND=10, N_CHOICE=16, N_POS=8, A_END=0, A_REFRESH=1, A_FREEZE=2, A_UPGRADE=3, N_POWERS=2, A_HERO_POWER=4, A_DARK_GIFT=6` … |
| HSBRSIM | 同文件同常量，逐项相同；`hsbg/env/action_space_std.py` 与 HSAI **完全一致**（diff 无输出） |
| 差异 | 无。动作编号、段布局、245 个原生动作的定义相同。 |

### O3 · 引擎卡表数据

| | |
|---|---|
| HSAI | `engine/crates/hsbg-engine/data/cards.json`，SHA256 `7a02d107…` |
| HSBRSIM | `rust/crates/hsbg-engine/data/cards.json`，SHA256 `dbc820cc…` |
| 差异 | **语义相同**：两边 `meta` 完全一致（`client_version=37.0.2.256589`、`overlay=37.0.2.json`、`hsdata_commit=31b034c…`），3983 张卡逐字段比对 **0 处不同**；归一化（排序键、紧凑序列化）后哈希均为 `51e11b0f18a87068…`。字节差异来自 JSON 格式化/键序。 |
| 另 | `data/p2/cards_37.json` 两边 SHA256 均为 `c7e6c159…`（完全相同）。 |

---

## 4. 奖励与信用分配

### T1 · 奖励公式

| | |
|---|---|
| HSAI | `hsbg/az/train/ppo.py:54` 导入 `RANK_UTILITY`；`ppo.py:212` `value_from_rank_logits_np = (p * RANK_UTILITY).sum()`；`ppo.py:1053-1056` `rank_utility_coef=1.0`、`terminal_strength_coef=0.0` |
| HSBRSIM | `ppo.py:57`、`ppo.py:209`、`ppo.py:1091-1094` —— 数值与语义相同 |
| 差异 | 奖励主干（名次效用 + 终局强度项）**一致**；差异只在采样侧的"局级权重"（见 T4）。 |

### T2 · 局级奖励权重（仅 HSBRSIM）

| | |
|---|---|
| HSAI | 无 |
| HSBRSIM | `ppo.py:1691 _episode_advantage_weights(game_ids, seats, turns, …)`；`build_batch_from_records(..., game_weight_mode/alpha/beta/floor/cap)`（签名多 5 个参数）；`metrics.py:36-60` 提供 `game_weights/weighted_mean/effective_sample_size` |
| 差异 | HSBRSIM 支持按局重加权优势函数；HSAI 无此参数与实现。 |

---

## 5. 数据采样

### T3 · `CollectConfig` 字段集

| | |
|---|---|
| HSAI | `ppo.py:307 class CollectConfig`，**74** 个字段；独有：`forbid_cancel`、`leaf_fused`、`mirror_strict`、`forced_tier_plan`、`forced_tier_carry`、`forced_tier_prob`、`forced_tier_teacher` |
| HSBRSIM | `ppo.py:304 class CollectConfig`，**96** 个字段；独有 26 个：`game_weight_mode/alpha/beta/floor/cap`、`forced_upgrade`、`forced_upgrade_margin`、`comp_buy_*`（13 个）、`forced_upgrade_max_tier`、`tier_spread`、`collect_discover_bias`、`collect_triple_bias`、`collect_chain_buy_bias`、`collect_chain_ids`、`collect_reroll_bias_t8t9`、`record_logp_mu` |
| 差异 | 两套互不相同的探索/约束开关：HSAI 是"强制升本 + 镜像严格 + 叶子融合"；HSBRSIM 是"comp-buy 定向购买 + 强制升级 + 局级权重 + 若干 collect bias"。 |

### T4 · 采样默认值

| 字段 | HSAI | HSBRSIM |
|---|---|---|
| `max_actions_per_turn` | `ppo.py:342` = **80** | `ppo.py:347` = **40** |
| `move_guard` | `ppo.py:346` = **True** | `ppo.py:352` = **False** |
| `move_cap` | `5` | `DEFAULT_MOVE_CAP`（常量） |

### T5 · 采样器主循环

| | |
|---|---|
| HSAI | `parallel_collect.py`：`_one_game` 内注入 forced-tier/teacher/mirror-strict |
| HSBRSIM | `parallel_collect.py:379 _max_opp_strength()` 新增；`_one_game` 内注入 comp_buy / forced_upgrade / game_weight |
| 差异 | 同一函数骨架，注入的策略偏置不同（见 T3 字段表）。 |

---

## 6. 训练算法与损失

### T6 · `PPOConfig` 字段集

| | |
|---|---|
| HSAI | `ppo.py:1042 class PPOConfig`，**71** 字段；独有 `forced_teacher_coef` |
| HSBRSIM | `ppo.py:1080 class PPOConfig`，**80** 字段；独有 10 个：`ceiling_aux_coef`、`discover_bc_coef`、`triple_bc_coef`、`bc_guard_ratio`、`discover_bc_scale`、`triple_bc_scale`、`upgrade_bc_coef`、`upgrade_bc_scale`、`upgrade_bc_guard_ratio`、`bc_ce_norm` |

### T7 · 损失函数

| | |
|---|---|
| HSAI | `ppo_losses(...)` 签名与 HSBRSIM 完全相同（`ppo.py:2464` vs SIM `2580`）；额外系数：`forced_teacher_coef` |
| HSBRSIM | 新增：`ceiling_aux_coef`（终局上限辅助损失）、`replication_bc_ce()`（`ppo.py:2542`）、`replication_bc_guard()`（`ppo.py:2568`）、`_rep_card_flags()`/`_ReplicationBias`（`ppo.py:3626-3785`） |
| 差异 | 两侧 PPO 主损失结构一致；HSBRSIM 多"终局上限头损失"和"复刻 BC（含 CE 与 guard）"两支，HSAI 多"强制教师损失"一支。 |

### T8 · 训练循环

| | |
|---|---|
| HSAI | `train_ppo(...)` `ppo.py:2831` |
| HSBRSIM | `train_ppo(...)` `ppo.py:3070`；签名相同 |
| 差异 | 循环骨架相同，内部按 T6/T7 的额外损失项与 T3 的采样开关分支不同。 |

---

## 7. 模型结构

### M1 · 卡表加载路径

| | |
|---|---|
| HSAI | `entity_model.py:95-108`：默认读 `engine/crates/hsbg-engine/data/cards.json` |
| HSBRSIM | 同函数：默认读 `data/p2/cards_37.json`（注释 "37.0.0 数据"） |
| 差异 | 静态特征与文本哈希的**数据来源文件**不同（两份文件卡 id/数值一致，见 O3；但字段集合与组织方式可能不同，取决于两边解析代码）。 |

### M2 · 终局上限辅助头（仅 HSBRSIM）

| | |
|---|---|
| HSAI | 无 `ceiling_head` |
| HSBRSIM | `entity_model.py:480 CEILING_HEAD_OUT = 4`；`attach_ceiling_head()`（零初始化 `Linear(d_model,4)`，`entity_model.py:489-508`）；`forward_ceiling()`（`entity_model.py:302-309`）；`forward_ceiling_model()`（`entity_model.py:511+`） |
| 差异 | HSBRSIM 的模型可挂 4 维终局头（`log1p(Σatk+hp)`、`tier/6` 等），与策略/价值共享 trunk；HSAI 无此头。 |

### M3 · 既有辅助头

| | |
|---|---|
| HSAI / HSBRSIM | 两边都有 `strength_head`（t+3 增量）、`process_head`（过程价值 3 维）、`move_from/move_to` 拆分、`problem` 头；代码位置基本对应（`entity_model.py` 同段） |
| 差异 | 除 M2 外，网络结构一致。 |

---

## 8. 评估机制

### E1 · 评测配置与采集项

| | |
|---|---|
| HSAI | `evaluate.py:57-64`：`mirror_pool`、`mirror_strict`、`collect_economy`、`collect_exposure`、`forced_tier_plan`、`forced_tier_carry`；`evaluate.py:111` 起有 `if config.mirror_strict:` 分支；结果里输出 `mirror`/`combat_exposure`/`end_turn_economy`/`forced_tier_actions` 等字段 |
| HSBRSIM | 上述字段与分支**全部删除**（`evaluate.py` 比 HSAI 少约 204 行） |
| 差异 | HSAI 的评测支持"固定回合末镜像库（严格 EOT 快照）"、"经济/曝光采集"、"强制升本探针"；HSBRSIM 的评测不含这些。 |

### E2 · 评测指标

| | |
|---|---|
| HSAI | `hsbg/eval/metrics.py`：`avg_rank_percentile` 等基础指标；另有 `duel_metrics.py`（HSAI 独有，304 行） |
| HSBRSIM | `metrics.py:35-171` 新增：`percentile_ranks`、`game_weights`、`weighted_mean`、`effective_sample_size`、`weighted_eval_summary`、`long_strong_subset` |
| 差异 | HSBRSIM 多了"加权评测 + 有效样本量 + 长局强局子集"指标；HSAI 多了 1v1 对局指标模块。 |

### E3 · 仅 HSAI 的评估/安全模块

`hsbg/eval/live_simulator_check.py`（实机↔模拟器状态比对）、`hsbg/az/train/strict_mirror.py`（严格镜像）、
`hsbg/az/train/pending_guard.py`（挂起菜单守卫）、`hsbg/az/train/checkpoint_io.py`、
`hsbg/az/train/collect_diagnostics.py`、`device_assignment.py`、`engine_quality.py`、
`local_batch_eval.py`、`metric_transfer.py`、`pipeline_safety.py` 等（共 74 个仅 HSAI 的 `hsbg/**` 文件）。

### E4 · 仅 HSBRSIM 的模块

`hsbg/az/train/game_rules.py`（环境变量驱动的规则覆盖：`HSBG_TIER_UPGRADE_DISCOUNT`、`HSBG_GOLD_BONUS_TURNS`、
`HSBG_BANNED_TRINKETS` 等）、`forced_upgrade.py`（强制升本规则）、`comp_buy.py`（定向购买）、
`comp_offline_policy.py`、`combat_sandbox.py`、`action_counts.py`。

---

## 9. 训练流程与工具

### P1 · 入口与编排

| | |
|---|---|
| HSAI | `hsbg/az/train/personal.py`（643 行，仅 HSAI）+ `workflows/`（121 文件）+ `deploy/slurm/*.sbatch`（26 个） |
| HSBRSIM | 无 `workflows/`、无 `deploy/`；入口在 `scripts/`（1,568 个文件）与 `benchmarks/`（39 文件）；Python 侧有 `hsbg/env/`（gym 风格）+ `hsbg/az` |
| 差异 | 训练编排层完全不同：HSAI 是"workflows + Slurm 提交"；HSBRSIM 是"scripts + 本地/集群脚本"。 |

### P2 · 测试

| | |
|---|---|
| HSAI | `tests/` 45 个 py 测试文件（含 `test_live_*` 实机链路 4 套）；引擎 41 个 `.rs`，独有 `server_rules_audit.rs` |
| HSBRSIM | `tests/` 101 个 py 测试文件；引擎 41 个 `.rs`，独有 `empty_mask.rs`（对应 R3）、`trinket_ban.rs`（对应 R2） |
| 差异 | 两边各自为对方的规则差异写了回归测试：HSAI 有"服务端规则审计"测试；HSBRSIM 有"空掩码""禁用饰品"测试。`mechanics.rs` 两边差异 186 行（断言随 R3/R8/R9 变化）。 |

### P3 · 数据与脚本体积

| | |
|---|---|
| HSAI | `data/` 7.6 MB / 20 文件；无 `scripts/` |
| HSBRSIM | `data/` 403 MB / 653 文件（含 `p2/`、`real_library/` 等）；`scripts/` 37 MB / 1,568 文件；`benchmarks/` 13 MB / 39 文件 |
| 差异 | HSBRSIM 附带完整数据管线与基准集；HSAI 只保留运行所需的最小数据集（卡表、饰品、池、archetype）。 |

---

## 10. 差异总表（供逐项审计）

| 编号 | 领域 | HSAI 位置 | HSBRSIM 位置 | 差异一句话 |
|---|---|---|---|---|
| R1 | 规则 | `config.rs:11,73,257`；`flow.rs:169/189/195/207/757` | `config.rs:79,282`（无 const） | 线上禁用 4 张卡是否默认生效 |
| R2 | 规则 | — | `config.rs:38-49`；`flow.rs:382/675`；`recruit.rs:1663` | 禁用饰品/升本折扣/逐回合金币 4 个新开关 |
| R3 | 动作合法性 | `actions.rs:229-231` | `actions.rs:219-231` | 空掩码时是否放行 `A_CANCEL` |
| R4 | 挂起菜单 | `actions.rs:154`；`flow.rs:534`；`recruit.rs:2523/2325` | `recruit.rs:2322/2357/2457` | 买不起饰品单的清理时机与可见性 |
| R5 | 回合末 | `flow.rs:795,814-840,1272` | `flow.rs:800-810` | 无解菜单：报错保留 vs 清空继续 |
| R6 | 回合末 | `flow.rs:844-849` | `flow.rs:849` | `end_turn_silent` 是否清空 pending |
| R7 | 手牌状态 | `state.rs:1050/1082/1089`；`engine.rs:226/3243`；`invariants.rs:59` | 无 | `queue_marks`/link 完整性与不变量检查 |
| R8 | 内容脚本 | `recruit.rs:631` 等 | `recruit.rs:631-645`；`buddies_a.rs:528`；`heroes_active.rs:98`；`spells_p2.rs:382` | 越界/阵营防御 |
| R9 | 内容脚本 | `trinkets_p8.rs:275-297`（`ATTACK_DONE`） | `trinkets_p8.rs:274-292`（`ATTACKS`） | Jar o' Gems 触发语义 |
| R10 | bot | `bot.rs:39-49` | `bot.rs:41`、`208-215` | 首选动作非法时的回退 |
| R11 | 注释 | — | `flow.rs:114-117/162-166` | 仅注释 |
| R12 | 注释 | `lib.rs:3-6` | `lib.rs:2-5` | 仅 fork 说明 |
| O1 | 观测 | `obs_std.py:24-30/191-205` | 同位置（版本文案不同） | 常量一致 |
| O2 | 动作空间 | `actions.rs` 常量 | 同 | 无差异 |
| O3 | 卡表 | `cards.json` `7a02d107…` | `cards.json` `dbc820cc…` | 语义相同、字节不同 |
| T1 | 奖励 | `ppo.py:54/212/1053-1056` | `ppo.py:57/209/1091-1094` | 一致 |
| T2 | 奖励权重 | — | `ppo.py:1691`；`build_batch_from_records` +5 参数 | 局级重加权 |
| T3 | 采样配置 | `CollectConfig` 74 字段 | 96 字段 | 两套不同开关集 |
| T4 | 采样默认值 | `80` / `True` | `40` / `False` | 每回合动作上限、move 守卫默认 |
| T5 | 采样器 | `parallel_collect._one_game` | 同 + `_max_opp_strength` | 注入的偏置不同 |
| T6 | 训练配置 | `PPOConfig` 71 字段 | 80 字段 | ceiling/replication BC 系数 |
| T7 | 损失 | `ppo_losses` | 同 + 2 组新损失 | ceiling aux、replication BC |
| T8 | 训练循环 | `train_ppo:2831` | `train_ppo:3070` | 骨架同，分支不同 |
| M1 | 模型输入 | `entity_model.py:95-108` → 引擎 cards.json | → `data/p2/cards_37.json` | 卡表来源不同 |
| M2 | 模型头 | 无 | `entity_model.py:476/486+` | 4 维终局上限头 |
| M3 | 其他头 | `strength/process/move/problem` | 同 | 一致 |
| E1 | 评测 | `evaluate.py:57-64/111` 等 | 删除 | mirror/economy/exposure/forced-tier |
| E2 | 指标 | `metrics.py` 基础指标 + `duel_metrics.py` | `metrics.py:35-171` 加权指标 | 指标集不同 |
| E3/E4 | 模块 | 74 个仅 HSAI | 6 个仅 HSBRSIM | 见 §8 |
| P1 | 编排 | `personal.py`+`workflows/`+`deploy/` | `scripts/`+`benchmarks/` | 入口不同 |
| P2 | 测试 | 45 py / 41 rs（含 `server_rules_audit`） | 101 py / 41 rs（含 `empty_mask`、`trinket_ban`） | 测试集不同 |
| P3 | 数据 | 7.6 MB / 20 文件 | 403 MB / 653 文件 + `scripts/` 37 MB | 附带数据不同 |

---

## 11. 未覆盖与限制

1. **未逐行比对** `ppo.py` 中 1,063 行差异的全部内容（仅比对字段集、签名、损失函数骨架与关键新增函数）；
   `parallel_collect.py`/`process_pool.py` 只比较了函数骨架与新增入口。
2. **未逐一核对 74 个"仅 HSAI"与 6 个"仅 HSBRSIM"模块的语义**（只列名与一句话用途）。
3. 未运行任何一方的测试或训练；所有结论来自静态代码与数据比对。
4. HSBRSIM 仓库含 `scripts/.remote`、`.tmp_*` 等运行残留目录，未纳入比对。
5. 引擎 `data/cards.json` 的字节差异原因（格式化/键序）已用归一化哈希确认，但**未定位到具体是哪个工具/步骤造成的序列化差异**。
6. HSBRSIM `obs_std.py` 注释中的"37.0.0.255790"与其引擎 `meta.client_version=37.0.2.256589` 不一致，本报告只记录该现象，未追溯原因。
