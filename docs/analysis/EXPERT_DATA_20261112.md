# 专家数据：`bc_samples` 断点的定位与打通（2026-11-12）

状态：**当前**。这条链路此前从未跑通，是"教会模型中后期运营"最主要的阻塞点。

---

## 1. 问题

诊断（见 [`OPPONENT_STRENGTH_20261112.md`](OPPONENT_STRENGTH_20261112.md) §6）指出：
模型**卖随从只有真人的 0.22 倍、搜牌 0.67 倍**，且 960 局里没有一局拼出 3 张同流派核心卡。
卖牌在纯终局名次奖励下当下像"亏价值"，收益要 5–10 回合后才兑现——**长程信用分配问题**。

针对它的现成手段是"用真人动作分布做软标签"：

| 模块 | 用途 | 依赖 |
|---|---|---|
| `hsbg/az/train/bc_aux.py`（1392 行） | 分层 BC 损失（9 类动作类型 CE + 卡牌身份 CE） | `data/firestone_bc/bc_samples.jsonl.gz` |
| `hsbg/az/train/board_problems.py` | 在**自对弈状态**上按 `(turn,tier)` 匹配真人参考分布，配 t+3 真实结果 | 同上 |
| `hsbg/az/train/dpo_upgrade.py` | 升本子集的人择 DPO | 同上 |
| `workflows/bc_pretrain.py`（929 行） | 纯监督预训练 | 同上 |

**全部只读、无一产出。** 代码树里 `grep bc_samples` 只有消费方，没有生产方。

## 2. 根因

`workflows/firestone_bc_parse.py`（718 行）是唯一的候选生产方，但它期望的输入是

```
<root>/player_<hash>/<game-uuid>/events.jsonl.gz
```

而本项目实际拿到的是 `xmlz/*.xml.zip`（每局一个 HSReplay XML）+ `pms/*.json`。
两者对不上，所以它一次都没跑出过东西——**这条链路是"写了但从未接线"**，
与 `build_human_curve_bank.py` 的情况完全同类。

## 3. 真正的解法：数据早就是对的形状

检查 `data/real_library/firestone-20261112-1000/actions.jsonl.gz` 发现，
297,884 条动作记录的**字段名与 `load_bc_samples` 要求的完全一致**：

```
action_type  action_index  card_id  turn  gold  tier  health  armor
shop_cards   hand_cards    board_cards        game_id  source
```

而且动作类型分布对 9 类白名单的覆盖率是 **96%**：

| 动作 | 条数 | 在白名单 |
|---|---:|---|
| play_minion | 80,944 | ✅ |
| sell | **54,319** | ✅ |
| play_spell | 50,412 | ✅ |
| reroll | 40,577 | ✅ |
| buy_minion | 22,842 | ✅ |
| buy_spell | 13,498 | ✅ |
| end_turn | 13,475 | ✅ |
| triple | 6,486 | ❌（白名单外，自动跳过） |
| hero_power | 5,312 | ✅ |
| upgrade | 4,795 | ✅ |
| dark_gift | 2,794 | ❌ |
| freeze | 2,430 | ❌ |

真正缺的只有两样，且都**可以按局关联、不需要伪造**：

1. **`mmr`** —— 动作流里没有，从 `library.jsonl` 按 `game_id` 关联；
2. **`hero`** —— 同上（`hero_card_id`）。

### 3.1 踩到的坑：英雄取错了玩家

`library.jsonl` 每局同时有 `self` 与 `opponent` 两行，**对手的英雄卡不同**。
第一版不分视角直接取第一行，导致 **10,776 局的 hero 与动作流不属于同一玩家**。
已改为只取 `perspective == 'self'` 的行。

## 4. 产物

`tools/build_bc_samples.py`（约 180 行，只做关联与字段改名，不改数值、不填缺失、不造样本）：

```
data/firestone_bc/bc_samples.jsonl.gz   5,013,577 bytes
```

复核（`load_bc_samples`，job 实测）：

| 口径 | 样本数 |
|---|---:|
| `mmr_min=0`（类别白名单后） | **282,717** |
| 默认 `mmr_min=6000` | **209,310** |

9 类全部有样本：`play_minion 77,489 / sell 54,318 / play_spell 50,411 /
reroll 40,577 / buy_minion 22,842 / buy_spell 13,498 / end_turn 13,475 /
hero_power 5,312 / upgrade 4,795`。

顺带自洽性检查：`sell` 54,318 条 ÷ 995 局 ≈ **54.6 次/局**，与 §6 里从 PMS 独立算出的
真人 **54.2 次/局** 一致——两条互不相干的数据源对上，说明解析没有系统性错位。

## 5. 下游已可运行

- `board_problems.build_reference_table()` 已实测成功：生成
  `ReferenceTable(floors=(10000, 8000, 6000), tables={10000: {'by_turn_tier_gold': ...}})`，
  键为 `(turn, tier, …, gold)`，值是该桶内真人的动作分布（含 `sell` / `reroll`）。
- 因此 `--bc-coef`（PPO 内辅助 BC）、`workflows/build_board_problems.py`（自对弈状态上的
  软标签）、`dpo_upgrade` 都不再有前置缺失。

## 6. 已知限制

1. **样本是胜者偏差**的（816/995 是第 1 名），学的是"赢家怎么打"，强度偏高。
2. **`shop_pos` / `hand_pos` / `board_pos` 未提供**：`_place_cards` 在缺位置时按列表下标
   落位，所以槽位顺序可能与真实 UI 槽位不同。对动作类型 CE 无影响，对"按槽位"的
   精细监督有影响。**未伪造位置**。
3. **`armor`、`triple`、`dark_gift`、`freeze` 未覆盖**（前两者白名单外 / 字段缺失）。
4. **`hero` 已修正为 self 视角**，但仍有 5 局 PMS 缺失（`xml_ok=995/1000`）。

## 7. 下一步

用 `--bc-coef` 做一次**低系数的辅助 BC**（不是纯 BC 预训练——历史上 9 类纯 BC 会把
moves/turn 从 2.4 推到 17.3、top4 从 96% 打到 29%，见 `bc_pretrain.py` 文档里的实测记录），
并用 §4 的机制判据（sell/reroll 每回合次数、≥2 核心卡比例）验证"换牌"是否松动。
