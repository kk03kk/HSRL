# 统一对局记录规范 v2（模拟器 / 实机）与模拟器整局验收

```yaml
status: current
owner: local
updated: 2026-11-11 12:35
schema: hsai.game-trace/v2
```

本文件定义**模拟器与实机共用的唯一一套对局记录**，并给出本次审计的问题清单、
修复内容和模拟器整局验收结果。代码入口：

| 角色 | 文件 |
|---|---|
| 规范（纯函数，双侧共用） | [`client/hsrl_agent/trace_schema.py`](../../client/hsrl_agent/trace_schema.py) |
| 异步写入器（单写线程，双侧共用） | [`client/hsrl_agent/live/trace.py`](../../client/hsrl_agent/live/trace.py) |
| 模拟器生产者 | [`client/tools/sim_decisions.py`](../../client/tools/sim_decisions.py) |
| 实机生产者 | `client/hsrl_agent/live/{controller,pipeline,runtime}.py` |
| 离线审计 / 跨端对照 | [`client/tools/trace_audit.py`](../../client/tools/trace_audit.py) |
| 回归测试 | [`tests/test_live_trace_unified.py`](../../tests/test_live_trace_unified.py) |

---

## 1. 修复前的问题（实测证据）

以 `artifacts/runs/live-e-20261111-004/match-0002`（13.7 MB，14 回合，第 2 名）为例：

| # | 问题 | 证据 |
|---|---|---|
| 1 | **等待轮询被当成决策**。同一状态、同一动作、缓存命中的重复读取也写进 `decisions.jsonl` | 753 行里 666 行 `reused=true`；T1 50 行里 46 行是 `end` 轮询 |
| 2 | **动作分布被污染**，`end` 占 85%，看起来像"模型一直点结束回合" | 按新规则重放后 T1 只有 `buy / play / end` 三条 |
| 3 | **没有执行后状态**：`decisions.jsonl` 只有执行前快照，`after` 只在 `actions.jsonl`，要靠 id 手工关联 | `summarize()` 只能给出 outcome，不能回答"这一步改变了什么" |
| 4 | **没有差值**：金币/等级/战团/手牌/商店的前后变化全靠事后重算 | 同上 |
| 5 | **模型输入不可回溯**：只存 `observation_sha256`，没有原始向量 | 无法逐位比较 sim×live 的输入 |
| 6 | **模拟器侧记录字段与实机不同名**（`real_observation.board[].card` vs `.card_id`、无 `phase`、无 `legal` 全量、无 `checkpoint_sha256`） | 旧 `client/tools/sim_decisions.py` |
| 7 | **实体身份错误**：商店/手牌是 `{"sid"|"hid": …, "item": {"Minion"|"Spell": …}}` 容器，旧代码直接读 `row["card"]`，拿到 `None`，意图退化成 `entity:None` | 本次修复前的记录 |
| 8 | **MoveGuard 返回值误用**：`guarded_mask()` 返回 `(mask, forced)`，旧代码把元组当掩码，`np.asarray` 直接抛异常或静默失效 | 真实卡死的直接原因之一 |
| 9 | **内部事件与执行混写**：脚本重试、分支、阻断与物理点击写在同一个 `actions.jsonl` | 统计口径无法区分 |
| 10 | **无完整性校验**：没有"每条决策都有执行、终态、无重复、回合单调"的检查 | 缺记录只能靠肉眼 |

---

## 2. 规范：一条流只承担一种语义

两个生产者写**同名、同义**的文件：

| 文件 | 内容 |
|---|---|
| `decisions.jsonl` | **模型真实决策**，一行一次；复用轮询、脚本重试不进这里 |
| `decisions-full.jsonl.gz` | 每次决策的完整快照 + 原始模型输出（gzip JSONL） |
| `actions.jsonl` | 每次**执行尝试**：手势/引擎计划、时序、`before`/`after`、`delta`、`status`/`outcome` |
| `waits.jsonl` | **等待轮询**：同状态同动作的重复读取 |
| `events.jsonl` | **内部事件**：`other_player_action`、`action_masked`、`turn`、`seat_eliminated`、`game_end` … |
| `observations.npy` / `observations-cards.npy` | 真实模型输入向量 `(N,3227)` float32 / `(N,124)` int32，按决策顺序 |
| `decisions-resolved.jsonl` | 派生：决策 + 执行 + 执行后状态 + 差值 |
| `summary.json` / `performance.json` / `audit.json` / `decision-report.md` | 聚合、耗时、完整性、人读报告 |

### 2.1 decision 行（关键字段）

```jsonc
{
  "schema": "hsai.game-trace/v2", "side": "sim",          // sim | live
  "visibility": "owner_full",                              // live: owner_public_projection
  "sequence": 42, "decision_index_in_game": 41, "decision_id": "d-…",
  "parent_decision_id": null,                              // 分支/目标续决策挂父 id
  "game_id": "sim-700001", "seat": 0, "role": "model",
  "turn": 7, "phase": "Recruit", "revision": 118,
  "action": 46, "kind": "play", "action_class": "play",
  "intent": "play BG36_360t3 @slot 3",
  "source": 403, "target": null, "position": 3,
  "source_card_id": "BG36_360t3",
  "policy_ms": 12.7, "reused": false, "repeat_index": 0,
  "binding_count": 37, "legal": [0,1,2,…], "legal_actions": [ … 最多 40 条 … ],
  "input": {                                              // 模型真实输入指纹
    "observation_sha256": "…", "mask_sha256": "…", "logits_sha256": "…",
    "vector": {"sha256":"…","shape":[3227],"dtype":"float32","nonzero":96,"abs_sum":…},
    "cards":  {"sha256":"…","shape":[124],"dtype":"int32", …}
  },
  "checkpoint_sha256": "5069aa62…",
  "state_key": "4c81e5a497b5da9f",                         // 同端内状态判据
  "board_signature": ["BG20_100:2:3:0:TAUNT", …],          // 跨端可比的战团指纹
  "pre": { …统一视图… },                                    // 执行前
  "inferred_view": {"pending":0,"approximation_fields":[…],"approximation_mode":"sim-native"},
  "plan": [ … ]
}
```

`pre` / `after` 是**同一形状**的我方视图：`turn/phase/gold/tavern_tier/upgrade_cost/
health/armor/hero/board/hand/shop/choices/controls/counters`，卡牌统一为
`entity_id/card_id/name/card_type/position/attack/health/tier/cost/golden/tags`。
实机 `client` 与模拟器 `S.sim_view(engine_state, seat, catalog)` 都过同一个
`trace_schema.view()`，键集合完全一致（有回归测试）。

### 2.2 execution 行

```jsonc
{"event":"execution","decision_id":"d-…","attempt":1,"channel":"mouse|engine",
 "physical":true,"status":"sent|receipt|applied|cancelled_before_press|blocked_before_gesture|delivery_unknown",
 "outcome":"applied|rejected|blocked|halted", "error":null,
 "plan":[…], "timing":{"elapsed_ms":4.0}, "native_trace":{…},
 "before":{…}, "after":{…}, "delta":{…}, "received_at_log_position":120}
```

`delta` 结构：`scalars`（gold/tier/health/… 的 `[前,后]`）、`board|hand|shop`
（`removed`/`added`/`order_changed`/`stats`，列表有上限并标 `omitted`）、
`counters`（引擎计数增量）、`changed`。**任一侧缺状态就是 `null`，绝不推断。**
`changed=false` 是一个有意义的结论：这一步在可见状态上什么都没改变
（模拟器实测：`end` 5 次、法术 `play` 17 次，见 §5.2）。

### 2.3 计数口径（避免重复统计）

| 名称 | 定义 |
|---|---|
| 决策数 | `decisions.jsonl` 行数（每个**新状态上的新意图**） |
| 重新推理数 | 其中 `reused=false`（真正跑了一次前向） |
| 等待轮询 | `waits.jsonl` 行数（同状态同动作的重复读取） |
| 有效动作 | 逐回合内动作身份（kind+source+target+position）与上一条不同的决策 |
| 模型真实动作 | `role=model` 且 `outcome=applied` 的执行行 |
| 其他玩家动作 | `events.jsonl` 的 `other_player_action`（模拟器 7 席） |

---

## 3. 两侧的固有差异（记录里显式标注，不假装等价）

| 维度 | 模拟器 | 实机 |
|---|---|---|
| `visibility` | `owner_full`（引擎真值，含对手手牌/随机池） | `owner_public_projection`（Power.log 公开投影） |
| `side` | `sim` | `live` |
| 执行通道 | `channel=engine`，同步返回 | `channel=mouse`，异步回执 |
| `phase` 取值 | `HeroSelect` / `Recruit`（战斗在 `EndTurn` 内结算） | `recruit` / `combat` / `choice` |
| `controls` | 空（无 UI） | 实机控件实体 |
| `entity_id` 体系 | 引擎 `eid`（法术用 `hid`/`sid`） | 客户端实体 id |
| `approximation_fields` | `controls`、`shop[].cost`、`hero.entity_id`、`choices` | 每次决策写 `approximation.json` 列出的不可重建字段 |
| 等待轮询 | 无（`waits.jsonl` 为空） | 有（倒计时/动画） |
| 面板 | 8 席全部由同一网络驱动（与训练自对弈一致） | 1 模型 + 7 Mean Bot 快照 |

`board_signature`、`observable()`（`turn/phase/gold/tavern_tier/upgrade_cost/health/armor`）、
`delta` 是跨端比较的公共词汇；`state_key` 只在同端内有意义。

---

## 4. 修复清单（本轮代码）

| 文件 | 改动 |
|---|---|
| `client/hsrl_agent/trace_schema.py` | **新增**。规范、统一视图、引擎动作解码（245→kind/槽位）、关键词位域解码、`delta`、`decision_id`、`join`、`audit`、逐回合统计、`sim_view` |
| `client/hsrl_agent/live/trace.py` | 重写。流分离（decisions/actions/waits/events/full/npy）、复用即等待、`pre/after/delta`、`repeat_index`、`decisions-resolved.jsonl`、`audit.json`、向量落盘、队列 2048、`_jsonable` 支持 `__slots__` 绑定视图 |
| `client/hsrl_agent/live/controller.py` | 3 处 `trace.decision(...)` 补 `seat/role/inputs`；`event()` 把重试/阻断/分支/回执等低频事件镜像进 `trace/events.jsonl`（`TRACE_EVENTS` 白名单，**不改任何决策/执行语义**） |
| `client/hsrl_agent/live/policy.py` | `decide()` 末尾记录 `last_inputs`（features/cards/mask + 摘要）；决策逻辑一字未改 |
| `client/hsrl_agent/live/runtime.py` | `GameTrace(out, side="live", store_vectors=True)` |
| `client/tools/sim_decisions.py` | 重写为 `GameTrace(side="sim")` 生产者；修正 MoveGuard 元组解包；补 `guard_forced_end`、`turn`/`seat_eliminated`/`game_end`/`combat` 事件、逐回合预算、`--max-decisions`；**其它 7 席改为引擎 bot（`game.bot_action`，不注入对手库）**；**采样前 `without_cancel` 清 Cancel(7)**（与 36h 长训 `forbid-cancel=1` 对齐），命中时写 `cancel_masked` 事件 |
| `client/tools/trace_audit.py` | **新增**。`check` / `replay` / `compare` 三个离线命令 |
| `tests/test_live_trace_unified.py` | **新增** 11 条回归 |
| `tests/test_live_resource_migration.py` | 两条硬编码 V1.1 的断言改为跟随 `config/live/model.json`（模型已换成 `league-latest-12311`） |

**未改动**：模型策略、掩码生成、动作绑定、游戏规则、实机点击/回执语义、服务器拥有的
`engine/`、`hsbg/`、`workflows/`、`tests/(非 test_live_*)`。

### 4.1 开销

`trace.decision()` 实测 **0.23–0.31 ms/次**（队列 2048，`events_dropped=0`），
相对 12.6 ms 的推理是 **~2%**；每局产物约 3–4 MB（131 决策）。写盘全在单写线程，
热路径只做 dict 组装与入队。

---

## 5. 模拟器整局验收（2026-11-11）

```
python client/tools/sim_decisions.py --games 1 \
  --checkpoint artifacts/models/league-latest-20261110-step012311/model.pt \
  --out artifacts/local/sim-trace-20261111-001 --seed 700001
```

| 项 | 值 |
|---|---|
| 模型 | `league-latest-12311`，`sha256 5069aa62…d29cd`，16,436,034 参数 |
| 引擎 | `hsbg_engine` `37.0.2.256589`，`source_hash c4c71014e3d41760` |
| 契约 | `N_ACTIONS=245`，`OBS_FLOATS=3227`，`OBS_CARDS=124`，`N_CARDS=3984` |
| 种子 / 席位 | `700001` / seat 0 |
| 结果 | **第 7 名**，被淘汰于 T11，对局共 **17 回合**（其余 7 席打到结束） |
| 模型决策 | **131** 次（全部 `reused=false`），**131** 条执行，**131** 条向量 |
| 其他玩家动作 | 1,566（单独事件流，不计入模型决策） |
| MoveGuard 屏蔽 | 12,154 个 move 动作位（405 条 `action_masked` 事件） |
| 完整性 | `audit.json: complete=true`，`missing_execution=0`，`unresolved_execution=0`，`write_errors=0`，`events_dropped=0` |
| 未截断战斗 | `truncated_combats=0` |
| 事件流 | `other_player_action` 1,566 · `turn` 17（回合切换）· `combat` 9（引擎战果）· `combat_damage` 10（跨回合生命差）· `action_masked` 405 · `action_budget_exhausted` 2 · `seat_eliminated` 1 · `game_end` 1 |
| 耗时 | 37 s（含 1,697 次前向；推理 p50 12.3 ms / p90 12.9 ms） |

### 5.1 逐回合动作类型（模型席位）

| 回合 | 金币 | 等级 | 决策 | 有效动作 | 动作类型 |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | 1 | choose×1（英雄选择） |
| 1 | 3 | 1 | 3 | 3 | buy×1, play×1, end×1 |
| 2 | 4 | 1 | 2 | 2 | upgrade×1, end×1 |
| 3 | 5 | 2 | 7 | 7 | buy×1, choose×2, dark_gift×1, play×2, end×1 |
| 4 | 6 | 2 | 7 | 7 | buy×1, choose×2, dark_gift×1, play×2, end×1 |
| 5 | 7 | 2 | 10 | 10 | choose×3, dark_gift×1, play×4, upgrade×1, end×1 |
| 6 | 8 | 3 | 3 | 3 | choose×1, upgrade×1, end×1 |
| 7 | 9 | 4 | 18 | 16 | buy×3, choose×3, play×6, refresh×4, sell×1, end×1 |
| 8 | 10 | 4 | 19 | 19 | buy×5, choose×3, play×7, refresh×3, end×1 |
| 9 | 10 | 4 | 36 | 27 | buy×3, choose×7, play×11, refresh×3, sell×1, **freeze×10**, end×1 |
| 10 | 10 | 4 | 25 | 14 | buy×2, choose×3, play×3, refresh×5, move×1, **freeze×10**, end×1 |

（`有效动作`＝动作身份与上一条不同的决策数；差值就是同一意图在新状态上的重读/重试。）

### 5.2 记录揭示的两个规则级异常（待服务器端决策，本轮未改）

1. **`Freeze` 是开关，模型可以原地打摆**：T9/T10 各 10 次连续 `freeze`，
   `delta` 显示商店 `FROZEN` tag 在 `1→无→1` 之间来回；金币与战团不变。
   实机桥接 `bind_actions()` 在"商店已全冻"时**不提供** Freeze，所以实机不会出现；
   模拟器只靠 80 动作/回合上限兜底。
2. ~~**法术 `Play` 有 68% 是空操作**~~ → **已撤回（2026-11-11 13:50 复核）**。真相是引擎的
   **两步施法**：`Play{hand,pos:0}` 只是"开始施法"，之后 `players[seat].pending` 变成
   `[{"Target": {"options":[...实体 id...]}}]`，合法动作只剩 `Cancel(7) + Choose(0..k)`；
   模型接着选 `Choose(k)` 选目标，`spells_cast+1` 与属性变化都发生在**第二步**。
   实测：37 次"无变化"的 play **全部**紧随一条 `choose`（37/37），其 delta 里
   `spells_cast/cards_played +1` 且目标随从属性变化。
   这是**记录缺陷**（`delta` 没有协议态、两次决策没有父子链），不是规则问题：
   `delta.scalars` 现已加入 `pending_kind`，两步动作以 `parent_decision_id` 串成一条链。
   实机同样是两步（`play` 带 targets → 再决策一次 target/branch），因此两边结构一致。

Freeze 仍是真问题（见 §6.6，已在录制器里按实机规则修掉），MoveGuard 只挡换位。

### 5.3 可还原性抽查

| 检查 | 结果 |
|---|---|
| 决策都有执行前状态 | 131/131 |
| 决策都有合法动作全量 | 131/131 |
| 决策都有输入向量指纹（含 sha256/形状/非零数） | 131/131 |
| 决策都有执行后状态 | 131/131 |
| 决策都有终态 outcome | 131/131（全部 `applied`） |
| `decision_id` 唯一 | 131/131 |
| 逐回合金币/等级单调可读 | 是（3→4→5→6→7→8→9→10，tier 1→2→3→4） |
| 流程覆盖 | 英雄选择（T0 `choose`）→ 招募（T1–T10）→ 战斗（9 场，引擎 `combat` + 生命差 `combat_damage`）→ 回合切换（17 次）→ 淘汰（T11）→ 最终结算（`game_end` 含 8 席名次） |
| 交叉校验 | T7 战斗引擎 `damage=10`，生命 `30→22`（差 8）：护甲 2 先吸收，两者一致 |

---

## 6. 实机侧：只做离线验证（本轮不启动真实游戏）

用新规则重放昨天那局原始记录（只读）：

```
python client/tools/trace_audit.py replay artifacts/runs/live-e-20261111-004/match-0002 --table
```

| | 旧记录 | 新规则 |
|---|---:|---:|
| `decisions.jsonl` 行 | 753 | **87** |
| 其中等待轮询 | — | 666（进 `waits.jsonl`） |
| 有效动作 | — | 69 |
| T0 英雄选择 | 13 | **1** |
| T1 | 50（`end`×46） | **3**（buy / play / end） |

旧档案校验会直接报出三类历史问题（这正是 `audit.json` 的作用）：
`reused rows still present: 666`、`decisions without execution: 671`、
`decisions without terminal outcome: 25`。**下一局实机将按新格式写入**，届时应为
`complete=true`。

---

## 6.5 面板对齐复跑：1 模型 + 7 引擎 bot + `forbid_cancel`（2026-11-11 13:30）

按用户要求把模拟器向长训口径靠：**另外 7 席改用引擎 bot（`game.bot_action`，不注入对手库）**、
**采样前清掉 `Cancel(7)`**。其余不变（贪心、MoveGuard 5/2、80 动作/回合、同一 seed 700001）。

| 项 | 第一次（7 席=同模型自对弈） | 第二次（7 席=引擎 bot） |
|---|---|---|
| 名次 | 第 7（T11 淘汰） | **第 1** |
| 对局回合 | 17（其余席位继续） | **15（打到结束）** |
| 模型决策 | 131 | **294** |
| 对手动作 | 1,566（同网络） | **971（引擎 bot）** |
| `Cancel(7)` 合法次数 | 17（未剔除） | **0**（37 条 `cancel_masked` 事件） |
| MoveGuard 屏蔽动作位 | 12,154 | **1,374** |
| 完整性 | `complete=true` | **`complete=true`** |
| 耗时 | 36 s | **14.9 s** |

逐回合（模型席位，金币=回合第一决策前）：

| 回合 | 金币 | 等级 | 决策 | 有效动作 | 动作类型 |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | 1 | choose×1 |
| 1 | 3 | 1 | 3 | 3 | buy×1, play×1, end×1 |
| 2 | 4 | 1 | 2 | 2 | upgrade×1, end×1 |
| 3 | 5 | 2 | 7 | 7 | buy×2, choose×1, play×3, end×1 |
| 4 | 6 | 2 | 6 | 6 | buy×1, choose×1, dark_gift×1, play×2, end×1 |
| 5 | 7 | 2 | 5 | 5 | choose×1, dark_gift×1, play×1, upgrade×1, end×1 |
| 6 | 8 | 3 | 14 | 13 | choose×5, buy×1, dark_gift×1, play×5, upgrade×1, end×1 |
| 7 | 12 | 4 | 28 | 19 | choose×5, buy×1, play×5, refresh×2, sell×1, move×2, upgrade×1, **freeze×10**, end×1 |
| 8 | 10 | 5 | 4 | 4 | choose×1, play×1, upgrade×1, end×1 |
| 9 | 10 | 6 | 27 | 17 | choose×4, buy×2, play×4, magnetize×2, refresh×4, **freeze×10**, end×1 |
| 10 | 13 | 6 | 18 | 14 | buy×4, play×4, magnetize×2, refresh×6, choose×1, end×1 |
| 11 | 10 | 6 | 33 | 32 | choose×11, buy×3, play×12, magnetize×2, refresh×4, end×1 |
| 12 | 10 | 6 | 33 | 20 | choose×5, buy×2, play×6, magnetize×2, refresh×7, **freeze×10**, end×1 |
| 13 | 13 | 6 | 46 | 36 | choose×6, buy×4, play×10, magnetize×2, refresh×6, move×5, sell×2, **freeze×10**, end×1 |
| 14 | 13 | 6 | 34 | 20 | choose×3, buy×2, play×4, magnetize×2, refresh×9, move×3, **freeze×10**, end×1 |
| 15 | 10 | 6 | 33 | 20 | choose×1, buy×3, play×6, refresh×7, move×5, **freeze×10**, end×1 |

结论：

1. **只改面板和 Cancel 规则，同一模型同一 seed 从第 7 变成第 1** —— 说明"名次"对面板的敏感度
   远大于对模型差异的敏感度，跨面板比名次必须先对齐面板。
2. `cancel_masked` 37 次证明规则真的生效；bot 的 971 个动作里**没有 cancel / freeze / move**，
   与自对弈对手（72 次 cancel、211 次 freeze）完全不同。
3. `freeze` 开关打摆（每回合 10 次）在新面板下依旧存在，仍待服务器端决定是否加合法性约束。

产物：`artifacts/local/sim-trace-20261111-002/`（含 `trace-header.json` 写明
`panel` / `decoding` / `rules`，可直接判断一局是哪个口径跑出来的）。

---

## 6.6 第三次复跑：加上"商店全冻不提供 Freeze"（2026-11-11 13:50）

录制器按实机 `client/hsrl_agent/live/actions.py:77` 的同一句话实现：
`if mask[2] and shop and all(槽位都已 FROZEN): mask[2]=False`，命中写 `freeze_masked` 事件。
同时修掉两条记录缺陷：`sim_view` 按 `pending` 类型解析（`Target` 的 options 是**实体 id**
不是卡牌 id，`Discover` 是随从对象）；`delta.scalars` 增加 `pending_kind`；
两步动作（play→choose）用 `parent_decision_id` 串成链。

| 项 | 第二次（bot 面板，无 freeze 规则） | 第三次（+ freeze 规则 + 协议态记录） |
|---|---|---|
| 名次 | 第 1 | **第 1** |
| 回合 | 15 | 15 |
| 决策 | 294 | **215** |
| `freeze` 决策 | **60** | **9**（`freeze_masked` 11 次） |
| 无可见变化的决策 | 47（play 37 + end 10） | **12（全是 end）** |
| 两步施法链 | 无父子关系 | **44 条决策带 `parent_decision_id`** |
| `cancel_masked` | 37 | 40 |
| 完整性 | complete | **complete** |
| 耗时 | 14.9 s | **12.4 s** |

结论：Freeze 打摆被消除（60 → 9，只在商店未全冻时出现），名次不变（仍第 1），
决策数下降 27%；"法术空操作"经复核是记录缺陷，已改成可追溯的两步链。
产物：`artifacts/local/sim-trace-20261111-003/`。

---

## 7. 下一步

1. 实机跑一局，用 `trace_audit.py check <match-dir>` 验收 `complete=true`；
2. 用 `trace_audit.py compare <sim-game> <live-match>` 做逐回合对照（动作类型、
   金币/等级节奏、`board_signature`）；
3. 把 §5.2 的两个规则问题交服务器端：是否给引擎加 Freeze/法术目标的合法性约束；
4. 训练侧（`hsbg/az/train`）若要复用本规范，直接 `from` 本模块即可（纯 stdlib +
   惰性 numpy）；服务器同步时按 `main` 拉取。
