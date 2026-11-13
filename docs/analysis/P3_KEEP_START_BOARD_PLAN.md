# P3 实现方案：`keep-start-board`（禁卖起始战团）——接线清单与陷阱

状态：**待实现**（2026-11-13 13:35 写好；不在训练作业跑着时推 digest 相关改动，见 D-037）

## 1. 语义（两种，都要能选）

- **V1 回合初保护**：本回合开始时在己方战团上的随从，本回合不可卖出。
- **V2 开局保护**：第 1 回合结束时的战团（"起始战团"），在前 N 回合内不可卖出（N 可配，默认 3）。

群友 README 只写 `keep_start_board = 1 (sell restriction)`，**没有说明是哪种**，所以两种都实现、
各跑一个短臂，用**真人场名次**判定（他们实测值 0.10–0.14 名）。

## 2. 关键陷阱：**必须同时接在训练和解码两侧，否则等于没接**

这是 `board_target` 踩过的坑（配了但没生效，静默跑完一整段）。`keep-start-board` 有**两条**决策路径：

| 路径 | 文件 / 位置 | 说明 |
|---|---|---|
| **训练 rollout** | `hsbg/az/train/parallel_collect.py`（`mask = mask_from_bytes(g.legal_mask(seat))` 附近，强制升本/`forbid-tier` 也在这里改 mask） | 训练时 learner 的动作在这里被约束；不改这里 = 训练根本没见过这条规则 |
| **评测/解码** | `hsbg/az/train/ppo.py::PPOGreedyPolicy.decide()`（`forbid_cancel` → `MoveGuard` → `PendingGuard` 之后） | 面板/实机走这里 |

另外 `hsbg/az/train/ppo.py` 里还有一处**训练用策略**的 mask 处理（`getattr(self.config, "forbid_cancel", False)` 那一段），
要用 grep 确认训练到底走 `PPOGreedyPolicy` 还是 `PPOPolicy` 的 `decide_batch`——**以实测为准**，
最稳的办法是：接上之后跑 1 轮，看 `action_hist["sell"]` 在受保护回合是不是**精确为 0**。

## 3. 需要的改动（最小集）

1. `CollectConfig`（`hsbg/az/train/ppo.py`）：`keep_start_board: int = 0`、`keep_start_board_mode: str = "turn"`
   （`turn` = V1，`game` = V2）、`keep_start_board_until: int = 3`（V2 的 N）。
2. `personal.py::OPTIONS`：加 `"keep-start-board"`、`"keep-start-board-mode"`、`"keep-start-board-until"`。
3. `workflows/train_entity_sync.py`：argparse 三项 + 传进 `CollectConfig`；启动期打印一行
   `[keep-start-board] mode=… until=…`（**必须打印**，便于核对真的开了）。
4. `hsbg/az/train/runlog.py::INJECTION_KEYS`：`keep_start_board*` 三项 + 一个
   `keep_start_board_blocked` 计数（被这条规则挡掉的卖出次数）——**没有这个计数就不算接对**。
5. 采集侧：每局维护 `protected`（V1：每回合初的快照；V2：第 1 回合末的快照 + 剩余回合数），
   在 mask 上把对应 `sell(16..22)` 位置置 False；挡掉时 `blocked += 1`。
6. 解码侧 `PPOGreedyPolicy`：同样逻辑（快照按 `(seat, turn)` 存，`reset()` 清空）。
7. 测试（`tests/test_keep_start_board.py`）：①`=0` 时 mask 与旧路径逐位一致；
   ②V1 下回合初的随从对应 sell 位被清掉、当回合新买的可以卖；
   ③V2 下第 1 回合末的战团在 N 回合内不可卖，N 回合后恢复；
   ④全被挡住时不会产生空 mask（EndTurn 始终合法）。

## 4. 实验设计（实现之后）

- **K1** = `keep-start-board-mode turn`（V1）
- **K2** = `keep-start-board-mode game --until 3`（V2）
- 两者都基于当前最强的 PPO 配方（H10 的 comp 奖励 + `anchor 0.02`），跑 5 段（`gpu` 分区 3 h 上限，见 D-041）。
- 判据：真人场 `mean_rank` 配对 CI 上界 < 5.4375；同时 `action_hist["sell"]` 在受保护回合应接近 0，
  `blocked` 计数应稳定为正（证明规则真的在起作用）。

## 5. 为什么值得做（证据）

- 群友实测：没有这两条掩码时平均名次差 **0.10–0.14**（README 原文）；
- 我们自己的诊断同向：`docs/analysis/` 里记录的"缺的是投资不是购买"、
  以及 T10–T11 断崖时板面只有对手 p75 的 0.44×——频繁换人正是"投不进去"的直接原因；
- 它是**规则型改动**，不引入新的可被单独满足的奖励，因此不会被 reward hacking 打穿。
