# 双模型 + 6 bot 配对评测（eval-duel）

本文定义「两个训练好的模型 + 6 个引擎启发式 bot、128 局、每回合 80 动作 / 5 换位」这套评测的
**协议、口径、门禁与产物**。入口是 `workflows/eval_two_models_six_bots.py`，规则参数在
`config/eval-duel-7357-vs-10004.json`，NUS 提交见 [docs/nus/EVAL_DUEL_DEPLOYMENT.md](nus/EVAL_DUEL_DEPLOYMENT.md)。

被测对象：

| 位置 | 模型 | 说明 |
|---|---|---|
| A（偶数局 seat0 / 奇数局 seat1） | `entity16m-nus-7357` | NUS 1080772 末段，累计 558,873,732 决策；AGENTS.md 批准的续训起点 |
| B（偶数局 seat1 / 奇数局 seat0） | `entity16m-p50-10004-820m` | P50 1085131 末段（820.3M 决策），同一 37.0.2 卡表 |
| 其余 6 座 | `hsbg_engine::bot` | 引擎内置启发式 bot（`game.bot_play`），与历史评测/P50 库同口径 |

两模型**同局对打**：同一 seed、同一 6 个 bot 面板，所以名次差是**配对样本**。

## 1. 为什么不用现成的评测入口

`hsbg/az/train/evaluate.py::evaluate_policy` 只支持「1 个 learner 座位 vs 7 座对手」；
`hsbg/az/train/personal.py evaluate` 把 `--games` 当**每座位局数**（8 座位 × N 局），
并且它的有效性门禁绑定镜像库注入（`mirror_applied > 0`）。本评测要的是
**2 个学习座位同时在场 + 6 个实时 bot**，没有库注入，所以单独实现，不改动上述任何口径：

- 引擎与 bot：**不改**（`engine/` 未改一行；bot 走 `game.bot_play`）；
- 模型：`load_checkpoint` 严格加载，并核对内嵌卡牌表与当前 `37.0.2.256589` 数据逐张量一致；
- 观测/动作：3227 floats + 124 cards / 245 动作，greedy argmax；
- 规则：每回合 80 动作上限、5 换位上限、同一 `(from,to)` 累计 2 次去抖（当前研究配方）。

## 2. 规则实现（与训练同口径）

| 规则 | 实现 | 说明 |
|---|---|---|
| 每回合 80 动作 | `DuelSeatPolicy`：同一 `(seat, turn)` 第 81 个决策在 `EndTurn` 合法时强制结束回合 | 与 `PPOGreedyPolicy` / trainer `_finish` 同一判据（`n_turn > limit and mask[0]`）；触发次数写 `rule_counters.*.forced_cap80` |
| 每回合 5 次换位 | 复用 `hsbg.az.train.move_guard.MoveGuard(cap=5, repeat_limit=2)` | 逆换位屏蔽、同动作 2 次去抖、cap 后屏蔽全部 49 个换位位 |
| 上限后的死锁 | `hard_cap=False`：若引擎只给换位动作，退回原掩码继续，记 `move_cap_overflow` | 训练侧 `hard_cap=True` 会抛错；评测不能让一次病态掩码中断整轮，故如实记账而不是静默放宽 |
| 强制升本 | 关闭（`forced_tier_plan=""`） | 正式评测不得带课程，否则 `protocol_sha256` 变化即混榜 |

`protocol` 里记录：模型 SHA、引擎/词表指纹、`training_source_sha256`、评测入口 SHA、
`games/seed_start/seed_end`、座位轮换规则、解码方式、三个规则参数、`max_steps`、
`forced_tier_plan=""`、`engine_unmodified=true`。任何一项变化都会改变 `protocol_sha256`，
**不同 `protocol_sha256` 的结果不能同榜**。

## 3. 记录口径

### 3.1 战团战力（主参考 P50）

- 单位是**一张回合末战团**：`body = Σatk + Σhp`（当前血量），再对样本取中位数；
- 主参考 `body_p50`；`P25/P75/P90/均值` 是辅助；
- `atk_p50` / `hp_p50` 是**另外两个量**，不能相加当作 `body_p50`（`tests/test_duel_eval.py`
  用反例锁定这条规则）；
- 每个 `(对局, 模型, 回合)` 只取**一帧**（模型执行 `EndTurn` 之前 —— 招募结束、战斗未结算），
  因此 `frames == games_at_turn`；随回合下降是因为早被淘汰的对局不再产生帧（**条件分布**）；
- `survivors` = 该回合末 `health > 0` 的帧数，与 `frames` 并列给出。

bot 参照：6 个 bot 座位的回合末战团。**最后一个结束回合的座位**在 `bot_play` 返回时引擎已结算
战斗，这一帧不计入参照（`bot_field.eot_skipped_frames` 如实计数），所以 bot 参照的 `frames`
每回合最多比模型少 1。

### 3.2 酒馆等级

逐回合 `tier_p50`、`tier_mean`、`tier_ge4/5/6` 占比、`gold_p50`；另有
`reach_tier`：每个目标等级（4/5/6）的**首个达标回合**（按对局统计：达到局数、占比、P50/P25/P75、最早）。

### 3.3 排名

逐局名次、`rank_counts`/`rank_distribution`（1..8）、平均/中位名次、前四率、冠军率、
名次效用 `u(r)=1−2(r−1)/7`、以**对局**为单位的 2000 次 bootstrap 95% CI；
按座位（seat0/seat1）拆分，用来看座位轮换是否抵消了先后手偏差。

### 3.4 配对差分

`Δ = rank_A − rank_B`（负值 = A 名次更好），同局、同 seed、同 bot 面板；
给出均值/中位/95% CI 与 A 更好 / B 更好 / 相同局数。**128 局下这是主判据**：
两个模型都远强于引擎 bot 时，前四率会饱和，只有配对差分还保留分辨力。

### 3.5 表现最佳的几局

- 「名次最优」：按名次升序、最终战团总值降序、seed 升序取前 5 局；
- 「最终战力最高」：按最终战团总值降序取前 3 局；
- 每局给完整最终战团：位置、卡牌中文名 + `card_id`、卡表等级、攻击、生命/上限、金色、关键词
  （从引擎 `Kw` 位掩码解出），以及该局的 `seed`、座位、名次、最终等级、血量。

## 4. 门禁（`report.json.valid`）

以下任一成立就 `valid=false`、`reasons` 写明原因，并让进程以非零码退出（**禁止入榜**）：

- 完成局数 ≠ `games`；
- `illegal > 0`（模型给出的动作不在合法掩码内）；
- `truncated > 0`（`max_steps` 内未结束）；
- `engine_bad > 0`（`truncated_combats` / `truncated_recruit_contexts` / `turn_limit_end` /
  `engine_stopped` 任一非零，见 `hsbg/az/train/engine_quality.py`）；
- 名次越界，或某模型没有任何决策/没有完整名次序列。

报告**总是先落盘**再退出，失败现场保留在输出目录里。

## 5. 产物

```
<out>/report.json   完整协议、模型审计、逐局明细、逐回合表、最佳局、规则计数
<out>/report.md     人读版本：结论摘要 + 等级/战力表 + 最佳局战团表 + 复现命令
```

`report.json` 里 `games[]` 保留逐局 seed/座位/名次/8 座 placement/决策数/守卫计数/引擎诊断，
便于事后做配对重采样或换统计口径，不需要重跑。

## 6. 本地证据（2026-11-07，CPU）

- 单测 23 项：`python -m pytest tests/test_duel_eval.py -q` 全绿（P50 口径、逐回合计数、
  bootstrap 可复现、协议敏感性、`valid` 门禁、markdown 渲染）。
- 冒烟 2 局与本地验证 16 局（`artifacts/runs/duel-local-validation`，CPU 4 线程 116 s）：
  `valid=true`，0 非法动作、0 截断、0 引擎异常，8307 个决策；
  16 局里 A 的平均名次 1.50、B 1.88，配对 Δ 均值 −0.375（95% CI [−0.75, 0.125]）。
- 这 16 局只用于**验证管线**：它证明帧时点、守卫计数、最佳局战团和门禁都在工作，
  **不构成棋力结论**（局数远小于 128，且是本机 CPU）。

## 7. 复现

```bash
# 本机/NUS 都可：先准备两份权重（复制 + 核 SHA）
python tools/stage_duel_models.py
# 正式 128 局
python workflows/eval_two_models_six_bots.py --out artifacts/runs/duel-eval-001
# 冒烟（不改协议常量，只覆盖局数/设备）
python workflows/eval_two_models_six_bots.py --out /tmp/duel --games 2 --device cpu
```

`--games/--seed/--model-a/--model-b` 会覆盖配置，覆盖后的值照常写进 `protocol`——
用它们跑的任何东西都**不是**本轮批准的 128 局协议，不能与正式结果同榜。
