# 中后期入场课程（LATE_GAME_ENTRY）

状态：**已实现，正在验证**（2026-11-12）。入口 `config/league-entry.json` +
`deploy/slurm/late_entry.sbatch`；前置门禁 `deploy/slurm/smoke_inject.sbatch`。

## 1. 为什么需要它（实测，不是推断）

`docs/analysis/HUMAN_GAP_20261112.md` 与后续逐回合拆解给出的四件事：

| 观测 | 数值 | 出处 |
|---|---|---|
| 模型 T12 板面 Σ(atk+hp) | 1,135 | human-gap 面板 |
| 真人（≥8k MMR、1 血吃鸡局）T12 p50 | **5,620** | 本库 `summary-m8k.json` |
| 真人 T14 p50 / T15 p50 | 27,234 / 44,268 | 同上 |
| 模型每局自生成镜像帧 | 8,872，`human_board_p=0` | 训练 history |
| 模型终局含 ≥3 张 S 级核心卡的比例 | **0 / 960 局** | composition 覆盖分析 |

也就是说：**模型从来没有在「真人有牌可打」的中后期状态上做过决策**，它的对手也
从来不是真人的阵容。奖励只认终局名次，而名次是在「自己打自己」的弱环境里比的——
赢自己的克隆不需要造阵容。

### 1.1 一个必须纠正的旧结论

此前把 `humancurve-pt10` 库称为「后期对手强度 2.9×/6.4× 真人」，这个说法**误导**：
`workflows/build_human_curve_bank.py` 的做法是把**我们自己快照**里每只随从的攻血按比
例重分配，使该回合总攻血落在真人分位——**牌面构成仍然是我们自己的弱阵容**，只是数字
被放大了。它既不是真人阵容，也不含核心引擎，因此：

- 它无法教「阵容」，只能教「数字大就赢」；
- 它把「对手强度天花板」这个指标做成了数字游戏，掩盖了真正的缺口；
- 它的参考分布**没有 MMR 下限**（`references()` 只按 `split=='train'` 过滤），
  所以连数字目标也偏低：该库 T12 实际约 3,300（= 1,135 × 2.9），只有 ≥8k 真人
  p50（5,620）的 **0.59 倍**。

结论：**换掉它**。新库直接用真人卡组的完整帧，见 §2。

## 2. 数据：真人对局库

来源 `data/real_library/firestone-20261112-1000/library.jsonl`（Firestone
`bgs-perfect-games` 1000 局解析，schema `hsbg.human_state_quasi_full`，一行 = 某座位
某回合末的**真实战团 + 手牌 + 酒馆等级 + 血量 + 金身/关键词位**）。

过滤工具：`tools/build_human_entry_bank.py`（**只过滤、不改数值**）

```bash
PYTHONPATH=engine/python:. python tools/build_human_entry_bank.py \
  --library data/real_library/firestone-20261112-1000/library.jsonl \
  --out-train data/real_library/firestone-bank-1000/train-m8k.jsonl \
  --out-eval  data/real_library/firestone-bank-1000/eval-m8k.jsonl \
  --summary   data/real_library/firestone-bank-1000/summary-m8k.json --min-mmr 8000
```

口径（全部写进 `summary-m8k.json`）：

- `split` 按游戏哈希分成 `train` / `evaluation`，训练与评测**不相交**；
- `stage ∈ {combat_start, recruitment_end_at_phase_switch}`（回合末战团）；
- 丢弃：空板、>7 只、非整数、`atk<0`、`hp<=0`、单随从 `atk+hp>100000`、
  全队 `Σ>250000`（阈值与 `hsbg.az.train.human_board_pool` 一致）；
- `mmr >= 8000`；`perspective=="opponent"` 的行没有本人 MMR，按**同局上传者**继承并
  标 `mmr_source="game_uploader"`。

结果：**1,168 行 train / 86 行 evaluation**，每回合 70–113 行（T3–T15）。
逐回合 p50（Σ 攻血）：

| turn | 3 | 5 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| p50 | 17 | 45 | 119 | 203 | 398 | 904 | 2,063 | 5,620 | 13,163 | 27,234 | 44,268 |

**这是「已吃鸡玩家」的分布**（feed 本身是 1 血吃鸡局），所以 p50 已经是「优秀玩家
的中位水平」——正好是我们要模型达到的目标，而不是要它去追 p99。

## 3. 机制：入场课程

`hsbg/az/train/parallel_collect.py` 新增两条路径，**默认全关**（`entry_p=0` 时逐位
等价旧配方）：

1. **`entry_pool` / `entry_p` / `entry_turn_min..max`**：每局抽一个入场回合 `T0`
   （确定性：`(cfg.seed, seed, gi)`）。`T0` 之前 **learner 席由引擎 bot 代打，不产生
   任何 learner 决策记录**；到 `T0` 回合开始、任何座位行动前，把一个真人完整状态
   （`apply_turn_snapshots`：战团 + 手牌 + 可推断 counters + 酒馆等级）注入 learner 席，
   之后才由策略接管。于是**每条训练记录都落在真人中后期状态上**。
2. **`human_board_min_turn` + `mirror_max_turn`**：把「镜像自对弈对手」关在 `T0` 之前，
   `T0` 之后改由**真人战团**提供 7 席对手（`human_boards` + `human_board_p=1` +
   `human_board_seats=7`）。这样中后期的竞争压力是真人阵容，而不是自己的克隆。

`entry_set_health`（默认 0）：是否把观测到的 `hero_hp` 也写回。默认**不写**——血量由
同一局自然演进，8 席名次赛保持对称，名次奖励才可比。

### 3.1 前置门禁

换库之后**必须**先跑：

```bash
sbatch deploy/slurm/smoke_inject.sbatch
```

它逐回合报告 `attempt/applied/board_match/hand_match/tier_match/ms`。2026-11-12 实测
（`train-m8k`，T8–T13，p50，full 保真度）：

```
[turn  8] attempt=6 applied=6 board_match=6 hand=4 tier=6 ms=4.6 str50=205
[turn 11] attempt=6 applied=6 board_match=6 hand=6 tier=6 ms=5.2 str50=2079
[turn 12] attempt=6 applied=3 board_match=3 hand=3 tier=3 ms=3.3 str50=5620
[totals] attempt=36 applied=22, board_match=22/22, tier 22/22, hand 17/22
```

即：**接受后的战团逐字段 100% 一致**；一次注入 3–5 ms（每局一次，可忽略）。
被拒的两类原因：`not_alive`（该席在 bot 代打阶段已淘汰，约 30% 局）与
`pending_choice`（引擎正处于选择态）。后者在真实采集循环里不会出现，且
`parallel_collect` 已对注入失败保留 `ENTRY_MAX_TRIES=3` 次下回合重试。

## 4. 它应该修好什么、不该指望什么

**应该修好**

- 让「后期名次」变成**有信息**的奖励：对手是真人阵容，赢就是真的更强；
- 让模型在**有牌可打**的状态上练习扩规模、找核心、换位、转型（这正是 0.20× 的缺口）；
- 训练算力集中在中后期：入场局每局 ~4–6 个 learner 回合，而不是 15 个。

**不该指望**

- 不会自动教会「从空场起手造阵容」——入场时阵容已由真人给定。要补这一环需要把
  `entry_sample` 往下推（p25/p50）或把 `entry_turn_min` 前移（T6/T4），这是 §5 的 S2。
- `entry_p=1` 的局里前期完全没有梯度，前期能力**只靠 anchor KL 保住**；每次评测都要
  跑前期回归（`docs/OPERATIONS.md` 的 human-gap 面板）。

## 5. 分阶段路线

| 阶段 | 入场 | 目标 | 验收 |
|---|---|---|---|
| **S1**（当前） | T9–11，p50 | 学会**运营一个已成型的真阵容**：扩规模、拿核心、换位、转型 | T12/T14 板面、≥2/≥3 核心卡比例、入场局平均名次、前期不退化 |
| **S2** | T6–8，p50→p25 | 学会**自己启动核心引擎**（从半成型阵容补齐） | 同上 + 「首次拿到 S 级核心卡的回合」前移 |
| **S3** | T1（全流程） | 端到端长训，专家数据正则 + 真人压力 | E1 正式榜 + 实机 |

**不要用单一指标验收**：卖牌次数、终端板面数值都可以被无意义地刷高。至少同时看
①入场局名次 ②板面 Σ 相对真人曲线 ③核心卡集合覆盖（≥1/≥2/≥3）④前期名次回归
⑤动作换挡（T6→T10 的 sell/refresh 斜率）。

## 6. 复现

```bash
# 1) 库 + 门禁（见 §2、§3.1）
# 2) 训练（多段；段间 --resume，运行期间冻结 hsbg/ 与 workflows/）
HSAI_LE_OUT=artifacts/runs/entry-s1-20261112 \
HSAI_LE_SEGMENTS=2 HSAI_LE_MINUTES=30 \
  sbatch deploy/slurm/late_entry.sbatch
```

`run.env` 记录 config / library / source / git SHA / `training_source_digest()` /
GPU / 节点，便于事后核对。**任何跨段实验期间不要提交 `hsbg/` 或 `workflows/`**
（续训守卫会以 `训练代码已变更` 拒绝，见 `docs/OPERATIONS.md` §3.5）。

## 追加：`--entry-min-tier`（2026-11-13，专家先验）

专家先验：当前版本高手 **T8 左右到五本**。而实测 champion 的升本曲线 T2/T5 与标准一致，
**T8 之后停住**（T8 4.33 / T10 4.44，真人 4.82 / 5.52；`upgrade` 使用率 T9 之后 ≈0）。
结构上 **15/33 个流派最高核心在 6 本、16/33 在 5 本、0/33 能在 4 本成型**，
所以停在 4 本等于拿不到真核心。

因此新增 `--entry-min-tier k`（默认 0 = 不过滤）：只使用**观测等级 ≥ k** 的入场帧，
让入场课程直接落在"五本位置"上。实现要点（`hsbg/az/train/human_board_pool.py`）：

- 等级取**观测值**（库行的 `tier` 字段）；等级未知（无 `board_detail` 的 board-only 行）**丢弃**
  ——宁可池子小，也不假装知道等级；
- 过滤计数落在 `pool.filtered`（`tier_below_min` / `tier_unknown`），作业启动时打印；
- 过滤后池子为空 → `train_entity_sync` **拒绝启动**（fail-loud，不拿空池子跑）；
- 6k 库 T8 实测有 **657 行 tier≥5**（占 64%），够 `p75` 采样。