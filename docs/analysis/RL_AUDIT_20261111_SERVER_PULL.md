# 服务器长训后强化学习模块变更审计（本地，2026-11-11）

状态：**当前**（本地 Agent 审计记录）
基线：`02ecc57` → 最新 `b8847ab`（fast-forward 12 个提交）
审计对象：`docs/AUDIT_2026-11-11.md`、`docs/PROJECT_STATE.md`、`docs/LEADERBOARD.md`、
`docs/sync/SERVER_TO_LOCAL_20261111.md`、`docs/sync/DECISIONS.md`（D-024…D-027），
以及 `hsbg/az/train/*`、`workflows/*`、`tests/*`、`config/*` 的实际 diff。

## 0. 同步与边界

| 项 | 结果 |
|---|---|
| 本地 HEAD | `02ecc57` → `b8847ab`（12 提交 fast-forward，无冲突） |
| 本地未提交工作 | 更新前 `git stash -u`、更新后 `stash pop`，P1/P2 修复与 trace 新文件全部保留 |
| 服务器是否改本地写权路径 | **否**：`client/`、`config/live/`、`tests/test_live_*` 零改动 |
| `check_ownership.py --strict` | 通过，local-owned 14 项 |
| `verify_workspace.py --skip-rust` | cards / engine-source / python 全绿，470 项测试 0 失败 |

### 0.1 本地环境坑（非回归，需记录）

在中文 Windows 上 `verify_workspace.py` 的 python 门禁**会假失败**：

- `tools/verify_workspace.py:76` 用 `text=True, encoding='utf-8'` 抓 pytest 输出；
- 但更深的子进程（`tests/test_az_ppo.py:388` 用 `text=True` 无 encoding、`tests/test_4159_long.py:195`
  用 `encoding='utf-8'`）对子进程 stdout 编码的假设**互相矛盾**，而子进程编码由
  `PYTHONIOENCODING`/locale 决定（本机默认 cp936）。

结果：不设变量 → `test_4159_long` 失败；设 `PYTHONIOENCODING=utf-8` → `test_az_ppo` 失败。
**唯一正确做法是 `PYTHONUTF8=1`（PEP 540 UTF-8 模式，同时把 `locale.getpreferredencoding()`
变成 utf-8）**，此时两者都通过。

结论：不是代码回归，也不是服务器代码问题；本地跑验收前先 `$env:PYTHONUTF8='1'`。
两处测试属服务器写权，本地不改，建议服务器把子进程统一成 `encoding='utf-8', errors='replace'`
或全仓依赖 UTF-8 模式。

## 1. 采样 / 自对弈：p75 镜像退化（A1，P0）

**问题**：36h 的“7 席联赛”实际上只在约 260 个固定对手棋盘上训练。

- 库 `train/snap_league.json.gz` 的 8,872 帧**全部 `seat=0`**；
- `MirrorSnapshotPool.get()` 在 p75 只保留“强度恰好等于分位点”的帧，turn 8 以后 ties=1；
- `StrictLibrary.sample()` 对席 1..7 找不到 own-seat 帧，全部回退 `by_turn[turn]=(0,)`；
- 于是 7 个对手席 + 512 局在 turn 8 以后拿到**同一张棋盘**；
- 质量门只数“数量”不校验“帧身份”，所以每轮 `mirror_applied == eligible_live_seats` 一直全绿。

**修复**（`hsbg/az/train/mirror_snapshots.py`）：

- 新增 `SAMPLE_BANDS` = `p50band(0.45–0.55) / p75band(0.70–0.80) / p90band(0.85–0.95)`，
  在分位带内**均匀取样**，保留难度目标但恢复多样性；
- `SAMPLE_MODES` 扩充；`train_entity_sync.py --mirror-sample` 直接用 `list(SAMPLE_MODES)` 做 choices；
- 新增 `_sorted_candidates()` 按桶缓存 strength 排序（`get()` 每回合每座位高频调用），`load()` 时清缓存。

**效果**：真实库 turn 8–16 从每 turn 1 帧恢复到 **~52 帧**。late turn 仍受源库规模限制
（turn 24 只有 2 帧），下一轮若要提升 late-game 需重建更大的 EOT 库。

**回归**：`tests/test_mirror_band.py`（新增）。

## 2. 动作规则：Freeze 空转（A7，P1）

**问题**：Freeze（action 2）是**最高频动作**，每回合平均 3.3–4.4 次，典型序列是连续按满
引擎 `freeze_presses_per_turn=10` 上限。10 是偶数，净效果等于没冻结。

**A/B 实证**（final，128 局 × seat 0/7）：

| 规则 | seat0 名次 | seat7 名次 | 总胜率 | 决策数 | Freeze 次数 |
|---|---:|---:|---:|---:|---:|
| 原版（10 上限） | 1.1406 | 1.2188 | 83.6% | 67,686 | 15,437 (22.8%) |
| `freeze_cap=1` | 1.1875 | 1.1875 | 83.6% | 53,951 (−20.3%) | 1,526 |
| 原版（另一 seed） | 1.2578 | 1.1484 | 83.2% | 67,536 | 14,931 (22.1%) |
| `forbid_unfreeze=1` | 1.2500 | 1.1562 | 83.2% | 56,748 (−16.0%) | 4,646 (8.2%) |

**修复**（不改模拟器规则，只让训练/评测与实机对齐）：

- `ppo.py` `CollectConfig.forbid_unfreeze`（默认 False，保持旧语义）；
- `PPOSelfPlayPolicy._mask()` 与 `PPOGreedyPolicy.decide()`：当 `mask[2]` 且
  `GLOBAL_FIELD_INDEX["frozen"] > 0.5` 时屏蔽 action 2；
- `PPOGreedyPolicy` 另加 `freeze_cap`（默认 0 = 精确游戏语义，仅审计/评测用）；
- 守卫后**空掩码显式 `RuntimeError`**，不再静默退化成 EndTurn；
- `personal.py` / `train_entity_sync.py` 暴露 `--forbid-unfreeze`（只接受 0/1）。

**与本地桥接的一致性**：本机 `facts.py` 已按引擎语义用
`g['frozen']=int(any(FROZEN))`（等价于引擎 `is_frozen() = shop.iter().any(|s| s.frozen)`），
`actions.py` 在全冻结时不暴露 action 2 —— 两端现在同一口径。

**注意**：`freeze_cap=1` 减少决策更多但“全冻结时仍能再按”，不是自然规则；
**推荐 `forbid_unfreeze`**，它才是与实机一致的规则。Freeze 仍占 ~8% 决策，属策略先验问题，
不要用硬 cap 掩盖。

## 3. PPO 更新循环与目标函数（A3、A6）

| 问题 | 修复 |
|---|---|
| KL 早停导致 0 次 optimizer step 的 iteration 仍推进 LR/updates/version | `ppo_trainer.py:377-382` 只在 `stats["updates"] > 0` 时 `scheduler.step()` 并 `updates += 1` |
| `build_batch_from_arrays` 在 worker 已构建批次上**重算 GAE**，丢弃已有 advantages/自定义奖励 | 改为 `prebuilt = all(k in t for k in ("advantages","returns","dones"))` 原样使用并在长度损坏时显式报错；旧 fallback 路径保留给未构建批次 |
| `_pre_ratio` 把 forced（old_logprob=0）样本计入管道保护，把均值比拉向 0 | `train_entity_sync.py:2414-2419` 先 `valid = ~batch.forced` 再采样；全 forced 返回 1.0 |
| 评测 logits 非有限未检查 | `PPOEvaluator` 显式 `RuntimeError("non-finite policy/value logits in evaluation")` |
| `profile.reload_s` 恒 0 | `train_entity_sync.py:2485-2486` 汇总各 worker 实测 `reload_s` |

**`max-updates` 语义澄清（重要，不是 bug）**：`ppo_trainer.py:59-68` 的 `max-updates` 只是
`LambdaLR` 的衰减视野，且按**全局 iteration** 计数。36h 起点已是 9691，所以第一轮 effective LR
就是 1.677e-6，第 10000 轮后固定 1.667e-6 —— 配置写的 `lr=2e-6` 从未生效。
下一阶段要重新走 2e-6→1.667e-6 **必须用 `workflows/fork_stage.py` 重写 scheduler**，只改 config 无效。

## 4. anchor 热身失效（A2，P1）

- `config/league-36h.json` 无 `reset-anchor`，`trainer.anchor_model` 由 `ppo_trainer.load` 恢复；
- 全局 iteration 从 9691 起，`i < anchor_iterations(250)` 永远为假 → `anchor_coef` 全程是 tail `0.02`；
- 但实测 `trainer.pt` 的 anchor 与 V1.1 参数 L2 距离仅 **0.476**（参数范数 913.9，约 0.05%），
  即 anchor 实际就在 V1.1 起点附近，**不是更旧的策略**；
- `anchor_kl` 0.0224→0.2268，末轮 `anchor_coef*kl≈0.0045` > policy loss `0.0030`，
  是一个持续但温和的旧策略正则。

结论：不是“拉向错误旧策略”的 P0；问题是 `anchor-beta=0.1`/`anchor-iterations=250` 完全没生效。
下一阶段用 `reset-anchor=1`（anchor=新阶段起点）。

## 5. 进程池与工程可靠性（A6）

- `process_pool._maybe_reload` 的 `weights_loaded` 消息**无 task_id**，永久留在 `_deferred`，
  每轮泄漏 8 条并线性扫描 → 已修（不再入 `_deferred`；reload 更新 version/path 并重置 norm）；
- `PPOTrainer.load` 传 config 时忽略 state 内 config → 修复；
- 观测只有宽度校验、没有字段级运行时指纹 → 补了字段级 gold test（见 §7）；
- checkpoint 架构门禁不统一（`entity18m` 的 n_heads、panel/eval_az 不走 audit_model）→ 记录为剩余风险。

**Actor 扩展 A/B**（final，512 局/iter，p75band，4 iter）：

| processes | decisions/s | wall/iter |
|---:|---:|---:|
| 8 | 7,300 | 17.7 s |
| 16 | 5,395 | 24.3 s |

结论：36h 的 8 进程**不是**低效配置；16 个 CUDA context 在单 actor GPU 上反而更慢。
下一轮保持 8；要扩展应加 actor GPU。

## 6. 评估协议（A4，P1）——本轮改动最集中

| 缺陷 | 修复 |
|---|---|
| `elo_win_prob` 把 Elo 差直接进 sigmoid（应除 `ELO_SCALE`），概率几乎全 0/1 | 改为 `sigmoid((score_i - score_j)/ELO_SCALE)`；旧报告 −98.6 Elo 显示 1.5e-43，正确约 0.362 |
| `seed_mode=shared` 下 bootstrap 按 pair 独立重采样 → CI 过窄 | shared 时改用 **seed-block bootstrap**（同一 seed 下标跨 pair 联合重采样），并校验各 pair 样本等长 |
| `_fit_bt` 优化失败静默返回 anchor（全平） | `hybr` 失败回落 `L-BFGS-B`（BT 似然是凸的），仍失败则 `RuntimeError`，拒绝输出假平局 |
| 冠军判定“CI 分离”拿 top 的 CI 下限比别人的点估计 | 改为 **rank CI**（bootstrap 里每次重采样都算一次排名），`build_e1_board.py` 用 rank CI 判分离 |
| challenge 报告未校验协议/模型 SHA | 加校验；缺失 rank CI 不再伪造 |
| `protocol_sha256` 只哈希 runner 文件 | 新增 `code_sha256 = semantic_code_sha256()`，覆盖实际语义代码 |
| `eval_latest_gauntlet.py` 协议串硬编码 legacy seat | 记录真实 `seat_mode`/`seed_mode`，pairs 用 `pair_seed_start` |
| 配置强制 8 个模型 | 放宽到 ≥2；新增 `seat_mode`(legacy/ordered56)、`seed_mode`(disjoint/shared) 校验与 `ordered56` 座位覆盖、枚举校验 |

产出：`fit_elo` 现在同时给 **Elo CI + rank CI + 修正后的 pairwise 概率**，`method` 字段写明
“game-level bootstrap; rank intervals”。

## 7. 观测编码：新增字段级 gold test

`tests/test_obs_std.py::test_global_fields_decode_to_engine_values` 是本次**观测侧唯一实质改动**
（编码逻辑本身没改）。旧测试只校验块宽度，同尺寸字段重排或改了除数会**静默同时污染训练与实机输入**。
新测试用 `game.set_player_from_json` 构造已知状态（health=37/armor=13/gold=9/tier=4/
spell_atk/gem_hp/tavern_atk/deity/board_len/hand_len/trinket_count/turn/alive…），
再用 `decode_blocks(features,cards)["global"]` 按 `GLOBAL_FIELD_INDEX` + `GLOBAL_FIELD_SCALE`
反解，逐字段 `approx(1e-4)` 对拍。

这是**本地实机侧最该复用的一类门禁**：实机投影要能对齐的正是这些字段。

## 8. 训练编排与下一阶段

新增/变更工具：

| 文件 | 作用 |
|---|---|
| `workflows/fork_stage.py`（新） | 完整 Trainer 分叉：保留 Adam/RNG/累计决策/update 编号，**只重写 LR 视野与 anchor**（解决 §3 的 `max-updates` 语义陷阱） |
| `workflows/audit_model_behavior.py`（新） | 行为审计：逐座位名次分布、动作组/逐动作直方图、连续重复率、forced 计数、**bf16 vs fp32 greedy argmax 翻转率** |
| `workflows/validate_league_checkpoint.py`（新） | 段后 dev/test 验证钩子，用与训练/库源局不相交的 seed + 严格镜像门 |
| `workflows/build_e1_board.py`（新） | 把 valid report 变成正式榜（rank CI + 冠军判定 + 协议/SHA 校验），不改原始 report |
| `workflows/eval_two_models_six_bots.py` | 新增 `semantic_code_sha256()` 供协议哈希 |
| `config/league-stage2.json`（新） | 下一轮配方：p75band + forbid-unfreeze=1 + `reset-anchor=1` + `max-updates=25000` + `anchor-iterations=15108` + seed 320000000 + 8 processes + 512 局/30min×72 段 |
| `deploy/slurm/*` | 新增 `league_stage2`、`e1_four_eval`、`audit_{actor_scaling,freeze_guard,unfreeze_guard,model_behavior,regression}`、`validate_league_checkpoint`、`league_36h_eval` |
| `hsbg/az/train/evaluate.py` | 新增 `collect_action_hist` / `collect_action_trace`（默认关闭，不进热路径） |
| `hsbg/az/train/personal.py` | 放开 `forbid-unfreeze` 与整组 `eval-*` 键；旧配置未写 eval 键时仍注入 `eval-panel=bot` / `eval-every-decisions=0`，保持旧行为 |
| `hsbg/az/train/parallel_collect.py` | collect meta 输出 `action_hist`（键为字符串） |

## 9. 资产与榜

**36h 作业**：`1102669` COMPLETED，72/72 段，`2026-11-11T16:46:33+08:00`，elapsed `1-11:44:43`，
final `v14858`，累计 **1,552,463,499** 决策，`illegal/truncated/dropped/worker errors=0`，
4 次 KL 早停、0 次零 step。trainer 在 cuda:0、8 actor 在 cuda:1（两张 H100 都用上）。

**E1 四模型校准**（6 对 × 448 局 = 2,688 局，ordered56，shared seed + seed-block bootstrap，
`valid=True` / `leaderboard_eligible=True`，0 illegal/truncated/engine_bad，1,490,692 决策；
r2 与 r1 **六对逐对完全一致**）：

| 排名 | 模型 | Elo | 95% CI | 排名 95% | 平均名次 | 前四率 | 冠军率 |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | `league-final-14858` | **1086.53** | [1062.49, 1111.01] | [1, 2] | **1.589** | 99.18% | 51.34% |
| 2 | `league-latest-13806` | 1068.45 | [1045.78, 1091.47] | [1, 3] | 1.644 | 99.11% | 49.48% |
| 3 | `league-latest-12311` | 1055.92 | [1032.16, 1079.91] | [2, 3] | 1.681 | 98.59% | 47.17% |
| 4 | `V1.1` | 1000.00 | [1000, 1000] | [4, 4] | 1.876 | 97.32% | 36.68% |

- 12311 → 13806 → final 单调上升，说明 36h **确实提升了头对头棋力**；V1.1 明显最后。
- final 对 13806 为 240–208（53.57%，Wilson CI 约 [48.9%, 58.2%]，**跨 0.5**）→
  只能写“点估计冠军”，不能写“已证明显著第一”。
- 正榜 `artifacts/models/e1-four-20261111/leaderboard.json`，协议
  `0fbf28d2f82c08062291ef6526275507dfeffa1f5db85159ac7531437f3f723d`。
- 旧 v2 的 12311 = 1067.68 与 E1 的 1055.92 **不可混榜**（协议不同）。

**bf16 vs fp32**：1800 个真实观测/模型，greedy argmax 翻转 **0.67%–1.00%**。
训练 leaf 用 bf16、评测/实机用 fp32 → 训练采样策略与评测 greedy 策略**不是逐动作一致**。
评测报告与实机结果都应标注“fp32 解码”。

## 10. 本地复核服务器对 `client/` 的静态审计（D-026）

服务器列了 7 项 train/live 不一致，本地逐条核对了代码，**抽查项全部成立**：

| # | 断言 | 本地核对 |
|---|---|---|
| 1 | `seen_boards`/`last_opponent` 恒空 | **成立**：`logs.py:62,81,116,139` 只有初始化/清空，全仓无赋值点；`facts.py:80-81` 却直接读入观测（影响 7×7×31=1519 浮点） |
| 2 | pending kind 把 ChooseOne(3)/Target(4)/Timewarp(8) 都归为 Discover(2) | **成立**：`adaptive.py:85-91` 只分 1/5/6/7，其余 `estimate(...,2,...)` |
| 5 | `MODEL_*` 标签缺失 | **成立**：`client/hsrl_agent/live/data/game_tags_255790.json` 中 `MODEL_MAX_GOLD_BONUS`/`MODEL_CHRONUM`/`MODEL_HAND_QUEUE_COUNT`/`MODEL_LOST_LAST`/`MODEL_TAVERN_ATTACK`/`MODEL_FROZEN` **全部 ABSENT**（`NEXT_OPPONENT_PLAYER_ID`、`PLAYER_LEADERBOARD_PLACE` 存在） |
| 6 | Freeze toggle / >16 选择 / trinket 可用性 | 与本地既有结论一致；`forbid_unfreeze` 规则已把动作侧对齐 |

第 3、4、7 项（任务/奖励从未重建、对手摘要 all-or-nothing、bf16/fp32）属同一批文件，
方向与已核项一致，**待用真实日志逐项量化频率**（服务器要求导出 `approximation.json` +
同刻 `decode_blocks` 对拍）。

## 11. 实机模型就绪（本地，本轮完成）

- 服务器 live 包 `artifacts/exports/league-final-14858-live.tgz`
  SHA256 `32ccdabdc7d43793fb7d358185741fbf082ba0f5c40a397d7967ade01859a22a`（56,288,249 B）
  —— **已下载且校验一致**；
- 解包到 `artifacts/models/league-final-20261111-step014858/`（model.pt / manifest.json /
  checkpoint.json / MODEL_CARD.md），`model.pt` SHA256
  = `ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7` —— **与服务器一致**；
- 结构核对：与 12311 **137 个张量、key 顺序、shape、dtype 全同**；
  `meta` = `arch=entity, d_model=384, n_layers=8, n_heads=8, seq_head=true, bc_card_head=true,
  bc_card_dim=3983, trainer_updates=14858` → 可直接替换的推理权重，不是新架构；
- `client/hsrl_agent/live/policy.py`：**新登记** `ba7e8bbc…` → `league-final-14858`
  （`parameters=16436034, auxiliary=False, required_client_build=255790,
  trained_engine_version=37.0.2.256589`）。**不登记会被 `validate_live_profile` 直接
  `Blocked('checkpoint is not an approved pinned policy')`**；
- `config/live/model.json`：pin 从 `league-latest-12311` 切到 `league-final-14858`，
  保留 V1.1 作为 fallback，`real_game_accepted` 仍为 `false`；
- `doctor --game-dir D:\Hearthstone --infer` **exit 0**：
  `checkpoint_ok=true`、`profile_ok=true`、`model_profile_compatible=true`、
  `synthetic_forward_ok=true`、`parameters=16436034`、`policy_id=league-final-14858`、
  `ready_for_user_test=true`；`live_ready=false`，唯一 blocker 是
  “current runtime still requires user-executed real-game acceptance”（即 E-0 人工验收，符合预期）。

## 12. 剩余风险与建议顺序

| 风险 | 级别 | 建议 |
|---|---|---|
| D-026 实机 obs/action 投影近似 | P1 | **上连续实机前必须先量化/接受**；先 E-0 单局看工程可靠性，E-0 名次不作棋力结论 |
| final 的 Freeze 倾向高（服务器约 22%） | P1 | 本地桥接全冻结不暴露 action 2，会自然抑制；E-0 观察决策速率与 Freeze 次数 |
| p75band late-turn 仍受限（turn 24 只有 2 帧） | P2 | 下一轮先用 band；要提升 late-game 需重建更大 EOT 库 |
| stage2 LR 视野/anchor warmup 是新配置 | P2 | 先跑 ≤2h 短程确认吞吐/稳定性，再上正式长训 |
| 60 秒级训练/评测代码自本轮起才修好 | — | 旧 E1 报告用旧代码；正式榜已用 r2（审计后冻结代码），旧榜不可混 |

**下一轮长训未启动**（服务器等用户确认）：`config/league-stage2.json` + `fork_stage.py` 已备好。
