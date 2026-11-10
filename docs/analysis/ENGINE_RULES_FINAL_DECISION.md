# 引擎规则最终判定与处置（2026-11-10）

依据：`ENGINE_RULES_AUDIT_REPORT.md` + 用户对 Q1–Q4 的回答 + 本轮只读实验与回归测试。
**本文件结论：不修改 HSAI 引擎代码。**

---

## 0. 结论

| 编号 | 规则差异 | 你的回答 | 代码事实 | 判定 | 处置 |
|---|---|---|---|---|---|
| **A1** | 两张 Kiri 饰品能否买到 | 仍禁用，按 HSAI | HSAI 默认剔除 `BG36_MagicItem_412/412t2`（`config.rs:11/73/257` + `flow.rs:169/189/195/207/757`） | ✅ HSAI 正确 | **不改** |
| **A2** | 罐装宝石触发次数/时机 | 按 37.0.2 卡面 | HSAI：`AttackDone` + 每战斗一次 + 仅友方野猪人攻击者（`trinkets_p8.rs:275-297`） | ✅ HSAI 与卡面一致 | **不改** |
| **A3** | 饰品单"买不起"怎么收场 | 极端场景，不必纠结 | 见 §2：正常流程**到不了**空掩码 | ✅ 无需处置 | **不改** |
| **Q4** | 目标在结算前消失 | 极罕见，低优先级 | HSAI 无越界防御（SIM 有） | 维持现状 | **不改** |

**本轮改动的文件：无。** 引擎源码、`client/`、`hsbg/` 全部未动；
因此 `source_hash`、模型兼容、动作空间（245）、观测（3227+124）、训练配置均保持不变。

---

## 1. 罐装宝石实现核对（逐条对照卡面）

卡面（37.0.2，引擎卡表 `text_zh`）：**「每场战斗中，在一个友方野猪人攻击后，本饰品对你的所有野猪人各使用一张鲜血宝石。」**

| 卡面要素 | HSAI 代码 | 是否符合 |
|---|---|---|
| 只在战斗内 | `if !c.combat { return }` | ✅ |
| 友方（`me.side`） | `*side != me.side` 过滤 | ✅ |
| 触发者是**野猪人** | `c.minion(*attacker).is(Races::QUILBOAR)` | ✅ |
| "攻击后"（攻击已结算） | 监听 `Event::AttackDone`（而非 `Attacks`） | ✅ |
| **每场战斗一次** | `combat_data(me.side, me.card) >= 1` 早退 + `add_combat_data(…,1)` | ✅ |
| 对**所有**友方野猪人各贴一张 | `friendly_where(…Races::QUILBOAR)` → `play_gem` 逐个 | ✅ |

对照：HSBRSIM 当前是 `Attacks`（伤害结算前）+ 无计数门槛（每次野猪人攻击都触发），且注释仍写 "after every 2"，
与其代码不符；上游 TavernGym 旧版是"每 2 次攻击"（有 `% 2` 与"还剩 2"显示）。
→ **HSAI 与 37.0.2 卡面一致，无需修改。**

已存在的回归测试（Rust）：`engine/crates/hsbg-engine/tests/patch_3702.rs:700`
`jar_of_gems_triggers_once_per_combat_on_the_first_quilboar_attack`。
（本地无 Rust 工具链，本轮未能执行，见 §5。）

---

## 2. 空 `legal_mask` 专项排查（本轮重点）

### 2.1 实验（只读，不改引擎）

脚本：`artifacts/local/engine-audit/probe_trinket_mask.py`（用 `state_json()` / `from_json()`，不触碰引擎源码）。

| 场景 | 结果 |
|---|---|
| 真实对局路径：5 个种子各打完整 bot 对局（931–1341 步） | **0 次空掩码**、**0 次不变量违例**，全部正常结束（15–18 回合） |
| 正常走到饰品回合（turn 6，种子 12345） | 挂起时金币 8，`legal_mask` 只有 4 个动作（四个饰品选项），无 EndTurn/Buy |
| 把该挂起状态的金币手工改成 0 再载入 | `needs_action=True` 且 `legal_mask` **全 False**（复现 HSBRSIM `063fe69` 描述的死锁形态） |
| 同上，金币改成 1 / 2 | 分别有 2 / 3 个合法选项（便宜的仍可买） |

### 2.2 可达性分析：正常流程为什么到不了

1. **所有创建点都在创建时检查"至少一个买得起"**：
   - 回合开始：`flow.rs:542-576`（`offer.is_empty() || 全部价格 > gold` → 整个报价跳过，注释 "an assumption"）；
   - Kiri 换装：`content/trinkets_p6.rs:162-170`（`d.cost <= 4 && d.cost <= gold`）；
   - 饰品换购：`content/trinkets_p6.rs:294-304`（`d.cost <= gold`，免费换购除外）；
   - 英雄技能/任务奖励：`heroes.rs:424`、`quests.rs:1799` 都走 `offer_trinkets()`，同一守卫。
2. **挂起期间只允许选项动作**（实测掩码只有 4 个 `Choose`），玩家无法同时买随从/刷新/升本把金币花掉。
3. **快照注入会拒绝带 pending 的帧**：`snapshot.rs:246-248`（`snapshot_pending(n)`）、`snapshot.rs:405-406`（`pending_choice(n)`）
   → 我们的镜像/评测注入路径也不会产生这种状态。

**结论**：空掩码只能由**手工构造或已损坏的状态**产生。按 `AGENTS.md` §2「工程错误严格停止并保留现场」的既定策略，
这种情况下报错（而不是静默改成"合法放弃"）是符合项目约定的；且修复需要改源码 → 改指纹 → 连锁影响，
故**本轮不改**。

### 2.3 若未来要改（仅记录，不执行）
最小改法与影响面：`legal_mask` 的 pending 分支在"没有任何合法选项"时放行 `A_CANCEL`（该动作编号 7 已存在），
并在 `cancel()` 中对"付费饰品单全部买不起"调用现有 `drop_unaffordable_trinket_offers()` 后返回 Ok。
- 动作空间：编号不变，只在该死状态新增一个合法动作；
- 观测：不变；
- 指纹：会变（需重建扩展 + 更新锚点），这是本轮不做的核心理由。

---

## 3. 兼容性评估（本轮结论：无影响）

| 对象 | 是否受影响 | 说明 |
|---|---|---|
| 模型 `artifacts/models/v1.1` | 否 | 引擎源码未改，`trained_engine_source_hash` 语义不变 |
| 动作空间 / 观测 | 否 | `N_ACTIONS=245`、`OBS_FLOATS=3227`、`OBS_CARDS=124`（实测） |
| 训练/评测协议 | 否 | 未改采样、奖励、损失 |
| 服务器训练 | 否 | 未推送任何引擎改动；服务器保持 `c4c71014` |

---

## 4. 发现的一个环境问题（不是代码问题，需单独处理）

**本地 `.venv` 里的 `hsbg_engine` 是旧构建**：

```text
.venv/Lib/site-packages/hsbg_engine/_core.pyd   2026-11-06 12:18   build_info.source_hash = c7b5f27d6f40d1d2
engine/.../content/trinkets_p8.rs               2026-11-09 18:45   （Jar o' Gems 修复）
```

也就是说：**本地离线跑的是修复前的引擎**，与当前源码（`c4c71014e3d41760`）不一致。
`tools/verify_workspace.py` 里**没有**"构建是否最新"的检查，所以这个问题不会被本地门禁发现。
本机没有 Rust 工具链（`cargo`/`rustc` 均不存在，`~/.cargo` 也不存在），**无法本地重建**。

影响范围与建议（不涉及代码改动）：
- 影响：本地离线回放/冒烟/实机推理用的是旧规则（仅 Jar o' Gems 等 11-09 修复项不同）；
- 对策：在有 Rust 的机器上重建一次扩展（服务器最方便），或本地安装 Rust 后 `maturin build`；
- 重建后需要同步更新锚点（`config/live/model.json` 的 `trained_engine_source_hash` 等）并重跑 `verify_workspace`。

---

## 5. 回归测试结果

| 项目 | 结果 |
|---|---|
| 实机链路测试 `tests/test_live_*.py` | **58 passed** |
| 引擎冒烟（5 局完整 bot 对局） | 0 空掩码、0 不变量违例；`db_info=(3983,'37.0.2.256589','37.0.2')`、`N_ACTIONS=245`、`OBS=3227+124` |
| 全量 `pytest tests/` | **22 个失败/错误（全部为预存在，与本轮无关）** |
| Rust 测试（含罐装宝石回归） | **未运行**：本机无 `cargo`/`rustc` |

22 个失败的根因（已核实，均非引擎规则）：
1. `test_move_guard.py`（6 个）、`test_az_ppo.py::test_engine_mcts_selfplay_ppo_smoke`、`test_parallel_collect.py`：
   `PPOSelfPlayPolicy` 的"强制动作规则"守卫（`max_actions_per_turn=80 / move_guard / move_cap=5`）由提交
   `e4d4dfc`（2026-11-09）引入，而相关测试最后修改是 2026-11-06 → 测试落后于代码；
2. `test_sustained_learning.py`（5 个）、`test_learning_experiment.py`（3 个）、`test_calibrated_opponents.py`（2 失败 + 2 错误）、
   `test_curriculum_long.py`、`test_checkpoint_cardtable_migration.py`：依赖服务器面板/大样本/特定环境，本地不具备；
3. `test_az_ppo.py::test_entry_script_help`：Windows 控制台编码导致 `subprocess(text=True)` 解码失败（`stdout=None`）。

（证明"预存在"：本轮 `git status` 只有两份未跟踪的报告文件，没有任何源码修改。）

---

## 6. 结论与后续

1. **引擎不需要修改**：A1/A2 已符合真实规则，A3 在正常流程不可达，Q4 按你的判断保持现状。
2. **本轮零代码改动** → 模型/动作/观测/训练/服务器全部不受影响。
3. 需要单独处理的**环境问题**：本地扩展是 11-06 旧构建（`c7b5f27d`），与源码（`c4c71014`）不一致，需在有 Rust 的机器重建。
4. 建议监测（不改代码）：训练/评测日志若出现 `mask 全 False` 或 `unresolvable end-of-turn choice`，按既定策略立即停机保留现场。
5. 22 个预存在测试失败属"测试与 API 漂移"，与引擎规则无关，可另开一轮对齐（不在本轮范围）。
