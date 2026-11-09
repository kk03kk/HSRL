> HSAI迁移说明：以下保留截至c4c71014e3d41760的原始fork差异记录。历史路径rust/在新工作区对应engine/；原字节版本保存在records/source-docs/hsbg/rust/FORK.md。

# FORK.md — hsbg_engine（AutoTavern TavernGym fork）

## 1. 来源与许可

| 项 | 值 |
|---|---|
| 上游 | `MCG-NJU/AutoTavern` |
| commit | `1c2086a997afde46d68c6c5992cab194ddcfce88`（TavernGym 37.0.2；初始 fork 基于 fb61201） |
| 许可 | MIT（`LICENSE` / `NOTICE` / `CITATION.cff` 原样保留） |
| 引擎数据 | AutoTavern schema，**37.0.2.256589**（3983 卡） |
| fork 日期 | 2026-10-26 |

## 2. 重命名

| 上游 | 本 fork |
|---|---|
| `crates/taverngym/` | `crates/hsbg-engine/` |
| crate `taverngym` | `hsbg-engine`（lib `hsbg_engine`） |
| `crates/taverngym-py/` | `crates/hsbg-engine-py/` |
| Python 包 `taverngym` | `hsbg_engine` |
| maturin `module-name = taverngym._core` | `hsbg_engine._core` |

## 3. 相对上游的实质改动

1. **数据来源**：`crates/hsbg-engine/data/cards.json` 原样取自固定上游提交。新表由 `tools/p2_data/check_carddb_3702.py` 校验。旧版 `data/p2/` 与生成器仅供历史复现。换数据后必须重建原生扩展。
2. **版本 meta 驱动**：`game_version()` / `server_rules_version()` 读 `cards.json.meta`；`db_info()` / `data_info()` 供一行核对。
3. **搜索**：新增 `src/search.rs`（逐步 MCTS，PyO3 `mcts_search`）与 `src/turn_search.rs`（回合内搜索；不进入 AZ 主路径）。
4. **大厅种族**：37.0.2 与上游一致。NAGA 不进入默认随机类型列表，显式配置可以强制加入。
5. **构建**：macOS `cargo build --release` 的 `-undefined dynamic_lookup` 链接参数；maturin 兼容。
6. **回合末场面快照镜像**（新增能力，默认不启用）：`src/snapshot.rs` 的 `Game::apply_turn_snapshots(snaps, advance)` +
   PyO3 `Game.apply_turn_snapshots`（同时暴露只读 `Game.pool_balance()` / `pool_species_violations()` /
   `invariant_violations()` 与 `snapshot_schema_version()`）。语义与白名单见 `docs/MIRROR_SNAPSHOT_DESIGN.md`
   「实现状态」：字段白名单替换 + typed EID 重映射 + `pool_delta_unknown`（旧持有物按 pool parts 归还、
   注入实体 `pool_copies=0`）。不改变任何既有机制/数值与默认行为（无调用时零影响）。
7. **人类战团（board-only）注入**（新增能力，默认不启用）：`src/snapshot.rs` 的
   `Game::inject_board(seat, minions, tier)` + PyO3 `Game.inject_board(seat, [[card, atk, hp], ...], tier=None)`。
   只替换 `board`（+ 可选 `tier`）与派生 `last_board`；hand/shop/资源/生命周期/配对全部保留；
   新随从 `alloc_eid()` 重分配且 `pool_copies=0`（`pool_delta_unknown`），旧 board 持有量归还共享池；
   指向被移除旧实体的 typed 引用（`link/group/src`、`hats`、`revert_golden`、`discover_targets`）就地清空；
   逐座位事务（非法输入不改状态/`next_eid`）；**不设置** `ended_turn`。语义与近似边界见
   `docs/HUMAN_BOARD_OPPONENT.md`。不改变任何既有机制/数值与默认行为（无调用时零影响）。

8. **37.0.2 嵌套收费饰品菜单死锁修复（1078314）**：上游已有“当前收费饰品菜单全部买不起则跳过”的规则，但只在 `choose(Trinket)` 后调用。购买邪恶邀请等产生发现时，下层菜单被遮住；发现结束后可能只剩 2 金，而 Kiri 替换菜单四项均为 3/4 金，形成零合法动作。`choose` 的所有分支、成功 `apply` 的队列/跨席效果收尾、`start_turn_player` 收尾均复用同一检查。免费菜单、有可购买选项的菜单以及其他待选均保留；不补金币、不放开 EndTurn、不清空未知菜单。新增真实饰品嵌套发现回归和免费/可负担/折扣恢复边界测试。卡表、动作编码、观测布局和序列化格式不变；源码指纹改变，旧库必须逐帧复验并发布新 manifest，不能原地改指纹。

9. **2026-11-04 在线禁用覆盖**：暴雪官方在线禁用表再次禁用两种 Kiri。固定上游 `1c2086a` 未包含这一变化，故默认 `RulesConfig.excluded_cards` 增加两种 Kiri、Putricide Sticker 和 Gentle Djinni（后二者原卡表已无候选资格）。卡表、词表不改。修复此前遗漏的饰品/黑暗赠礼候选池过滤及饰品资格检查；历史配置可显式 `excluded_cards=[]`，不删除旧实现或旧数据。来源：https://us.forums.blizzard.com/en/hearthstone/t/bans-what-and-where/151289 。
10. **待选不能静默丢弃**：原生 Bot 的 pending 推荐经过合法掩码，空目标且可取消时推荐 Cancel；真正零合法动作不伪造成功。`end_turn_silent` 遇到 pending 原样拒绝推进。普通回合末自动选择仍沿用上游的随机策略；零合法且不可取消、选择链超过 256 或选择执行失败时返回错误并保留待选，不再 `pending.clear()` 继续作战。该自动选择策略及“不可负担收费菜单跳过”仍是模拟器假设，不能标作暴雪已核实机制。
11. **源码审计门禁**：`tools/p2_data/check_engine_upstream.py` 对照 `personal/configs/engine-upstream-lock.json` 核对全部引擎/绑定 Rust 源码及上游测试文件清单。固定上游规范化后相同文件直接锁定，所有 fork 差异登记原因；源码变化需重新审计并更新清单。该离线门禁不自动查询最新远端。

12. **关联手牌生命周期**：真实 Bot 对局 seed 8700010 复现：弃置激活效果移走配对卡，另一张 link 悬空；三连消耗也漏清。修复弃置、三连与临时卡到期的反向配对引用清理。seed 8700018 在显式历史池中进一步复现：发牌立即三连后仍用已移除 hid 建立新配对；set_hand_link 现仅引用实际手牌或等待队列中的实体。增加对应三项原生回归及动作间关联引用不变量。只解除已消失的实体关系，不把弃置等同于打出，不连带弃掉幸存牌。新原始 Bot 帧若再有 unknown_ref，建库失败关闭，不再排除后继续发布。

13. **满手队列追加与质量诊断**：随机种子1078368在Timewarp开启时复现，满手时空购买向hand_queue追加而没追加QueueMark，重建时丢掉受保护队列的hid/link/gifts。修复时空购买与实验伙伴两处同类入口，追加默认标记但保留已有项；定向回归保留配对身份，六配置随机测试覆盖。PyO3新增diagnostic_counts常数时间返回战斗/招募截断、回合上限、诊断停止，供Python采集与评测失败关闭；观测与动作空间不变。

14. **37.0.2 Jar o' Gems 每战斗一次修复（2026-11-09 审计）**：37.0.2 将饰品文本从“after 2 friendly minions attack each combat”改为“After a friendly Quilboar attacks each combat”，语义是每个战斗首次友方野猪人攻击后触发一次；原实现既用了 `Attacks`（攻击伤害前）而不是 `AttackDone`（攻击结算后），又删掉了旧计数却没有加入每战斗门槛，导致每次野猪人攻击都在伤害前触发。现改用 `AttackDone`，并用 `Ctx::combat_data` 记录本战斗是否已触发；非野猪人攻击不消耗额度，新战斗重置。新增 `patch_3702::jar_of_gems_triggers_once_per_combat_on_the_first_quilboar_attack`。卡表、动作、观测和序列化格式不变；源码指纹由 `c7b5f27d6f40d1d2` 变为 `c4c71014e3d41760`，旧库必须复验并发布新 manifest。

245 动作、3227+124 观测与卡牌下标保持上游契约。上游是第三方模拟器；上述测试不构成与真实服务器机制完全等价的证明。

## 4. 数据

- `data/cards.json` 只允许由 `tools/p2_data/gen_carddb_37.py` 生成；禁止手改。
- 当前数据版本：`37.0.2.256589`；`game_version()` 实测返回该值。
- 换数据：替换 JSON → `cd rust && python -m pip install -e .` → `cargo test --release -p hsbg-engine`。

## 5. 测试/构建

```bash
cd rust
python -m pip install -e .
cargo build --release -p hsbg-engine
cargo test --release -p hsbg-engine
python -m pytest ../tests/test_rust_engine.py -q
```

当前（2026-11-09 审计）：Rust 942 passed / 0 failed / 1 ignored（43 个测试二进制）；Python 381 passed / 14 skipped；`N_ACTIONS=245`、`OBS_FLOATS=3227`、`OBS_CARDS=124`。

## 6. 规则权威

- 机制/数值以 `hsbg_engine` + 固定上游 37.0.2.256589 数据为准；AutoTavern 是第三方重建，不是官方服务端。
- 任何机制修正必须：写失败测试 → 改 Rust → 跑全量 Rust 测试 → 记录到本文件（动机/证据/影响）。
- 不保留其它引擎作为 fallback。
