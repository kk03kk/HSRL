# 模拟器规则审计（2026-11-09）

状态：**当前审计报告**。范围仅限 `engine/` 模拟器与 37.0.2 正式服规则；未启动新的正式评测或训练。

## 1. 正式服版本与来源

| 项 | 值 |
|---|---|
| 客户端版本 | `37.0.2.256589` |
| 服务器规则版本 | `37.0.2` |
| 正式服发布日期 | 2026-11-03 |
| 核实时间 | 2026-11-09 08:20–09:10 +0800 |
| 最新客户端构建 | HearthstoneJSON `/v1/256589` 为最新；未发现 256589 之后的客户端构建 |

来源与交叉验证：

- 官方补丁公告：`https://hearthstone.blizzard.com/en-us/news/24307311/3702-patch-notes`
- Hearthstone Wiki：`https://hearthstone.wiki.gg/wiki/Patch_37.0.2.256589`
- HearthstoneJSON 最新数据：`https://api.hearthstonejson.com/v1/latest/`，目录为 `/v1/256589`
- 37.0.0 / 37.0.1 热修复：`https://hearthstone.wiki.gg/wiki/Patch_37.0.0.254477`
- 36.6.3 / 36.6.0：`https://hearthstone.wiki.gg/wiki/Patch_36.6.3.253932`、`https://hearthstone.wiki.gg/wiki/Patch_36.6.0.251952`
- 在线禁用表：`https://us.forums.blizzard.com/en/hearthstone/t/bans-what-and-where/151289`

截至审计时间，没有发现 37.0.2 之后的新客户端补丁或新的 Battlegrounds 数值热修复。37.0.2 的已知改动均按正式服已实装处理。

原始来源摘要与测试计数：`records/research/simulator-audit-2026-11-09/sources.json`。

## 2. 模拟器与上游状态

| 项 | 值 |
|---|---|
| 引擎 | `engine/`，`hsbg_engine`（AutoTavern / TavernGym fork） |
| 上游 | `MCG-NJU/AutoTavern` |
| 上游 HEAD | `1c2086a997afde46d68c6c5992cab194ddcfce88`（2026-11-04，37.0.2） |
| 上游同步 | 已同步到 37.0.2；无更新提交可拉取 |
| 数据 | `37.0.2.256589`，3983 卡，词表/动作/观测布局未变 |
| 接口 | 245 原生动作；3227 floats + 124 cards |
| 审计前源码指纹 | `c7b5f27d6f40d1d2` |
| 审计后源码指纹 | `c4c71014e3d41760`（Jar o' Gems 修复后） |

上游 37.0.2 数据本身也保留了本节发现的 `Jar o' Gems` 语义问题；该修复只存在本 fork，未修改固定上游卡表。

## 3. 内容覆盖

由 `examples/content_report.rs` 在 37.0.2 数据上生成：

| 类别 | 已实现 | 当前候选 |
|---|---:|---:|
| 商店随从（Tier 1–6，含纯关键词） | 244（239 scripted + 5 keyword-only） | 244 |
| 英雄（英雄技能已脚本化） | 116 | 116 |
| 酒馆法术 | 70 | 70 |
| 饰品 | 234 | 234 |
| 黑暗赠礼 | 47 | 47 |
| Tier 7 | 12 | 12 |
| Timewarp | 253 | 253 |

说明：

- 当前候选数已排除 2026-11-04 在线禁用（两种 Kiri、Putricide Sticker、Gentle Djinni）和 37.0.2 从池中移除的 Expert Aviator。
- Naga 自 36.6.0 起临时退出酒馆随从池，37.0.2 未回归；因此 Naga 专属随从/法术（如 Spitescale Special、Shifting Tide、Queen's Command、Sea Witch Zar'jira）不计入当前候选。
- Duos 专属随从/法术按 solo 模拟器范围排除。
- `Rot Hide Gnoll` 的客户端 pool tag 为 `2`，默认 `strict_pool_tag` 将其排除；这是既有模拟器假设，未凭猜测加入。
- 覆盖率是“注册/可进入池”的覆盖，不等于逐条机制已被正式服证明。
- 非当前池差异：`BG28_Reward_505` Tumbling Disaster 的引擎文本为 +1/+1，HearthstoneJSON 256589 为 +4/+4；Wiki 注明它已在 33.6.2 从奖励池移除，引擎 `POOL_REWARDS` 也未包含它，因此不影响当前对局。该条保留为历史数据差异，未按猜测改动。

## 4. 37.0.2 逐项核对

### 4.1 随从池

| 项 | 数据 | 行为/测试 | 结论 |
|---|---|---|---|
| Murcules 回归 | `BG27_023` 在 pool | 已脚本化；随机手牌随从 +3/+3 | 通过 |
| Expert Aviator 移除 | `BG34_140` 不在 pool | 脚本保留但不可进入池 | 通过 |
| Dark Paradox T2 | 6/2，+2 Health | `paradox_played` +2/g | 通过 |
| Dark Paradox T3 | 4/4，+4/+4 per Battlecry | `apply_bonuses` 4/g | 通过 |
| Harbinger Aph'lass | Tier 5 | 数据 Tier 5；discard 触发 +2/+1 | 通过 |
| Sacrificial Wrathguard | 6/3，+2/+1 | 脚本 2/1；Activate 提升 | 通过 |
| Eredar Escapist | 5/10，5 次伤害 | 计数阈值 5 | 通过 |
| Ashen Corruptor | 4/8，+1/+2 | 非对称 Tavern 临时 buff | 通过 |
| Crimson Vindicator | 9/7，Windfury | 数据含 Windfury；Rally 正常 | 通过 |
| Living Azerite | 6/6，+3/+3 | 代码 +3/+3（注释旧值已过时） | 通过 |
| Ultraviolet Ascendant | 8/8，+4/+4 | 代码 +4/+4 | 通过 |
| Unbound Tempest | 3/9，3 个 Elemental | 阈值 3 | 通过 |
| Spark Snapper | 8/8，2/2 Satellite | 代码 2/2 | 通过 |
| Glambot | 5/5，5/5 Satellite | 代码 5/5 | 通过 |
| Shamanic Tidecaller | 6/9，+3/+4 | 代码 3/4（golden 6/8） | 通过 |
| Twilight Tidehunter | 3/7，+9/+9 | 代码 9/9 | 通过 |
| Murcules 效果 | 手牌随从 +3/+3 | `patch_3702` 随机目标测试 | 通过 |
| Enterprising Escapee | 7/5，spend 5 | 计数 5；golden 提前 2 回合 | 通过 |
| Ship Master Eudora | 10/3，+8/+8 | 代码 8/8；golden 8/8 twice | 通过 |
| Hooktusk | 5/5，your Pirates +1/+1 | 代码包含自身及所有 Pirates | 通过 |
| Hot-Air Surveyor | Tier 5 | 数据 Tier 5 | 通过 |
| Sanguine Refiner | 4/9，+1/+1 | 代码 1/1 | 通过 |
| Jailbird Juggernaut | 13/17 | 数据 13/17 | 通过 |
| Obsidian Ravager | 11/11 | 数据 11/11 | 通过 |
| Futurefin | 17/23 | 数据 17/23 | 通过 |
| The Last One Standing | 10/10，+10/+10 并 improve | 代码 10 起，combat/recruit 均记录提升 | 通过 |
| Champion of Sargeras | 12/12，+12/+12 | 代码 12/12 | 通过 |
| Stalwart Kodo | 15/30，不同随从，双倍最大属性，3 次 | 代码 2x（golden 3x），3 次/战斗 | 通过 |

### 4.2 酒馆法术

| 项 | 数据 | 行为/测试 | 结论 |
|---|---|---|---|
| Misplaced Tea Set | 2 费，+3/+3 | 代码 3/3 | 通过 |
| Perfect Vision | 1 费 | 数据 1 费，set 20/20 | 通过 |
| Wave of Gold | +3/+3；Golden 再 +3/+3 | 代码 3/3 + golden 3/3 | 通过 |
| Natural Blessing | +2/+2 | 代码 2/2 | 通过 |

### 4.3 饰品

37.0.2 的饰品改动分三类核对：

- **池移除**：Spell-powered Wrench；Quilligraphy Set / Fridge Magnet / Writhing Tentacles（Greater）。数据中均 `pools=[]`。
- **费用调整**：8 个 Lesser 涨价、14 个 Lesser 降价、9 个 Greater 涨价、4 个 Greater 降价；全部由 37.0.2 数据驱动，代码按 `CardDef::cost` 使用。
- **文本/效果调整**：27 项。逐项核对后，数值、窗口、目标限制均与数据一致；`Wand of Divination` 的 Quillboar 大厅禁用、`Relentless Deflector` 与 Furtiveness 的禁配、`Toughened Shield` 的 Quil/Naga 大厅限制均有对应实现或测试。

发现并修复 1 项行为错误：

| ID | 问题 | 严重度 | 修复 |
|---|---|---|---|
| `BG30_MagicItem_546` Jar o' Gems | 37.0.2 文本为“每场战斗中，在一个友方野猪人攻击后”触发一次；代码既用 `Attacks`（伤害前）而不是 `AttackDone`（攻击结算后），又在每次野猪人攻击时都触发，失去每战斗一次的门槛 | **高**（野猪人/饰品的经济与属性增长被显著放大，且宝石在攻击伤害前生效） | 改用 `AttackDone` + `combat_data` 每战斗一次门槛；新增时序/次数回归测试 |

修复后的行为：

1. `Attacks`（伤害前）不触发；必须等到 `AttackDone`（攻击结算后）；
2. 非野猪人攻击不消耗次数；
3. 第一次友方野猪人攻击触发一次，对每个友方野猪人各打一张鲜血宝石；
4. 同一战斗中后续野猪人攻击不再触发；
5. 新战斗重新获得一次触发额度。

### 4.4 黑暗赠礼

逐项核对的数据/行为：

- Fresh Perspective 窗口 4–5；Battle Scars 7–9；Charisma 7–10；Golemancy 9–12；Sharpened Sword / Toughened Shield 3–5；Worship 从 11 起。
- Desecration +3/+2；Incubation +6/+6；Hostility +15 Attack；Resistance +15 Health；Toughened Shield +2 Health。
- Relentless Deflector 不可与 Furtiveness 搭配。
- 第 7 回合起，每次 Dark Gift Discover 至少 2 个“最常见类型”候选（`dark_gift_slots` 中两个 typed 位置）。

以上均有数据或既有回归测试覆盖；未发现新的不一致。

## 5. 近期补丁核对

| 补丁 | 与模拟器相关的内容 | 结果 |
|---|---|---|
| 37.0.2.256589 | 26 个随从、4 个法术、4 个饰品池移除、35 个费用调整、27 个饰品文本/效果、13 项 Dark Gift、统计/动画 | 数据全部为 37.0.2；发现并修复 Jar o' Gems 时序/次数错误 |
| 37.0.1 服务端热修复 | Hooktusk 与 Timewarped Nellie's Ship；Leeroy/Toreth/Heroic Broodmother 的 Dark Gift 禁配；Recruit 阶段死亡 Reborn；Gunpowder Courier 文本；Mask of Ancient Ones 属性；Hoarding Hyena 死循环；Stormpike/Frostwolf 双重触发 | 数据/禁配/测试已覆盖；Mask/Hyena 有定向测试；其余按代码检查 |
| 37.0.0.254477 | 无 Battlegrounds 数值改动；上述 BG 热修复项 | 已处理 |
| 36.6.3.253932 | Deity 门槛 3→4；Aberration 不再保底；随从/法术/饰品平衡 | 数据为 37.0.2；关键值和机制已实现 |
| 36.6.2 热修复 | Dark Gift 条件、Efficiency 排除等 | 代码包含相应排除/测试 |
| 36.6.0.251952 | Aberration 加入；Naga 暂时退出随从池 | 当前仍生效；Naga 专属内容不计入池 |
| 36.2.0（Season 14） | Dark Gift、Activate、Lockbox、Fishbait、Venomous 新规则 | 数据/脚本已覆盖 |

## 6. 测试与验证

审计前基线：

- `cargo test --release -p hsbg-engine`：43 个 Rust 测试二进制，941 passed / 0 failed / 1 ignored。
- 忽略项：`p3_cards_a` 中 “Discover Naga” 用例——当前 Naga 不在池，属于预期跳过。

修复后：

- `cargo test --release -p hsbg-engine`：942 passed / 0 failed / 1 ignored（新增 Jar 回归）。
- 新增 `patch_3702::jar_of_gems_triggers_once_per_combat_on_the_first_quilboar_attack`。
- 全量 Rust 套件未出现回归。
- 原生扩展已用修复后源码重建：`build_is_current=True`，`source_hash=c4c71014e3d41760`，`db_info=(3983, 37.0.2.256589, 37.0.2)`。
- 工作区验证 `tools/verify_workspace.py` 全绿：卡表摘要、上游源码锁、Rust、Python 四项均通过；Python 381 passed / 14 skipped / 0 failed。
- 审计中发现并修正一个既有的部署链测试漂移：`p50_forced_preflight.sbatch` 已被改为 30 分钟，测试仍期待 1 小时；测试已对齐当前部署文件，与本模拟器修复无关。

源码锁：

- `config/engine-upstream-lock.json` 已更新 `content/trinkets_p8.rs` 的 fork 摘要和修复原因；`check_engine_upstream.py` 通过。

## 7. 剩余风险

- 模拟器是第三方重建；100% 池覆盖和测试通过不等于对正式服服务端的逐位等价证明。
- 仍有未公开或未定位的规则：Dark Gift 的候选权重/回退、部分随机目标选择、隐藏触发顺序、饰品/任务奖励的边界条件。
- `Rot Hide Gnoll` 的 pool tag 2 排除是既有假设；如果 37.0.2 实际在池，会造成 1 个 Tier 1 随从缺失。需要实机或更权威 tag 证据才能改动。
- `data/bg_*.json`（客户端 legacy encoder 使用）仍是 37.0.0.255790 兼容快照，不是引擎数据；不影响当前训练/评测，但实机桥接前应重新生成。
- Jar o' Gems 修复改变了规则源码指纹。旧的 bot 库、评测 manifest 和 `source_hash=c7b5f27d6f40d1d2` 的结果不能与新引擎混用；下一轮必须先重建原生扩展，再对旧库逐帧复验并发布新 manifest。
- 本次没有进行真实客户端对局日志对比；规则正确性证据主要来自官方补丁、卡牌数据、定向测试和代码审查。

## 8. 结论

模拟器数据已对齐 **37.0.2.256589**，当前池内随从、法术、饰品、黑暗赠礼和英雄均有实现；37.0.2 的数值/文本改动已同步。审计发现并修复了 `Jar o' Gems` 的高严重度触发时序/次数错误。

在以下前提下，模拟器可以进入下一轮评测/训练：

1. 使用重新构建后的原生扩展（源码指纹 `c4c71014e3d41760`）；
2. 对旧 bot 库/评测库执行逐帧复验并发布新 manifest；
3. 不把旧 `c7b5f27d6f40d1d2` 协议下的结果与新协议结果混榜；
4. 对 Naga、Duos、Rot Hide Gnoll 等已知边界明确标注，不把测试通过当成正式服完全等价证明。
