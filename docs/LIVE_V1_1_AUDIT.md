# V1.1 实机全链路审计与数据体系

状态：**当前文档；离线审计/修复/验证已完成，等待用户下一轮实机测试。**  
范围：本地 `client/` 真实游戏桥接；不改模型权重、训练协议、奖励或模拟器规则。  
上一局证据：`artifacts/runs/live-v1.1-test-1/`（只读保留）。

## 1. 结论

主链路可用：V1.1 权重与 SHA 一致，真实客户端状态能进入 3227+124 模型输入，模型动作能经过身份绑定和几何规划发送，61/61 个已发送动作获得服务端因果回执，无内部异常、无未决输入的 blind retry。

上一局不是崩溃：进程正常退出、`Power.log`/事件归档完整；唯一显式失败是
一个五选一发现面板没有标定几何，脚本安全暂停而非点击。

审计发现并修复的核心问题：

| 优先级 | 问题 | 根因 | 修复 |
|---|---|---|---|
| P0 | 选择面板等待时因对手公开状态更新取消已授权点击 | 点击前用完整模型状态哈希重验，包含无关对手摘要 | 新增 `execution_stamp`：只绑定阶段/回合/交互/实体身份/授权动作，保留完整模型状态记录 |
| P0 | 同一回合可产生与训练不一致的 Move 序列 | 实机只粗略限制 8 次换位，训练固定 5 次、同动作 2 次、禁逆换位 | 实机解码头对齐 `max_actions=80, move_cap=5, repeat=2, inverse=blocked` |
| P1 | 五选一/六选一没有几何条目，整段暂停 | 标定表只有 1–4 项 | 派生可审计运行时行；点击前仍必须通过绿色/金色横梁与实体顺序证明 |
| P2 | 点击前 JPEG 编码阻塞输入 | 2560×1600 同步 JPEG 约 31–62 ms | 独立低开销帧队列异步编码，输入门槛不等待磁盘 |
| P1 | 缺少逐决策可重建数据 | 只有共享 event JSONL，模型全快照不在局目录 | 每局写 compact + full decision、action、summary、performance、Markdown |
| P1 | 没有真实转型对照工具 | 无 S→A→S' 公开状态接口 | 新增离线 `hsgb.eval.live_simulator_check` 与门禁分类 |

## 2. 上一局异常还原

时间线（`agent-events.jsonl`，74 次真实推理、49 次缓存复用）：

- 19:25:00 hero 选择：模型 `choose`，点击 + 确认，19:25:02 `applied`。
- 19:25:22–19:25:34 第一回合购买、打出、EndTurn，全部有服务端回执。
- 19:27–19:43 正常招募/战斗循环；共 66 个 input stage，61 个 applied。
- 19:28:20 一次 `cancelled_before_press`：悬停后完整状态哈希变化，脚本在按下前取消；没有盲点。
- 19:35:11 `uncalibrated layout:discover/choices/5`：5 选一没有几何条目，输入暂停约 15 秒；19:35:25 服务端自动/外部选择后才继续。
- 19:43:09 `PLAYSTATE=CONCEDED` 后本局结束；结果 `4` 名，`exit.json` 正常、archive complete。

性能（离线复算，不重新启动游戏）：

| 指标 | 值 |
|---|---:|
| 真实推理 p50 / p90 / max | 18.47 / 20.61 / 71.38 ms |
| 决策到 dispatch p50 / p90 | 287 / 345 ms |
| dispatch 到 input stage p50 / p90 | 2.2 / 49.5 ms |
| 完整原生输入 p50 / p90 | 377 / 397 ms |
| 同帧同步 JPEG | ~62 ms |
| 异步帧入队（新） | ~0.01 ms |

## 3. 正确性设计

### 3.1 状态和动作身份

`execution_semantics` 保留：

- game、build、phase、turn、pending 类型和来源；
- mode、local player、ready、warnings、options/playback 标志；
- board/hand/shop/choices/controls/branches 的完整实体身份与顺序；
- interaction、目标英雄、target preview、全部可执行 Binding。

不进入点击门槛的只有对手摘要、全局统计、牌表估算等模型上下文。它们仍完整保存在
`decisions-full.jsonl.gz`，用于复盘和训练分析。

这避免了两类错误：

- 把无关的对手 UI 更新当作点击失效（上一局曾因此取消一次合法点击）；
- 为了速度而忽略真正的实体位移、授权变化或目标变化（仍会 `Blocked`）。

### 3.2 动作语义

- 训练/评测固定 `forbid_cancel=1`；实机只在取消为唯一可用动作时保留 Cancel，避免
  未训练动作位被 argmax 选中。
- Move 段按训练规则在采样前清掩码；只有收到 `applied` 才计入回合动作，取消的
  pre-press 不消耗配额。
- `bind_actions` 只从真实 server Options/EntityChoices 产生动作，不估算合法动作。

### 3.3 复杂交互

- 发现/饰品/任务/英雄技能/目标/分支各自使用标定 HUD 与实体槽。
- 选择行点击前要求同一实体顺序、两个新鲜帧、横梁支持和悬停后证明。
- 目标/分支必须来自 pinned server option/sub-options；候选实体集合变化即暂停。
- 拖拽执行前再次校验起点/终点窗口归属；终点释放前再次验证。

## 4. 每局数据体系

每次 `run`/`autoplay` 新目录包含：

| 文件/目录 | 内容 |
|---|---|
| `session.json` | run token、模型 SHA、日志会话、事件起点 |
| `decisions.jsonl` | 一决策一行的紧凑索引：序号、turn/phase、动作、意图、合法动作、推理 ms、摘要 |
| `decisions-full.jsonl.gz` | 每次决策的完整真实客户端快照、模型视图、approximation、合法动作和决策 |
| `actions.jsonl` | 物理计划、点击/拖拽坐标、原生输入轨迹、执行阶段、前后真实状态摘要、因果结果 |
| `summary.json` | 决策/执行/成功/失败计数和逐决策索引 |
| `performance.json` | 推理、输入、成功率和每秒有效动作等可量化指标 |
| `decision-report.md` | 人类可读的逐步决策表与数据边界 |
| `trace-status.json` | 异步写入队列/丢帧/错误统计 |
| `latency.json` | 投影、视觉证据、派发前检查分项耗时 |
| `wait-summary.json` | 等待类别累计秒数 |
| `visual-evidence/` | 选择行、派发前、未标定布局等有限截图证据 |
| `session-archives/` | 本局 Power.log/LoadingScreen/GameNetLogger 字节前缀与 SHA |
| `coverage.json` / `approximation.json` | 未知字段与显式近似项 |
| `agent-events.jsonl` | 原有完整事件流（兼容旧审计） |

模型决策、动作执行、状态、性能通过 `decision_id` 关联；每个 decision 行有
`sequence` 和 `decision_id`，子目标/分支决策用 `parent_decision_id`。

写盘是单线程有界队列；控制线程只做入队。队列满时记录 `events_dropped` 或
`frames_dropped`，不会阻塞点击，也不会静默丢数据。

查看方式：

```text
artifacts/runs/<run>/decision-report.md
artifacts/runs/<run>/performance.json
artifacts/runs/<run>/summary.json
artifacts/runs/<run>/decisions.jsonl
artifacts/runs/<run>/actions.jsonl
```

离线重建：

```bash
python tools/audit_live_run.py --run artifacts/runs/<run> --out artifacts/local/<run>-audit.json
```

## 5. 模拟器校验可行性

### 能严格比较的

- 动作身份：真实 server option/choice/来源/目标/slot 与模拟器动作索引；
- 已授权动作的前置实体身份和阵营；
- Buy/Sell/Play 的源区域离开；
- turnover/tier/gold/health/armor 等客户端公开标量；
- 战团 card-id 多重集与顺序（若模拟器快照可精确恢复）。

### 能部分比较的

- 随从攻击/生命、关键词、附着数：客户端缺失的私有字段必须保持 unknown；
- 商店卡与价格：客户端显示价格不等于引擎全局价格。

### 不能直接比较的

- 共享池随机抽牌、隐藏手牌、未来商店；
- 客户端未公开的临时效果、内部触发队列、随机目标；
- 战斗随机性/动画中间态、异步触发顺序（无服务器内部日志时）。

`hsbg/eval/live_simulator_check.py` 已实现离线比较框架：

- 输入是 trace 的 `actions.jsonl` 与 `decisions-full.jsonl.gz`；
- 先检查 live 自身不变式（buy/sell/play 源是否离开）；
- 只有显式重建了 simulator before/after 才对标量/战团多重集做严格比较；
- 随机/隐藏效果归类为 `unverifiable`，不当作模拟器 Bug；
- 输出 `checks`、`differences`、`strict_equivalent` 和字段级差异报告。

当前**未**把真实公开快照自动转换成完整模拟器 `Game` 状态；这是有意保守：缺少的池、
隐藏手牌和触发队列不能猜测。下一步应选取可精确注入的子集（回合末公开战团）做回归库，
而不是伪造完整状态。

## 6. 测试与准备状态

- `tests/test_live_execution_trace.py`：执行语义、5/6 选一几何、训练动作规则、异步 trace。
- `tests/test_live_simulator_check.py`：live 不变式、随机差异分类、离线报告。
- `tests/test_live_resource_migration.py`：V1.1 pinned SHA、build gate、cancel/move 规则。
- 离线 rehearsal：55/55 历史快照完成模型推理 + 几何/Controller dry-run，未创建原生后端。
- 未启动真实游戏、未发送输入、未训练、未改模型权重。

已知边界：`verify_workspace` 的整体 Python 套件在当前离线环境仍有与本任务无关的
Rust 扩展指纹/训练旧测试失败；本次实机链路相关测试单独全绿。正式 NUS 验证以计算节点
完整套件为准。

## 7. D 阶段工程改造与实机验证（2026-11-10）

状态：**当前**。目标：把"一局失败就整批停工、未知投递空等 120 s、名次竞态、赛后续局"
四类问题修掉；LLM 不进输入热路径，监督与分类全部确定性。

| 变更 | 文件 | 作用 | 验证 |
|---|---|---|---|
| 事件分类 + 失败预算 + 心跳 | `client/hsrl_agent/live/supervisor.py`（新增） | 把每局结束归为 ok/S0/S1/S3/S2；同一签名重复即停批；5 s 心跳 `health.json` | `tests/test_live_supervisor.py`；实机 `batch-stop.json` 记录 S3 |
| 批处理不再一票否决 | `practical.py:_autoplay_locked` | 每局独立目录、逐局分类、写 `matches.json`/`incidents.jsonl`；只在 S0 或预算耗尽时停批；`Blocked` 也被分类 | 实机：match-0001 ok → 自动进入 match-0002 |
| 未提交对账 | `controller.py:unsent_reconciliation` | `client_submission_position` 未推进 ⇒ 客户端从未提交，2.5 s 后归档为 unsent（不再等 120 s） | 实机 canary-3 触发 2 次 `expired_unsent_input`（proof=client_never_submitted） |
| 已提交但未被服务端采纳 | `controller.py:rejected_reconciliation` | 已提交但无 root、且**同一动作仍被授权** ⇒ 6 s 后归档为 not applied | 单测 3 例；实机待 E 阶段确认 |
| 陈旧账本 | `controller.py:retire_stale_inflight` | 从上一局恢复的 `inflight.json` 若属于更早回合，开局即归档，不再毒化下一次 attach | 单测/实机待确认 |
| 重复手势闸门 | `controller.py:_gesture_allowed` | 只有"被回执确认应用过"的同一手势在 3 s 内才被抑制；被吞掉的首次点击永远可重试（无饥饿） | 单测 4 例 |
| 名次竞态 | `runtime.py:settle_placement` | 终局等待名次批稳定（两次相同 + 0.15 s），写 `placement_source` | 实机 2 局：result.json 名次与游戏内结果屏一致（第二名/第三名） |
| 赛后自动续局 | `lobby.py:result_banner` + `startup.reverify_control` | 识别"第 X 名！"结算横幅并以横幅为锚点消歧后点击一次 | 实机 2 次成功续局（banner → 开始） |
| 大厅未知覆盖页 | `startup.wait_for_match` | 20 s 无可标定控件 → 记录证据 + 立即 S3 失败；**不再**按 Escape（实测 Escape 会打开游戏菜单，二次按下打开了商店） | 实机探针 `artifacts/local/d-phase/overlay-probe/` |
| 等待分类 | `execution_status.py` | 新增 not_decision_phase / control_settle / end_turn_control / choice_highlight / duplicate_gesture / stale_decision | 实机 `wait-summary.json` |
| 归档去重 | `tools/live_archive_index.py`（新增） | 按 source+offset+sha256 合并同一份日志的多次拷贝 | `artifacts/local/archive-index/runs.json`：37 段 / 45 份拷贝 / 8 份冗余 |

实机验证记录（同一客户端会话 `Hearthstone_2026_11_10_13_24_18`）：

| 尝试 | 目录 | 结果 |
|---|---|---|
| canary-1 | `artifacts/runs/live-v1.1-d-phase-001` | 完整一局：LOST / 第 2 名 / 12 回合；`placement_source=stable`；0 次回执超时 |
| canary-2 | `live-v1.1-d-phase-002` | 结算横幅识别成功但复验锚点写错（只认"点击继续"）→ S3；已修 |
| canary-3 | `live-v1.1-d-phase-003` | 新对账生效（2 次 unsent 归档）；第 9 回合一次 **play 已提交但始终没有 root**（动作仍被授权，疑似服务端未采纳）→ 120 s 后 halt，暴露 `rejected_reconciliation` 需求 |
| canary-4 | `live-v1.1-d-phase-004` | 附着同一局到结束：LOST / 第 3 名 / `placement_source=stable`；未决输入归档 |
| canary-5 | `live-v1.1-d-phase-005` | 陈旧账本 + 新闸门 + 赛后续局全路径；本人把 `--seconds` 设成 900，第 9 回合到点退出（配置失误，不是缺陷） |
| canary-6 | `live-v1.1-d-phase-006` | 附着打完同一局：**WON / 第 1 名 / 15 回合**，`placement_source=stable`，结果屏"第一名！！"确认 |

已知遗留：`other` 等待桶仍按"上一条状态"归因，不代表真实阻塞时长；`rejected_reconciliation`
与陈旧账本清理尚缺实机复现；Power.log 附魔聚合（离线阵容攻血）未做。

