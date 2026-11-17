# HSAI 文档索引

状态：**当前生效**（2026-11-17 更新）。当前入口以本索引为准；专题分析均带日期与适用范围。
历史文档在 `records/archive/<topic>-<YYYYMMDD>/`，**不是启动命令**。

## 新人从这 5 份读起（顺序照抄）

| 顺序 | 文档 | 内容 | 读完你能回答 |
|---:|---|---|---|
| 1 | [`CHARTER.md`](CHARTER.md) | **项目目标 / 研发模式 / 一轮迭代怎么走** | 我们在做什么？谁做决定？一轮怎么推进？ |
| 2 | [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) | **训练配方（冻结版）**：每个参数的值、出处、为什么；强制升本；对手场目标表 | 训练怎么跑？为什么是这个数？ |
| 3 | [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) | **评测标准与判胜规则**：M1/M2 + 6 本回合数 + 名次面板 + 同协议 | 怎么证明"变强了"？ |
| 4 | [`PROJECT_STATE.md`](PROJECT_STATE.md) | **当前状态**：模型谱系与成绩、正在跑什么、下一步 | 现在到哪一步了？ |
| 5 | [`../AGENTS.md`](../AGENTS.md) | **操作权威约定**：放哪、红线、自主权边界、技术口径 | 怎么动手？什么不能做？ |

## 运作与架构

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 单 Agent 双环境：职责、三层结构 |
| [ENGINEERING.md](ENGINEERING.md) | 模块边界、冻结发布、模型登记、完整动作证据、环境与验收 |
| [OPERATIONS.md](OPERATIONS.md) | 日常运作：开工开场、提交作业、**§10 一轮迭代操作清单** |
| [architecture/TRANSPORT.md](architecture/TRANSPORT.md) | SSH 隧道、git over SSH、GitHub 定位、大文件传输 |

## 查证与专题

| 文档 | 内容 |
|---|---|
| [DECISIONS.md](DECISIONS.md) | 决策留痕（`D-###`，**只追加不改历史**） |
| [analysis/MODEL_CAPABILITY_RESEARCH_20261117.md](analysis/MODEL_CAPABILITY_RESEARCH_20261117.md) | 当前模型能力研究：全量动作、训练账本、AutoTavern差异与下一轮建议（未实施） |
| [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) | **哪些方向已否证、不要重跑** + 踩过的技术坑 |
| [DATA_BANKS.md](DATA_BANKS.md) | 真人数据资产清单（战团库、入场库） |
| [nus/ENVIRONMENT.md](nus/ENVIRONMENT.md) | NUS 环境限制、GPU 容量、QOS 上限 |
| [nus/ENGINE_RELEASES.md](nus/ENGINE_RELEASES.md) | 引擎发版记录（含源码指纹） |
| [nus/INCIDENTS.md](nus/INCIDENTS.md) | 历史故障与防复发规则 |
| [COMP_TARGET.md](COMP_TARGET.md) | 阵容覆盖奖励的设计（**已判负**，保留实现说明） |
| [EXPERT_QA.md](EXPERT_QA.md) | 专家问答清单（部分未答） |
| [battlegrounds_strategy_guide.md](battlegrounds_strategy_guide.md) | 用户持续维护的唯一现役HSReplay攻略；旧 `config/strategy_cards.json` 有未解析卡，不能替代正文 |
| [analysis/TRACE_UNIFIED_20261111.md](analysis/TRACE_UNIFIED_20261111.md) | 统一对局记录规范 `hsai.game-trace/v2` |

## 事实的唯一来源（有冲突时以这一列为准）

| 事实 | 唯一来源 |
|---|---|
| 项目目标 / 研发模式 / 一轮怎么走 | [`CHARTER.md`](CHARTER.md) |
| 训练配方（超参、强制升本、对手场、健康红线） | [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) |
| 评测指标定义、样本量、判胜规则、协议对齐 | [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) |
| 当前最强模型、正在跑什么、下一步 | [`PROJECT_STATE.md`](PROJECT_STATE.md) |
| 操作红线、自主权边界、放哪、传输 | [`../AGENTS.md`](../AGENTS.md) |
| 决策为什么这么定 | [`DECISIONS.md`](DECISIONS.md)（只追加） |
| 哪些方向已否证 | [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) |
| 引擎版本 / 指纹 / 卡表 | [`nus/ENGINE_RELEASES.md`](nus/ENGINE_RELEASES.md) |
| 登录节点规格 / GPU 容量 / QOS 上限 | [`nus/ENVIRONMENT.md`](nus/ENVIRONMENT.md) |
| 服务器实时状态（作业/GPU/HEAD） | 不写文档，用 `python ops/hsai_ops.py state` |

**已退休**（不要再当现状引用，原文在 `records/archive/`）：
`LEADERBOARD.md`（E1 bot 场 Elo）、`EVAL_DUEL.md`、`nus/EVAL_DEPLOYMENT.md`、`nus/GPU_SELECTION.md`、
`LIVE_V1_1_AUDIT.md`、`E_PHASE_RUNBOOK.md`、`SIMULATOR_AUDIT_2026-11-09.md`、`AUDIT_2026-11-11.md`、
`analysis/REPORT_20261114_ARENA.md`（闭环难度控制器探索线，已被校准快照场取代）。

## 文档规则

- `docs/` 根目录只放**当前有效**的规范与入口；`docs/analysis/` 放**日期化的一次性分析**（历史证据）、
  `docs/nus/` 放服务器侧事实、`docs/architecture/` 放架构。
- 被取代的文档：整批移进 `records/archive/<topic>-<YYYYMMDD>/`（**不删除**），并在下方"已退休"里登记。
- **改文档后必须自查**：索引里每条链接都要能打开（2026-11-15 曾出现 8 个指向已归档文件的死链）。
- 同一事实只允许有一个"唯一来源"，其他文档只能引用。
- 服务器侧实时状态不写进文档，用 `python ops/hsai_ops.py state` 取。
