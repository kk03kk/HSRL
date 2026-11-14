# HSAI 文档索引

状态：**当前生效**（2026-11-14 重排）。本目录只放**当前有效**的文档。
历史文档在 `records/archive/<topic>-<YYYYMMDD>/`，**不是启动命令**。

## 新人从这 5 份读起（顺序照抄）

| 顺序 | 文档 | 内容 | 读完你能回答 |
|---:|---|---|---|
| 1 | [`CHARTER.md`](CHARTER.md) | **项目目标 / 研发模式 / 一轮迭代怎么走** | 我们在做什么？谁做决定？一轮怎么推进？ |
| 2 | [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) | **训练配方（冻结版）**：每个参数的值、出处、为什么 | 训练怎么跑？为什么是这个数？ |
| 3 | [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md) | **评测标准与判胜规则**：M1/M2 + 名次面板 + 比协议 | 怎么证明"变强了"？ |
| 4 | [`PROJECT_STATE.md`](PROJECT_STATE.md) | **当前状态**：冠军是谁、正在跑什么、下一步 | 现在到哪一步了？ |
| 5 | [`../AGENTS.md`](../AGENTS.md) | **操作权威约定**：放哪、红线、自主权边界、技术口径 | 怎么动手？什么不能做？ |

## 运作与架构

| 文档 | 内容 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 单 Agent 双环境：职责、三层结构、与旧架构的对应 |
| [OPERATIONS.md](OPERATIONS.md) | 日常运作：开工固定开场、改代码、训练、拉模型、实机、应急 |
| [architecture/TRANSPORT.md](architecture/TRANSPORT.md) | SSH 隧道、git over SSH、GitHub 定位、大文件传输（含实测吞吐） |

## 专题

| 文档 | 内容 |
|---|---|
| [DECISIONS.md](DECISIONS.md) | 决策留痕（`D-###`，只追加不改历史） |
| [LEADERBOARD.md](LEADERBOARD.md) | 当前排行榜规则与分数 |
| [EVAL_DUEL.md](EVAL_DUEL.md) | 2 model + 6 bot 对局协议、口径、门禁 |
| [nus/ENVIRONMENT.md](nus/ENVIRONMENT.md) | NUS 环境限制、Slurm、GPU、恢复 |
| [nus/GPU_SELECTION.md](nus/GPU_SELECTION.md) | **GPU 选型实测基准与调度约束**（含 MIG 切片陷阱） |
| [nus/INCIDENTS.md](nus/INCIDENTS.md) | 历史故障与防复发规则 |
| [nus/EVAL_DEPLOYMENT.md](nus/EVAL_DEPLOYMENT.md) | NUS 评测提交与监控 |
| [nus/ENGINE_RELEASES.md](nus/ENGINE_RELEASES.md) | 引擎发版记录 |
| [analysis/](analysis/) | 代码级审计与逐日实验记录（**历史证据，不是启动命令**） |

## 实机专题

| 文档 | 内容 |
|---|---|
| [LIVE_V1_1_AUDIT.md](LIVE_V1_1_AUDIT.md) | V1.1 实机审计、执行修复、每局数据体系 |
| [E_PHASE_RUNBOOK.md](E_PHASE_RUNBOOK.md) | E 阶段连续实机验收手册 |

## 历史审计（结论仍被引用，但不是启动命令）

| 文档 | 内容 |
|---|---|
| [SIMULATOR_AUDIT_2026-11-09.md](SIMULATOR_AUDIT_2026-11-09.md) | 37.0.2 规则审计、覆盖、修复与测试 |
| [AUDIT_2026-11-11.md](AUDIT_2026-11-11.md) | 36h 复盘 + 全链路审计、E1 校准与长训准入 |
| [battlegrounds_strategy_guide.md](battlegrounds_strategy_guide.md) | 酒馆战棋策略参考 |

## 已退休

`docs/sync/` 与 `docs/SYNC_MODEL.md` 已归档到本机的
`records/archive/sync-protocol-20261112/`（该目录被 `.gitignore` 忽略，只在本地磁盘，
另一台机器上没有；退休原因与恢复方法见那里的 `README.md`）。
**不要再向已退休的 `docs/sync/` 写东西。**

## 文档规则

- `docs/` 不存放一次性运行日志、旧排行榜、旧部署命令或已完成复盘。
- 运行产物放 `artifacts/`；原始证据放 `records/`；大文件不进 Git。
- 每份文档标注状态（当前 / 计划 / 归档）、协议与入口；新实验用新目录，旧协议不覆盖、不混榜。
- 一次性分析写 `docs/analysis/<TOPIC>_<YYYYMMDD>.md`。
- 服务器侧实时状态不写进文档，用 `python ops/hsai_ops.py state` 取。
