# HSAI 双端项目整理审计

2026-11-21 · D-203。生产与研究入口按七个模块归入[架构图](../ARCHITECTURE.md)。本报告记录核查边界、版本身份和归档位置。

## 范围与发现

- 开工时 Windows 与 NUS 主检出均为 `codex/autonomous-research-20261118` 的 `1a215cf9`，两端干净，状态为 `IN SYNC`，NUS 队列为空。跨端 Doctor 7 项通过。Git 清点 1,412 个跟踪文件，覆盖引擎、Python、客户端、训练和评测入口、配置、测试、资料及研究摘要。结合各入口实际引用检查关键路径，历史结果按已有封存抽查。
- Windows 同一仓库有主检出和独立研究、部署工作树。NUS 登记 287 个工作树条目，包含大量冻结复现目录。冻结的配方、引擎与环境应从对应发布读取。已发现独立 Windows 实机工作树有未跟踪的其他研究文档，按原样保留。
- 主检出仍承载旧规则基线。Windows CPU 环境实测 Python 3.13.5、Torch 2.13.0+cpu，基线引擎返回 3,227 浮点、124 卡槽与 245 原生动作。Windows 默认部署与 NUS 五轮训练各用 Player150 的独立构建。项目状态和文档首页此前同时出现多个不同年代的“当前”结论，已按工作树及适用协议标明。根目录依赖元数据与 v1.4 配方也有历史背景，复现按实际运行记录和版本化依赖锁。
- Windows 初始文档检查有 17 处可点链接指向机器私有产物，造成唯一一项基线测试失败。四份研究报告保留了原文件名、证据位置与数据结论，私有链接改成仓库相对路径文本。
- 主检出有 226 个工作流文件、144 个 Slurm 目录文件，正式、兼容与一次性实验入口并存。独立发布引用原路径，因此本轮用配置、决策和报告定位各入口。独立实机工作树也保留在已有路径。
- `state` 包含 NUS 全项目容量统计，明显慢于 `probe` 和 Git、队列单独查询。开工采样的 NUS 项目表观规模为 431G，账号所在共享文件系统使用率为 96%。本轮保留原状态实现。整库异地备份、历史环境从零复建及 3286 正式实机验收均待确认。

首次工作区检查用了工具默认输出，覆盖了本地生成报告 `artifacts/local/validation.json`。旧内容待确认。最终检查使用单独报告路径，正式评测目录和原始对局记录均未写入。

## 版本锚点

| 锚点 | 主检出基线 | Player150 现役部署与封存 |
|---|---|---|
| 引擎锁与运行指纹 | [主锁](../../config/engine-upstream-lock.json)，37.0.2.256589 / `c4c71014e3d41760` | NUS 冻结发布 `player150-five-02`，提交 `cb9587ac0cfa4b1875415277652392043a43e082`。37.0.3 / `8bd1f4785018c474`，锁文件 SHA-256 `03031a660c1461696b5007bf528257ab9054baba3b5bc1c709b7ec3f3739b4d5` |
| 实机 pin | [主检出旧 pin](../../config/live/model.json) 指向 14858，摘要 `ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7` | Windows 控制台读取 `artifacts/local/r1-player150-live/config/live/model.json`。3286 默认、3236 手动回退。3286 摘要 `778cc6f4ca9ba6175ec6f54c2241a030a70ce4e35b7ac7120492666360ba86df`，本地登记与部署两份权重均复算一致 |
| 模型 manifest 与恢复 | 旧模型和 v1.4 恢复索引见[工程手册](../ENGINEERING.md) | Windows `artifacts/models/player150-round05-3286/manifest.json`，观测摘要 `ad4c812c2446b1c768064ea0cf5ed8eeb4cad0625744ec4c68e2dbb27cbeb485`。NUS 第五轮 `train/fit/train/snapshots/final_003286/` 存在 `model.pt`、`trainer.pt`、`record.json` |
| 正式评测协议 | v1.4 与更早 E1 见[评测协议](../EVAL_PROTOCOL.md)和[历史评测](MODERNIZATION_20261117.md) | 固定场 `5361375f96441316237a792205b5ff76587bff27a9fd80d7cad25307075927d9`；独立真人战团场 `24d387ec449479425bd356b61c639b35f6095b4a32cab700519a0a62a04581f4`。Slurm 1173164、1173165、1173166 均为 COMPLETED / 0:0 |

Player150 Doctor 的引擎元数据同时保留原始卡表的 `game_version` 37.0.2.256589 和 `server_rules_version` 37.0.3。4,967 是模型观测宽度，原生基座仍为 3,227。Player150 在 NUS 使用独立环境 `artifacts/environments/player150-20261120-01`。五轮配方、封存路径及评测回执从[训练记录](PLAYER150_FIVE_20261120.md)、[晋级交接](PLAYER150_PROMOTION_20261121.md)查阅。0750 共享预算与各自原生预算属于另两个协议，见[配对报告](ASD_POLICY0750_PAIRED_20261121.md)。

## 归档与目录

| 内容 | 位置与处理 |
|---|---|
| 旧架构与状态、阵容覆盖奖励、专家先验 | 两端 `records/archive/project-organization-20261121/docs/` 保留四份原件。退出活跃导航的 `COMP_TARGET.md`、`EXPERT_QA.md` 可由原 Git 提交复原。 |
| Windows 早期运维快照 | `records/archive/project-organization-20261121/artifacts/state/`，85 份早于 2026-11-17 的既有状态快照。活跃状态、近期快照和日志仍在原目录。 |
| Windows 工作草稿 | 同一归档的 `artifacts/local/`，5 份旧提交说明草稿和 1 份暂存计划。归档根的 `index.json` 含 95 个文件的原路径、长度与 SHA-256。一次性 pytest 缓存从项目根移到同一归档的 `pytest-cache/`。 |
| 旧项目和历史模型 | 旧项目原档位于 `artifacts/archives/`。在 Windows 增设 `artifacts/models/historical/project-organization-20261121/index.json`，用 NTFS 硬链接登记旧 S14 best、旧客户端 BC、Stage 2 proposal 权重及其既有 checkpoint。原路径继续可用。此前 9 份历史权重见 `artifacts/models/historical/player150-promotion-20261121/index.json`，全局旧模型登记见 `artifacts/models/registry.json`。 |
| 历史研究、恢复和原始证据 | 用 [决策台账](../DECISIONS.md) 和 [文档索引](../README.md) 查报告，再定位原 `artifacts/runs/`、NUS 冻结发布、恢复目录与 `records/evidence/`。已封存权重、训练目录、战团库、恢复代和原始评测按其原路径保留。 |

本轮界定到入口和关键依赖，未逐个重放所有历史模型、实验和回放。运行代码、配置、冻结发布、工作树和游戏目录未改动，归档权重沿用原内容与路径。整理期间不启动训练、评测或实机输入。

## 验证结果

- Windows 全量 Python 回归 1,176 通过、8 跳过。源码、根目录与大文件门禁、卡表检查、引擎源码锁、依赖一致性与 84 份 Markdown 链接检查均通过。本轮未重编译 Rust 或提交计算节点测试。
- Windows 离线 Doctor 实际加载 3286，通过权重校验、客户端可执行文件核对和合成推理。回执记录 `native_input_acquired=false`，正式实机验收仍待确认。
- 95 份文档、历史快照与草稿全部复核归档摘要。3 处新登记历史模型的 4 个文件逐项核对原路径、文件身份和摘要，此前 9 份历史权重也复算摘要一致。
