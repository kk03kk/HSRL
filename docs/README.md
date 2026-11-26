# HSRL 文档索引

首次阅读顺序建议为 [项目首页](../README.md)、[研究流程](WORKFLOW.md)、[项目目录](PROJECT_LAYOUT.md)、[37.0.3 研究记录](reports/37.0.3.md)。

## 执行与验收

| 文档 | 用途 |
| --- | --- |
| [研究流程与迁移手册](WORKFLOW.md) | 阶段输入、产物、晋级条件、踩坑和下一补丁迁移 |
| [项目目录](PROJECT_LAYOUT.md) | 根目录职责、源代码入口、生成产物和运行约定 |
| [NUS 操作手册](NUS.md) | 登录、冻结检出、构建、GPU 资源和故障排查 |
| [模拟器基线](SIMULATOR.md) | 37.0.3 溯源、动作预算和实机回执、完整对局门槛 |
| [策略教师](STRATEGY.md) | 33 套攻略覆盖、站位策略、种子和代表对局 |
| [酒馆战棋攻略](battlegrounds_strategy_guide.md) | 英雄、饰品、各流派核心及历史策略教学 |
| [BC 与 PPO 准入](BC.md) | 自然示范、纠错、18M 模型、独立验收和冻结初始化 |
| [首轮 PPO](PPO.md) | 配方、整轮追溯、第 208 轮概率精度故障与恢复 |
| [PPO 硬件对照](PPO_HARDWARE.md) | 整卡、并发、采样、训练吞吐和同权重核对 |
| [战团库](WARBAND.md) | 库生成、压力分层、原局隔离、牌池及虚拟席处理 |
| [九小时训练运行手册](OVERNIGHT.md) | 提交、恢复、自动决策、报告和结果检查 |
| [37.0.3 研究记录](reports/37.0.3.md) | 主要结果、限制、训练曲线及实验证据位置 |
| [PPO 288 轮独立验收详表](reports/37.0.3-ppo288.md) | 权重、配对区间、每回合战力、全流派与动作审计 |
| [仓库工作约定](../AGENTS.md) | 编辑、发布、作业、冻结资产与版本迁移约定 |

## 上游引擎与评测参考

| 文档 | 用途 |
| --- | --- |
| [中文入口](upstream/README.zh-CN.md) · [English entry](upstream/README.en.md) | 上游快速开始、TavernGym 和 TavernBench、原论文引用 |
| [TavernGym](TAVERNGYM.md) | Python／Rust 接口、动作、观测、配置和性能 |
| [引擎架构](ARCHITECTURE.md) | 游戏状态、招募与战斗、脚本调度 |
| [卡牌内容覆盖](CONTENT.md) | 可实现的随从、英雄、法术、饰品与其他内容 |
| [时间扭曲酒馆](TIMEWARP.md) | 条件入口和扩展规则 |
| [37.0.2 补丁记录](PATCH_37.0.2.md) | 历史客户端卡表变迁与覆盖 |
| [TavernBench](TAVERNBENCH.md) | 于 36.6.3 封存的上游任务、标签和评分方式 |

阶段报告记录实验当时的引擎指纹、种子与配置。跨阶段读取指标时以同一表格的对手、补丁和样本定义为准。
