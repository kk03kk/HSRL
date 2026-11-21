# HSAI 当前状态

## 正式模型与执行入口

**2026-11-21 · D-198。** 当前登记模型为 `player150-round05-3286`。训练暂停。Windows 控制台默认选择 3286，3236 为手动回退。D-199 实机试验已停，3286 的正式实机验收待确认。

| 项目 | 入口 |
|---|---|
| Windows 默认入口 | `client/start-ai-control.cmd` 委托给 `artifacts/local/r1-player150-live/`。离线使用 `client/start.ps1`。生效 pin 在独立工作树的 `config/live/model.json`。[部署交接](analysis/PLAYER150_PROMOTION_20261121.md) |
| Player150 模型与恢复 | Windows 本地登记目录为 `artifacts/models/player150-round05-3286/`。NUS 权威模型与 Trainer 位于五轮训练第 5 轮的 `snapshots/final_003286/`。[训练记录](analysis/PLAYER150_FIVE_20261120.md) |
| Player150 规则 | 37.0.3，独立构建指纹 `8bd1f4785018c474`。4,967 浮点数、124 卡槽、245 原生动作。模型、引擎锁、观测和正式协议摘要见[晋级记录](analysis/PLAYER150_PROMOTION_20261121.md)。 |
| 主检出旧基线 | [引擎源码锁](../config/engine-upstream-lock.json)对应 37.0.2.256589 和 `c4c71014e3d41760`。[旧 pin](../config/live/model.json) 指向 14858。 |
| 实时作业 | `python ops/hsai_ops.py state` 或 `python ops/hsai_ops.py jobs`。 |

## 最近研究结论

| 决策 | 结论与证据 |
|---|---|
| D-202 海盗首次机会接管 | 复现 576 条旧轨迹。两场共 320 个开局，两回合接管的盲验终局效用差 −0.243［−0.375，−0.107］。保留 3286 和冻结教师，未进入蒸馏。[实验记录](analysis/PIRATE_TAKEOVER_20261121.md) |
| D-201 0750 配对与机制验证 | 960 局独立配对，在两场复核发现海盗、磁力机械和圣盾龙，另精确复放 24 局。0750 保留为冻结专项参照。[协议与结果](analysis/ASD_POLICY0750_PAIRED_20261121.md) |
| D-200 3286 对原版 R1 | 两场各 192 对，名次差区间跨零。同一成熟野猪人起点 T13 P50 提高 43.2%，自然成型稳定提升待确认。[策略评测](analysis/PLAYER150_STRATEGY_EVAL_20261121.md) |
| D-199 实机速度与动作回执 | 第二局到 T8 按用户要求停机。动作位置、客户端提交与服务器回执三方匹配 87/90，三笔保留为外部结算。控制台仍用 fast。[实机研判](analysis/PLAYER150_LIVE_SPEED_20261121.md) |
| D-198 3286 晋级 | 五轮累计 294,360,994 次训练决策。固定场平均名次 3.938→3.508，独立真人战团场 3.406→3.164。通过 237 个历史决策点的离线回放。[正式结果](analysis/PLAYER150_FINAL_REVIEW_20261121.md) |

早期研究从[决策台账](DECISIONS.md)和[专题索引](README.md)查阅。常用复现入口包括
[37.0.3 群友路线](analysis/PEER_3703_ROUND_20261120.md)、[3265 真人场训练](analysis/HUMAN_PPO_FINAL_20261119.md)、
[v1.4 工程与评测](analysis/MODERNIZATION_20261117.md)及 [D-081 封存研究](analysis/AUTONOMOUS_RESEARCH_20261118.md)。
整理前的时间线、作业表和历史版本锚点保留在 `records/archive/project-organization-20261121/docs/PROJECT_STATE.md`，
也可通过 `git show 1a215cf9:docs/PROJECT_STATE.md` 查询。

[架构与模块研究顺序](ARCHITECTURE.md) · [目录、证据与归档审计](analysis/PROJECT_ORGANIZATION_20261121.md)
