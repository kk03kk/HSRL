# 文档索引

优先从当前流程和部署入口开始。历史文档为兼容旧实验与打包脚本保留，不代表当前推荐配置。

## 当前使用

| 文档 | 用途 |
|---|---|
| [完整训练流程](TRAINING_PIPELINE_CURRENT.md) | 模型、数据分割、BC、库PPO、验收与限制 |
| [NUS部署](NUS_ENTITY_LIBRARY_START.md) | entity-library-v2上传、CPU准备、A100资格训练和结果查看 |
| [操作预算](TRAINING_ACTION_BUDGET.md) | 200次训练工程预算与模型输入 |
| [补丁校验](PATCH_37_VERIFICATION.md) | 36.6.3基准上的两次公告修复范围 |
| [战团采集器](WARBAND_COLLECTOR.md) | 按对局/回合手动录入战团 |
| [对手记录采集](OPPONENT_CAPTURE.md) | 从现有日志提取对手资料 |
| [离线工具](LOCAL_OFFLINE.md) | 回放、诊断和离线动作适配 |
| [离线输入契约](OFFLINE_INPUT_CONTRACT.md) | 玩家观测、动作意图与执行回执接口 |

## 模拟器与上游基准

- [原生架构](ARCHITECTURE.md)
- [TavernGym接口](TAVERNGYM.md)
- [内容覆盖](CONTENT.md)
- [时空扭曲机制](TIMEWARP.md)
- [TavernBench](TAVERNBENCH.md)：上游数据和结果，不能当作HSRL模型成绩。

## 历史阶段

以下入口只用于复查对应版本，请勿代替当前部署文档：

- [最初PPO部署](NUS_PPO_START.md)、[repair2](NUS_REPAIR2_START.md)、[repair3](NUS_REPAIR3_START.md)
- [论文启发阶段](NUS_PAPER_STAGE_START.md)、[小模型试训](NUS_PAPER_PILOT_START.md)
- [首模型诊断](NUS_FIRST_MODEL_DIAGNOSIS.md)、[恢复训练](NUS_PILOT_RECOVERY.md)
- [阶段执行计划](EXECUTION_PLAN.md)

个人审计草稿、机器清单、客户端提取文件、运行日志与checkpoint留在本地，不随Git仓库发布。
