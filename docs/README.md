# HSAI 文档索引

本目录只保留**当前有效**的文档。历史实验、旧部署、旧排行榜与已完成的复盘统一归档到
`records/archive/<topic>-<YYYYMMDD>/`（见 `docs/SYNC_MODEL.md` §4）。归档文档不是启动命令。

## 当前必读

| 文档 | 内容 |
|---|---|
| [PROJECT_STATE.md](PROJECT_STATE.md) | 当前作业、模型结论、下一步 |
| [SYNC_MODEL.md](SYNC_MODEL.md) | **三端（本地/GitHub/NUS）文件管理、真源、同步与协作规范** |
| [INVENTORY_LOCAL.md](INVENTORY_LOCAL.md) | 本地盘点：有什么、缺什么、哪些待服务器确认 |
| [sync/README.md](sync/README.md) | 两个 Agent 的异步协作区与状态板 |
| [sync/SERVER.md](sync/SERVER.md) | 交给服务器 Agent 的盘点任务书（待对方填写） |
| [sync/DECISIONS.md](sync/DECISIONS.md) | 三端管理的决议记录（只追加） |
| [LIVE_V1_1_AUDIT.md](LIVE_V1_1_AUDIT.md) | V1.1 实机审计、执行修复、每局数据体系、D 阶段改造与实机验证 |
| [E_PHASE_RUNBOOK.md](E_PHASE_RUNBOOK.md) | E 阶段连续实机验收手册（计划；E-0 canary → E-1 三局） |
| [SIMULATOR_AUDIT_2026-11-09.md](SIMULATOR_AUDIT_2026-11-09.md) | 37.0.2 规则审计、覆盖、修复与测试 |
| [ELO_EVAL.md](ELO_EVAL.md) | 8 模型 / 2 model + 6 bot pairwise Elo 方案 |
| [EVAL_DUEL.md](EVAL_DUEL.md) | 2 model + 6 bot 对局协议、记录口径、门禁 |
| [nus/ENVIRONMENT.md](nus/ENVIRONMENT.md) | NUS 环境限制、Slurm、GPU、恢复 |
| [nus/INCIDENTS.md](nus/INCIDENTS.md) | 历史故障与防复发规则 |
| [nus/EVAL_DEPLOYMENT.md](nus/EVAL_DEPLOYMENT.md) | NUS 评测提交与监控流程 |

## 机器清单

| 文件 | 生成方式 |
|---|---|
| [sync/manifests/local.json](sync/manifests/local.json) | `python tools/sync_manifest.py --side local --out docs/sync/manifests/local.json` |
| `sync/manifests/server.json` | 服务器侧同名命令（**待生成**） |

## 文档规则

- `docs/` 不存放一次性运行日志、旧排行榜、旧部署命令或已完成复盘。
- 运行产物放 `artifacts/`；原始证据放 `records/`；大文件不进 Git。
- 每份文档标注状态（当前 / 计划 / 归档）、协议与入口；新实验用新目录，旧协议不覆盖、不混榜。
- 跨端问题写进 `docs/sync/`，不要写进贡献者文档。
