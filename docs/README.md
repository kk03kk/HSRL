# HSAI 文档索引

本目录只保留当前阶段有用的文档。历史实验、旧部署、旧排行榜和已完成的复盘统一归档到：

```
records/archive/docs-2026-11-09/
```

归档清单：[MANIFEST.md](../records/archive/docs-2026-11-09/MANIFEST.md)。

根目录只保留：

- `README.md`：项目介绍
- `AGENTS.md`：代理工作方式与硬约束

## 当前必读

| 文档 | 内容 |
|---|---|
| [PROJECT_STATE.md](PROJECT_STATE.md) | 当前作业、模型结论、下一步 |
| [SIMULATOR_AUDIT_2026-11-09.md](SIMULATOR_AUDIT_2026-11-09.md) | 37.0.2 正式服/模拟器规则审计、覆盖、修复与测试 |
| [ELO_EVAL.md](ELO_EVAL.md) | 8 模型 / 2 model + 6 bot pairwise Elo 方案 |
| [EVAL_DUEL.md](EVAL_DUEL.md) | 2 model + 6 bot 对局协议、记录口径、门禁 |
| [nus/ENVIRONMENT.md](nus/ENVIRONMENT.md) | NUS 环境限制、Slurm、GPU、恢复 |
| [nus/INCIDENTS.md](nus/INCIDENTS.md) | 历史故障与防复发规则 |
| [nus/EVAL_DEPLOYMENT.md](nus/EVAL_DEPLOYMENT.md) | NUS 评测提交与监控流程 |

## 文档规则

- `docs/` 不存放一次性的运行日志、旧排行榜、旧部署命令或已完成的复盘。
- 运行产物放 `artifacts/`；原始证据放 `records/`。
- 新实验使用新目录；旧协议不覆盖、不混榜。
- 归档文档只作为证据，不能当作当前启动命令。
