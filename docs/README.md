# HSAI 文档索引

开工先读 [AGENTS.md](../AGENTS.md)，按 [OPERATIONS.md](OPERATIONS.md) 核对双端状态。
[PROJECT_STATE.md](PROJECT_STATE.md) 记录当前模型和研究交接，[ARCHITECTURE.md](ARCHITECTURE.md) 给出七个模块和阅读顺序。

| 模块 | 规范和入口 | 研究与历史证据 |
|---|---|---|
| 模拟器与卡牌规则 | [发版记录](nus/ENGINE_RELEASES.md)、[主检出源码锁](../config/engine-upstream-lock.json) | [规则校验与修复结果](analysis/SIMULATOR_PARITY_QUALIFICATION_20261121.md)、[实现与补丁验收流程](analysis/SIMULATOR_RULE_PARITY_20261121.md) |
| 观测、动作与模型 | [工程验收](ENGINEERING.md)、[现役攻略](battlegrounds_strategy_guide.md) | [观测差异与战团库](analysis/PLAYER150_OBSERVATION_REVIEW_20261121.md)、[实机观测逐帧核对](analysis/PLAYER150_LIVE_OBSERVATION_20261121.md)、[3286 观测与动作初审](analysis/PLAYER150_OBS_ACTION_AUDIT_20261121.md)、[Player150 五轮配方](analysis/PLAYER150_FIVE_20261120.md)、[0750 兼容性](analysis/ASD_POLICY0750_COMPAT_20261121.md) |
| 回放、对手与课程数据 | [数据资产](DATA_BANKS.md) | [真人回合面板](analysis/HUMAN_TURN_BASELINE_20261118.md)、[对手实际覆盖](analysis/HUMAN_FIELD_BY_TURN_20261119.md) |
| 策略训练与恢复 | [训练配方](TRAINING_RECIPE.md)、[冻结发布与恢复](ENGINEERING.md) | [历史负结果](NEGATIVE_RESULTS.md)、[持续学习研究](analysis/CONTINUAL_RL_ROADMAP_20261121.md) |
| 评测与研究证据 | [评测协议](EVAL_PROTOCOL.md)、[对局记录规范](analysis/TRACE_UNIFIED_20261111.md) | [Player150 正式结果](analysis/PLAYER150_FINAL_REVIEW_20261121.md)、[0750 配对](analysis/ASD_POLICY0750_PAIRED_20261121.md)、[海盗接管盲验](analysis/PIRATE_TAKEOVER_20261121.md) |
| Windows 实机桥接 | [3286 部署与离线入口](analysis/PLAYER150_PROMOTION_20261121.md) | [实机速度与回执](analysis/PLAYER150_LIVE_SPEED_20261121.md) |
| 发布、运维与资产 | [运维手册](OPERATIONS.md)、[传输说明](architecture/TRANSPORT.md)、[NUS 环境](nus/ENVIRONMENT.md) | [故障记录](nus/INCIDENTS.md)、[项目整理审计](analysis/PROJECT_ORGANIZATION_20261121.md) |

## 状态、决策与归档

- [项目宪章](CHARTER.md)说明项目目标与研发流程。[决策台账](DECISIONS.md)保存只追加的 D-编号记录。
- 复现使用对应的冻结源码、已执行配方、运行记录及权重和库的摘要。实时 HEAD、队列与 GPU 状态用 `python ops/hsai_ops.py state` 查看。
- `docs/analysis/` 保存带日期的研究报告。报告内 `artifacts/`、`records/` 路径指明 Windows 或 NUS 的取证位置。
- `records/archive/project-organization-20261121/docs/` 保存整理前架构和状态页、已判负的 `COMP_TARGET.md`、早期专家先验清单 `EXPERT_QA.md`。干净克隆可用 `git show 1a215cf9:docs/<文件名>` 查询原文。
- 早先已归档的 `LEADERBOARD.md`、`EVAL_DUEL.md`、`LIVE_V1_1_AUDIT.md`、`E_PHASE_RUNBOOK.md`、`analysis/REPORT_20261114_ARENA.md` 等继续保存在 `records/archive/`。
- 文档修改后执行 `python ops/check_docs.py`。
