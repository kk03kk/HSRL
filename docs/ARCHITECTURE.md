# HSAI 项目架构

HSAI 在 Rust 酒馆战棋模拟器中训练和评测策略，通过独立的 Windows 客户端接入真实游戏。[当前状态](PROJECT_STATE.md)记录现任模型和执行入口，[文档索引](README.md)提供按模块查阅的入口。

## 1. 七个核心模块

| 模块 | 职责与实现 | 阅读入口 |
|---|---|---|
| 1. 模拟器与卡牌规则 | Rust 实现招募、战斗、卡牌、合法动作和八席对局，PyO3 暴露单局与批量接口。卡表和源码锁固定规则版本。 | [Rust 内核](../engine/crates/hsbg-engine/)、[Python 绑定](../engine/crates/hsbg-engine-py/src/lib.rs)、[源码锁](../config/engine-upstream-lock.json) |
| 2. 观测、动作与模型 | 将状态映射为数值观测、卡牌槽和合法动作掩码，实体模型输出动作分布与价值。版本化观测、词表顺序和动作语义共同决定兼容性。 | [观测](../hsbg/env/obs_std.py)、[动作](../hsbg/env/action_space_std.py)、[实体模型](../hsbg/az/train/entity_model.py)、[Player150 输入](analysis/PLAYER150_FIVE_20261120.md) |
| 3. 回放、对手与课程数据 | 将 Firestone 真人回放及模拟对局转成回合末战团、入场状态、模仿样本和训练、留出面板。核对数据来源、分割、回合覆盖与注入质量。 | [数据资产](DATA_BANKS.md)、[Firestone 建库](../workflows/build_firestone_eot.py)、[真人战团采样](../hsbg/az/train/human_board_pool.py) |
| 4. 策略训练与恢复 | Slurm 启动采集与 PPO 更新。教师、BC、课程、奖励和对手按配方启用。模型权重用于推理或分叉，完整 Trainer 另保存恢复所需的优化器、调度器和随机状态等。 | [训练入口](../workflows/train_entity_sync.py)、[PPO](../hsbg/az/train/ppo_trainer.py)、[检查点](../hsbg/az/train/checkpoint_io.py)、[配方](TRAINING_RECIPE.md) |
| 5. 评测与研究证据 | 固定权重、种子、规则、预算和对手，执行分片面板及专项配对。报告逐回合 M1/M2、样本数、名次及共同存活成长，封存逐局与逐动作证据。 | [评测入口](../workflows/evaluate_suite.py)、[证据实现](../hsbg/evaluation/)、[协议](EVAL_PROTOCOL.md)、[Player150 正式结果](analysis/PLAYER150_FINAL_REVIEW_20261121.md) |
| 6. Windows 实机桥接 | 读取日志与界面，从公开状态推理。就绪、合法性、提交和回执门禁约束输入。Doctor、决策与回放可独立离线运行。 | [控制台](../client/start-ai-control.cmd)、[离线入口](../client/start.ps1)、[桥接实现](../client/hsrl_agent/live/)、[3286 部署](analysis/PLAYER150_PROMOTION_20261121.md) |
| 7. 发布、运维与资产 | 用冻结 Git worktree、版本化环境、Slurm 配方和内容摘要追踪运行。同步两端源码，登记模型，留存恢复资产及研究证据。 | [冻结发布](../ops/releases.py)、[跨端运维](../ops/hsai_ops.py)、[Slurm 配方](../deploy/slurm/)、[工程验收](ENGINEERING.md) |

```text
卡牌规则 → 模拟器 → 观测、动作与模型 → 采集、PPO 与检查点 → 独立评测
真人或模拟回放 → 战团、课程与模仿样本 → 训练输入／留出评测
评测与用户晋级决策 → 版本化模型 pin → Windows 公开状态与回执
发布、环境与资产清单贯穿整条链路
```

## 2. 双环境与三层结构

Windows 管理源码、CPU 回归、实机桥接和本地证据。NUS 登录节点处理 Git、调度和小型元数据，计算节点承担引擎构建、训练、建库与大规模评测。通过 SSH 同步源码与传输资产，GitHub 保存 Git 里程碑。

| 层 | 常用路径 |
|---|---|
| 源码与配置 | `engine/`、`hsbg/`、`client/`；`workflows/` 编排、`tools/` 构建与审计、`tests/` 回归、`config/` 配方与锁定、`data/` 小型规则数据 |
| 执行与运行环境 | `ops/`、`deploy/`、角色依赖锁；NUS `artifacts/releases/` 冻结作业源码，Windows `artifacts/local/r1-player150-live/` 承载已安装部署 |
| 产物与证据 | `artifacts/{models,libraries,runs,archives}/` 保存运行与恢复资产，`records/{evidence,archive}/` 保存原始证据与历史资料，`docs/analysis/` 保存研究报告 |

主检出保留 37.0.2 基线，`config/live/model.json` 指向旧 14858。默认 Windows 控制台委托到独立工作树中的 3286 pin，使用冻结的 37.0.3 构建和 Player150 观测与预算。具体标识见[状态页](PROJECT_STATE.md)和[版本锚点审计](analysis/PROJECT_ORGANIZATION_20261121.md)。操作见[运维手册](OPERATIONS.md)及[传输说明](architecture/TRANSPORT.md)。

## 3. 逐模块研究顺序

1. **模拟器与接口（1 → 2）**。先对齐引擎、卡表、动作编号和观测布局，再追踪一个决策的输入、掩码与输出。
2. **数据与评测（3 → 5）**。用现有证据复算一份成绩，核查来源切分、对手场、协议、提前淘汰及共同存活样本。
3. **训练与恢复（4 → 7）**。对照一次已封存训练，沿轨迹、奖励、PPO 和检查点核查学习信号与接续语义。
4. **实机桥接（6）**。最后对照离线快照和实机回执，核查公开观测、动作确认和时序。
