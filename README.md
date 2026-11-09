# HSAI

HSAI 是一个面向 Hearthstone Battlegrounds 的强化学习与评测工作区，基于 AutoTavern / HSBRSIM 的 Rust 模拟器和 `hsbg/az` 训练栈，进行实体网络、BC/PPO、固定 Bot 库评测和实机桥接研究。不是从零原创的模拟器。

当前项目已经运行在 NUS 服务器上：

```text
~/projects/HSAI
```

**代理直接在 NUS 登录节点工作，所有开发、测试、评测、训练、Slurm 提交与监控都在服务器完成，不再需要本地传权重或回传结果。** 用户只负责大方向，代理自主规划和执行。

## 当前阶段

更新：2026-11-09

- 1095539 强制升本长训已完成：累计 `810,501,828` 决策，`time_budget` 停止。
- 11 个代表模型已在 live-v1/dev 完成 512 局评测；**没有任何模型显著优于 7357**。
- live-v1 点估计最佳：`m1085131-v9786`（3.799）；当前 run 最佳：30M `v7641`（3.930）。
- 当前 final `v9691` 为 3.965；240M `v9595` 在 p50 96 局最好，但 live-v1 排 8/11。
- A/B `1098924` 已按用户要求取消，没有产生结果。
- 2026-11-09 完成 37.0.2 模拟器规则审计：修复 `Jar o' Gems` 每战斗触发次数错误；Rust 942 passed / 0 failed / 1 ignored。
- 2026-11-09 旧战团库已全部归档；8 模型 pairwise Elo 配置、权重 staging/审计、16 局冒烟已完成；正式 1,792 局等待启动确认。

当前状态与结论见 [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)。

## 从这里开始

1. [docs/README.md](docs/README.md)：当前文档索引。
2. [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)：当前作业、模型结论、下一步。
3. [docs/SIMULATOR_AUDIT_2026-11-09.md](docs/SIMULATOR_AUDIT_2026-11-09.md)：37.0.2 模拟器审计报告。
4. [docs/ELO_EVAL.md](docs/ELO_EVAL.md)：2 model + 6 bot pairwise Elo 方案。
5. [docs/EVAL_DUEL.md](docs/EVAL_DUEL.md)：2 model + 6 bot 对局协议。
6. [docs/nus/ENVIRONMENT.md](docs/nus/ENVIRONMENT.md)：NUS 环境、Slurm、GPU、恢复。
7. `AGENTS.md`：代理工作方式和硬约束。

## 当前技术口径

- 模拟器：`engine/`，酒馆 **37.0.2.256589**，源码指纹 `c4c71014e3d41760`。
- 原生动作：**245**；Python 扩展声明 274，但不能使用引擎未实现的后 29 维。
- 观测：**3227 floats + 124 cards**。
- 模型：16.44M entity 网络；必须在词表顺序、动作语义、布局、卡表和指纹上兼容。
- 训练主线：一个学习席位 + 七个严格 EOT 快照席位，`sims=0`；不把搜索/BC/旧客户端当成当前训练栈。
- 当前续训起点：**7357 完整 Trainer**（累计 `558,873,732`、Adam `545,990`）；完整 Trainer 包含 Adam、调度器、RNG、anchor 和累计计数，权重文件不能代替。
- 当前评测：live-v1/dev 固定 512 局；p50 只做辅助；不同 `protocol_sha256` 不能混榜。
- 下一阶段：2 model + 6 bot pairwise Elo，方案见 [docs/ELO_EVAL.md](docs/ELO_EVAL.md)。

## 工作区结构

```text
engine/      Rust 游戏规则、战斗、快照事务、PyO3 接口和当前卡表
hsbg/        观测/动作、网络、BC/PPO、并行采集、SHM、评测
workflows/   建库、迁移、训练、评测、实验入口
deploy/      NUS Slurm 入口与计算节点脚本
config/      训练/评测配置与源码锁
data/        兼容静态数据；不能覆盖当前引擎卡表
client/      实机桥接；public-approx，不等价于模拟器精确状态
tests/       契约、规则和离线回归
artifacts/   模型、库、运行输出；不入 Git
records/     原始证据、迁移记录、研究归档
docs/        当前有效文档；历史文档在 records/archive
README.md    本文件
AGENTS.md    代理工作方式
```

## NUS 运行约束

登录节点只有约 **1 GiB 虚拟内存**。重任务只能在 Slurm 计算节点：

- 可以：Git、`sbatch/sacct/squeue`、`tail`、小 JSON 标准库读取。
- 不可以：`pytest`、Torch、模型加载、Rust 编译、pip 构建、建库、训练、大型 JSON 解析。

作业规则：

- 只用 `sbatch`，作业内不要包 `srun`。
- 不 unset / 不手工改 `CUDA_VISIBLE_DEVICES`；GPU 设备以原始 Driver UUID 和实际计算证明为准。
- 新作业使用新输出目录；已存在且非空的目录拒绝覆盖。
- 排队/运行期间冻结代码、依赖、引擎、库和权重。
- 工程错误严格失败；正常棋力波动记录但不自动止损。

## 当前结果摘要

| 模型 | live-v1 512 mean rank | 说明 |
|---|---:|---|
| `m1085131-v9786` | **3.799** | 点估计最佳，不显著优于 7357 |
| `m1080772-v7357` | **3.828** | 稳健基线 |
| `m1095539-v7641` 30M | 3.930 | 当前 run live 最佳 |
| `m1095539-v9691` final | 3.965 | 当前 final |
| `m1095539-v9595` 240M | 4.107 | p50 96 局最佳，但 live 差 |
| `m1095539-v7936` 60M | 4.557 | 已确认灾难点 |

完整排行榜：

```text
artifacts/models/leaderboard-v1/reports/20261109-live-v1.json
```

## 文档与归档

- 根目录只保留 `README.md` 和 `AGENTS.md`。
- `docs/` 只放当前有效文档，索引见 [docs/README.md](docs/README.md)。
- 历史文档与旧部署已归档到：

```text
records/archive/docs-2026-11-09/
```

- 运行产物在 `artifacts/`；原始证据在 `records/`。归档文档不是当前启动命令。

## 当前下一步

1. 用户确认 `config/eval-elo-8.json`、8 个模型 SHA、engine `c4c71014e3d41760` 和协议 SHA `837231e0...`。
2. 启动 `sbatch deploy/slurm/elo_8.sbatch`（1 × H100-47、8h、28 对 × 64 局 = 1,792 局）。
3. 监控作业；失败保留现场并用新目录重跑。
4. 输出 pairwise Elo、bootstrap CI、两两胜率矩阵和门禁报告。
5. 根据结果决定下一轮训练来源和配方；不把 p50、live-v1、Elo 的绝对分数混榜。

来源：[AutoTavern](https://github.com/MCG-NJU/AutoTavern)、[HSBRSIM](https://github.com/Gallo13th/HSBRSIM)。历史原作者和许可证声明保留在原始证据与源码中。
