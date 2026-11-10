# AGENTS.md —— 给本地 / 服务器 Agent 的工作约定

本文件对**本地 Windows Agent** 和**NUS 服务器 Agent** 同时有效。两边职责不同，
但都必须遵守同一套文件、同步、评测与安全规则。未核实的服务器事实一律标 **待确认**。

## 0. 阅读顺序

1. `README.md`（架构与用法）
2. `docs/README.md`（文档索引）
3. `docs/PROJECT_STATE.md`（当前作业与结论）
4. `docs/SYNC_MODEL.md`（三端管理规范）
5. `docs/sync/LOCAL.md` / `docs/sync/SERVER.md` / `docs/sync/DECISIONS.md`（跨端协作）
6. 按任务读对应专题文档；历史文档只在 `records/archive/`，不是启动命令。

## 1. 三端职责（写权唯一）

| 端 | 拥有写权 | 禁止 |
|---|---|---|
| 本地 Windows `D:\HSAI` | `client/` 实机桥接、`config/live/`、实机证据 `artifacts/runs/live-*`、`records/`、`docs/sync/LOCAL.md`、`docs/live/**`、`docs/LIVE_V1_1_AUDIT.md`、`docs/E_PHASE_RUNBOOK.md`、`docs/analysis/**` | 改**服务器拥有的代码**（`engine/ hsbg/ workflows/ deploy/ data/ tests/(除 test_live_*) tools/`）；本地编译模拟器；长训、建库、大规模评测 |
| GitHub `kk03kk/HSRL` `main` | 源码/配置/卡表/文档/清单（≤5 MB 单文件） | 权重、库、数据、日志、压缩包 |
| NUS `~/projects/HSAI` | **模拟器内核、动作空间、观测编码、奖励、模型与损失、采样与自对弈、评测、训练编排、资产（权重/库/数据）**、`docs/sync/SERVER.md`、`docs/sync/ENGINE_RELEASES.md`、`docs/`（除本地那几份） | 实机操作；改 `client/` 与 `config/live/`；在登录节点跑重活 |

**任何资产只有一个端能写。** 需要跨端改同一文件时，先改 `main`，另一端 pull。
完整职责矩阵与迁移步骤见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。
本地可用 `python client/tools/check_ownership.py --strict` 自检是否越界。

## 2. 与用户的工作方式

- 用户只给大方向；Agent 自主规划、执行、验证、提交（Slurm）、监控、归档。
- 不因正常棋力波动反复请示；目标明确就继续推进。
- 只有**不可逆且明显超出目标**的操作才先确认：删除历史证据、改游戏目录、force-push、清空工作区、正式长训/大规模评测启动。
- **没有用户明确要求时：不 commit、不 push、不创建 PR。**
- 工程错误（非法动作、截断、引擎异常、注入失败、非有限损失、状态损坏）严格停止并保留现场；
  正常棋力波动只记录，不自动止损。

## 3. NUS 硬约束（服务器 Agent 必读，本地 Agent 引用）

- 登录节点约 **1 GiB** 虚拟内存：只做 Git、`sbatch/sacct/squeue`、`tail`、小 JSON 标准库读取。
- `pytest`、Torch、模型加载、Rust 编译、pip 构建、建库、训练、大型 JSON 解析**只能**在计算节点。
- 作业内**不要**包 `srun`（本站出现过 srun 把双卡缩成单卡）。
- 不 unset / 不手工改 `CUDA_VISIBLE_DEVICES`；GPU 身份以原始 Driver UUID 与 Torch 实际计算为准。
- 新作业用新输出目录；已存在且非空的目录拒绝覆盖。
- 排队/运行期间冻结代码、依赖、引擎、库和权重；要改就新目录、新作业号。
- 失败先用**新目录**重跑并保留旧现场，不覆盖、不删除。

## 4. 技术口径（唯一）

- 模拟器：`engine/` 的 `hsbg_engine`，酒馆 **37.0.2.256589**，源码指纹 `c4c71014e3d41760`。
- 动作：原生 **245**；Python 扩展声明 274，引擎未实现的后 29 维**不能**用于训练或实机输出。
- 观测：**3227 floats + 124 cards**。
- 模型兼容必须核对词表顺序、动作语义、布局、卡表与指纹，不能只对形状。
- 训练栈：`hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- 续训起点：**7357 完整 Trainer**（累计 `558,873,732`、Adam `545,990`）；完整 Trainer 含 Adam/调度器/RNG/anchor，权重文件不能代替。
- 版本锚点：`config/engine-upstream-lock.json`、`config/live/model.json`（模型 SHA）、
  模型目录 `manifest.json`、评测 `protocol_sha256`。任何记录必须写清这四个。

## 5. 评测口径

- **P50**：每张战团"总攻击+总生命"的中位数；必须给 P50 与样本量，均值/P25/P75/P90 只作辅助。
- 不能把攻击中位数与生命中位数相加冒充总攻血 P50。
- 牌面强度 ≠ 战斗胜率；棋力看名次、前四率、冠军率与配对差。
- **live-v1**：真实 Mean Bot 快照面板（1 模型 + 7 bot）；**p50 库**：训练/辅助面板，不能单独当棋力结论；
  **pairwise Elo**：2 焦点模型 + 6 bot，只按两焦点相对名次计分。
- 不同 `protocol_sha256` 不能混榜。

## 6. 当前阶段（2026-11-10）

- 训练：`1095539` 已完成，**当前无正式训练/评测作业**；不要擅自取消 `1097487` 的交互 allocation。
- 模型结论：live-v1 无模型显著优于 7357；`v9786` 点估计最好（3.799）。
- 实机：V1.1 夜间 7 局 + D 阶段 6 次 canary 已完成；**D 阶段工程改造完成并停下等确认**。
- 下一步（需用户授权）：E 阶段连续实机验收；pairwise Elo 1,792 局启动。
- 服务器侧事实（commit、模型、库、作业产物）：**全部待确认**，见 `docs/sync/SERVER.md`。

## 7. 跨端协作协议

- 目录 `docs/sync/`：`LOCAL.md`（本地写）、`SERVER.md`（服务器写）、`DECISIONS.md`（双方追加）、
  `manifests/{local,server}.json`（`tools/sync_manifest.py` 生成）。
- 每份 Markdown 头部：`status: pending|draft|confirmed`、`owner`、`updated`、`base_commit`。
- 流程：写自己的文件 → push → 对方 `pull --rebase` → 对方在**自己的文件**回答并在 `DECISIONS.md`
  追加 `D-###` → 复核后改 `status: confirmed`。
- 禁止改写对方文件；禁止在 `docs/sync/` 放二进制；冲突以编号最大的 accepted 决议为准。
- 每次大动作（换模型/卡表/引擎指纹、跑完一晚实机、跑完一轮评测）后重跑 manifest 并提交。

## 8. 文件与文档规则

- 根目录只允许：`README.md AGENTS.md pyproject.toml requirements.txt .gitignore .gitattributes`。
- `docs/` 只放**当前有效**文档；被取代的移动到 `records/archive/<topic>-<YYYYMMDD>/`（不删除）。
- `config/` 只放当前配置；历史配置进 `records/archive/`。
- `records/` 是证据（整目录不进 Git，除 `research/long*` 摘要）；`artifacts/` 全部不进 Git。
- 命名：运行目录 `artifacts/runs/<side>-<purpose>-<YYYYMMDD>-<seq>`；
  模型目录含 `model.pt + manifest.json + checkpoint.json + MODEL_CARD.md`；
  评测报告 `models/<lib>/reports/<YYYYMMDD>-<protocol>.json`。
- 大文件阈值：单文件 ≤ 5 MB、单提交 ≤ 20 MB；压缩包/权重/日志/PDF 一律不提交。
- 文档中不写本地 `D:\HSAI` 之外的旧路径作为"当前路径"；历史路径只作证据。
- 新文档要写状态（当前/计划/归档）、协议和入口，避免把归档命令当现行命令。

## 9. 安全边界

- 不修改游戏目录；不执行历史未决输入账本（`live-state/inflight.json` 之类）。
- 不删除旧项目、模型、日志、库或原始证据；`artifacts/archives/` 是旧项目唯一副本。
- 不伪造 `plan.json`、checkpoint、SHA、卡表或恢复状态；不绕过源码锁、卡表、注入、数据质量与引擎异常门禁。
- 不在日志或文档写密钥、token、私密路径。
- 禁止 `reset --hard`、`git clean`、全仓格式化、force-push。

## 10. 两端具体差异

| 事项 | 本地 Windows | NUS |
|---|---|---|
| Python | `.venv`，3.13.5，torch **cpu** | conda/venv，GPU（H100），Slurm 计算节点 |
| 运行入口 | `client/live_245.py`（doctor/autoplay/decide/replay） | `workflows/*` + `deploy/slurm/*.sbatch` |
| 能跑 | 实机、离线回放（只读镜像）、`tests/test_live_*.py` | 模拟器编译与测试、训练、建库、评测、Slurm |
| 不能跑 | 模拟器开发/编译/审计、训练栈测试、长训、大评测 | 实机、真实游戏输入 |
| 独有证据 | `artifacts/runs/live-*`、`records/evidence` | `artifacts/models/*`、库、评测报告、Slurm 日志 |

## 11. 2026-11-10 架构调整（现行）

- **服务器统一维护模拟器与强化学习系统**（规则/动作/观测/奖励/模型/采样/评测/训练编排/资产）；
- **本地只做真实游戏桥接**（读屏、推理接入、自动操作、执行验证、日志采集），
  保留最小运行依赖（`client/` + `obs_std`/`action_space_std`/`common` 常量 + `hsbg_engine` 轮子），
  **不再维护独立的模拟器开发环境、不再本地编译审计**；
- 唯一例外：引擎升级时由本地做一次**部署构建**（Windows 轮子只能在本地产出），
  版本由 `docs/sync/ENGINE_RELEASES.md`（服务器写）驱动；
- 实机日志可回传服务器用于规则验证，走 tar.gz + sha256，**不进 Git**；
- 规则审计结论不变：除非发现重大规则错误或训练稳定性问题，不修改模拟器。
  详见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 与 [`docs/sync/SERVER_HANDOFF_20261110.md`](docs/sync/SERVER_HANDOFF_20261110.md)。
