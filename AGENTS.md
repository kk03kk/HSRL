# AGENTS.md —— HSAI 单 Agent 工作约定

本文件是 HSAI 项目的**唯一权威约定**。它约束一个 Agent，同时管理两个环境：
本地 Windows `D:\HSAI` 与 NUS 计算集群 `~/projects/HSAI`。

未核实的事实一律标 **待确认**，不得写成既定事实。

## 0. 阅读顺序

1. 本文件（规则与红线）
2. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)（双环境职责与三层结构）
3. [`docs/OPERATIONS.md`](docs/OPERATIONS.md)（日常工作流，含开工固定开场）
4. [`docs/architecture/TRANSPORT.md`](docs/architecture/TRANSPORT.md)（SSH / git / GitHub / 大文件）
5. [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)（当前作业与结论）
6. 按任务读专题文档；历史文档在 `records/archive/`，**不是启动命令**。

## 1. 基本模式

- **一个决策者，两个执行环境。** 服务器上没有自主进程；它是被 `ssh` 驱动的执行端。
- **用户只提目标**，不当中转站。Agent 自主规划、执行、验证、归档。
- **每次开工先跑** `python ops/hsai_ops.py state`，看到 `IN SYNC` 才动手（见 OPERATIONS §1）。
- **代码走 git over SSH，大文件走 `ssh cat`，GitHub 只做里程碑异地备份。**
- 禁止用 scp 覆盖源码；禁止让服务器留未提交改动过夜。

## 2. 东西放哪

| 资产 | 服务器 | 本地 | 进 Git |
|---|:--:|:--:|:--:|
| 源码 `engine/ hsbg/ workflows/ deploy/ client/ tests/ tools/ config/ docs/` | ✅ | ✅ | ✅ |
| 模型 `model.pt` | ✅ 全部 | 只放线上 pin 的一个 | ❌ |
| Trainer `trainer.pt` | ✅ | ❌ | ❌ |
| 战团库 `artifacts/libraries/` | ✅ | ❌ | ❌ |
| 训练产物 `artifacts/runs/` | ✅ | ❌ | ❌ |
| 实机证据 `artifacts/runs/live-*`、`records/evidence/` | 可导入 | ✅ | ❌ |
| `data/firestone_replays/`、`data/real_library/` | ✅ | ❌ | ❌ |
| `artifacts/`、`records/` 整体 | 各自 | 各自 | ❌ |

判据：**读代码就能跑 → 放两边；读数据才能跑 → 只在服务器；必须在真实游戏边上 → 只在本地。**
详见 `docs/architecture/TRANSPORT.md` §4。

- 根目录只允许：`README.md AGENTS.md pyproject.toml requirements.txt .gitignore .gitattributes`。
- `docs/` 只放当前有效文档；被取代的移进 `records/archive/<topic>-<YYYYMMDD>/`（不删除）。
- 大文件阈值：单文件 ≤ 5 MB、单提交 ≤ 20 MB；权重/库/日志/压缩包/PDF 一律不提交。
- 命名：运行目录 `artifacts/runs/<side>-<purpose>-<YYYYMMDD>-<seq>`；
  模型目录含 `model.pt + manifest.json + checkpoint.json + MODEL_CARD.md`；
  评测报告 `models/<lib>/reports/<YYYYMMDD>-<protocol>.json`。

## 3. NUS 硬约束

- 登录节点实测 **4 核 / 16 GB**，多人共享（旧文档写的"约 1 GiB"是错的）。
  只做 git、`sbatch/sacct/squeue`、`tail`、小 JSON 标准库读取。
- **测试、Torch、模型加载、Rust 编译、pip 构建、建库、训练、大评测只能在计算节点。**
  `ops/hsai_ops.py` 的登录节点守卫默认拦截这些命令。
- 作业内**不要**包 `srun`（本站出现过把双卡缩成单卡）。
- 不 unset / 不手工改 `CUDA_VISIBLE_DEVICES`；GPU 身份以原始 Driver UUID 与 Torch 实际计算为准。
- 新作业用新输出目录；已存在且非空的目录拒绝覆盖。
- 排队/运行期间冻结代码、依赖、引擎、库和权重；要改就新目录、新作业号。
- 失败先用**新目录**重跑并保留旧现场，不覆盖、不删除。
- 分区与时限、GPU 型号见 `docs/OPERATIONS.md` §3.2（随集群变化，用实测值，不抄旧文档）。

## 4. 技术口径（唯一）

- 模拟器：`engine/` 的 `hsbg_engine`，酒馆 **37.0.2.256589**，源码指纹 `c4c71014e3d41760`。
- 动作：原生 **245**；Python 扩展声明 274，引擎未实现的后 29 维**不能**用于训练或实机输出。
- 观测：**3227 floats + 124 cards**。
- 模型兼容必须核对词表顺序、动作语义、布局、卡表与指纹，不能只对形状。
- 训练栈：`hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- **冻结训练契约**：`max-actions-per-turn=80`、`move-guard=1`（cap 5 / repeat 2）、
  `forbid-cancel=1`。改它需要用户明确授权。
- 续训起点：**7357 完整 Trainer**（累计 `558,873,732`、Adam `545,990`）；
  完整 Trainer 含 Adam/调度器/RNG/anchor，**权重文件不能代替**。
- 版本锚点：`config/engine-upstream-lock.json`、`config/live/model.json`（模型 SHA）、
  模型目录 `manifest.json`、评测 `protocol_sha256`。**任何记录必须写清这四个。**

## 5. 评测口径

- **P50**：每张战团"总攻击+总生命"的中位数；必须给 P50 与样本量，均值/P25/P75/P90 只作辅助。
- 不能把攻击中位数与生命中位数相加冒充总攻血 P50。
- 牌面强度 ≠ 战斗胜率；棋力看名次、前四率、冠军率与配对差。
- **live-v1**：真实 Mean Bot 快照面板（1 模型 + 7 bot）；**p50 库**：训练/辅助面板，
  不能单独当棋力结论；**pairwise Elo**：2 焦点模型 + 6 bot，只按两焦点相对名次计分。
- 不同 `protocol_sha256` 不能混榜。
- **E1 评测只跑一轮**：4 模型 × 6 对 × 448 局（ordered56）即完成排名，报告 valid/eligible
  后立即停止。**不得**为"确认复现"再跑第二轮，**不得**为冠军单独跑挑战赛。
  只有报告 invalid、工程错误或协议缺陷才重跑，且先向用户说明。

## 6. 自主权边界

**自主可做**：读、分析、审计、改代码、写测试、跑测试、建目录、短程 smoke（≤ 2 h）、
提交/取消**自己**的作业、拉权重、写文档、整理归档、向 GitHub 推送里程碑。

**必须先问**：

| 项 | 理由 |
|---|---|
| 删除资产 / 数据 / 证据 | 不可逆 |
| force-push、重写他人分支历史 | 不可逆，且会毁掉异地备份的意义 |
| 正式长训（> 6 h） | 占卡、改模型走向 |
| 大规模评测（> 1000 局） | 占卡 |
| 改游戏目录 | 破坏用户资产 |
| 改 `config/live/` 的模型 pin | 直接影响线上实机行为 |

- 不因正常棋力波动反复请示；目标明确就继续推进。
- 工程错误（非法动作、截断、引擎异常、注入失败、非有限损失、状态损坏）严格停止并保留现场；
  正常棋力波动只记录，不自动止损。
- **没有用户明确要求时：不创建 PR。**

## 7. 安全红线

- 禁止 `reset --hard`、`git clean`、全仓格式化、force-push。
- 不删除旧项目、模型、日志、库或原始证据；`artifacts/archives/` 是旧项目唯一副本。
- 不修改游戏目录；不执行历史未决输入账本（`live-state/inflight.json` 之类）。
- 不伪造 `plan.json`、checkpoint、SHA、卡表或恢复状态；不绕过源码锁、卡表、注入、
  数据质量与引擎异常门禁。
- 不在日志或文档写密钥、token、私密路径。
- **跳板机密码永不进对话、永不进命令行**（会进会话记录和进程列表）。

## 8. 记录与文档

- 每次大动作（换模型/卡表/引擎指纹、跑完一晚实机、跑完一轮评测、结构性架构改动）后：
  更新 `docs/PROJECT_STATE.md`，在 `docs/DECISIONS.md` 追加一条 `D-###`（只追加、不改历史），
  并在里程碑点推送 GitHub。
- 一次性分析写 `docs/analysis/<TOPIC>_<YYYYMMDD>.md`，写清状态、协议与入口。
- 文档里不把旧路径当"当前路径"；历史路径只作证据。
- 归档目录不更新、不参与流程。

## 9. 当前阶段（2026-11-12）

- 架构：单 Agent 双环境管理已上线；`docs/sync/` 散文协议已归档，GitHub 降级为异地归档。
- 训练：无正式作业在跑（队列里只有一条 `no-shell` 分配）。`stage2` 配置已含 `turnband`
  采样与 `forbid-unfreeze` 选项，但**没有任何运行在使用它，48h 启动需用户批准**。
- 模型：E1 正式榜第一是 `league-final-14858`（点估计；对 13806 不显著）。
  实机 pin 同为 `league-final-14858`，`real_game_accepted: false`。
- 实机：`league-final-14858` 两局终局有效（第 6 名 / 第 5 名），无可比优势；
  与 D-026（实机观测是 `public-approx-v1` 有损投影）一致，待取证。
- 服务器侧动态事实以 `python ops/hsai_ops.py state` 的实测输出为准，不抄文档。

## 10. 两端差异速查

| 事项 | 本地 Windows | NUS |
|---|---|---|
| Python | `.venv` 3.13.5，torch **cpu** | `.venv` 3.12.3，torch CUDA |
| 入口 | `client/live_245.py`（doctor/autoplay/decide/replay） | `workflows/*` + `deploy/slurm/*.sbatch` |
| 能跑 | 实机、离线回放、`tests/test_live_*` | 模拟器编译与测试、训练、建库、评测、Slurm |
| 不能跑 | 模拟器开发/编译/审计、训练栈测试、长训、大评测 | 实机、真实游戏输入 |
| 独有证据 | `artifacts/runs/live-*`、`records/evidence` | `artifacts/models/*`、库、评测报告、Slurm 日志 |
| 运维入口 | `ops/hsai_ops.py` | 无（被 ssh 驱动） |
