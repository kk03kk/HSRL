# AGENTS.md —— HSAI 单 Agent 工作约定

本文件是 HSAI 项目的**唯一权威约定**。它约束一个 Agent，同时管理两个环境：
本地 Windows `D:\HSAI` 与 NUS 计算集群 `~/projects/HSAI`。

未核实的事实一律标 **待确认**，不得写成既定事实。

## 0. 阅读顺序

1. 本文件（规则与红线）
2. [`docs/CHARTER.md`](docs/CHARTER.md)（**目标 / 研发模式 / 一轮迭代怎么走**）
3. [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md)（**训练配方，已冻结**）
4. [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)（**评测标准与判胜规则**）
5. [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md)（当前冠军、正在跑什么、下一步）
6. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)（双环境职责与三层结构）
7. [`docs/OPERATIONS.md`](docs/OPERATIONS.md)（日常工作流，含开工固定开场）
8. [`docs/architecture/TRANSPORT.md`](docs/architecture/TRANSPORT.md)（SSH / git / GitHub / 大文件）
9. 按任务读专题文档；历史文档在 `records/archive/` 与 `docs/analysis/`，**不是启动命令**。

## 1. 基本模式

- **一个决策者，两个执行环境。** 服务器上没有自主进程；它是被 `ssh` 驱动的执行端。
- **用户定方向，Agent 执行、分析、验证。** 方向、目标、任何 GPU 作业的提交、以及训练配方/评测
  标准的变更，都由用户决定；Agent 负责把方向落成可验证的结果，发现问题时**提建议**而不是自己改道。
  （2026-11-14 变更：不再有"Agent 自主规划多天研发路线"的模式，详见 [`docs/CHARTER.md`](docs/CHARTER.md)。）
- **每次开工先跑** `python ops/hsai_ops.py state`，看到 `IN SYNC` 才动手（见 OPERATIONS §1）。
- **代码走 git over SSH，大文件走 `ssh cat`，GitHub 只做里程碑异地备份。**
- **隧道自愈**：用户已登记跳板机密码时，Agent **自己**重建，不要等用户、也不要问。
  固定四步（预检 VPN → 重建 → 独立探活 → 仍 DOWN 才读日志），实测 6–10 s，
  **权威步骤见 [`docs/OPERATIONS.md`](docs/OPERATIONS.md) §1.3**。
  两条硬禁令：不要自己拼 `ssh -N -f -L`；**不要给建隧道脚本接管道**
  （常驻 ssh 会继承管道写端永不关闭，命令卡死而隧道其实已建好）。
  尚未登记密码时才需要用户跑 `ops\tunnel.ps1 -Setup`。
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
  **唯一例外（用户 2026-11-15 授权，见 D-063）**：`deploy/slurm/stage_arena.sbatch` 里的
  **MIG UUID 重定向** —— Slurm 只给数字编号时 CUDA 会把 `0,1` 当两张物理卡（第 2 张不存在），
  所以脚本用 `nvidia-smi -L` 里**本作业已分配**的 MIG UUID 重写 CVD，让两张切片都能用。
  约束：只能用当前分配报出来的 UUID、不能凭空指定、必须把原值/UUID/结果写进 `run.env`。
- 新作业用新输出目录；已存在且非空的目录拒绝覆盖。
- 排队/运行期间冻结代码、依赖、引擎、库和权重；要改就新目录、新作业号。
- 失败先用**新目录**重跑并保留旧现场，不覆盖、不删除。
- **远端脚本一律纯 ASCII**；文件内容只用 `write`/`edit` 工具读写，不经 PowerShell
  `Get-Content`/`Set-Content` 中转（本机 ACP=936，会毁 UTF-8 并可能吃掉引号）。
  详见 [`docs/OPERATIONS.md`](docs/OPERATIONS.md) §2.1 与 `D-050`。
- 分区与时限、GPU 型号见 `docs/OPERATIONS.md` §3.2（随集群变化，用实测值，不抄旧文档）。

## 4. 技术口径（唯一）

- 模拟器：`engine/` 的 `hsbg_engine`，酒馆 **37.0.2.256589**，源码指纹 `c4c71014e3d41760`。
- 动作：原生 **245**；Python 扩展声明 274，引擎未实现的后 29 维**不能**用于训练或实机输出。
- 观测：**3227 floats + 124 cards**。
- 模型兼容必须核对词表顺序、动作语义、布局、卡表与指纹，不能只对形状。
- 训练栈：`hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- **冻结训练契约**（唯一真值 `hsbg/az/train/eval_policy.py::TRAINING_CONTRACT`）：
  `max-actions-per-turn=100`、`move-guard=1`（cap **2** / repeat 2）、`forbid-cancel=1`、
  `forbid-unfreeze=1`、`freeze-cap=1`（每回合只能按一次 Freeze）、`ban-trinkets=1`、`sims=0`。
  **2026-11-17 用户改过 100/2/1（D-070）**：改它需要用户明确授权，且会让 `protocol_sha256` 变（换榜）。
- 续训/分叉：完整 Trainer（`trainer.pt`，含 Adam/调度器/RNG/anchor）才等于"精确续训"，
  **权重文件不能代替**；但**带 `--trainer-state` 会用 state 里的 `base_lrs` 覆盖 `--lr`**。
  当前配方故意只继承权重、重建 Adam（为了拿到论文的 6e-06），详见 `docs/TRAINING_RECIPE.md` §2.2。
  该训练器**没有 `--resume`**，续训参数是 `--trainer-state`。
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

**自主可做**（不占 GPU）：读、分析、审计、改代码、写测试、跑测试（CPU）、建目录、
拉权重、写文档、整理归档、向 GitHub 推送里程碑。

**必须先问**：

| 项 | 理由 |
|---|---|
| **任何消耗 GPU 的作业**（训练、评测、建库、smoke、探针） | 用户 2026-11-14 定的硬规则：占卡；须先给配置、用户点头 |
| 删除资产 / 数据 / 证据 | 不可逆 |
| force-push、重写他人分支历史 | 不可逆，且会毁掉异地备份的意义 |
| 正式长训（> 6 h） | 占卡、改模型走向 |
| 大规模评测（> 1000 局） | 占卡 |
| 改训练配方 / 评测标准的任何参数 | 配方已冻结（`docs/TRAINING_RECIPE.md`）；变更由用户决定 |
| 改游戏目录 | 破坏用户资产 |
| 改 `config/live/` 的模型 pin | 直接影响线上实机行为 |

- **不自主规划多天研发路线**；发现方向性问题时写建议、等用户决策（见 `docs/CHARTER.md`）。
- 工程错误（非法动作、截断、引擎异常、注入失败、非有限损失、状态损坏）严格停止并保留现场；
  正常棋力波动只记录，不自动止损，也不因此改配方。
- **没有用户明确要求时：不创建 PR。**

## 7. 安全红线

- 禁止 `reset --hard`、`git clean`、全仓格式化、force-push。
- 不删除旧项目、模型、日志、库或原始证据；`artifacts/archives/` 是旧项目唯一副本。
- 不修改游戏目录；不执行历史未决输入账本（`live-state/inflight.json` 之类）。
- 不伪造 `plan.json`、checkpoint、SHA、卡表或恢复状态；不绕过源码锁、卡表、注入、
  数据质量与引擎异常门禁。
- 不在日志或文档写密钥、token、私密路径。
- **跳板机密码永不进对话、永不进命令行**（会进会话记录和进程列表）。
  唯一允许的落盘形式，是用户**自己**执行 `ops\tunnel.ps1 -Setup` 后由 DPAPI 加密的
  `%LOCALAPPDATA%\HSAI\tunnel.cred`。Agent 不读它、不打印它、不传输它、不把它写进任何
  日志或文档；只调用已经封装好的 `ops/tunnel.ps1` / `hsai_ops.py tunnel`。

## 8. 记录与文档

- 每次大动作（换模型/卡表/引擎指纹、跑完一晚实机、跑完一轮评测、结构性架构改动）后：
  更新 `docs/PROJECT_STATE.md`，在 `docs/DECISIONS.md` 追加一条 `D-###`（只追加、不改历史），
  并在里程碑点推送 GitHub。
- 一次性分析写 `docs/analysis/<TOPIC>_<YYYYMMDD>.md`，写清状态、协议与入口。
- 文档里不把旧路径当"当前路径"；历史路径只作证据。
- 归档目录不更新、不参与流程。

## 9. 当前阶段（**as of 2026-11-17，不再在此复制参数真值**）

> 2026-11-17 审计结论：本节曾经是"配方快照"，但连续漂移了一整代（写着强制升本是
> `2:2,4:3,5:4,7:5,8:6` / prob 1.0 / 教师关 / lr 1.2e-05，而实际生效的是
> `2:2,5:3,7:4,8:5,10:6` / prob 0 / 教师全程 / lr 6e-06）。按本节复现会跑出
> **D-065 已经判负的那套配方**。所以：**参数真值只允许出现在下面这两处**。

- **参数真值的唯一来源**：`.sbatch`/训练器脚本本身 ＋ [`docs/DECISIONS.md`](docs/DECISIONS.md)
  的**最新一条** `D-###`。要看当前配方，读 [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md)
  的"指针"段，再看 `deploy/slurm/stage_arena.sbatch` 的实际取值。
- **研发模式**：用户定方向、Agent 执行/分析/验证；不自主规划多天路线（`docs/CHARTER.md`）。
- **当前最强模型 / 冠军**：以 `docs/DECISIONS.md` 最新条为准；本文件**不再**写模型名。
- **判胜口径**：用户 2026-11-17 明确"**不用单一指标判胜**"，主看逐回合 M1 与增长曲线，
  名次只作参考（[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)）。
- **GPU**：`h100-47` 是 MIG 切片，用 MIG UUID 重定向后不抽签（§3 的例外条款，D-063）。
- **实机**：`real_game_accepted: false`，待 D-026 取证。
- 服务器侧动态事实以 `python ops/hsai_ops.py state` 的实测输出为准，不抄文档。
- **历史审计与失效结论**：[`docs/analysis/AUDIT_20261117.md`](docs/analysis/AUDIT_20261117.md)
  登记了"因 Bug 而不可靠"的历史结论清单（含 D-068 的加重护手封禁在 v1.3 三轮里未生效）。

## 10. 两端差异速查

| 事项 | 本地 Windows | NUS |
|---|---|---|
| Python | `.venv` 3.13.5，torch **cpu**（**已安装 `hsbg_engine` 37.0.2.256589**，`import hsbg_engine` 可用） | `.venv` 3.12.3，torch CUDA |
| 入口 | `client/live_245.py`（doctor/autoplay/decide/replay） | `workflows/*` + `deploy/slurm/*.sbatch` |
| 能跑 | 实机、离线回放、`tests/test_live_*`；**引擎相关测试也能跑**（`pytest tests` 全量 877±，2026-11-17 实测 rc=0，含 `test_calibrated_opponents` 这类真实对局测试）；小规模 CPU 评测 smoke（2 局约 7 s） | 模拟器编译与测试、训练、建库、评测、Slurm |
| 不能跑 | **编译 Rust**（无 MSVC/`cl.exe`）、训练（torch 是 CPU 版）、长训、大评测 | 实机、真实游戏输入 |
| 独有证据 | `artifacts/runs/live-*`、`records/evidence` | `artifacts/models/*`、库、评测报告、Slurm 日志 |
| 运维入口 | `ops/hsai_ops.py` | 无（被 ssh 驱动） |
