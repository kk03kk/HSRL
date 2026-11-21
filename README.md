# HSAI

Hearthstone Battlegrounds 的**强化学习与模型能力研究**工作区；真实游戏自动化暂时搁置。模拟器基于
AutoTavern / HSBRSIM 的 Rust 引擎（`engine/`）。**由一个 Agent 同时管理两个环境**：
本地 Windows 负责脚本、远程管理及未来实际验收，NUS 服务器负责模拟器、训练、评测与大规模资产。

```text
             ┌───────────────────────────────┐
             │  GitHub kk03kk/HSRL           │  异地归档（里程碑推送）
             └───────────────▲───────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │            git over SSH（随时）          │
┌───────▼──────────────┐                   ┌──────▼───────────────────┐
│ 本地 Windows D:\HSAI │                   │ NUS ~/projects/HSAI      │
│ 实机桥接 client/     │                   │ 模拟器内核 engine/       │
│ 运维层 ops/          │◄──── ssh cat ────►│ 训练栈 hsbg/az           │
│ 实机证据             │   权重/大文件      │ 权重 / 库 / 数据 / 产物  │
└───────▲──────────────┘                   └──────────────────────────┘
        │ 用户只在这里对话
```

接入方式、隧道原理、实测吞吐见 [docs/architecture/TRANSPORT.md](docs/architecture/TRANSPORT.md)。
完整职责与三层结构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，日常流程见
[docs/OPERATIONS.md](docs/OPERATIONS.md)，规则见 [AGENTS.md](AGENTS.md)。

## 当前入口

动态训练与模型状态只看 [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) 和实时
`python ops/hsai_ops.py state`；本页不复制作业号、冠军或下一轮训练配置。

- [工程架构与验收](docs/ENGINEERING.md)：冻结源码发布、模型登记、完整动作证据、环境复建。
- [模型能力与下一轮研究](docs/analysis/MODEL_CAPABILITY_RESEARCH_20261117.md)：历史训练账本、完整动作分析、机制限制与待批准实验。
- [唯一现役HSReplay攻略](docs/battlegrounds_strategy_guide.md)：以用户持续维护的正文版本为准，不混用归档攻略。
- [训练配方](docs/TRAINING_RECIPE.md)与[评测协议](docs/EVAL_PROTOCOL.md)：科学设计的约束。
- [现代化与正式评测报告](docs/analysis/MODERNIZATION_20261117.md)：本轮工作、证据、结果与限制。
- 引擎仍为 37.0.2.256589 / `c4c71014e3d41760`；原生动作 245，观测 3227 floats + 124 cards。
- 实机 pin 不因候选登记或离线评测自动改变；真实游戏验收仍独立进行。

## 目录结构

| 目录 | 内容 | 进 Git |
|---|---|:--:|
| `engine/` | Rust 模拟器 `hsbg-engine`、37.0.2 卡表、规则测试 | ✅ |
| `hsbg/` | 观测/动作编码、网络、BC/PPO、并行采集、评测 | ✅ |
| `client/` | 实机桥接：`hsrl_agent/live/`（当前）、`bg_agent/`、`legacy_policy/`（历史） | ✅ |
| `workflows/` | 建库、迁移、训练、评测入口 | ✅ |
| `tools/` | 运维脚本（`verify_workspace.py`、`live_archive_index.py`…） | ✅ |
| `ops/` | **跨端运维层**：`hsai_ops.py`（state / doctor / sh / push / pull / model-pull） | ✅ |
| `tests/` | 契约/规则/离线回归（`test_live_*.py` 覆盖实机链路） | ✅ |
| `deploy/` | NUS `setup`、`bootstrap_nus.sh`、`slurm/*.sbatch` | ✅ |
| `config/` | **仅当前配置**：`live/`（实机 profile + 模型 SHA）、评测、训练、引擎锁 | ✅ |
| `data/` | 卡表、饰品、池、archetype | ✅ |
| `docs/` | 当前有效文档（索引见 [docs/README.md](docs/README.md)） | ✅ |
| `records/` | 证据、迁移、源文档快照、研究摘要（`research/long*` 例外） | ❌ |
| `artifacts/` | 模型、战团库、数据集、运行产物（`runs/live-*` = 实机证据） | ❌ |

> 根目录只允许 `README.md AGENTS.md pyproject.toml requirements.txt .gitignore .gitattributes`。

## 快速开始

### 本地（Windows，代码/测试）

```powershell
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe tools/verify_workspace.py --skip-rust
.venv\Scripts\python.exe -m pytest tests -q
```

新机器先按 [环境复建步骤](docs/ENGINEERING.md#5-环境与可选外部源码) 安装角色锁定依赖和引擎轮子。
Rust 构建与 GPU 验证在 Slurm 计算节点进行。

### 实机（真实游戏自动化）

在已安装实机工作树的 Windows 本机，双击 `client/start-ai-control.cmd` 打开控制台。默认使用 Player150 末代 3286，下拉菜单可手动选择 3236。权重、离线回放和训练状态见[部署与停训交接](docs/analysis/PLAYER150_PROMOTION_20261121.md)。

前置：`D:\Hearthstone` 安装正确（`Hearthstone.exe` 与 profile SHA 一致）、游戏已登录并停在大厅、
分辨率 2560×1600 / DPI 144。

```powershell
python client/live_245.py doctor --game-dir D:\Hearthstone --infer
python client/live_245.py autoplay --game-dir D:\Hearthstone `
    --out artifacts/runs/live-<purpose>-<date>-001 --seconds 3600 --games 1
```

- `--games 0` = 直到 STOP/F8；`--games N` = N 局（失败按 `--max-failures` / `--max-same-failure` 预算处理）。
- 每局产物：`decisions.jsonl`、`decisions-full.jsonl.gz`、`actions.jsonl`、`summary.json`、`performance.json`、
  `status.json`、`health.json`、`result.json`、`session-archives/`（Power.log 字节前缀）。
- 审计：`python tools/audit_live_run.py --run artifacts/runs/<dir> --out <json>`。

### 跨端运维（本地一个入口管两端）

```powershell
python ops/hsai_ops.py state                 # 双端状态快照（隧道/HEAD/漂移/作业/模型）
python ops/hsai_ops.py doctor                # 七点自检
python ops/hsai_ops.py push                  # 本地 main -> 服务器
python ops/hsai_ops.py pull                  # 服务器 -> 本地
python ops/hsai_ops.py jobs                  # Slurm 队列 + 最近记账
python ops/hsai_ops.py model-pull <name>     # 拉模型到本地，逐文件 SHA256 校验
python ops/hsai_ops.py sh "<命令>"            # 远端执行（带登录节点守卫）
python ops/hsai_ops.py tunnel                # 隧道断了时打印重建命令
```

前置：隧道在跑（`probe` 返回 `ok: true`）。原理、恢复与实测吞吐见
[docs/architecture/TRANSPORT.md](docs/architecture/TRANSPORT.md)。

## 文档导航

| 文档 | 说明 |
|---|---|
| [AGENTS.md](AGENTS.md) | **唯一权威约定**：放哪、红线、自主权边界、技术口径 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 单 Agent 双环境职责与三层结构 |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | 日常工作流：开工开场、改代码、训练、拉模型、实机、应急 |
| [docs/architecture/TRANSPORT.md](docs/architecture/TRANSPORT.md) | SSH 隧道、git over SSH、GitHub 定位、大文件传输 |
| [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) | 当前作业、模型结论、下一步 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 决策留痕（`D-###`，只追加） |
| [docs/analysis/](docs/analysis/) | 一次性分析与专题审计（历史审计已归档到 `records/archive/docs-20261114/`，不再当"当前路径"用） |
| [docs/nus/ENVIRONMENT.md](docs/nus/ENVIRONMENT.md) | NUS 环境、Slurm、GPU、恢复 |

## 项目边界

- 规则与学习系统只在服务器改；实机桥接只在本地改（不是因为无权，而是避免两端不一致）。
- 实机桥接是 `public-approx-v1`：显式近似，不等价于模拟器精确状态。
- 不修改游戏目录、不执行历史未决输入账本、不伪造 plan/checkpoint/SHA。
- 自主权边界与红线见 [AGENTS.md](AGENTS.md) §6/§7。

来源：[AutoTavern](https://github.com/MCG-NJU/AutoTavern)、[HSBRSIM](https://github.com/Gallo13th/HSBRSIM)，历史原作者与许可证声明保留在原始证据与源码中。
