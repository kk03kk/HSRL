# HSAI

Hearthstone Battlegrounds 的强化学习 + **真实游戏自动化**工作区。模拟器基于
AutoTavern / HSBRSIM 的 Rust 引擎（`engine/`）。**由一个 Agent 同时管理两个环境**：
本地 Windows 负责真实游戏桥接与运维，NUS 服务器负责模拟器、训练、评测与 GPU 资产。

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

## 当前状态（2026-11-12）

- 架构：**单 Agent 双环境管理已上线**。`ops/hsai_ops.py` 是唯一远端入口；GitHub 降级为
  异地归档；`docs/sync/` 散文协作协议已退休并归档。
- 模拟器：酒馆 **37.0.2.256589**，源码指纹 `c4c71014e3d41760`，原生动作 **245**，观测 **3227 floats + 124 cards**。
- 训练：`1102669 hsai-league-36h` **已完成**（72/72 段；final `v14858`、累计 `1,552,463,499`；`illegal/truncated/dropped=0`）。final 资产在 `artifacts/models/league-final-20261111-step014858/`。**当前没有正式作业在跑。**
- 审计：2026-11-11 全链路审计已完成，报告 `docs/AUDIT_2026-11-11.md`；修复了镜像 p75 退化、Freeze 空转、评测 Elo 尺度/shared bootstrap 等问题。
- 评测（引擎 `c4c71014`）：E1 四模型校准正式报告 r2（2,688 局，与 r1 六对逐对一致）：`league-final-14858` 点估计冠军 **1086.53**（CI [1062.49, 1111.01]），`13806` 1068.45，`12311` 1055.92，`V1.1` 1000；final vs 13806 为 240–208，不显著。按用户规则只跑一轮、不做挑战赛。
- 旧结果：v2 榜（12311 1067.68）协议与 E1 不同，不能混榜；11 模型 live-v1 锚定旧引擎 `c7b5f27d`，只作历史证据。
- 实机：pin 已更新为 `league-final-14858`，`real_game_accepted: false`；两局终局有效（第 6 / 第 5 名），尚未显示相对 12311 的优势，与 D-026（`public-approx-v1` 有损投影）一致，待取证。
- 下一阶段：下一轮长训使用 `config/league-stage2.json`（turnband + forbid-unfreeze + stage fork，2×H100，94×30min），**等待用户确认后再启动**。

详细状态：[docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)。

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
python -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt          # numpy / torch / pytest / maturin
python tools/verify_workspace.py         # BOM、源指纹、卡表、契约门禁
python -m pytest tests/test_live_*.py -q # 实机链路离线回归
```

### 实机（真实游戏自动化）

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
| [docs/LIVE_V1_1_AUDIT.md](docs/LIVE_V1_1_AUDIT.md) | V1.1 实机审计、D 阶段改造与实机验证 |
| [docs/SIMULATOR_AUDIT_2026-11-09.md](docs/SIMULATOR_AUDIT_2026-11-09.md) | 37.0.2 模拟器规则审计 |
| [docs/LEADERBOARD.md](docs/LEADERBOARD.md) | 当前唯一排行榜规则与分数 |
| [docs/EVAL_DUEL.md](docs/EVAL_DUEL.md) | 2 model + 6 bot 对局协议 |
| [docs/nus/ENVIRONMENT.md](docs/nus/ENVIRONMENT.md) | NUS 环境、Slurm、GPU、恢复 |

## 项目边界

- 规则与学习系统只在服务器改；实机桥接只在本地改（不是因为无权，而是避免两端不一致）。
- 实机桥接是 `public-approx-v1`：显式近似，不等价于模拟器精确状态。
- 不修改游戏目录、不执行历史未决输入账本、不伪造 plan/checkpoint/SHA。
- 自主权边界与红线见 [AGENTS.md](AGENTS.md) §6/§7。

来源：[AutoTavern](https://github.com/MCG-NJU/AutoTavern)、[HSBRSIM](https://github.com/Gallo13th/HSBRSIM)，历史原作者与许可证声明保留在原始证据与源码中。
