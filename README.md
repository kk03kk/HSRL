# HSAI

Hearthstone Battlegrounds 的强化学习 + **真实游戏自动化**工作区。模拟器基于
AutoTavern / HSBRSIM 的 Rust 引擎（`engine/`），**模拟器与训练/评测系统统一由 NUS 服务器维护**，
本地 Windows 只负责真实游戏桥接（读屏、推理接入、自动操作、执行验证、日志采集），
代码与文档经 GitHub 同步。

```text
本地 Windows (D:\HSAI)  ──push/pull──  GitHub kk03kk/HSRL  ──pull/push──  NUS ~/projects/HSAI
  实机桥接 client/ + config/live/          唯一代码与文档真源              模拟器内核与规则
  读屏 / 推理 / 操作 / 回执对账             （不含任何大文件）              动作空间 / 观测 / 奖励
  实机证据 artifacts/runs/live-*                                           模型 / 损失 / 采样 / 评测
                                                                           训练编排 / 权重 / 库 / 数据
```

**职责边界（2026-11-10 起）**：规则与学习系统只在服务器改，实机桥接只在本地改。
完整矩阵、迁移步骤与例外（引擎 Windows 轮子由本地做部署构建）见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 当前状态（2026-11-11）

- 模拟器：酒馆 **37.0.2.256589**，源码指纹 `c4c71014e3d41760`，原生动作 **245**，观测 **3227 floats + 124 cards**。
- 训练：`1102669 hsai-league-36h` **已完成**（72/72 段，`2026-11-11T16:46:33+08:00`；final `v14858`、累计 `1,552,463,499`；`illegal/truncated/dropped=0`）。final 资产在 `artifacts/models/league-final-20261111-step014858/`。
- 审计：2026-11-11 全链路审计已完成，报告 `docs/AUDIT_2026-11-11.md`；发现并修复镜像 p75 退化、Freeze 空转、评测 Elo 尺度/shared bootstrap 等问题。
- 评测（**当前引擎 `c4c71014`**）：E1 四模型校准正式报告为 r2（2,688 局，协议 `0fbf28d2…`；与 r1 六对逐对一致）：`league-final-14858` 点估计冠军 **1086.53**（CI [1062.49, 1111.01]），`13806` 1068.45，`12311` 1055.92，`V1.1` 1000；final vs 13806 为 240–208，不显著。按用户规则只跑一轮、不做挑战赛。
- 旧结果：v2 榜（12311 1067.68）是过渡榜，协议与 E1 不同，不能混榜；11 模型 live-v1 锚定旧引擎 `c7b5f27d`，只作历史证据。
- 实机：V1.1 已完成 7 局夜间测试 + 6 次 D 阶段 canary；**E1 冠军 final 14858 已交接本地**（live 包 + 步骤见 `docs/sync/SERVER_TO_LOCAL_20261111.md`），当前 live pin 待本地更新，先跑 E-0 canary。本地实机 obs/action 差异见 `docs/sync/DECISIONS.md` D-026。
- 下一阶段：本地用 final 14858 做 E-0 canary（交接见 `docs/sync/SERVER_TO_LOCAL_20261111.md`）；下一轮长训使用 `config/league-stage2.json`（p75band + forbid-unfreeze + stage fork，2×H100，72×30min），**等待用户确认后再启动**。

详细状态：[docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)。

## 目录结构

| 目录 | 内容 | 进 Git |
|---|---|:--:|
| `engine/` | Rust 模拟器 `hsbg-engine`、37.0.2 卡表、规则测试 | ✅ |
| `hsbg/` | 观测/动作编码、网络、BC/PPO、并行采集、评测 | ✅ |
| `client/` | 实机桥接：`hsrl_agent/live/`（当前）、`bg_agent/`、`legacy_policy/`（历史） | ✅ |
| `workflows/` | 建库、迁移、训练、评测入口 | ✅ |
| `tools/` | 运维脚本（`sync_manifest.py`、`verify_workspace.py`、`live_archive_index.py`…） | ✅ |
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

### 服务器（NUS，训练/评测）

```bash
cd ~/projects/HSAI && git pull --rebase
sbatch deploy/slurm/<job>.sbatch      # 作业内不要再包 srun；GPU 以原始 Driver UUID 为准
squeue -u "$USER"; sacct -S today -X -o JobID,JobName,State,Elapsed
```

硬约束：登录节点约 1 GiB 虚拟内存，只做 Git/`sbatch`/`tail`/小 JSON；训练、评测、建库、编译只在计算节点。

### 三端同步

```bash
python tools/sync_manifest.py --side local  --out docs/sync/manifests/local.json     # Windows
python tools/sync_manifest.py --side server --out docs/sync/manifests/server.json    # NUS
```

两份 manifest 的 diff 就是 drift 报告；协作规则见 [docs/SYNC_MODEL.md](docs/SYNC_MODEL.md)
与 [docs/sync/README.md](docs/sync/README.md)。

## 文档导航

| 文档 | 说明 |
|---|---|
| [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) | 当前作业、模型结论、下一步 |
| [docs/SYNC_MODEL.md](docs/SYNC_MODEL.md) | 三端文件管理、真源、同步与协作规范 |
| [docs/INVENTORY_LOCAL.md](docs/INVENTORY_LOCAL.md) | 本地盘点（有什么、缺什么、待确认） |
| [docs/sync/SERVER.md](docs/sync/SERVER.md) | 交给服务器 Agent 的盘点任务书 |
| [docs/LIVE_V1_1_AUDIT.md](docs/LIVE_V1_1_AUDIT.md) | V1.1 实机审计、D 阶段改造与实机验证 |
| [docs/SIMULATOR_AUDIT_2026-11-09.md](docs/SIMULATOR_AUDIT_2026-11-09.md) | 37.0.2 模拟器规则审计 |
| [docs/LEADERBOARD.md](docs/LEADERBOARD.md) | 当前唯一排行榜规则与分数 |
| [docs/EVAL_DUEL.md](docs/EVAL_DUEL.md) | 2 model + 6 bot 对局协议 |
| [docs/nus/ENVIRONMENT.md](docs/nus/ENVIRONMENT.md) | NUS 环境、Slurm、GPU、恢复 |

## 项目边界

- 模型/训练协议/奖励由服务器侧负责；本地不改模型权重与训练语义。
- 实机桥接是 `public-approx-v1`：显式近似，不等价于模拟器精确状态。
- 不修改游戏目录、不执行历史未决输入账本、不伪造 plan/checkpoint/SHA。
- 没有用户明确要求时不提交 commit、不 push、不创建 PR。

来源：[AutoTavern](https://github.com/MCG-NJU/AutoTavern)、[HSBRSIM](https://github.com/Gallo13th/HSBRSIM)，历史原作者与许可证声明保留在原始证据与源码中。
