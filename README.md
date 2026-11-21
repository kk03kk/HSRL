# HSAI

HSAI 是《炉石传说：酒馆战棋》的强化学习与模型能力研究项目。Rust 模拟器用于训练和评测策略，Windows 客户端用于离线验证和实机桥接，NUS 集群承担训练、建库与大规模评测。

## 从哪里开始

| 要查什么 | 入口 |
|---|---|
| 项目结构、模块依赖和研究顺序 | [架构](docs/ARCHITECTURE.md) |
| 当前模型、部署和最近研究结论 | [项目状态](docs/PROJECT_STATE.md) |
| 训练、评测、回放与历史证据 | [按模块查文档](docs/README.md) |
| 双端协作、状态核对和同步 | [项目约定](AGENTS.md)、[运维手册](docs/OPERATIONS.md) |
| 冻结发布、模型登记和恢复 | [工程验收](docs/ENGINEERING.md) |

## 本地检查

```powershell
python ops/hsai_ops.py state
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe ops/check_docs.py
.venv\Scripts\python.exe tools/verify_workspace.py --skip-rust --json "artifacts/state/workspace-$(Get-Date -Format yyyyMMdd-HHmmssfff).json"
```

环境复建见[工程手册](docs/ENGINEERING.md#5-环境与可选外部源码)。Rust 构建与 GPU 验证在 NUS 计算节点进行。

## Windows 部署

已安装 Player150 部署工作树时，`client/start-ai-control.cmd` 打开控制台。当前工作状态和默认模型见[项目状态](docs/PROJECT_STATE.md)。离线环境检查使用 `client/start.ps1 -GameDir D:\Hearthstone`，历史决策与回放命令见[部署交接](docs/analysis/PLAYER150_PROMOTION_20261121.md)。

控制台读取独立工作树中的模型 pin。主检出的 `config/live/model.json` 是 14858 旧入口配置。实机状态来自公开日志与界面近似，验收以实际回执和逐动作证据为准。

## 目录速查

| 路径 | 内容 |
|---|---|
| `engine/`、`hsbg/` | Rust 模拟器、Python 观测、模型、训练与评测实现 |
| `workflows/`、`tools/`、`tests/` | 建库、训练、评测编排，审计工具与回归 |
| `client/` | 实机和离线入口、桥接、校准与兼容代码 |
| `ops/`、`deploy/` | 双端同步、发布、环境初始化与 Slurm |
| `config/`、`data/` | 引擎、运行配置与小型规则数据，原始大型回放按环境保存 |
| `docs/` | 活跃规范、状态、决策与日期化研究报告 |
| `artifacts/`、`records/` | 权重、训练、证据和归档，少数已审阅的研究摘要随 Git 保存 |

部分引擎源码源自 AutoTavern 和 HSBRSIM。原作者与许可声明保留在源码和历史证据中。
