# 三端文件管理与同步规范（Local ↔ GitHub ↔ NUS）

状态：**当前**。适用于本地 Windows 工作区、GitHub `kk03kk/HSRL`、NUS `~/projects/HSAI`。
未在服务器上核实的内容一律标 **待确认**，不得写成既定事实。

## 1. 三端职责

| 端 | 负责 | 不负责 |
|---|---|---|
| **本地 Windows**（`D:\HSAI`） | 代码开发、模拟器维护、实机桥接（`client/`）、真实游戏验证、本地小规模冒烟、证据留存 | 长训、建库、大规模评测 |
| **GitHub**（`kk03kk/HSRL`，`main`） | **唯一代码/文档同步中心**：源码、配置、卡表、测试、文档、清单 | 任何大文件（权重/库/数据/证据） |
| **NUS**（`~/projects/HSAI`） | 训练、评测、建库、数据生成、Slurm 作业、服务器侧 Agent | 实机操作、真实游戏证据 |

## 2. 唯一真源（single source of truth）

| 资产 | 真源 | 进 Git | 本地保留 | 服务器保留 |
|---|---|:--:|---|---|
| `engine/ hsbg/ client/ workflows/ tools/ tests/ deploy/` | GitHub | ✅ | 工作副本 | 工作副本 |
| `docs/`（当前有效文档） | GitHub | ✅ | 写 | 读 |
| `config/`（含 `config/live/` 实机 profile） | GitHub | ✅ | 写 | 读 |
| `data/`（卡表/池/饰品） | GitHub | ✅ | 写 | 读 |
| 训练产物（Trainer、checkpoint、曲线） | **服务器** | ❌ | 只拉需要的 | ✅ |
| 战团库 `library-v1`、bot 库 | **服务器** | ❌ | 仅用于实机的子集 | ✅ |
| 评测排行榜原始报告 | **服务器** | 仅摘要（<1 MB） | 摘要 | ✅ |
| 实机证据（`artifacts/runs/live-*`、截图、Power.log 归档） | **本地** | ❌ | ✅ | ❌ |
| `.venv`、Rust `target/`、OCR onnx | 各端自建 | ❌ | ✅ | ✅ |

> 一句话：**代码和说明走 GitHub，算力产物留在服务器，实机证据留在本地。**
> 任何一份文件只允许一个端拥有写权。

## 3. 同步规则

1. **代码只经 GitHub**：本地 → push → 服务器 pull；服务器上的改动必须回到 `main`（分支/PR），禁止两端长期各改一份。
2. **大文件不上 Git**：单文件 ≤ 5 MB、单提交 ≤ 20 MB。当前仓库 690 个文件 / 22 MB，是健康基线。
3. **每次大动作后跑清单**：换模型、换卡表、换引擎指纹、跑完一晚实机、跑完一轮评测，两端各执行一次
   `python tools/sync_manifest.py --side <local|server> --out docs/sync/manifests/<side>.json`，
   提交到 Git。**两份 manifest 的 diff 就是 drift 报告**。
4. **版本锚点先行**：任何评测/实机记录必须写明四个锚点——
   `engine-upstream-lock.json` 的源码指纹、`config/live/model.json` 的模型 SHA、
   模型目录里的 `manifest.json`、评测协议的 `protocol_sha256`。
5. **禁止**：把权重/库/数据/日志塞进 Git；在仓库里放压缩包；两端各自"就地"改同一个共享文件。

## 4. 目录规范

```text
<repo>/
  README.md  AGENTS.md  pyproject.toml  requirements.txt  .gitignore  .gitattributes
  engine/ hsbg/ client/ workflows/ tools/ tests/ deploy/ config/ data/ docs/
  records/        # 证据与历史（整目录不进 Git，除 records/research/long* 摘要）
  artifacts/      # 运行产物（整目录不进 Git；权重、库、数据集、runs、local）
```

- 根目录只允许上面 6 个文件，其余一律归入子目录。
- `docs/` 只放**当前有效**文档；被取代的文档移动到 `records/archive/<topic>-<YYYYMMDD>/`。
- `records/source-docs/` 放上游源码文档快照与论文；`records/evidence/` 放原始证据。
- `artifacts/` 命名：`runs/<side>-<purpose>-<YYYYMMDD>-<seq>`；模型目录必须含
  `model.pt + manifest.json + checkpoint.json + MODEL_CARD.md`；
  评测报告 `models/<lib>/reports/<YYYYMMDD>-<protocol>.json`。

## 5. 两个 Agent 的协作协议（不能直接对话）

固定目录 `docs/sync/`，**每端只写自己的文件**：

| 文件 | 写者 | 用途 |
|---|---|---|
| `docs/sync/LOCAL.md` | 本地 Agent | 本地盘点、状态、待确认清单 |
| `docs/sync/SERVER.md` | 服务器 Agent | 服务器盘点、状态、待确认清单 |
| `docs/sync/DECISIONS.md` | 双方追加 | 决策记录（`D-###`，只追加不删改） |
| `docs/sync/manifests/{local,server}.json` | `tools/sync_manifest.py` 生成 | 可比对的机器清单 |

每份 Markdown 头部固定五行：

```yaml
status: pending | draft | confirmed
owner: local | server
updated: YYYY-MM-DD HH:MM
base_commit: <git rev-parse --short HEAD>
```

**一轮交互**：A 在自己的文件里写问题/结论 → push → B `pull --rebase` → B 在 `SERVER.md`
回答、在 `DECISIONS.md` 追加决议 → push → A 复核后把 `status` 改成 `confirmed`。
冲突时以 `DECISIONS.md` 里编号最大的决议为准。

**红线**：不改写对方的文件；不在 `docs/sync/` 放二进制；不在服务器上执行历史未决输入账本。

## 6. 常见误区（本项目已经踩过的）

- 文档写了路径但文件在另一端：`artifacts/models/leaderboard-v1/`、`records/research/autonomous-2026-11-08/`
  在本地均不存在（评测在 NUS 跑）。→ 现在统一在 `docs/sync/LOCAL.md` 标 `server-only / 待确认`。
- 根目录堆压缩包：`v1.1-inference.tgz` 已归档到 `records/archive/model-transfer/` 并加 `*.tgz` 忽略。
- 把别的工具的配置当项目文件：`soclaas*.{json,toml}` 已移到 `records/archive/local-tooling-soclaas/`。
- 历史配置留在 `config/`：`config/historical/autotavern/*` 已移到 `records/archive/config-historical-autotavern/`。
