# 本地工作区盘点（2026-11-10）

状态：**当前**，只覆盖 `D:\HSAI` 本地；服务器内容标 **待确认**。
生成方式：目录遍历 + `git ls-files` + 关键文件 SHA256 + 导入冒烟，未运行训练/评测/游戏。

## 1. 结论

- **项目完整可用**：引擎可导入（`hsbg_engine.N_CARDS = 3984`）、V1.1 权重在位且 SHA 与 `config/live/model.json` 一致、卡表/实机 profile/OCR 模型齐全、实机链路端到端跑通过（2026-11-10 D 阶段 6 次 canary）。
- **没有因为旧项目被删除而缺文件**：唯一受影响的是两个 profile 里的 `provenance.source` 元数据（指向已删除的 `D:\HearthstoneAI\...`，仅作来源说明，运行时不读）；旧项目本身已在本仓库 `artifacts/archives/` 有归档副本。
- **本地缺 4 条文档引用路径**（都属于"服务器侧产物"，见 §5），已在文档里改标 `server-only / 待确认`。
- 体积集中在 `artifacts/`（24.0 GB）与 `records/`（5.2 GB），**都不进 Git**；Git 只承载 22 MB / 690 个源码与文档文件。

## 2. 顶层结构

| 目录/文件 | 体积 | 内容 | 进 Git | 真源 |
|---|---:|:--:|---|
| `engine/` | 4.6 MB / 152 文件 | Rust 模拟器 `hsbg-engine`（37.0.2.256589）、卡表、规则测试 | ✅ | GitHub |
| `hsbg/` | 3.4 MB / 137 文件 | 观测/动作编码、网络、BC/PPO、并行采集、评测 | ✅ | GitHub |
| `client/` | 6.1 MB / 247 文件 | 实机桥接：`hsrl_agent/live/*`（当前）、`bg_agent/`、`legacy_policy/`（历史） | ✅ | GitHub |
| `workflows/` | 2.4 MB / 121 文件 | 建库、迁移、训练、评测入口 | ✅ | GitHub |
| `tools/` | 0.4 MB / 42 文件 | 运维脚本（含新增 `sync_manifest.py`、`live_archive_index.py`、`audit_live_run.py`） | ✅ | GitHub |
| `tests/` | 1.9 MB / 93 文件 | 契约/规则/离线回归（含 `test_live_supervisor.py` 22 例） | ✅ | GitHub |
| `deploy/` | 28 文件 | NUS `setup`、`bootstrap_nus.sh`、26 个 `slurm/*.sbatch` | ✅ | GitHub |
| `config/` | 0.1 MB / 24 文件 | **仅当前配置**：`live/`（实机）、评测、训练、引擎锁 | ✅ | GitHub |
| `data/` | 7.6 MB / 20 文件 | 卡表（`bg_cards.json`、`card_zhcn.json`、`p2/cards_37.json`）、饰品、池、 archetype | ✅ | GitHub |
| `docs/` | 10 文件 | 当前有效文档（含本文件、`SYNC_MODEL.md`、`sync/`） | ✅ | GitHub |
| `records/` | 5.2 GB / 1477 文件 | 证据、迁移、源文档、研究摘要 | ❌（仅 `research/long*` 摘要例外） | 本地 |
| `artifacts/` | 24.0 GB / 12378 文件 | 模型、库、数据集、运行产物、本地实验 | ❌ | 见 §3 |
| `.venv/` | 417 MB | Python 3.13.5 + torch 2.13.0+**cpu**、numpy 2.4.0、rapidocr、onnxruntime | ❌ | 各端自建 |

## 3. `artifacts/` 明细

| 子目录 | 体积 | 内容 | 备注 |
|---|---:|---|---|
| `models/` | 9.2 GB | `v1.1/`（64.7 MB，实机用）、`library-v1/`（3.4 GB，39 个 cohort + `index.csv`/`SHA256SUMS.txt`）、`historical/`（3.9 GB 旧训练）、`duel/`、`best/`、`nus-1080772/` | **真源在服务器**；本地只保证 `v1.1/` 与实机所需子集 |
| `datasets/` | 3.5 GB | firestone、friend-reference、legacy-autotavern/hsdata/catalogs、library-metadata | 训练/评测输入，真源服务器 |
| `imports/` | 3.7 GB | 5 个 NUS 产物导入包（`nus-1080772*`、`nus-duel-eval-1092254`、`nus-long-002-prechecks`） | 服务器产物落地副本 |
| `archives/` | 3.2 GB | 旧项目归档：`autotavern/`、`hearthstoneai/`、`hearthstonerl/`、`hsbg/` | **旧项目已删除，这是本地唯一副本** |
| `runs/` | 1.4 GB | 实机 `live-*`（昨夜 7 局 + D 阶段 6 次）、duel 冒烟、`slurm/` | 实机证据 = 本地真源 |
| `local/` | 2.6 GB | 本地实验/冒烟（`forced-tier-smoke*` 2.1 GB、rehearsal×10、audit-night×2、`d-phase/`） | 可清理候选，未删 |
| `client/` | 30 MB | OCR onnx（det 9.5 MB / rec 20.3 MB / cls 0.6 MB）+ manifest | 各端自建，可由脚本重新下载 |

## 4. 关键资产核对（本地实测）

| 项 | 结果 |
|---|---|
| 引擎导入 | ✅ `import hsbg_engine`，`N_CARDS = 3984` |
| V1.1 权重 | ✅ `artifacts/models/v1.1/model.pt`，`68a8a587…0848e`，与 `config/live/model.json` 一致 |
| 训练/评测产物 tarball | ✅ `v1.1-inference.tgz` 内 `model.pt` 同 SHA（已归档到 `records/archive/model-transfer/`） |
| 实机 profile | ✅ `config/live/autoplay-255790.json`（`review_state=reviewed`，client 2560×1600，dpi 144） |
| 卡表 | ✅ `data/bg_cards.json`、`data/card_zhcn.json`、`data/p2/cards_37.json` |
| 战团库 | ✅ `artifacts/models/library-v1/{manifest.json,index.csv,SHA256SUMS.txt,cohorts,models}`（39 cohort） |
| GPU | ⚠️ 本地 `torch.cuda.is_available() = False`（CPU-only，符合分工：训练在服务器） |
| Python 环境隔离 | ⚠️ `.venv` **不是完全隔离**：`pyvenv.cfg` 里 `include-system-site-packages=false`，但 `.venv/Lib/site-packages/external-runtime.pth` 把 `D:\Anaconda\Lib\site-packages` 加进了 `sys.path`。因此 `torch 2.13.0+cpu` 实际来自 Anaconda base，venv 内只装了 numpy/onnxruntime/opencv/hsbg-engine/editable 包 |
| 依赖告警 | ⚠️ `pip check` 唯一告警：`numba 0.61.0` 需要 `numpy<2.2`，实际 numpy 2.4.0。numba 来自 Anaconda base，被 `datashader/hvplot/pandas/xarray` 的可选依赖带入；**HSAI 代码不使用 numba/datashader/hvplot**（仓库内 0 处引用），故不处理。若将来要可复现环境，应新建纯净 venv 并按 `requirements.txt` 安装 torch |
| 游戏目录 | ✅ `D:\Hearthstone` 存在（本轮盘点时游戏进程未启动，仅检查目录） |

## 5. 缺失 / 待确认（不要在服务器核实前当事实）

| 路径（被文档引用） | 本地 | 判断 |
|---|---|---|
| `artifacts/models/leaderboard-v1/reports/20261109-live-v1.json` | ❌ | **应只在服务器**（11 模型 live-v1 评测在 NUS `1098876/1098877` 跑）→ 待确认 |
| `artifacts/models/leaderboard-v1/manifest-2026-11-09.json` | ❌ | 同上 → 待确认 |
| `records/research/autonomous-2026-11-08/TRAINING_REVIEW.md` | ❌ | 服务器研究报告 → 待确认（本地只有 `research/{firestone-20261106,long1080772,long1085131}`） |
| `artifacts/runs/p50-forced-tier-long-20261108-135823/` | ❌ | 服务器 p50 长跑输出 → 待确认 |
| `records/archive/docs-2026-11-09/`（README/AGENTS 曾引用） | ❌ | 归档目录此前根本不存在；本次新建 `records/archive/` 并已修正文档 |
| 服务器 `~/projects/HSAI` 的 commit、模型、库、作业产物 | — | **完全未核实**，见 `docs/sync/SERVER.md` |

## 6. 本次本地整理（可逆移动，未删任何数据）

| 动作 | 从 | 到 |
|---|---|---|
| 归档历史配置（25 个） | `config/historical/autotavern/*` | `records/archive/config-historical-autotavern/` |
| 归档本机另一工具配置 | `soclaas-model-catalog.json`、`soclaas.config.toml` | `records/archive/local-tooling-soclaas/` |
| 归档模型传输包（54 MB，SHA 已核对） | `v1.1-inference.tgz` | `records/archive/model-transfer/v1.1-inference-20261109.tgz` |
| 归档旧工作区快照 | `workspace.json`、`workspace-tree.txt` | `records/archive/workspace-snapshots-20261106/` |
| 归档论文 | `docs/research/papers/autotavern.pdf` | `records/source-docs/papers/autotavern.pdf` |
| 忽略规则 | — | `.gitignore` 增加 `*.tgz`、`soclaas*.{json,toml}` |

移动后根目录只剩 `README.md / AGENTS.md / pyproject.toml / requirements.txt / .gitignore / .gitattributes`。

## 7. 清理记录（2026-11-10 已执行）

实际释放 **2,240 MB**（`artifacts` 23,958 MB → 21,718 MB）。明细见
`records/archive/cleanup-20261110.txt`；原则：**只删可重建的 .pt 快照、仓库副本和重复回放**，
证据/权重/库/旧项目归档一律保留。

| 目标 | 体积 | 理由 |
|---|---:|---|
| `artifacts/local/forced-tier-smoke*` 的 `.pt`（14 个） | 2.1 GB | **已删除**：同一 7357 快照在 `library-v1`、`nus-1080772`、`7357-update-restore` 各有一份；json 摘要保留 |
| `artifacts/local/publish-*`、`github-release-check` | 64 MB | **已删除**：仓库快照副本，Git 历史里已有 |
| 8 份重复 rehearsal | 26 MB | **已删除**：保留 `rehearsal-v1.1-round2-fix` 与 `rehearsal-v1.1-final-audit2` |
| `.pytest_cache/` | 0 MB | **已删除** |
| `artifacts/archives/*`（3.2 GB） | 3.2 GB | **保留**（旧项目唯一副本） |
| `artifacts/models/historical`（3.9 GB） | 3.9 GB | **保留**（旧训练权重，本地无法重建；如要腾空间可只留 manifest） |
