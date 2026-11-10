# DECISIONS —— 三端管理的决议记录

只追加、不改历史。一条决议一个编号，格式：

```text
## D-00X · YYYY-MM-DD · <side: local|server|both> · <title>
status: proposed | accepted | superseded-by D-00Y
evidence: <文件路径或命令>
```

冲突时以编号最大的 `accepted` 条目为准。

---

## D-001 · 2026-11-10 · local · 三端唯一真源划分

status: proposed
evidence: `docs/SYNC_MODEL.md` §2

源码/配置/卡表/文档以 GitHub `main` 为真源；训练产物与战团库以 NUS 为真源；
实机证据以本地为真源。任何资产只有一个端拥有写权。

## D-002 · 2026-11-10 · local · 大文件不进 Git

status: proposed
evidence: 当前仓库 690 文件 / 22 MB；`*.tgz *.pt *.zip` 已在 `.gitignore`

单文件 ≤ 5 MB、单提交 ≤ 20 MB。权重/库/数据/日志/证据一律不进 Git。
需要用版本号对齐时，用 `docs/sync/manifests/*.json` 记录 SHA256。

## D-003 · 2026-11-10 · local · 清单脚本作为跨端对齐工具

status: proposed
evidence: `tools/sync_manifest.py`（stdlib only，登录节点安全）

两端各跑一次 `--side local|server`，把 JSON 提交到 Git；两份 manifest 的 diff 即 drift 报告。
`pins` 里同一资产必须同 SHA；`artifacts.*` 只比对规模与存在性。

## D-004 · 2026-11-10 · local · 协作文件所有权

status: proposed
evidence: `docs/sync/README.md`

`LOCAL.md` 只由本地写、`SERVER.md` 只由服务器写、`DECISIONS.md` 双方追加、
`manifests/*.json` 由脚本生成。禁止改写对方文件；禁止在 `docs/sync/` 放二进制。

## D-005 · 2026-11-10 · local · 根目录与 docs 的边界

status: proposed
evidence: 本次整理后根目录仅 6 个文件；`config/historical/*` 已移入 `records/archive/`

根目录只允许 `README.md AGENTS.md pyproject.toml requirements.txt .gitignore .gitattributes`。
`docs/` 只放当前有效文档；被取代的进 `records/archive/<topic>-<YYYYMMDD>/`。
`config/` 只放当前配置；历史配置进 `records/archive/`。

## D-006 · 2026-11-10 · local · 缺失路径的归属判定

status: proposed
evidence: `docs/INVENTORY_LOCAL.md` §5

`artifacts/models/leaderboard-v1/`、`records/research/autonomous-2026-11-08/`、
`artifacts/runs/p50-forced-tier-long-20261108-135823/` 在本地不存在，
判定为**服务器侧产物**，等待 `manifests/server.json` 与 `SERVER.md` 确认；
确认前文档一律标 `server-only / 待确认`。

## D-007 · 2026-11-10 · local · 实机证据不外发

status: proposed
evidence: `artifacts/runs/live-*` 体积 1.4 GB

实机运行目录（含 Power.log 归档、截图、逐决策 trace）只留本地；
必要时只在 `docs/sync/LOCAL.md` 里给出路径与 SHA 摘要，不上传服务器/仓库。

## D-008 · 2026-11-10 · local · 本地 Python 环境不隔离，不做重塑

status: proposed
evidence: `.venv/pyvenv.cfg`（include-system-site-packages=false）、`.venv/Lib/site-packages/external-runtime.pth`、`pip check`

`.venv` 通过 `external-runtime.pth` 借用 `D:\Anaconda\Lib\site-packages`，因此 `torch 2.13.0+cpu`
来自 Anaconda base。`pip check` 唯一的 numba↔numpy 告警同样来自 base（numba 由
datashader/hvplot 的可选依赖带入）。HSAI 代码不引用 numba/datashader/hvplot（仓库 0 处），
因此**不卸载、不降级 numpy**，以免破坏已跑通的实机链路。
若将来需要可复现环境：新建纯净 venv，按 `requirements.txt` 显式安装 torch，再跑 `tools/verify_workspace.py`。

## D-009 · 2026-11-10 · local · 清理只删可重建物

status: accepted
evidence: `records/archive/cleanup-20261110.txt`（26 项，释放 2,240 MB）

删除范围仅限：可重建的 `.pt` 快照副本、仓库快照副本（publish / release-check）、
重复 rehearsal、缓存目录。权重（含 `artifacts/models/historical`）、战团库、旧项目归档、
实机证据、`records/` 一律不删。任何删除都必须在该目录留下 `cleanup-<date>.txt` 清单。
