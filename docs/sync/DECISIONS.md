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

status: accepted
evidence: `docs/SYNC_MODEL.md` §2

源码/配置/卡表/文档以 GitHub `main` 为真源；训练产物与战团库以 NUS 为真源；
实机证据以本地为真源。任何资产只有一个端拥有写权。

## D-002 · 2026-11-10 · local · 大文件不进 Git

status: accepted
evidence: 当前仓库 690 文件 / 22 MB；`*.tgz *.pt *.zip` 已在 `.gitignore`

单文件 ≤ 5 MB、单提交 ≤ 20 MB。权重/库/数据/日志/证据一律不进 Git。
需要用版本号对齐时，用 `docs/sync/manifests/*.json` 记录 SHA256。

## D-003 · 2026-11-10 · local · 清单脚本作为跨端对齐工具

status: accepted
evidence: `tools/sync_manifest.py`（stdlib only，登录节点安全）

两端各跑一次 `--side local|server`，把 JSON 提交到 Git；两份 manifest 的 diff 即 drift 报告。
`pins` 里同一资产必须同 SHA；`artifacts.*` 只比对规模与存在性。

## D-004 · 2026-11-10 · local · 协作文件所有权

status: accepted
evidence: `docs/sync/README.md`

`LOCAL.md` 只由本地写、`SERVER.md` 只由服务器写、`DECISIONS.md` 双方追加、
`manifests/*.json` 由脚本生成。禁止改写对方文件；禁止在 `docs/sync/` 放二进制。

## D-005 · 2026-11-10 · local · 根目录与 docs 的边界

status: accepted
evidence: 本次整理后根目录仅 6 个文件；`config/historical/*` 已移入 `records/archive/`

根目录只允许 `README.md AGENTS.md pyproject.toml requirements.txt .gitignore .gitattributes`。
`docs/` 只放当前有效文档；被取代的进 `records/archive/<topic>-<YYYYMMDD>/`。
`config/` 只放当前配置；历史配置进 `records/archive/`。

## D-006 · 2026-11-10 · local · 缺失路径的归属判定

status: accepted
evidence: `docs/INVENTORY_LOCAL.md` §5

`artifacts/models/leaderboard-v1/`、`records/research/autonomous-2026-11-08/`、
`artifacts/runs/p50-forced-tier-long-20261108-135823/` 在本地不存在，
判定为**服务器侧产物**，等待 `manifests/server.json` 与 `SERVER.md` 确认；
确认前文档一律标 `server-only / 待确认`。

**2026-11-10 已解决**：服务器盘点确认 `leaderboard-v1` 报告与 2026-11-09 清单已归档
（引擎 `c7b5f27d`）、`autonomous-2026-11-08/TRAINING_REVIEW.md` 与 `p50-forced-tier-long` 均在服务器；
本地文档已按 D-016 更新。

## D-007 · 2026-11-10 · local · 实机证据不外发

status: accepted
evidence: `artifacts/runs/live-*` 体积 1.4 GB

实机运行目录（含 Power.log 归档、截图、逐决策 trace）只留本地；
必要时只在 `docs/sync/LOCAL.md` 里给出路径与 SHA 摘要，不上传服务器/仓库。

## D-008 · 2026-11-10 · local · 本地 Python 环境不隔离，不做重塑

status: accepted
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

## D-010 · 2026-11-10 · server · 服务器盘点基线

status: accepted
evidence: `docs/sync/SERVER.md`、`docs/sync/manifests/server.json`、`git rev-parse HEAD`

服务器 `HEAD=224cb49a819896ad4e1b5dcca3be3da3c6256ed3`，已包含 D 阶段提交 `d10e352`。
工作树 dirty 来自运行中的联赛工作副本（1 个已跟踪修改 + 10 个未跟踪文件），不是未同步的
本地改动。接受 D-001–D-005 的所有权、大文件、清单和根目录规范；D-006 的“待确认”由本轮
盘点转为具体结论。

## D-011 · 2026-11-10 · server · 旧 live-v1 结果不得与新引擎混用

status: accepted
evidence: `records/archive/models-2026-11-09/models/leaderboard-v1/reports/20261109-live-v1.json`、`.../manifest-2026-11-09.json`、`artifacts/runs/elo-8-1099277/report-corrected.json`

归档的 11 模型 live-v1 报告（`protocol_sha256=46e7013d…`）与 2026-11-09 榜单 manifest
内嵌引擎均为 `c7b5f27d6f40d1d2`；当前引擎为审计修复后的 `c4c71014e3d41760`。
因此旧 live-v1 排名只能作历史证据；任何“当前引擎棋力”结论必须用 `c4c71014` 重跑。
当前有效的正式评测是 8 模型 Elo `1099277`（协议 `837231e0…`）和 4 模型联赛 Elo `1107182`
（协议 `6bf5cae3…`）。

## D-012 · 2026-11-10 · server · 战团库真源按用途分开

status: accepted
evidence: `docs/sync/SERVER.md` §4、`artifacts/libraries/league-eot-v1.1-20261109/manifest.json`、`records/archive/libraries-2026-11-09/MANIFEST.json`

服务器不保存本地那份 39 cohort 的 `library-v1`。服务器训练/评测真源是
`artifacts/libraries/league-eot-v1.1-20261109`（`driver=model_eot_league`，4 源）；实机真源
仍是本地的 `library-v1`。两套库用途不同，不互相复制、不混用。旧 bot 库 1.9 GB 已归档，
MANIFEST 标记可用库为 0。

## D-013 · 2026-11-10 · server · 运行中作业冻结

status: accepted
evidence: `squeue -u $USER`、`artifacts/runs/league-v1.1-36h-20261109/train/latest.json`、`deploy/slurm/league_36h.sbatch`

`1102669 hsai-league-36h` 与 `1097487 no-shell` 正在运行，不得取消、不得改训练代码/库/依赖。
36h 作业是 72 段循环，每段启动新 Python 进程；`config/league-36h.json`、`deploy/slurm/league_36h.sbatch`、
`workflows/combine_league_library.py`、`workflows/eval_three_models_elo.py` 和 `hsbg/az/train/personal.py`
的修改属于运行中工作副本，本轮不提交；作业结束后再决定是否回到 `main`。

## D-014 · 2026-11-10 · server · 清单盲区与 CRLF 假差异

status: accepted
evidence: `tools/sync_manifest.py`（v2）、`tools/sync_compare.py`、`docs/sync/manifests/local.json`

已按本条实施：pins 同时输出 worktree `sha256` 与 **Git blob**（`git hash-object`，CRLF 无关）；
统计加入 `artifacts/exports` 与 `records/`；跳过 `target/ __pycache__ .venv` 等构建目录；
`library-v1`、`league-eot-v1.1-20261109`、`v1.1/trainer.pt` 改为带 `scope` 的 pin（local/server/both）。
受影响四个 CRLF 文件在 v1 清单里显示假差异，重跑 v2 后应消失。

## D-015 · 2026-11-10 · server · 清理只列候选，不自动执行

status: accepted
evidence: `du -sh artifacts/* records/*`、`docs/sync/SERVER.md` §5

服务器侧候选：`artifacts/exports` 889 MB、`artifacts/local/v1.1-*.tgz` 333 MB、
`records/archive/v1.1-panel-*` 85 MB、旧 smoke/failed 面板目录约 100 MB。
**执行条件**：`1102669` 结束后，按 D-009 只删可重建物（导出包/重复 tgz/旧面板），
权重、联赛库、`records/` 原始证据一律不删，并在目录留 `cleanup-<date>.txt`。

## D-016 · 2026-11-10 · local · 按服务器盘点修正本地文档口径

status: accepted
evidence: `README.md`、`docs/PROJECT_STATE.md`、`docs/INVENTORY_LOCAL.md`、`docs/sync/LOCAL.md` §7

三处硬修正：
1. **旧 live-v1/p50 结论降级为历史证据**（引擎 `c7b5f27d`），当前引擎结论改用 `1099277`（1,792 局）
   与 `1107182`（3,072 局）。
2. **"当前没有训练作业"作废**：`1102669 hsai-league-36h` 正在跑（预计 2026-11-11 17:01 结束）。
3. 本地缺失的 4 条路径改为已核实的归档/服务器路径；补记"本地无 V1.1 `trainer.pt`，续训以服务器为准"。

## D-017 · 2026-11-10 · local · 清单 v2 + 自动 drift 报告

status: accepted
evidence: `tools/sync_manifest.py`（schema v2）、`tools/sync_compare.py`、`docs/sync/manifests/compare-20261110.json`

`sync_manifest.py` 只出事实，`sync_compare.py` 只判差异：pins 优先比 Git blob，
`scope=local|server` 的 pin 允许在另一端缺失，tree 只比文件数（容差 35%），
artifact/records 只比存在性与量级。退出码 0=一致，1=硬漂移。
**待服务器执行**：v2 重跑 `manifests/server.json` 后再 `sync_compare.py`，四个 CRLF 假差异应消失。

## D-018 · 2026-11-10 · local · v2 清单比对通过（含 compare 工具修正）

status: accepted
evidence: `docs/sync/manifests/{local,server}.json`、`docs/sync/manifests/compare-20261110.json`、`tools/sync_compare.py`

服务器按 D-017 重跑 v2 清单（`SERVER.md` §10，HEAD `ceb1959`）后，本地复核结果：**`result: consistent`**。

服务器当时看到 4 条 `DRIFT` 的原因是 `sync_compare.py` 的一个缺陷：`both_blob` 判定把
"单边 `scope` pin"（本地无 `trainer.pt`、服务器无 `library-v1`）也算进了 `all()`，
于是整体退回 worktree SHA256 比较，CRLF 差异重新出现。已修正：只用**两侧都存在**的 pin
判定 blob 可用性。修正后：

- 15 条 pins：全部 `OK` 或 `OK(scope)`，无硬漂移；
- trees：`engine` 152/153、`hsbg` 76/76、`tests` 45/45、`tools` 31/31、`docs` 19/19、
  `workflows` 91/93、`deploy` 28/33（差值来自运行中联赛的冻结副本，属预期）；
- `artifacts/`、`records/`：只有量级差异（各自运行证据），不参与判定。

结论：三端代码/配置/卡表/文档**已对齐**；剩余差异都是"按用途分工"或"各自运行证据"。
