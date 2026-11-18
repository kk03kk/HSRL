# DECISIONS —— 项目决议记录

只追加、不改历史。一条决议一个编号，格式：

```text
## D-00X · YYYY-MM-DD · <side: local|server|both> · <title>
status: proposed | accepted | superseded-by D-00Y
evidence: <文件路径或命令>
```

冲突时以编号最大的 `accepted` 条目为准。

> **2026-11-12 迁移说明**：本文件原位于 `docs/sync/DECISIONS.md`，随双 Agent 协作协议
> 退休而移到 `docs/DECISIONS.md`。D-001 … D-027 里的 `evidence:` 路径**保持原样不作修改**
> （只追加不改历史），其中指向 `docs/sync/**`、`docs/SYNC_MODEL.md`、`docs/INVENTORY_LOCAL.md`、
> `tools/sync_manifest.py` 的引用已经归档到 `records/archive/sync-protocol-20261112/`；
> 恢复方法见该目录的 `README.md`。历史上 `side` 字段的 local/server 现在表示"该决议由哪一端
> 提出"，不再表示所有权边界。

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

## D-019 · 2026-11-10 · local · 架构调整：服务器统一开发，本地专注实机

status: accepted（由 D-020 接受）
evidence: `docs/ARCHITECTURE.md`、`docs/sync/SERVER_HANDOFF_20261110.md`、`AGENTS.md` §1/§10/§11

1. **写权重划**：`engine/ hsbg/ workflows/ deploy/ data/ tools/ tests/`（除 `test_live_*`）、
   `pyproject.toml`、`requirements.txt`、`README.md`、`AGENTS.md`、`docs/`（除本地那几份）→ **服务器**；
   `client/ config/live/ artifacts/ records/ docs/sync/LOCAL.md docs/live/** docs/analysis/**` → **本地**。
2. **本地不再**做模拟器开发、编译、规则审计与训练栈测试；**唯一例外**是引擎升级时的
   Windows 轮子"部署构建"（服务器无法产出 Windows 产物）。
3. **实机日志回传**：`live-<date>-<seq>.tar.gz` + `.sha256` → 服务器 `artifacts/imports/live-<date>/`，
   **不进 Git**。
4. **引擎发版记录**：由服务器维护 `docs/sync/ENGINE_RELEASES.md`（`日期 | commit | source_hash | db_info | 摘要`），
   本地据此决定是否重建轮子。
5. 本次架构文档由本地起草（过渡期），自本次提交起 `docs/` 归服务器维护；规则审计结论不变（不改模拟器）。

## D-020 · 2026-11-10 · server · 接受架构调整并完成 P0 回归

status: accepted
evidence: `docs/sync/SERVER.md` §11、`docs/sync/ENGINE_RELEASES.md`、`artifacts/runs/server-regression-20261110-01/`

服务器接受 D-019 的写权划分和 `docs/ARCHITECTURE.md`：`engine/ hsbg/ workflows/ deploy/ data/ tools/ tests/`
（除 `test_live_*`）及模拟器/训练类文档归服务器；`client/ config/live/ artifacts/runs/live-* records/`
仍归本地。P0 已完成：

- 全量回归 `1109107`：cargo 942 passed / 0 failed / 1 ignored；pytest 9 failed / 0 errors；
  `verify_workspace` 仅 python gate 失败，原因均为测试仍用旧动作口径（`max=40`、`move_cap=1/2/3/99`），
  而生产策略已强制 `max=80 / move_guard=1 / cap=5 / repeat=2 / forbid_cancel=1`。
- 当前扩展 `build_info().source_hash=c4c71014e3d41760`，`build_is_current=True`；
  `ENGINE_RELEASES.md` 首行已记录；服务器无 Windows 构建机，Windows 轮子仍由本地部署构建。
- v2 清单重跑：pins=15，`sync_compare.py` 返回 `consistent`（exit 0），仅剩本地 manifest 的 HEAD 过期提示。
- 下一步：修 9 个测试口径（服务器 own `tests/`），P1 等本地 `live-*.tar.gz` 到达后做规则校验。

## D-021 · 2026-11-10 · server · 动作规则测试口径修复，回归全绿

status: accepted
evidence: `tests/test_move_guard.py`、`tests/test_parallel_collect.py`、`tests/test_curriculum_long.py`、
`tests/test_az_ppo.py`、`artifacts/runs/server-regression-20261110-02/`、`docs/sync/SERVER.md` §11.1

修复提交 `1c6de55`。9 个失败全部是测试仍用旧动作口径，已按生产强制规则对齐：
`max_actions_per_turn=80`、`move_guard=True`、`move_cap=5`、`move_repeat_limit=2`、`forbid_cancel=True`；
关闭守卫与 MCTS `search_fn` 改为断言拒绝；cap/repeat 序列、lockstep meta 和无 Move mask 用例同步更新。

复跑 `1109179`（计算节点）：pytest 0 failed / 0 errors，cargo 942 passed / 0 failed / 1 ignored，
`verify_workspace.py` 全 gate PASS。服务器全量回归自此为绿色；未改训练/评测生产代码。

## D-022 · 2026-11-10 · server · 最新联赛模型保存、gauntlet 评测与实机选择

status: accepted
evidence: `artifacts/models/league-latest-20261110-step012311/`、`artifacts/runs/gauntlet-12311-128-20261110/`、`docs/sync/SERVER.md` §12

保存 `1102669` 的 `step_012311`（累计 1,175,791,607；model SHA `5069aa62…`，trainer SHA `60dcaf05…`）。
`1109300` 用 128 局/对打 4 个 elo4 模型：4 对全胜，合计 305–207（59.57%，CI 55.27–63.87%）；
对 `milestone-1033972345` 为 69–59（53.91%，CI 45.31–62.50%，不显著）。
服务器选择 `league-latest-12311` 作为本机实机探索模型；保守回退为 `milestone-1033972345`。
权重与报告在 `artifacts/`，不进 Git；本地通过 `league-latest-12311-live.tgz`（SHA `f5f28bac…`）下载。

## D-023 · 2026-11-10 · server · 接受本地 Windows 部署构建（c4c71014）

status: accepted
evidence: `config/live/engine.json`、`client/tools/build_engine.ps1`、`ad4778d`

本地完成 S3 部署构建：Windows wheel SHA `524d6871e436d38750edb669b9d4afd7e281c334adf44086a60cf73ad9266e9f`，
`expected_source_hash=c4c71014e3d41760`，`matches_server=true`。服务器 `build_info().source_hash`
同为 `c4c71014e3d41760`，`db_info=(3983, "37.0.2.256589", "37.0.2")`。两端引擎锚点一致；
服务器不修改 `config/live/`，本地不需要重新审计模拟器。

## D-024 · 2026-11-11 · server · 单一排行榜 v2 + 简化入榜规则

status: accepted
evidence: `docs/LEADERBOARD.md`、`artifacts/models/leaderboard-v2/leaderboard.json`、`docs/sync/SERVER.md` §13

用户决定只保留一张榜：旧 `leaderboard-v1` 文件归档到
`records/archive/leaderboards-20261111/`，旧 8/4 模型报告保留为证据/锚点但不再作为当前榜；
弱模型以后删除/归档。

简化规则：旧模型分数冻结在 4 模型 Elo；新模型跑 128 局/对 gauntlet，用 fixed-opponent
Bradley-Terry MLE 求唯一分数 + 5,000 bootstrap；排名按点估计，CI 同时报告。
`league-latest-12311` 得 **1067.68**（95% CI [1036.97, 1099.46]），5 模型第一；
对旧第一 `milestone-1033972345` 直接 128 局为 69–59（53.91%，CI 跨 0.5），所以是
用户规则下的点估计第一，不是显著超越。`13806` 尚未评测，暂不入榜。

## D-025 · 2026-11-11 · server · 13801 删除、14556 保存与训练结束前冻结

status: accepted
evidence: `artifacts/cleanup-20261111.txt`、`artifacts/models/league-latest-20261111-step014556/manifest.json`、`artifacts/exports/league-latest-14556-live.tgz`、`docs/sync/SERVER.md` §14

按用户要求删除 `league-latest-20261111-step013801` 及其 live 导出，只保留 13806；清理清单记录 SHA/字节。
保存当前最新训练快照 `step_014556`（累计 `1,506,838,367`；model `14d7c4d5…`，trainer `91afb637…`）并生成推理包。
36h 作业结束前冻结训练代码/库/权重，仅更新文档；作业结束后保存 final，并按用户确认的新排名规则 v1
（联合 Bradley-Terry + 排名区间 + 冠军挑战 + 连接锚点）提交 E1 校准。

## D-026 · 2026-11-11 · server · 实机观测/动作投影审计发现（需本地取证）

status: pending-local
evidence: 服务器对 `client/hsrl_agent/live/` 的静态审计（文件/行号见下）；本会话未操作真实游戏

服务器审计在本地拥有的实机桥接代码中发现以下 train/live 不一致。它们不是服务器可改的代码，
请本地 Agent 用真实 Power.log/决策轨迹逐项确认，并回传证据；确认后再决定修法。

1. **对手已见战团/最后对手恒为空**：`logs.py` 只初始化/清空 `seen_boards`、`last_opponent`，
   仓库内没有赋值点；`facts.py:80-81` 读取它们。训练侧 `engine/.../obs.rs:410-417` 使用
   `board_shown/scouted` 真实填充。影响 7×7×31=1519 个浮点与 `OPP_SEEN_BOARD_LEN`。
2. **pending kind 分类**：`adaptive.py:85-91` 把 ChooseOne(3)、Target(4)、Timewarp(8) 都归为
   Discover(2)；训练侧 `obs.rs:217-227` 有 9 类 one-hot，且 ChooseOne 的 `choice_items` 走
   特殊路径。影响每个选择菜单的 9 维 one-hot，ChooseOne 还影响 choice 浮点块。
3. **任务/奖励从未重建**：`facts.py:112` 只在任务未激活时清空 `view['quests']`，没有构造分支；
   `owned_rewards` 无构造点。训练侧 `obs.rs:210-213,238-243` 有真实任务进度和任务/奖励卡槽。
4. **对手摘要 all-or-nothing**：`facts.py:69-83` 只有在恰好 7 个对手且每个都有 health/tier 时
   才写入 `view['opponents']`；任一缺失就整体退化为 `[None]*7`，`global.alive` 回退为 8。
   训练侧保留死亡对手槽位和公开的 leaderboard 信息。
5. **MODEL_* 标签缺失**：`facts.py:125-128` 查 `MODEL_MAX_GOLD_BONUS` 等标签，但
   `client/.../data/game_tags_255790.json` 没有这些键；`adaptive.py:37-38,61-78` 把
   `max_gold_bonus/hand_queue_len/effects_len/tavern_atk/tavern_hp/chronum` 和多个 item 私有
   状态固定为中性值。`readiness.py` 已把其中 6 个列为 unresolved。
6. **动作可用集差异**：训练侧 Freeze 是 toggle（全冻结时可解冻），实机 `actions.py:68-78`
   在全冻结时不暴露 action 2；训练/评测 >16 选择会截断，实机 >16 直接 Blocked；
   实机 trinket/Timewarp 可用性没有按 gold/Chronum 过滤，引擎 `recruit.rs:2393-2406` 会过滤。
7. **bf16/fp32 差异**：训练 leaf 用 bf16，评测/实机 fp32；服务器实测同模型 greedy argmax
   翻转率约 **0.8%–1.0%**（V1.1 0.78%，12311 1.0%，1800 个真实观测/模型）。不是阻塞项，
   但实机结果应标注“fp32 解码”，不能声称与训练采样逐动作一致。

请求本地：
- 用一局含战斗、任务、ChooseOne/目标菜单、冻结/解冻、>16 选择的真实日志，跑
  `AdaptiveProjector` 并导出 `approximation.json` 中实际被置零/近似的字段；
- 对比同一时刻引擎等价状态的 `obs_std.decode_blocks` 数值（至少对手棋盘、任务、pending kind、
  choice 块）；
- 给出每项的出现频率与是否可在真实日志中重建；不能重建的明确写“近似”。
服务器侧不会改 `client/`；确认后由本地决定修法与实机验收顺序。

## D-027 · 2026-11-11 · server · E1 单轮规则、final 冠军、实机交接

status: accepted
evidence: `docs/AUDIT_2026-11-11.md` §7–8、`docs/LEADERBOARD.md`、
`docs/sync/SERVER_TO_LOCAL_20261111.md`、`artifacts/models/e1-four-20261111/leaderboard.json`

用户 2026-11-11 明确：
1. **E1 只跑一轮**：4 模型 × 6 对 × 448 局（ordered56）即完成排名；报告 valid/eligible
   后立即停止。不得为“确认复现”再跑第二轮，不得为冠军单独跑挑战赛；只有 invalid/
   工程错误/协议缺陷才重跑，且先请示。
2. **冠军 = `league-final-14858`**（E1 点估计第一 1086.53；与 13806 的 240–208 不显著，
   只称“点估计冠军”）。实机直接使用 final。
3. 服务器已删除 challenge 入口/配置；正式榜为
   `artifacts/models/e1-four-20261111/leaderboard.json`，协议
   `0fbf28d2…`。r2 与 r1 六对逐对一致，r2 作为正式报告。
4. 实机交接：`docs/sync/SERVER_TO_LOCAL_20261111.md`；live 包
   `artifacts/exports/league-final-14858-live.tgz`
   SHA256 `32ccdabd…`，model SHA `ba7e8bbc…`；本地更新 `config/live/model.json`
   后先跑 E-0 canary；D-026 的 obs/action 差异仍需本地取证。
5. 下一轮长训未启动；`config/league-stage2.json` 等只作准备，等待用户确认。

## D-028 · 2026-11-12 · both · 单 Agent 双环境管理，退役双 Agent 协作协议

status: accepted
evidence: `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/OPERATIONS.md`、
`docs/architecture/TRANSPORT.md`、`ops/hsai_ops.py`、
`records/archive/sync-protocol-20261112/README.md`

用户 2026-11-12 授予完整规划与执行权，要求把双端管理彻底整理成单 Agent 模式。

1. **一个决策者，两个执行环境。** 服务器上不再有自主 Agent；它是被 `ssh` 驱动的执行端。
2. **职责不再靠"写权"隔离**，改为按"什么只能在这里产生"划分：实机只在本地，GPU 只在
   服务器。两端都保留全量源码副本。
3. **退役 `docs/sync/` 散文协议**：LOCAL.md / SERVER.md / README.md / 两份交接文档 /
   manifests / `tools/sync_manifest.py` / `docs/SYNC_MODEL.md` / `docs/INVENTORY_LOCAL.md`
   全部归档到 `records/archive/sync-protocol-20261112/`（不删除，git 历史完整保留）。
   `DECISIONS.md` 上移为 `docs/DECISIONS.md`，`ENGINE_RELEASES.md` 上移为
   `docs/nus/ENGINE_RELEASES.md`。
4. **CODEOWNERS 式约束改为自检提示**：`client/tools/check_ownership.py` 保留，但语义从
   "禁止本地修改服务器代码"降级为"提示这个文件通常由哪端维护"。

## D-029 · 2026-11-12 · both · 传输架构：git over SSH 为主，GitHub 降为异地归档

status: accepted
evidence: `docs/architecture/TRANSPORT.md`、`ops/hsai_ops.py`、
实测：`model-pull` 67.9 MB / 6.8 s = 10.0 MB/s；`trainer.pt` 243 MB / 13.7 s = 17.7 MB/s

1. **代码同步 = git over SSH**。本地加远端
   `server ssh://hsai-nus/home/h/hengzhi/projects/HSAI`，双向 fetch/push。
   服务器仓库设 `receive.denyCurrentBranch=updateInstead`：工作区干净时推送会更新工作区，
   **有未提交改动时推送被拒绝**。已在隔离沙箱实测拒绝行为，并在真实仓库实测正向更新
   （`304cb0b..5683aa0`）。漂移不再可能被一次"看起来成功"的推送掩盖。
2. **GitHub 保留但降级**：只作**异地归档**，在里程碑点推送，不再作为日常同步通道或
   Agent 间沟通渠道。理由：本地与服务器两份副本都在会坏的硬盘上，需要一份不在那两台
   机器上的副本；而绕第三方同步比直连慢、且多一个故障点。
3. **大文件永不进 Git**：权重/库/回放走 `ssh cat` 流式传输 + 逐文件 SHA256 校验。
   实测链路约 140 Mbit/s，1 GB 约 1 分钟。
4. **不用 scp 覆盖源码**；不用 rsync（本地未安装，且两端没有"目录必须一致"的诉求）。

## D-030 · 2026-11-12 · both · 服务器 12 个未提交文件定性：真实工作，验证后提交

status: accepted
evidence: commit `304cb0b`；`tests/test_mirror_band.py` 6 passed；
`hsbg/az/train/mirror_snapshots.py`、`workflows/league_health.py`、
`deploy/slurm/league_stage2_48h.sbatch`

退役服务器 Agent 留下 5 个已跟踪改动 + 7 个未跟踪文件，在服务器上飘了约 24 小时。
逐个读 diff 后定性为**真实工作**，不是垃圾：

1. `turnband:<lo-hi=pct,...>` 回合相关分位带镜像采样（带严格校验与 6 个测试）；
2. `validate_league_checkpoint.py --forbid-unfreeze`（默认 0，评测与训练/实机对齐）；
3. `workflows/league_health.py` 分段健康门禁（纯标准库，不 import torch）；
4. `deploy/slurm/league_stage2_48h.sbatch`（94×30min + 周期验证 + 健康门禁）；
5. 三份 stage2 配置与 1,026 行策略指南；`data/firestone_replays/`（741 MB）与
   `data/real_library/`（15 MB）加入 `.gitignore` 留在服务器。

**没有任何正式作业在使用这些文件，也没有启动 48h 训练。** `turnband` 是否采用、
48h 是否启动，仍需用户批准（D-027 第 5 条不变）。

## D-031 · 2026-11-12 · local · 登录节点规格更正：4 核 / 16 GB

status: accepted
evidence: `ssh hsai-nus 'nproc; free -g'`、`ops/hsai_ops.py doctor`

旧文档（`AGENTS.md` §3、`README.md`、`docs/PROJECT_STATE.md` §8）写"登录节点约 1 GiB
虚拟内存"。实测为 **4 核 / 16 GB**，旧数字是错的。

结论方向不变：4 核且多人共享，仍然不允许在登录节点跑 pytest / torch / 编译 / 建库 /
训练。`ops/hsai_ops.py` 新增登录节点守卫，把这条从"文档要求"变成"默认拦截"。

## D-032 · 2026-11-12 · local · 实机链路 UTF-8 决定性输出

status: accepted
evidence: commit `eec4896`、`workflows/run_4159_long.py`、
`tests/test_4159_long.py::test_login_source_precheck_uses_stdlib_without_slurm_or_audit_plan`

该测试在 Windows 上长期失败：子进程以 `-S` 通过管道输出 JSON，中间含中文 `scope`
字段，Python 用 cp936 编码而测试按 UTF-8 解码；解码异常让 `result.stdout` 变成 `None`，
最终报成误导性的 "JSON object must be str, bytes or bytearray, not NoneType"。

修法是在脚本入口把 stdout/stderr 钉死为 UTF-8（Linux 上本来就是 UTF-8，无行为变化）。
`ops/hsai_ops.py` 同样处理，避免 Agent 看到的服务器输出变成乱码。

## D-033 · 2026-11-12 · local · 隧道自愈：DPAPI 凭据 + SSH_ASKPASS

status: accepted
evidence: `ops/tunnel-auth.ps1`、`ops/tunnel.ps1`、`docs/architecture/TRANSPORT.md` §6、
`tests/test_ops_guard.py::test_ps1_files_with_non_ascii_must_be_utf8_with_bom`

用户 2026-11-12 指示"启用隧道自愈"，把唯一的人工步骤消掉。

1. **做法**：用户执行一次 `ops\tunnel.ps1 -Setup`，在安全提示里输入跳板机密码；密码用
   Windows DPAPI（`CurrentUser` 作用域）加密写 `%LOCALAPPDATA%\HSAI\tunnel.cred`。
   `ops\tunnel-auth.ps1` 在 Setup 时编译 `C:\ProgramData\HSAI\hsai-askpass.exe`，
   ssh 通过 `SSH_ASKPASS` + `SSH_ASKPASS_REQUIRE=force` 取用。
2. **Agent 边界**（AGENTS.md §7 同步收紧）：Agent 绝不读取、打印、传输该凭据，也不把它
   写进任何日志或文档；只调用封装好的 `ops/tunnel.ps1` / `hsai_ops.py tunnel`。
   凭据状态只能通过 `-Status` 得知（返回长度，不返回内容）。
3. **实测硬约束**（`OpenSSH_for_Windows_9.5p2`）：
   - `SSH_ASKPASS_REQUIRE=force` 必须显式设置，否则 Win32-OpenSSH 不会使用 askpass；
   - askpass 的 **exe 路径必须全 ASCII**，`C:\Users\黄\...` 下会
     `CreateProcessW failed error:2`；故 exe 放 `C:\ProgramData\HSAI\`，密文留用户目录；
   - 含中文的 `.ps1` **必须带 UTF-8 BOM**，否则 Windows PowerShell 5.1 按 ANSI 解码会
     打乱行结构（曾把 C# here-string 整段当 PowerShell 代码）。已加回归测试。
4. **验证**：用假密码走通全链路——助手编译成功、DPAPI 往返一致、密文不含明文、
   ssh 确实调起助手并得到 `Permission denied (password)`、`-Forget` 清理干净。
   真实密码由用户自行登记，Agent 全程不接触。
5. **代价与回退**：跳板机密码从此常驻本机（加密态），风险模型从"每次手输"变成"本机 +
   本 Windows 账户被攻破即失守"。`-Forget` 随时退回手动模式。NUS 改密后自愈失败会报
   `Permission denied`，重跑 `-Setup` 即可。
6. **实施中发现的三个新约束**（已写进 `ops/tunnel.ps1` 与 TRANSPORT §6.3）：
   - **`sjump` 没有 shell**：认证成功后返回 `This account is currently not available.`
     这是 `nologin` 而非密码错误。**验证密码必须用真实的端口转发**，不能用
     `ssh hsai-bastion 'echo ok'`——第一版 `-Setup` 就是因此误报"密码错误"，其实密码是对的。
     已改为 `-Setup` / `-Verify` 都起临时转发探通再拆。
   - **`ssh -f` 配管道在 Windows 上死锁**：fork 出的子进程继承 stdout，调用方永远等不到
     EOF（实测挂满 120 s）。改用 `Start-Process -WindowStyle Hidden` + 输出重定向到
     `C:\ProgramData\HSAI\tunnel-<port>.log`，靠轮询端口判成败。
   - 含中文的 `.ps1` 必须 UTF-8 BOM（见第 3 条）。
7. **真凭据实测（2026-11-12 14:45）**：`-Verify` 临时端口演练通过；`-Force -Port 2222`
   杀掉现有隧道后**无人值守重建成功**，`doctor` 七点全通过。目标状态达成。

## D-034 · 2026-11-12 · local · 训练主线改为「真人状态注入 + 分阶段中后期课程」

status: accepted
evidence: `docs/LATE_GAME_ENTRY.md`、`docs/analysis/HUMAN_GAP_20261112.md`、
`tools/build_human_entry_bank.py`、`tools/smoke_human_inject.py`、
`deploy/slurm/smoke_inject.sbatch`、`config/league-entry.json`

用户 2026-11-12 授予全项目自主权（取消时长与审批上限），要求"找到最快、最可靠、最有效
的方式让模型真正变强"，并最看重 33 个流派攻略与真人专家数据的利用率。

1. **诊断结论**（实测，非推断）：模型 T12 板面 Σ(atk+hp)=1,135，真人 ≥8k MMR 吃鸡局
   T12 p50=**5,620**（0.20×）；960 局里 **0 局**达到 ≥3 张 S 级核心卡；训练对手
   `human_board_p=0`（全是自对弈镜像）。即"从没在真人有牌可打的中后期状态上做过决策"。
2. **纠正旧结论**：`humancurve-*` 库所谓"后期对手 2.9×/6.4× 真人"是把**我们自己的牌**
   按比例放大攻血得到的，牌面构成仍是我们的弱阵容，且参考分布没有 MMR 下限——连数字目标
   也只有 ≥8k 真人 p50 的 0.59 倍。它不能教阵容，只能教"数字大就赢"。**改用真人完整帧**。
3. **新数据资产**：`data/real_library/firestone-bank-1000/train-m8k.jsonl`（1,168 行，
   ≥8k MMR、train split、只过滤不改数值）+ `eval-m8k.jsonl`（86 行，evaluation split，
   与训练不相交）。逐回合 p50 见 `docs/LATE_GAME_ENTRY.md` §2。
4. **新训练机制**（`hsbg/az/train/parallel_collect.py`，默认全关 = 逐位等价旧配方）：
   - `entry_p/entry_turn_min..max`：`T0` 前 learner 由 bot 代打且**不产生记录**，到 `T0`
     把真人完整状态（战团+手牌+counters+酒馆等级）注入 learner 席，之后策略接管；
   - `human_board_min_turn` + `mirror_max_turn`：`T0` 之后对手换成真人战团，镜像关在
     `T0` 之前；
   - `learner_final_str_*`：每轮 learner 终局板面 Σ，按终局回合分桶——直接对真人同回合
     p50 求比值的**学习曲线**。
5. **前置门禁**：换库必跑 `sbatch deploy/slurm/smoke_inject.sbatch`。实测接受后战团逐字段
   100% 一致、3–5 ms/次；T9 会被引擎 pending choice 100% 挡住（32/32），故入场重试
   必须"一回合只试一次"（已修）。
6. **验收四件套**（刻意不设单一指标）：入场局名次、板面强度对真人曲线比值、核心卡覆盖
   （≥1/≥2/≥3）、前期名次回归。卖牌次数/终端数值可被无意义刷高，只作辅证。



## D-035 · 2026-11-12 · local · 入场课程的病灶是升本节奏：启用 tier-target 奖励

**背景（新增证据，推翻上一轮的方向判断）**

- 用 `tools/inspect_gap_panel.py` 把 `gap-e3c-20261112/panel.json` 逐回合拆开看：
  E3 与 champion 在 T2–T7 的板面与 tier **几乎完全一致**，崩的是 T8–T12 的升本节奏
  ——learner T10 平均 tier `4.62`，champion `5.25`，真人 `5.52`，慢了**整整一级**；
  板面强度随之在 T14 掉到 champion 的 0.70 倍（比值 0.107 → 0.075）。
- 因此"把入场回合提前"（F1，入场 T6–8）是错方向：F1 四件套里**三项全退**——
  `mean_rank 1.177→1.417`、T10 之后 tier 卡在 4.5–4.7、`≥1 S` 核心卡 0.292→0.170。
  F1 已撤销，现场保留在 `artifacts/runs/human-gap-f1-20261112/`。
- 机制解释：入场课程在 `T0` 之前把 learner 交给 bot 且**不产生 learner record**，
  于是"T8 之前升到 5 级"从未进入梯度；而 `board-target` 奖励按回合比板面，
  升本当期掉板面 → "别升本、买属性"成了局部最优。

**决定**

1. **启用 `tier-target` 奖励**（`hsbg/az/train/tier_target.py`，早就实现、从未启用）：
   每个新回合起点读一次 tier，低于目标扣 `coef*0.02`，首次达标一次性加 `coef*0.15`，
   目标表用真人曲线取整 `7:4,9:5,12:6`，`coef=1.0`。这是**有意的策略偏置**，
   不是势函数塑形（不满足最优策略不变性），所以只在本阶段课程里开。
2. **工程配套**：`personal.py` 的 `OPTIONS` 加 4 个 `tier-target-*` 键（否则 config
   JSON 会被当成"未知配置项"直接拒掉）；`runlog.py` 的 `INJECTION_KEYS` 加 13 个
   `tier_*` 键，`analyze_entry_run.py` 加 `t_hit% / t_rw / bt_rw` 三列。
   理由就是 `board_target` 踩过的坑：**配了但没进 history，会静默跑完一整段**。
3. **两个配方**：G2 = `entry-p 0.5`（一半整局 / 一半入场）+ board-target 0.30 +
   tier-target 1.0；G3 = `entry-p 1.0` + 同样两个奖励（= E4 的单变量对照）。
4. **验收工具落地为代码**：`tools/accept_gap_panel.py` 把四件套做成逐项 PASS/FAIL
   （排名 / 固定回合板面比值 / 核心卡覆盖 / 前期回归），并单独打印 T8/T10/T12 的
   tier 差作为机制变量。判据写在工具里，避免每次靠肉眼挑好数字。
5. **同步约束重申**：`training_source_digest()` 覆盖 `hsbg/az/train/*.py` 与
   `workflows/train_entity_sync.py`，所以**段与段之间不能推这两个目录的改动**；
   本轮为此先撤销只跑了 3 分钟的 G1，再推 tier-target 改动，然后重开 G2/G3。

## D-036 · 2026-11-13 · local · 过程奖励被打穿：先修环境（对手强度），再修奖励

**新证据（决定性负结果，96 局/模型）**

- 打开 `tier-target` 后训练 208 轮（G2）与 146 轮（G4），tier 目标确实达成
  （T12 平均 5.98 / 5.93，base 5.36，真人 5.78），但整局面板**四件套全 FAIL**：
  `mean_rank 1.177→1.458/1.833`、T12 板面 `1193→803/855`（−33%）、
  `≥1 S 核心卡 0.292→0.083/0.146`。
- 42 轮时的读数（`gap-g23a`）是"机制对了、板面没坏"，到 208 轮才露出退化 ——
  **≤50 轮的短窗口读数系统性偏乐观**。

**机制**：`tier-target` 每达标一级给 `+0.15`，不要求板面跟随；升本花 9–11 金币、
当回合板面必然下降，而 `board-target` 的当回合惩罚量级只有 0.15 的几分之一。
PPO 于是收敛到捷径「把金币全砸在升级上、不买随从」。

**更根本的原因**：`base` 在 7 个引擎 bot + 真人 p50 战团下面已是
`mean_rank 1.177 / win 0.854` —— **名次奖励饱和、没有梯度**。饱和的主奖励无法纠正
任何可独立满足的过程奖励。

**决定**

1. **不再往饱和奖励上叠过程奖励**。`tier-target` 保留在代码里但默认 0，本轮不启用；
   若将来再用，必须加"板面对真人曲线比值达到下限"的门控（未实现，列入待办）。
2. **先修环境**：新作业 B3（`1124567`）= 混合课程 + `human-board-sample auto`
   （T≤8 p50 / T9–12 p75 / T13+ p90）+ **纯名次奖励**。对手变强 → 弱阵容直接掉名次，
   名次信号重新有信息量。
3. **H1（`1124485`）继续**：它在 G2 配方上加了 `comp-target 0.5`。
   该奖励要求**持有**核心卡，天然与"不买随从"相反，可以检验它能否抵消 tier 捷径。
4. **验收纪律**：一律跑满 6 段预算再判定；短窗口读数只当"机制是否生效"的冒烟信号，
   不得据此宣布有效。

## D-037 · 2026-11-13 · local · 流程事故：往 main 推「会改 digest 的代码」杀掉了三个在跑的作业

**事故**：04:20 左右为了推文档，`hsai_ops.py push` 把当时**已提交在 main 上**的
代码改动（`f36ef91`：`aggregate_tier_target` 补门控字段、`comp_target` 反刷分门槛）
一起推到了服务器。`training_source_digest()` 覆盖 `hsbg/az/train/*.py` 与
`workflows/train_entity_sync.py`，于是 G5/F4/G6 在各自下一个分段边界全部因
`ValueError: 训练代码已变更；请另开实验，不能伪装为精确续训` 而 **FAILED**：

| 作业 | 已跑完 | 结局 |
|---|---|---|
| `1125152` G5 | 63 轮 / 59:48 | FAILED（续训守卫） |
| `1125153` F4 | 62 轮 / 59:52 | FAILED（续训守卫） |
| `1125235` G6 | 20 轮 / 20:42 | FAILED（续训守卫） |

快照都保留着（`keep-snapshots 3`），没有丢数据；损失的是这三个臂的剩余训练时间。

**根因**：我把"改代码"和"推文档"混在同一条 main 上。push 推的是 main 的 HEAD，
不是"我这次想推的那几个 commit"，所以文档提交会把挂着的代码提交一起带过去。

**新规则（硬性）**：

1. **有作业在跑时，digest 相关改动一律不进 main**：先提交到侧分支
   `pending-digest/YYYYMMDD`，等所有作业结束再 merge 到 main 并推送。
   digest 相关 = `hsbg/az/train/*.py` 与 `workflows/train_entity_sync.py`。
2. **push 前先看清带过去的是什么**：`git log --oneline origin/main..HEAD` 逐条过一遍，
   确认没有 digest 相关文件。
3. 安全的目录（可以随时推）：`docs/`、`config/`、`tools/`、`tests/`、
   `workflows/eval_human_gap.py`（评测侧，不在 digest 里）。
4. 事故后**不重试原作业**（那会伪装成精确续训），一律用**新目录重新开跑**；
   本轮已按此把三个臂改成 `-b` 版本重开（`1125421/1125423/1125425`）。

**副产物**：重开时顺手修了 `tier-gate-ref` 缺 `T7` 的缺陷
（配置层传 `7:150,8:220,...`，不必改代码）。

## D-038 · 2026-11-13 · local · 运维禁令：`sh --` 里不要用 shell 变量拼 sbatch 参数

**事故**：为一次开三个臂，我写了
`for c in "g5 config/league-entry-g5.json" ...; do set -- $c; sbatch --export=ALL,HSAI_LE_OUT=...$1...,HSAI_LE_CONFIG=$2 ...; done`。
结果是：**每轮循环跑了两次**（一次 `$1` 正确、一次 `$2` 变成 `config`），
共提交 6 个作业；更糟的是**留下来的三个作业里 `HSAI_LE_CONFIG` 根本没生效**，
`run.env` 里写着 `config=config/league-entry.json`（= 默认 S1 配方），
也就是说 g5b/f4b/g6b 跑的是**错的配方**，白烧了约 6 分钟 GPU。

**根因**：`ops/hsai_ops.py sh` 这一层的引号/展开语义与本地 PowerShell 叠加，
`$1`/`$2` 既不是可靠的远端变量，也不是稳定的本地字面量。

**新规则（硬性）**

1. `sbatch` 一律**一条命令一个作业**，`--export` 里的路径**写字面量**，
   不用 `$1/$2/$(...)` 或 for 循环拼装。
2. 提交后**必须先看 `run.env`**（`config=`、`segments=` 两行）确认配方正确，
   再让它跑；不要等一小时后才发现跑错。
3. 需要批量提交时，在**本机**生成完整的 bash 脚本，用 `git` 送到服务器执行
   （脚本本身进版本库、可复查），而不是在 `sh --` 里现场拼。

**本轮结果**：g5b/f4b/g6b 全部 scancel + 删目录，按规则 1 重开为
`entry-g5c-20261113`（`1125428`）与 `entry-g6c-20261113`（`1125429`），
`run.env` 已确认分别指向 `config/league-entry-g5.json` / `-g6.json`。

## D-039 · 2026-11-13 · local · 发现并修正 anchor 日程：我所有的 fork 都比 champion 被约束 5 倍

**发现**：`workflows/train_entity_sync.py` 的 anchor 日程是

```python
anchor_beta_i = anchor_beta if i < anchor_iterations else anchor_beta_tail
```

- **champion 的配方**（`config/league-36h.json`）：`anchor_beta=0.1, anchor_iterations=250,
  tail=0.02` → **250 轮后就降到 0.02**；它一共训练了 5,316 轮，也就是**绝大部分时间在 0.02**。
- **我这一夜所有从联赛 fork 的臂**（B3/C1/D1/F4/G5/G6/G7/G8/H1/H2/H3/H4）：
  `anchor_iterations=15108`（这是 S1 配方里的值），而每个臂只跑 100–200 轮
  → **全程停在 0.1，永远到不了 0.02**。

也就是说：这些 arm 的 KL 锚是 champion 稳态的 **5 倍**，每步都被拉住。
这解释了一个反复出现的困惑——**为什么十几个臂的改动都"有方向、没幅度"**：
D1 名次只从 1.333 回到 1.177、H2 覆盖虽然翻倍但板面只掉 12%……

反过来，E 系列（`anchor_beta=0.02` 恒定）**确实动了**（tier 超额达成、板面 −33%），
只是被 `tier-target` 的捷径带偏。**两代结果的差异有相当一部分来自 anchor，而不是奖励本身**。

**决定**

1. 主线配方改用 **champion 自己的日程**：`anchor-beta 0.1` + `anchor-iterations 250`
   + `anchor-beta-tail 0.02`。已生成 `config/league-entry-h6.json` 并启动（`1126696`）。
2. 今后任何"fork 后没有变化"的实验，**先核对 anchor 日程**再归因给奖励。
3. 记录口径：`history.jsonl` 里若只有 `anchor_kl` 递增而无策略位移，
   第一嫌疑就是 anchor 没衰减；报告实验时必须写明 `anchor-beta/-iterations/-tail` 三元组。

## D-040 · 2026-11-13 · local · 修正 D-039：`anchor-iterations=250` 在本轮跑不到，等于没改

**D-039 的修法无效**：`train_entity_sync.py` 里判断用的是**本轮运行的迭代序号** `i`：

```python
trainer.config.anchor_coef = (args.anchor_beta if i < args.anchor_iterations
                              else args.anchor_beta_tail)
```

而这一夜的每个臂只跑 **120–200 轮**，所以把 `anchor_iterations` 从 15108 改成 250
**仍然到不了阈值** —— `H6/H7/H8` 实际还是全程 `anchor_beta=0.1`。
（实测：H6 收在 `step_015016`，即 158 轮 < 250。）

**正确的做法**是把阈值压到本轮必然越过的位置：`anchor-beta 0.02` +
`anchor-iterations 1` + `anchor-beta-tail 0.02` —— 即**恒定 0.02**，
也就是 champion 在 5,316 轮里绝大部分时间所处的稳态。

已按此重开两个臂：
- `1128610 H10` = H2 配方（comp-target 1.0，min_hits 2）+ anchor 0.02；
- `1128615 H11` = H7 配方（+board-target 0.30）+ anchor 0.02。

**教训**：改日程类参数时必须核对"阈值是否在本轮会被越过"，不要只看配置文件里的数字。

## D-041 · 2026-11-13 · local · 接力链的段数与时限：`gpu` 分区最多 5 段

**事故**：`H2` 长训链（`1125994`，8 段 × 20 min）在 `03:00:29` 被 Slurm **TIMEOUT** 杀掉。
原因是 `--minutes 20` 是**软上限**：每段会先跑完当前 iteration 再退出，实测每段 20 → 22–25 分钟，
再加每段加载模型的 ~40 秒。8 段 × ~22.5 min ≈ **180 min**，正好撞上 `gpu` 分区的 3 小时上限。

**规则（硬性）**：

1. `--partition=gpu`（MaxTime 3 h）上，**最多 5 段 × 20 min**，并显式 `--time=03:00:00`；
   要更长就用多次接力（每轮结束后我手动交下一棒，`HSAI_LE_RESUME=1` 接同一目录）。
2. 每棒结束后**先看 `sacct` 的 State**：`TIMEOUT`/`FAILED` 都要在下一棒里 `--resume` 续上，
   而不是当成"跑完了"——快照还在，续跑是合法的。
3. 段数与时限的对应关系写进提交命令的注释里，避免下次再撞。

## D-042 · 2026-11-13 · local · 隧道重建固定为「5 秒预检 + 脱离式启动 + 有界轮询」，禁止盲等

**背景**：用户指出"每次重连 VPN 后 Agent 都会卡很久"。复盘查出**三个独立原因**，
此前被我混为一谈（详见 `docs/nus/TUNNEL_FAST_PATH.md`）：

1. `tunnel.ps1` **成功之后包装进程不退出**（日志已打印 `up`），前台等它 → 工具超时；
2. `ssh -N` 是包装进程的**子进程**，`job_kill` 包装进程会**连隧道一起杀掉**（实测两次）；
3. **VPN 没生效时 TCP 假通**（`sjump:22` 返回 True 但 banner 永不到），
   脚本失败路径要跑满 `Wait-Tunnel` 8 次 ≈ **50 s**，这段时间全是空等。

**固定流程（每步都有硬上界）**

| 步 | 命令 | 上界 | 判据 |
|---|---|---|---|
| A 预检 | `ssh -o ConnectTimeout=5 -o BatchMode=yes -o PubkeyAuthentication=no -o PasswordAuthentication=no hengzhi@sjump.comp.nus.edu.sg true` | 6 s | `Permission denied` = 路通；`timed out during banner exchange` = VPN 没好，**停** |
| B 起隧道 | `Start-Process powershell -WorkingDirectory 'D:\HSAI' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','ops\tunnel.ps1','-Force' -WindowStyle Hidden` | 0.07 s 返回 | 不等待、不杀 |
| C 轮询 | `powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status`（每 1.5 s，≤16 次） | 25 s | 见 `UP` 立刻进 D |
| D 探活 | `python ops/hsai_ops.py probe` / `state` | 5 s | `host=xlogin2` 且 `IN SYNC` |

**2026-11-13 12:55 实测**：A 0.34 s → B 0.07 s → C **6.0 s 报 UP** → D 通过，**合计 ≈7 s**。

**硬性禁令**：不对隧道做 `job_kill`/`Stop-Process`；不给 `tunnel.ps1` 接管道或 `*> 文件`；
不在同一个调用里等 `-Force` 返回；VPN 未确认前不重建；任何一步超过 25–30 s 无结果就掐掉看现场。

**附带澄清**：`state` 报 `DRIFT` 且两端 `dirty=0` 通常是**本地有未推送提交**，
不要据此做 `reset`/`checkout`。

## D-043 · 2026-11-13 · local · 专家先验转为正式输入：升级曲线断在 T5 之后，覆盖奖励反压低等级

status: accepted
evidence: `docs/analysis/EXPERT_PRIORS_20261113.md`、`docs/analysis/COMP_SHARED_CORES_20261113.md`、
`artifacts/runs/gap-priors-base-20261113/`、`artifacts/runs/gap-priors-h10-20261113/`、
`tools/analyze_comp_structure.py`、`config/league-entry-m1.json`

用户（版本 37.0.2.256589 的高水平实战者）书面回复了 23 条专家问题，并明确 9 条为**确认先验**；
`config/strategy_cards.json` 的 33 个流派与 S/A/B 评级**由其本人编写**，不再作为"待怀疑的外部攻略"。

**当场测到的三件事（96 局真人场，同种子）**

1. 升本曲线 T2=1.92 / T5=3.05 与高手标准（T2 二本、T5 三本）一致，之后停住：
   T8 4.33 / T10 4.44 / T12 4.79（真人 4.82 / 5.52 / 5.78）；`upgrade` 使用率
   0.48(T6) → 0.29(T8) → 0.02(T9) → 0.00(T11/T12)。
2. 每回合结束金币 p50 在 T5–T12 **全部为 0.0** ⇒ 从不存钱，升五本所需的 9 金币从未攒到。
3. `comp-target`（h10）把 T10 平均等级压到 **4.08**（champion 4.44）：该奖励让模型留在
   低本刷便宜核心卡。而结构分析显示 **15/33 个流派最高核心在 6 本、16/33 在 5 本、
   0/33 能在 4 本成型**（铜须被 12 个流派列为核心、巴琳达 7 个）——
   **模型在结构上拿不到真核心**，这才是"覆盖≠强度"反复出现的机制解释。

**决定**

1. 先验只进**配置 / 课程 / 指标**，不进动作掩码；不硬编码升本回合、卖牌次数、阵容标签
   （用户红线：目标是学会经济与成长的因果，不是执行攻略）。
2. 下一阶段主线候选 `M1`（`config/league-entry-m1.json`，未启动）：入场收窄到 **T8 + p75
   + 只注入 tier≥5 的真人帧**，`comp-target` 降 0.3 且 T10 起计分，`human-board-min-turn` 9→7。
   `entry-min-tier` 尚未登记进 `personal.OPTIONS`，接线前该配置 fail-loud 拒绝启动。
3. 更正 §5.24 的 churn 结论：`action_mix_by_turn` 的分母缺陷已修（`mean/used_rate/mean_used`
   三口径），真实值是 T3/T4/T5 每局卖 **0.04/0.01/0.01**，卖牌是后期现象。
4. `keep-start-board`（K1/K2）按专家意见降级：V1 不再作候选主线，V2 只作可选临时约束；
   代码与 4 条单测留在侧分支。
5. 本项目此后**凡"游戏正解"类问题先问用户**，凡"模型是否做到 / RL 会不会学歪"类问题才做实验。
## D-044 · 2026-11-13 · local · 评测前必须 pin 快照；多模型面板的 `--export` 用 `\;`

status: accepted
evidence: 作业 `1131577`（FAILED: `checkpoint not found: .../step_015001/model.pt`）、`1131581`（COMPLETED，两模型面板 + `accept_gap_panel` 全链路通过）

**事故**：想用 `entry-h10` 的 `step_015001` 做面板，但 `keep-snapshots 3` 已经把它轮转删除
（当时最新是 `step_015031`）。`eval_human_gap.sbatch` 在加载阶段就 `checkpoint not found` 退出——
失败很快、没有浪费 GPU，但白占一次排队。

**规则（硬性）**

1. **面板/评测引用快照前先 `ls` 该 run 的 `train/snapshots`**，并把要评的快照
   `cp -r` 到 `artifacts/runs/pin-<arm>-<step>/`（已有的 `pin-*` 目录就是这个用途）。
   直接引用 `train/snapshots/step_XXXXX` 会在作业排队期间被轮转掉。
2. 多模型面板（`HSAI_GAP_MODELS=a=...;b=...`）必须让 sbatch 收到**字面分号**：
   在 PowerShell 里把整条命令放进**双引号字符串**、分号写成 `\;`
   （`"...HSAI_GAP_MODELS=base=x/model.pt\;h10=pin-y/model.pt,..."`）。
   裸写会被 PowerShell 当语句分隔符；单引号会被 `hsai_ops.py sh` 丢掉。
3. 提交后**先看 `run.env` 与 `panel.log` 头**再等结果（D-038 的同一条纪律）。
## D-045 · 2026-11-13 · local · 两个长臂的最终判定为负：覆盖类与板面类奖励都换不来强度；主线转向"五本入场课程"

status: accepted
evidence: `artifacts/runs/gap-h10l1-humfield-20261113/`、`artifacts/runs/gap-h10l1-bot-20261113/`、
`artifacts/runs/entry-h10-20261113`（COMPLETED 2:03:07）、`artifacts/runs/entry-l1-20261113`（COMPLETED 1:59:47）

两个作业按用户要求跑到自然结束，双场 96 局面板（快照 h10@15031 / l1@14961，最终快照均为 15046）：

| 臂 | 配方 | 真人场 mean_rank（配对 CI） | bot 场 mean_rank | T10 板面 | T10 tier | `≥1 S` | 前期 |
|---|---|---|---|---|---|---|---|
| base | champion | **5.4375** | **1.177** | 1.000 | 4.44 | 0.292 | — |
| h10 | comp-target 1.0 + anchor 0.02（174 轮） | 5.8125 (+0.375 [−0.03,+0.77]) | **1.354** (+0.177，CI [+0.02,+0.35] **显著变差**) | −11.4% | **4.07** | **0.579** | −6~−14% |
| l1 | h10 + `terminal-strength-coef 0.3`（128 轮） | 5.6354 (+0.198 [−0.19,+0.57]) | 1.167 (−0.010，中性) | −7.6% | 4.22 | 0.438 | −8~−15% |

**三条结论**

1. **`comp-target` 的代价被钉死在"等级"上**：覆盖率 0.292→0.579（近 2×）的同时，
   T10 平均等级 4.44→**4.07**、板面 −11%、名次变差（bot 场显著）。机制清楚：它奖励"持有核心卡"，
   而模型能拿到的核心卡集中在低本，于是它选择留在低本刷卡。
   **H10 在 83 轮时还是"覆盖 1.95×、板面持平"，到 174 轮就翻负**——短窗口乐观的又一次实例。
2. **直接奖励"终局板面厚度"（`terminal-strength-coef 0.3`）无效**：L1 的 T10 板面 −7.6%，
   名次中性。板面类奖励至此**四次证否**（Σ 口径 / `board-target` / `terminal-strength` / 加倍）。
3. 两臂都出现**前期回归**（T3–T7 −6~−15%），说明这些后期奖励在 anchor=0.02 的自由度下
   连早期行为一起带偏了。

**决定**：主线转向 **课程而非奖励**——`M1/M0`（`config/league-entry-m1.json` / `-m0.json`）：
入场收窄到 **T8 单回合 + p75**，`comp-target` 与 `tier-target` **全关**，
`human-board-min-turn 7`；两臂唯一差别是 `entry-min-tier`（M1=5 只用五本以上的入场帧，M0=0 不过滤）。
6k 入场库的注入门禁已通过（`smoke-inject-6k-20261113`：6/6 applied、tier_match 6/6、3.7 ms）。
## D-046 · 2026-11-13 · local · 同一事故第二次：把 `evaluate.py` 随文档推进 main，判死 M2/M3；改为**代码级守卫**

**事故**：19:20 左右为记录 `hand_n` 改了 `hsbg/az/train/evaluate.py`（digest 相关），
用 `git add -A` 和文档一起提交并 `push` 到 main。正在跑的 **M2（`1132980`）/ M3（`1132981`）**
在各自第二个分段边界（约 20 分钟处）双双 FAILED：
`ValueError: 训练代码已变更；请另开实验，不能伪装为精确续训`。
快照保留（`keep-snapshots 3`），损失约 40 分钟 GPU ×2。

这是 **D-037 的同一条规则第二次被违反**——问题不在"不知道规则"，而在"靠记性执行"。

**决定：把规则变成代码，不再靠记性**

`ops/hsai_ops.py` 的 `push` 现在内置 **digest 守卫**：

1. 取 `git diff --name-only server/main..HEAD`；
2. 若含 `hsbg/az/train/**.py` 或 `workflows/train_entity_sync.py`；
3. 就查 `squeue -u $USER`，**只要有 RUNNING 的 `hsai-*` 作业就拒绝推送**，并在错误里列出
   作业号与三条处理方式（等作业结束 / 放侧分支 / 确实要中止就加 `--force-digest`）。

实测（2026-11-13 19:25）：`_is_digest_path` 对 `hsbg/az/train/ppo.py` 与
`workflows/train_entity_sync.py` 返回 True、对 `config/*.json` 返回 False；
`_running_training_jobs()` 正确返回 `['1133079(hsai-late-entry)', '1133069(hsai-late-entry)']`。

**副作用（重要）**：一旦把侧分支 merge 进本地 main 但还没推送，
**后续所有 push 都会被守卫拦下**（因为 `server/main..HEAD` 里始终有 digest 文件）。
正确顺序是：**等作业结束 → merge → 立刻 push**，中间不要夹别的推送需求。

**本轮复跑**：M3 由加速版 `N1`（引擎入场 + `games 256`）接手，M2 由 `M2b`
（均匀 p50 + `games 256`）接手，19:20 起在 h100-47 上跑，5 段 × 20 min。

## D-047 · 2026-11-13 · local · 隧道重建第二次翻车：常驻 ssh 继承了调用者的 stdout 管道；流程文档二源冲突一并消除

**事故**：用户重连 VPN 后要求"严格按流程重建隧道"。我执行 `python ops/hsai_ops.py tunnel`，
**卡住 4 分钟无输出被用户打断**，用户明确要求"深度确认清楚再搞，并写进文档"。本次没有
损失 GPU 时间，但这是 D-042 之后**同一件事第二次**翻车。

**两个独立根因（都对上了实测）**

1. **代码根因：常驻 ssh 继承了调用者的 stdout 管道。**
   `cmd_tunnel` 用 `subprocess.run(..., stdout=PIPE)` 收集 `tunnel.ps1` 的输出；而
   `tunnel.ps1` 用 `Start-Process` 拉起的是**常驻** `ssh -N`（隧道本身，永不退出）。
   Windows 的句柄继承把这个管道写端交给了 ssh，Python 永远等不到 EOF；
   `subprocess.run` 超时后还会再 `communicate()` 一次，继续死等。
   **直接测证**（23:12，一次性 2233 端口演练，不碰 2222）：
   `powershell -File ops\tunnel.ps1 -Port 2233` 的 stdout 被工具管道接住时，
   命令**不返回**，直到我手动 `Stop-Process` 掉那条 ssh 才返回，
   elapsed=**53.8 s，正好等于我杀它的时刻** → 写端确实被 ssh 持有。
2. **文档根因：两份"怎么建隧道"互相矛盾。**
   `AGENTS.md` §1 与 `OPERATIONS.md` §7/§10 说"跑 `hsai_ops.py tunnel`"（暗示可以在一次调用里等），
   而 `docs/nus/TUNNEL_FAST_PATH.md` + D-042 的结论是"绝不等待、必须脱离式启动 + 轮询"。
   我照做了前者、违反了后者。**"知道规则"不等于"有权重的单一流程"**，这是比 bug 更要紧的问题。
3. **附带发现（会静默毁脚本）：`ops/tunnel.ps1` 的 UTF-8 BOM 被编辑器改写时丢掉。**
   Windows PowerShell 5.1 读无 BOM 的中文脚本会按 ANSI 解码并**吞掉换行**：本次注释行
   `# …refused。` 把下一行 `if (-not $Force) {` 吞进注释，5.1 报"意外的标记 }"——
   自愈脚本变成**必定失败**，而 pwsh 7 解析同一文件却报 0 错，极易漏掉。
   仓库已有守卫测试 `test_ps1_files_with_non_ascii_must_be_utf8_with_bom`，是它证明了这次是我改坏的；
   BOM 已按字节补回（`<#` 前三个字节），5.1 下重新可解析。

**决定**

1. **代码**：`cmd_tunnel` 改为把子进程输出重定向到**日志文件**
   `artifacts/state/tunnel-selfheal.log`（`stdin=DEVNULL`，有界 `--timeout`），跑完只读文件尾；
   `tunnel.ps1` 的 `New-Tunnel` 给 ssh 补 `-RedirectStandardInput`（空文件）与 `-ConnectTimeout=20`，
   并在成功时打印 `ssh pid` 与日志路径。新增 `--port` 供演练（不碰 2222）。
   **实测**：杀掉 2222 转发后 `python ops/hsai_ops.py tunnel` **10.2 s 返回**、rc=0、
   `probe` 独立确认 `host=xlogin2`；随后 `state` 报 `tunnel : OK`。
2. **测试**：`tests/test_ops_guard.py` 增加两条守卫——
   `cmd_tunnel` 不得对子进程用 `stdout=PIPE`（AST 级检查）、
   `tunnel.ps1` 必须同时重定向 ssh 的 stdin/stdout/stderr。
3. **文档单一化**：`OPERATIONS.md` §1.3 成为**唯一权威流程**，写成四步
   （A 预检 VPN → B 重建〔B1 `hsai_ops.py tunnel` 6–10 s；B2 脱离式 `Start-Process` + 1.5 s 轮询 ≤25 s〕
   → C 独立探活 → D 仍 DOWN 才读日志分类），每条禁令都附实测证据；
   `docs/nus/TUNNEL_FAST_PATH.md` **降级为证据附录**（实测数字/日志指纹），
   `AGENTS.md` §1、`TRANSPORT.md` §1.1/§7、`OPERATIONS.md` §7/§10 全部改为指向 §1.3。

**判据（以后照这个验收隧道流程）**：重建全程 **≤10 s** 且**单次工具调用内返回**；
返回后必须能用 `probe`/`state` 独立证明 `host=xlogin2`；任何一次超过 30 s 无结果，
就掐掉读 `C:\ProgramData\HSAI\tunnel-2222.log` 尾部，而不是继续等。
## D-048 · 2026-11-14 · local · 续训路径上**配置的 `lr` 会被 trainer state 覆盖**：L3 因此是空跑，已改 L4（fresh-Adam 分叉）

**事故**：我以为 L3（`config/league-long-l3.json`，`lr 4e-6` + `anchor 0.02`）是"lr ×2 + anchor 放松"的
解冻臂，实际它只改了 anchor。证据是 L3 日志第一行：

```
[lr] base=[2e-06] last_epoch=14858 lr_min_ratio=0.833333 max_updates=10000
     -> effective=1.666666667e-06 (config lr=4e-06 ratio=0.833333 max_updates=25000)
[lr][warn] 续训沿用 trainer state 的 lr_min_ratio/max_updates，命令行值被忽略
```

**根因（代码事实，不是猜测）**：`late_entry.sbatch` 走 `personal.py`，而它**总是**带 `--trainer-state`；
`workflows/train_entity_sync.py:1920` 于是走 `PPOTrainer.load(...)`，由
`ppo_trainer.py:427-434` 用 trainer state 里的 `scheduler.base_lrs` / `lr_min_ratio` / `max_updates`
覆盖配置。只有**不带 `--trainer-state`** 的 fresh 分支（`:1949`
`PPOTrainer(model, p_cfg, lr_min_ratio=..., max_updates=...)`）才会用配置的 lr。
仓库里那条 `[lr][warn]` 早就写了这件事，我在配 L3 时只读了配置语义、没读这条警告。

**代价**：L3 白跑 13 轮（约 6 min 一张卡）后被 `scancel`；更大的代价是**今天所有课程臂
（N1/M2b/M2c/S1b/S1c/L1/L3）的实际 lr 都是 1.667e-6**，所以 §5.56 的"配对差恰好 0.0"
可能根本不是"配方无效"，而是"策略几乎没动"。

**决定**

1. **新增 `deploy/slurm/lr_fork.sbatch`**：直接调 `train_entity_sync.py`，**不带 `--trainer-state`**
   （fresh Adam 分叉，从 champion 权重起跑），`HSAI_LR` 控制 lr，`HSAI_LR_RESUME=1` 支持链式续训。
   provenance/`run.env` 里显式写 `mode=fresh-adam-fork (NO --trainer-state)`，与"续训"区分开。
2. **L4 = `1135112→1135113→1135114`**（`artifacts/runs/entry-l4-20261114`）：
   `fresh-adam-fork` + **lr 3e-6** + anchor 全程 0.02，其余与 L1 完全相同。
   **已核实生效**：L1 迭代行 `lr=1.67e-06`，L4 迭代行 `lr=3e-06`。
3. **规矩**：以后凡是"改 lr"的实验，必须二选一——
   ①用 `lr_fork.sbatch`（fresh Adam）；②先做 trainer fork 工具显式改 `base_lrs`。
   **不许**再靠配置 `lr` 猜测续训行为。凡是"改配方"的实验，落配置后必须回到日志里
   **看到那一行为准**（`[lr]` / `[anchor]` / `[keep-start-board]` 之类），配了没中一律按失败处理。
4. 顺带记录：`l3` 保留为"anchor-only"证据（13 轮，不作结论）；`config/league-long-l3.json` 不删，
   但标注为"lr 未生效，勿复用作解冻臂"。

## D-049 · 2026-11-14 · local · 用 `python -c "…"` 改文件时，PowerShell 会吃掉 `${…}`；改文件一律走编辑工具

**事故**：给 `deploy/slurm/spell_l5.sbatch` 加参数化时，我用
`python -c "t.replace('--games 256', '--games ${HSAI_LR_GAMES:-256}')"`。
**PowerShell 把 `${HSAI_LR_GAMES:-256}` 当自己的变量展开成空**，写进文件后变成
`--games \ --iterations …` 与 `--processes \ --workers 1` —— 参数直接消失。
而这个脚本正是 04:20 起 L5a 与所有 resume 作业要用的；若不发现，那一轮会白占卡甚至行为异常。
（正在跑的 L4/L5c 不受影响：Slurm 提交时已快照脚本。）

**根因**：PowerShell 的插值发生在**双引号内**，连 `python -c` 的字符串都会先被展开；
`${name:-default}` 恰好是 PowerShell 的合法语法（`${}` 里可带表达式），所以不会报错，而是**静默变空**。

**决定（与 D-038 同族，收紧一层）**

1. **改文件一律用编辑工具**（write/edit），不用 `python -c` / `-replace` / `Set-Content` 拼字符串。
2. 若确需脚本改文件：内容先落到文件（用 write 工具），再 `python script.py` 读取，**不经过 shell 插值**。
3. 改**启动器/作业脚本**后必须做两道核对：① 服务器上 `bash -n <脚本>`；② `grep` 出关键 flag
   （`--lr` / `--games` / `--processes` / 新增开关）**逐行肉眼确认**。
4. 任何"参数化"改动后，先在**冒烟规模**（8 局 / 2 进程 / 8 min）跑一次，再上 150 分钟正式段。

## D-050 · 2026-11-14 · local · 远端脚本一律纯 ASCII：PowerShell 的默认编码会毁 UTF-8 并吃掉引号

**事故**：写 L4 边界脚本 `artifacts/state/at_l4_boundary2.sh` 时，用 PowerShell 把它包进
`cat > /tmp/b2.sh <<'EOF' … EOF` 再经 `hsai_ops.py sh -f` 送远端。远端 `bash -n` 报
`syntax error near unexpected token '('`，指向的却是一行**完全合法**的
`echo "@@L4_RESUME_JOB=$res (afterany:$pan)"`。真正坏掉的是第 6 行：
`echo "@@ABORT: L4 1135112 仍是 $st，边界未到"` 的中文被毁，行尾引号被吃成 `?`
→ 引号不闭合 → 报错飘到 40 行之后。

**根因（已实测复现，不是推测）**：`hsai_ops.py sh` 是**字节透明**的
（`remote_run` 用 `input=script.encode("utf-8")` 走 ssh stdin，远端原样落盘）。
坏在本地这一环：Windows PowerShell 5.1 的 `Get-Content` / `Set-Content` 默认按
**ANSI 代码页**解码（本机 `ACP=936`），BOM-less UTF-8 中文先被当 GBK 解成乱码再存回：

```
Get-Content default : 浠嶇劧鏄?OK      # 原文 "仍然是 OK"
Get-Content -Encoding UTF8 : 仍然是 OK
```

一个多字节字符被吃成 `?` 时，若它紧邻引号就会把引号一起吃掉。

**决定**

1. **文件内容只走编辑工具**（`write` / `edit`），不经 PowerShell 文本 cmdlet 中转；
   必须用 PowerShell 读写时显式 `-Encoding UTF8` / `[IO.File]::ReadAllText($p, UTF8)`。
2. 交给 `hsai_ops.py sh -f` 的脚本**一律纯 ASCII**（注释、`echo` 写英文）。
   这条是**纵深防御**：即使编码环节再出问题，最多是乱码，不会把脚本变成语法错误。
3. 新写的远端脚本先 `bash -n` 再执行（`cat > /tmp/x.sh <<'EOF' … EOF` + `bash -n /tmp/x.sh`）。
4. 判据：远端报 `syntax error` 但**指向的行看着没问题** ⇒ 先怀疑**更早某行**的非 ASCII
   字符，而不是去改那一行。

## D-051 · 2026-11-14 · local · digest 推送守卫收窄到"会 resume 的训练作业"，并补上 PENDING

**问题**：`hsai_ops.py push` 的 digest 守卫原来判据是"队列里有任何 `hsai-*` 且 `RUNNING`"
就拒绝推送。两个方向都不对：

1. **太宽**：`hsai-human-gap`（真人场面板）、`hsai-bench`、建库作业**从不读 trainer state**，
   改 digest 对它们零影响。夜班整晚都有面板在跑 ⇒ 代码一行都推不上去，
   只能等到"恰好没有评测"的窗口，实践中等于没有窗口。
2. **太窄**：只看 `RUNNING`，漏掉 **`PENDING`**。`--dependency=afterany` 链里的下一段
   在提交时就已经带着"旧 digest"排队；一旦被调度就以新代码启动、拿旧 trainer state
   续训 ⇒ **开始即失败**，比 RUNNING 更隐蔽（2026-11-14 夜班 `1136176` / `1136229`
   就是这种待命状态）。

**决定**

- 守卫判据改为：作业名以 `hsai-spell` / `hsai-lrfork` / `hsai-late` 开头，且状态是
  `RUNNING` 或 `PENDING` ⇒ 拒绝推送（这些是唯一会 resume 的类型）。
- 拒绝信息里直接列出 `jobid(名称,状态)`，不再只说"有作业在跑"。
- 5 条测试钉住双向：RUNNING 训练要拦、PENDING 链要拦、评测/建库放行、别人的作业放行、
  已结束的训练作业放行。

**注**：这不是放宽纪律，而是把纪律对准它真正要保护的东西——"精确续训能力"。

## D-052 · 2026-11-14 · local · **根因：名次奖励在我们所处的板面区间里对"棋力"不敏感**；训练前必须先测信号（新门禁）

**背景**：一晚上四个不同形状的奖励实验（加大塑形 / 终局代理 / 动作条件补贴 / 有界望远镜增量）
全部无效或更差；同配方继续训练稳定退化。用户提出的假设是"对手太难 ⇒ 后期拿不到有效奖励 ⇒
只优化前期"。我把这个假设做成了可测量的形式，结论是**成立且更尖锐**。

**决定性测量**（`gap-hires-20261114-0314`，768 局，champion；板面按卡表 atk+hp 逐局累加，
**在共同回合、死亡之前**测量）：

| 测量 | 结果 |
|---|---|
| 名次 ~ 死亡回合 | r = −0.806，**R² = 0.65** |
| 名次 ~ **T8 板面**（n=639，仅取活过 T10 的局） | r = −0.011，**R² = 0.000** |
| 名次 ~ **T9 板面**（n=540） | r = −0.001，**R² = 0.000** |
| 名次 ~ **T10 板面**（n=286，仅取活过 T12 的局） | r = −0.062，**R² = 0.004** |
| 同一死亡回合（T11）内按板面四等分的平均名次 | 5.33 / 4.83 / 4.83 / 5.06（**不单调**，幅度 0.5 名） |

**⇒ 控制住"死在第几回合"之后，板面强度对名次几乎没有解释力（R²≈0）。**

**机制**：战斗结果是板面比值的**阶跃函数**。champion 逐回合战斗胜率实测：
T7 53.5% / T8 70.1% / T9 **84.3%** / T10 54.3% / T11 **17.4%** / T12 **13.3%** / T13 **2.7%**；
受伤均值 T9 1.3 → T11 13.2 → T12 18.8。
同时真人曲线在 T10→T12 **加速**（p50：879→2232→4983，**+467%**），我们近似线性（740→1047→1434，**+94%**），
比值从 0.84 掉到 **0.29**。在我们生活的这个比值区间里，1434 和 2000 面对 4983 都是"整队被清空"
⇒ **板面的边际作用为零** ⇒ 名次只剩"活多久"这一个可学维度。

**由此统一解释今晚全部现象**
1. 四种奖励形状无效：它们都长在同一个对棋力不敏感的指标上。
2. 继续训练稳定退化：唯一有梯度的方向是"减少暴露"，于是学出"薄板、少挨打、多熬一轮"
   （自建 T8 板面 270→206→168 与名次严格同序）。
3. 唯一"没变差"的干预是**换场**（`l7mixb` 46% 真人/54% 镜像：5.344，Δ +0.044，首次非零吃鸡）
   ——因为它把对手拉回到"对局有悬念"的区间。
4. "太难/太简单"的矛盾由此解开：**信号只在比值≈1 的窄带里有效，两端都是阶跃的平坦段**。

**新门禁（D-052，强制）**

> 任何 ≥1 小时的训练开跑之前，必须先在**该训练使用的场**里测
> `R²(板面 → 名次)`（用与本文相同的口径：共同回合、死亡前测量、控制死亡回合）。
> **R² ≈ 0 则不许开长训** —— 那等于用一把看不见棋力的尺子量进步。
> 报告里同时给逐回合战斗胜率与受伤均值。

**下一步方向（按此根因排序，不按愿望）**
1. **后期入场课程**：把 `--entry-turn 8` 改成 **T10–T12**，并注入**真人 p75 量级**的板子
   （T12 p50 已 4983，而现在的 T8 注入仅 ≈266）⇒ 让它在"板面同量级"的局面里反复练中后期运营。
   入场库够用：T10/T11/T12 分别 850/867/895 帧（比现在用的 T8 的 657 帧更多）。
   launcher 已参数化 `HSAI_ENTRY_TURN_MIN/MAX/HSAI_ENTRY_SAMPLE`（默认 8..8/p75 逐位不变）。
2. **能力匹配的对手**（PFSP 思路，按**能力**而非时钟退火），把难度钉在"胜率 40–60%"的带里。
3. **直接奖励留存随从的滚雪球**（真人 +690/回合 vs 我们 +40）：死了拿不到、卖掉的算不上，
   有界且不可作弊——即使名次不敏感，这个量本身就编码"运营"。
4. 停止：调奖励系数 / 调锚 / 调 lr / 加长同配方训练。
**门禁已可执行（2026-11-14 11:30）**：`tools/measure_reward_signal.py <panel.json>`
（控制死亡回合后的 `R²(板面→名次)` + 逐回合战斗胜率 + PASS/FAIL，阈值 `SIGNAL_R2_MIN=0.05`）。
在现有面板上的**实测**：

| 模型 | R²(死亡回合) | **signal R²** | T8/T9/T10/T11 偏相关 r |
|---|---:|---:|---|
| champion | 0.650 | **0.005** | −0.054 / −0.020 / −0.067 / −0.062 |
| `l4`@301 | 0.632 | **0.003** | +0.018 / +0.042 / +0.024 / −0.057 |
| `l7mixb` | 0.664 | **0.009** | −0.078 / −0.053 / −0.004 / −0.095 |

**全部 FAIL（<0.05）** ⇒ 在这个场里，名次奖励对板面的信息量约等于零，
而死亡回合单独解释 63–66% 的名次。这就是"练了这么久没有提升"的定量答案。
后续任何训练都必须在**训练所用的场**上过这条门禁，或在机制指标（留存增重、T11–T13 存活、
T12 板面比值）上先看到改善。

## D-053 · 2026-11-14 · local · **配置的 coef ≠ 实际付出的量级**：奖励实验必须从 collect stats 反算

**事故（第二次同类）**：`l6comp` 的 `comp-target-coef 0.35` 按设计应有 ~17.5% 名次全距的量级，
我据此把"comp-target 无效"写成了"**换奖励形状完全没改变结局**"。复查 `history.jsonl` 才发现：

```
comp_target_coef=0.35   comp_target_best_mean=0.038   comp_target_reward_sum≈5.6/轮
```

`best_mean=0.038` 意味着"阵容完成度"平均只到 3.8%（模型几乎从不凑齐 2 张核心卡），
于是**实际付出的量级 ≈ 0.35 × 0.038 ≈ 0.013，只有名次全距 2.0 的 0.7%** ——
远低于我自己定的"每个奖励项 ≥5% 全距"门槛。**它不是"无效"，它是"几乎没发钱"。**

同一类错误已经出现过一次：§5.71 的 `shaping_coef 0.1` 只占 0.16%（无效结论被迫作废）。

**因此修正今晚的结论**

| 奖励形状 | 实际量级 | 是否构成有效检验 |
|---|---|---|
| `Φ=log Σ` coef 20 | 30% 但 `adv_std` 4.51 | ❌ 方差机器，不是信号 |
| 终局板面 1.0 | 35% | ✅ **唯一真正被测过的形状** → 显著更差（拿板面换命） |
| 动作条件法术补贴 | 0（零频动作） | ❌ 结构上不可能触发 |
| comp-target 0.35 | **0.7%**（realized） | ❌ **量级不足，等于没测** |

⇒ 「四种奖励形状同一种结局」这个说法**作废**。准确说法是：
**只有"终局板面"被真正测过（且有害）；另外两个是零频/零量级的空跑，一个被方差毁掉。**
奖励空间**没有被穷尽**。

**新规矩（写死）**

1. 任何奖励实验，**开跑前**写清"设计量级 = coef × 该项的理论上界 / 2.0"；
2. **跑完第一个 iteration 就要从 collect meta 反算实际量级**（例如
   `comp_target_best_mean`、`retained_growth_reward_sum`、`reward_shaping_mean`），
   与设计量级差 3 倍以上就**先修触发条件，不要拿结论说话**；
3. 报告里必须同时给"设计量级"和"实际量级"两列。

## D-054 · 2026-11-14 · local · **训练场难度远超模型能力，是"学不会中后期"的直接原因**：改动态难度匹配（不乘倍率）

**用户提出的方向**：先降前期压力让模型活到中后期，再用"略高于自身"的对手做渐进式挑战，
难度随实际反馈调整而不是按时间表；先学会玩中后期，再回头优化完整对局名次。

**这条方向被数据证实，而且比预想更极端。** 用 `tools/pool_strength_curve.py` 把真人战团库
（`firestone-bank-6k/train-m6k.jsonl`，`boards=12689`，`turns=1..18`）按回合求出强度分位
（Σ(atk+hp)），把模型的自建板面放到同一把尺子上：

| 回合 | p10 | p25 | p50 | p75 | p90 | 我方自建板面 | 我方所处分位 | 旧训练场给的对手 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 78 | 111 | **160** | 241 | 372 | 168–270 | **≈p50–p75** | 引擎 bot |
| 10 | 260 | 423 | **791** | 1669 | 3101 | 740–1047 | **≈p50–p60** | p50 (791) |
| 11 | 415 | 909 | **1945** | 4346 | 8844 | ~1047 | **≈p28** | p50 (1945) |
| 12 | 739 | 1772 | **4472** | 9445 | 18432 | 1434 | **≈p13** | p50 (4472) |

三条结论：

1. **前期不是短板**：T8 我们已经站在真人 p50–p75。所以"前期能力退步"不是当前主要矛盾；
2. **T10 之后断崖**：人类 T10→T12 中位从 791 涨到 4472（+467%），我们只从 ~900 到 1434（+59%）；
3. **旧训练场在 T12 给我们的是 p50 = 3.1 倍于自身的对手**。每场必输 ⇒ 战斗恒为"被清场"，
   **没有任何可学信号**（与 D-052 的 R²≈0 完全一致：既然总被清场，板面 1434 和 2000 无差别）。

⇒ 这不是"训练不够久"，是**课程错误**：把学习者长期泡在"打不过"的对手里。

**决定（写死）**：把"对手强度"从一个固定分位改成一个**闭环控制量**。

- **绝对刻度锚**：对手强度统一用 `Σ(atk+hp)` 的**真人战团分位**表达（"能打赢 p40 的真人战团"
  是一个跨运行可比、与训练进度无关的绝对刻度）。曲线由 `tools/pool_strength_curve.py`
  从战团库生成（`config/arena_v1.json` 指向它），提交进仓库以便复现。
- **不乘倍率**：绝不给真人战团身材乘系数。乘 1.1 会造出牌库里不存在的分布（4 星随从变 2 倍
  身材），模型会学成"打数据怪"。改为在真人库里按目标强度**就近选真实战团**
  （`HumanBoardPool.sample_at_strength_pick`，候选回合窗口 `[t-2, t+1]`，形状真实）。
- **三段闭环**（`hsbg/az/train/opponent_arena.py`）：early(T1–7) 以"活到 T9 的比例"为反馈，
  mid(T8–11)/late(T12+) 以"战斗败北率"为反馈（目标胜率 0.70/0.60/0.50，死区 0.05）。
  实测胜率**高于**目标 ⇒ 上调对手强度（每次 ≤6%），低于 ⇒ 下调；**没有任何按 iteration 的排期**。
- **难度棘轮**：mid/late 同时达标且活到 T9 达标，连续 3 轮 ⇒ 把"活到 T9"的目标从 0.90 逐步
  收紧到 0.70（**恢复前期压力是掌握度的函数，不是时间的函数**）。
- **档位必须跨段持久化**：`arena_state.json` 落在 run 目录，续训接着用；否则每段都从头降难度。

**怎么判断"真的学会了"而不是"对手变弱了"**（用户 Q6 的答案）：

1. 训练难度会自适应，所以**训练胜率永远不是进度信号**；进度信号是**能打赢的分位本身**
   （`arena_*_str` ↔ 曲线分位）——它上升意味着"我们能打赢更强的真人战团"，与训练场无关；
2. **逐回合遥测**（`tel_own_str_*`/`tel_tier_*`/`tel_cum_dmg_*`/`tel_reach*`）：自建板面、
   升本节奏、掉血速度、到达率，全部是与对手强度无关的绝对量；
3. **冻结标尺**：最终仍用 `eval_human_gap`（768 局、赢家场、配对 bootstrap）做验收，
   训练场怎么变都不影响这把尺子；`--eval-human-quantile` **不改**。

**默认关闭、逐位等价**：`--arena-config` 为空时 `arena_on=0`，采集侧走原分位路径；
遥测默认开（`turn_telemetry=True`），只多一次/回合 `player_json`，不改奖励与损失。

## D-055 · 2026-11-14 · local · **"没掉血" ≠ "赢了"**：战斗归因必须用引擎的 `won_last/lost_last`

**事故**：难度控制器的第一版把"本回合血量没降"当作"赢了这场战斗"。后果在 A/B 的
arena 臂上 3 个 iteration 内就爆出来：

```
arena_late_str  2026 -> 3323（= 我方 T12 板面 1240 的 2.7 倍）
arena_ratchet   3（reach_target 已经被收紧到 0.75）
```

也就是：**控制器一路加码，把模型推回"每场必输"的老坑**——正是 D-054 要修的那个病。
根因是两个口径错误叠加：

1. **无战斗回合**（第 1–2 回合、特殊回合）不掉血 ⇒ 被算成胜利；
2. **平局**不掉血 ⇒ 也被算成胜利。

两点一起，使 `1 - losses/combats` 变成"没输的比例"，系统性高于真实胜率；比例控制器
看到"总是高于目标"，就一直往上调。

**修法**：`_sample_turn` 改用引擎 `player_json` 里的 ``stats.won_last / lost_last / tied_last``：

- `combats` 只统计**有胜负**的战斗（平局不进分母）⇒ `1 - losses/combats` 才是真胜率；
- `ties` 单独计数（`arena_ties_*` 进 history），便于核对平局比例；
- 旧引擎没有这三个字段时退回"每回合算一场"（记录在注释里，不静默改变语义）。

**通用教训（写死）**：任何"胜负/成功率"型反馈都必须问一句
**分母是不是真的"发生了这件事"**。本项目已经因为这个坑损失过两次：
`comp_target` 的 coef 与实际量级（D-053），以及这次的胜率分母（本条目）。

## D-056 · 2026-11-14 · local · **研发模式改为"用户定方向、Agent 执行"**，不再自主规划多天路线

**变更**（用户直接指令）：此前是"Agent 自主规划、自主决策、连跑多天"。现在改为：

1. 用户定总体方向；Agent 执行、分析、验证；
2. **任何消耗 GPU 的作业（训练、评测、建库、smoke、探针）都必须先给配置、用户点头**；
3. Agent 发现问题时写建议、由用户决定，**不自己改道**。

**固定迭代流程**（`docs/CHARTER.md` §2.2）：训练 → 同协议评测 → 与冠军对比 →
有进步则在新模型上重建战团库继续 / 没进步则归因并提出建议。

**落地**：新建 `docs/CHARTER.md`；改写 `AGENTS.md` §1/§6/§9（自主可做收窄到"不占 GPU"，
"必须先问"新增"任何 GPU 作业"与"改配方参数"）。

**为什么**：多天自主规划会让方向漂移、且大量 GPU 花在无人确认的假设上。改完后每轮都有
"评测 → 与冠军对比"这个客观闸门。

## D-057 · 2026-11-14 · local · **训练配方按 AutoTavern 论文 Table 7 冻结**（取代冠军那套实际生效的 1.667e-06/1024 批）

**背景**：冠军（`league-36h`）走 `personal.py` + `--trainer-state`，`PPOTrainer.load()` 用 state 里的
`scheduler.base_lrs` 覆盖了 `--lr`，**实际生效 lr≈1.667e-06**、有效批 1,024、anchor 250 轮。
这与论文配方不是一回事。

**决定**：按论文（[arXiv:2610.33845](https://arxiv.org/html/2610.33845v1) §6 + Table 7）重配并**冻结**：

| 项 | 值 |
|---|---|
| lr | 6e-06 → 5e-06（`--lr-min-ratio 0.8333`） |
| 有效 minibatch | 8,192 = minibatch 1024 × grad-accum 8 |
| KL anchor | β=0.1，**前 100 轮**，tail 0.02 |
| clip / target-KL / λ / entropy / grad-clip | 0.2 / 0.05 / 0.995 / 0.002 / 1.0 |
| max-updates | 4,650（≈论文一个 stage 的退火长度） |
| 其余 | games 512、epochs 1、procs 8、sims 0 |

**两处与论文的已知差异**（用户 2026-11-14 确认接受）：
1. 起点是冠军权重 + **新建 Adam**（论文每 stage 从上一 stage 继续）——因为带 `--trainer-state`
   会把 lr 覆盖回 1.667e-06，论文的 6e-06 就失效；
2. 奖励除终局名次效用外，额外挂一个**小升本节奏项**（论文没有）。

**冻结规则**：不每轮改。只有发现具体问题（工程错误 / 指标异常 / 机制缺陷）才由 Agent 提议、
用户决定。落地文档 `docs/TRAINING_RECIPE.md`。

**同时确立的对照口径**：论文一轮 ≈100 万决策、一个 stage = 300 轮 = 3.1 亿决策；
我们一轮 ≈12.4 万决策，一次 3 小时作业 ≈620 轮 ≈7,700 万决策 ≈ 论文一个 stage 的 1/4。
**所以"跑一次 3 小时"不等于"训练完成"。**

## D-058 · 2026-11-14 · local · **`h100-47` 是 MIG 切片**：双卡要看落点，脚本内置硬门禁

**事故**：`stage_arena.sbatch` 申请 `--gres=gpu:h100-47:2`，Slurm 记账 `gres/gpu=2` 正常，
但作业在 `actor GPU越过当前可见设备范围0..0` 处崩溃——`torch.cuda.device_count()==1`。

**根因**：`h100-47` 不是整卡，是 **MIG 3g.47gb 切片（46.4 GiB）**。同一节点上两张切片可能落在
**同一张物理卡**（`CUDA_VISIBLE_DEVICES=0,1` 的数字编号按物理卡解释，第 1 张不存在 → 只看到 1 张），
也可能分属两张物理卡（此时 `device_count==2`，是真双卡）。

**实测**：xgpi11 同卡 → 1 张；xgpi13 分卡 → 2×46.4 GiB（真双卡）。
冠军那条 36h 的历史记录 `gpus=[1]` 说明它当时抽到了分卡。

**账号 QOS 上限**（`normal`，按用户）：`h100-47=2`、`h100-96=1`、`a100-80=1`、`a100-40=2`、
`nv=8`、`gres/gpu=8` ⇒ **"两张整卡 H100"这个账号拿不到**。

**修法**：`stage_arena.sbatch` 在建输出目录**之前**做双卡硬门禁：`device_count < 2` 就 `exit 3`，
由外部重投换节点（约 1 分钟一轮）。`deploy/slurm/gpu_probe.sbatch` 留作诊断工具。

**通用教训**：Slurm 的 `AllocTRES` 说"给了 2 张"不等于**进程真的能用 2 张**；
凡是要多卡的作业，启动第一件事必须是问框架 `device_count()`，而不是相信记账。

## D-059 · 2026-11-14 · local · **第一轮：候选双场验证真进步（−0.27 名次）；顺带否证"升本太慢是关键缺陷"**

**一轮的完整闭环跑通了**（新研发模式的第一次实践）：

| 环节 | 结果 |
|---|---|
| 训练 | 作业 1140935，双卡 MIG 分属两张物理卡，303 轮 / 3,900 万决策 / 164.6 分钟，非法动作/截断/无效局**全程 0** |
| 训练场评测 | 512 局：`mean_rank` 2.615 → **2.352**，吃鸡 41.6% → **48.6%**，配对 bootstrap CI [+0.064, +0.463] 排除 0 |
| held-out 评测 | 512 局（455 帧，源自 `eval-m6k.jsonl`，**从未参与训练**）：`mean_rank` 2.918 → **2.637**，吃鸡 41.0% → **44.3%**，CI [+0.061, +0.512] 排除 0 |
| 判胜 | **IMPROVED**（两场效果量 −0.264 / −0.281，几乎一致 ⇒ 不是主场优势） |

**为什么要额外做 held-out**：本轮训练用的场就是评测用的场，冠军从未在这个场地上练过 ⇒ 有主场优势。
论文的规矩是"考试库绝不用于训练；训练用另一批同源战团"。于是从 `eval-m6k.jsonl` 按**同样的校准系数**
另建 `arena-cal-v3hold-snap` 做交叉验证。**规则已写进 `docs/EVAL_PROTOCOL.md` §4 规则 6**：
判真进步必须用 held-out 场，否则只能叫"主场成绩"。

**顺带否证的一个假设**：本轮加的**升本节奏奖励没有把 tier 拉上去**——两场都显示候选的平均 tier
比冠军低 0.47~0.74（T8–T18），而中位板面高 1%~16%。**"升本慢"并不是这个模型的缺陷**：
它把金币换成板面之后，名次、前四率、吃鸡率都变好了。这条已写进 `docs/NEGATIVE_RESULTS.md` A 区。
（注意：本轮同时改了对手场 + 超参 + 奖励三处，**不能把这 0.27 名次归给任何单一原因**。）

**两个流程纪律**：
1. 下一轮**只改一个变量**，否则无法归因；
2. 两处口径 bug 已修：`--max-updates` 的单位是**训练轮**不是优化步数（K4，本轮退火只走了 6.5%），
   以及 M1 退化判定必须要求 `n≥100`（曾用 n=20 的 T19 把结论误判成 SUSPICIOUS）。

## D-062 · 2026-11-15 · local · **强制升本（一半强制+一半教师）成功把 tier 推到 6，但 T12 板面反而降了 15%**

**做法**（用户定）：`--forced-tier-plan 2:2,4:3,5:4,7:5,9:6`（T9 必到 6 本）+ `--forced-tier-prob 0.5`
（一半强制）+ `--forced-teacher-until 40M --forced-teacher-coef 0.05`（另一半写 DAgger 标签）+
升本软奖励关闭（coef 0）。对手场换成 `arena-cal-v5-snap`（前期压到几乎必赢、T11/T12 抬到 1.18/1.23 倍）。

**结果**（作业 1142148，633 轮 / 7,400 万决策 / 164 分钟；末轮 `lr=5e-06` 证明 K4 修正确实生效）：

| 指标 | 上一版 arena-v3 | **arena-v5** |
|---|---|---|
| T10 平均 tier | 4.46 | **6.00** |
| T12 平均 tier | 4.50 | **6.00**（486/488 局在 6 本） |
| **T12 中位战力（KPI）** | 1,434 | **1,222（−15%）** |
| T8 中位战力 | 251 | 179 |
| T13 中位战力 | 1,836 | 1,637 |
| **T14 中位战力** | 2,325 | **2,348（反超）** |
| **T16 中位战力** | 3,170 | **3,554（+12%）** |
| 存活到 T14 | 75.6% | 60.4% |
| 名次均值 | 2.637 | 3.057（场地更难，不可直接比） |

**判断**：
1. **强制升本完全成功** —— tier 从 4.4 稳到 6.0，99.6% 的局 T10 就到 6 本，`missed` 只占 0.3%
   （2560 个目标里 ~8 个）。这套"硬约束 + 教师标签"比软奖励有效得多。
2. **但冲本是有代价的**：T1–T9 几乎不买随从，到 T12 只有约 3 个回合的 6 本采购，板面追不回来
   ⇒ **T12 中位战力从 1,434 掉到 1,222**。这正好回答了本轮的问题："先冲本再补板面"在 T12 这个
   时点上不划算，**收益要到 T14 之后才兑现**（T14 打平、T16 +12%、T18 +8%）。
3. **教师预算 40M 在全程约 55% 处用尽**（实际总量 ~74M），之后是纯自学；这是设计内的渐进交棒，
   但如果要全程带教师，下一轮应把 `--forced-teacher-until` 设到 80M。

**下一轮的候选方向**（等用户定，本轮不改）：
- (a) 把强制计划**推后一档**（如 `3:2,5:3,6:4,8:5,10:6`），让 T9–T11 有回合补板面；
- (b) 保持 6 本时点，但把 **T12 库目标从 1,800 降到 1,400 左右**，让它先"够得着"再慢慢抬；
- (c) 教师预算加到 80M，让它把"该升本"学得更牢。

## D-063 · 2026-11-15 · local · **双卡不用抽签了：用 MIG UUID 重写 CUDA_VISIBLE_DEVICES**（用户授权破 CVD 红线）

**问题**：`h100-47` 是 MIG 3g.47gb 切片，每节点 4 片。申请 2 片时 Slurm 总挑编号最小的连续两片，
而节点映射固定为 **gres `0,1 → 物理卡0`、`2,3 → 物理卡1`** ⇒ 大概率同卡。更要命的是 Slurm 只设置
**数字** `CUDA_VISIBLE_DEVICES=0,1`，CUDA 把它解释成"物理卡0 和 物理卡1"，第 2 张不存在 ⇒
`torch.cuda.device_count()==1`。2026-11-15 连续申请 **12 次都是同卡**，双卡完全靠运气。

**解法**：从 `nvidia-smi -L` 读出**本作业已被分配**的 MIG UUID，用它重写 CVD，CUDA 就按 MIG 实例
枚举。探针 1143745 实测：同卡分配下 `device_count` **1 → 2**（两张都是 46.4 GiB）；
正式作业 1143756 实测：Slurm 给 `0,1`，重写后 **split 模式**，吞吐 **8,900 决策/秒**
（与跨物理卡持平；共享卡降级时只有 5,200–5,900）。

**这是破例**：`AGENTS.md` §3 原本写"不手工改 `CUDA_VISIBLE_DEVICES`"。用户 2026-11-15 明确授权：
"那是我之前的规矩太严格了……全部授权"。已把例外条款写进 §3。

**约束（写死）**：
1. 只能用 `nvidia-smi` 在**当前分配**下报出来的 UUID，不能凭空指定、不能扩大资源；
2. 恰好读到 2 个才启用，否则自动降级为共享卡（并打印 `gpu-mode: SHARED fallback`）；
3. 原始 CVD、UUID、改写后 `device_count`、`actor_gpus` **全部写进 `run.env`** 可审计；
4. `HSAI_ARENA_MIG_REWRITE=0` 可整体关掉。

**遗留的诚实说明**：两张切片在同一张物理卡时共享显存带宽/L2，理论上比跨卡略吃亏；但实测吞吐
（8,900）与跨卡（8,900–9,300）持平，可视为无损。

## D-064 · 2026-11-15 · local · **v6：100% 强制升本成功（6 本 7.36 回合/局）；下一轮 lr 2× + 后期加压**

**v6 结果**（作业 1143756，599 轮 / 164 分钟 / 约 7,800 万决策，`illegal=0` 全程）：

| 指标 | v5（prob 0.5 + 教师） | **v6（prob 1.0，教师关）** |
|---|---|---|
| T12 中位战力 | 1,222 | **1,344（+10%）** |
| T16 中位战力 | 3,554 | **3,675（+3.4%）** |
| T8 平均 tier | 5.06 | **5.84** |
| **6 本停留回合数/局** | ~5.2 | **7.36** |
| 存活到 T14 | 309/512 | **408/512** |
| 吃鸡率 | 41.0% | **45.9%** |
| mean_rank | 3.057 | **2.596** |

**但 prob=1.0 有明确代价（记为 K7）**：被强制的决策 `logprob=0`、不进策略梯度，所以
**"按升级"这个动作从未由策略采样** ⇒ 这一轮**没有产出"自己会升 6 本"的能力**，而且评测也施加了
同一强制计划，所以那个 tier 5.95 是"带拐杖的成绩"。**要验证它自己会不会升，必须跑一次不带强制的评测。**

**用户 2026-11-15 决定**：
1. **学习率 6e-06 → 1.2e-05（2×）**。依据：v6 实测 KL 0.0038–0.0045（只用到 `target-KL 0.05` 的 8–9%）、
   `clip_frac` 4.4%、`ent` 1.43→1.15（在收敛，不是没学）⇒ 步长明显偏小。**`target-KL` 保持 0.05 不动**
   （它是唯一真正拦过灾难的闸）。止损线：KL 早停 > 10% 或 `ent` < 0.8 就退回 9e-06。
2. **后期加压**（用户原话"拉高一丝，试一下会不会炸"）：场地目标 T12 起比上一轮实测高
   1.15→1.30×（T12 1,550 / T14 2,860 / T16 4,700 / T18 6,400），前期不动。
   目的：逼它从"线性堆身材"转向"经济滚雪球"——线性买牌够不着这个高度。
3. **规模先不扩**：每轮固定 3 小时，慢慢累积（用户："每次都 3 个小时，就慢慢来，不着急"）。

**观察到的行为**（v6 终局阵容 512 局）：T4+T5+T6 牌占 **73.7%**，核心是「梦魇茶客」(47.5%)+
机械(901)/龙(662)；512 个阵容**全不相同**（没有退化成照抄一套），但**没有机制协同**，是纯堆身材。
（**2026-11-15 更正**：核实牌面后，"没有机制协同"是误判——梦魇茶客产「乱放的茶具」酒馆法术、
胡蒙格斯放大酒馆法术、蓄能女沙皇吃法术施放、希望信使吃圣盾掉落、坠落的飞天魔像吃亡语触发，
是一张真协同网。见 D-065。）

## D-065 · 2026-11-16 · local · **v9 三轮连训：tier 目标完美达成，但棋力没有进步（进入平台期）**

**配置**：起点 arena-v8 → 三轮各自接上一轮产物（seed +1）；强制计划延长到
`2:2,4:3,5:4,7:5,8:6,10:6,12:6,14:6,16:6,18:6`（T2–T19 零空档）；`--freeze-cap 2`；
lr 1.2e-05 / target-kl 0.15 / entropy 0.008 / anchor 0.02 全程；训练场固定 `arena-cal-v8-snap`、
考试场固定 `arena-cal-v7hold-snap`。

**三轮训练全部干净**：547 / 561 / 567 轮，各 ~164 分钟；`illegal/invalid/truncated = 0`；
KL 早停 1 / 0 / 0 次；吞吐 8,600–9,300 决策/秒；`ent` 从 1.15 升到 1.40–1.49（探索确实变多）。

**结果（四代统一在新 10 档计划下、同场 512 局）**：

| | T12 | T16 | T18 | mean_rank | 前四 | 吃鸡 |
|---|---:|---:|---:|---:|---:|---:|
| **arena-v8（基准）** | **1,381** | **3,729** | **5,553** | **2.604** | 82.4% | 44.7% |
| R1 | 1,346 | 3,628 | 4,566 | 2.695 | 80.3% | 44.5% |
| R2 | 1,306 | 3,687 | 5,037 | **2.531** | 82.4% | **47.9%** |
| R3 | 1,299 | 3,457 | 4,933 | 2.725 | 77.9% | 45.9% |

**结论**：
1. **tier 目标彻底达成**：T10 起平均 tier = **6.00**（旧计划是 5.89–5.94），每一局都到 6 本。
   这是"强制计划延长到 T18"的功劳，**不是三轮训练的功劳**（v8 在新计划下同样 6.00）。
2. **棋力没有进步**：三轮之间 mean_rank 在 2.531–2.725 之间摆动（跨度 0.19），
   **大于它们与 v8 的差距（0.07）** ⇒ 差异基本是**轮间噪声**，不是真实提升。
   三轮合计约 5,400 轮（600M 决策）**没有产出可测的进步**。
3. **v8 仍是最强**（或与 R2 并列）；R2 的吃鸡率 47.9% 是四代最高。
4. **没有提高战团库压力**：最强的 v8 在 T18 只有 5,553 / 库 6,416 = **87%**，
   T16 是 78%，离"明显超过"很远，按用户授权条件不该加压。

**判断**：现有配方（100% 强制升本 + 校准场 + 高熵高 KL）**已经到平台**。
下一步需要换杠杆，而不是继续同配方叠加轮数。候选见 `PROJECT_STATE` §7。


## D-060 · 2026-11-14 · local · **评测口径定稿：每轮只测一次、默认 held-out 场、M1 为主判据**

**用户决定**（2026-11-14）：判模型强弱只看**逐回合中位战力 M1**；每轮训练**只评测一次**，
训练场或没见过的场都可以，不要为了稳妥跑两次。

**我的补充并被接受**：既然只跑一次，就**固定用 held-out 场** —— 成本和训练场完全一样
（512 局 / 1 卡 / ~11 分钟），但顺手去掉"候选在自己训练场上占便宜"这个隐患。
本轮为验证隐患是否存在跑过两次：训练场 2.615→2.352（−0.264）、held-out 2.918→2.637（−0.281），
**效果量几乎一致 ⇒ 隐患很小**，所以以后不必再跑两次。

**落地**：`docs/EVAL_PROTOCOL.md` §1 改为"M1 主判据 + 名次面板并列交叉检查（免费附带）"、
"每轮一次、默认 held-out"；§4 规则 6 保留"考试场必须 held-out"。

## D-061 · 2026-11-14 · local · **撤回一个错误诊断：分段续训不会破坏 anchor 保护期**；并修掉两处真实缺陷

**撤回**：我曾断言"`start_i = trainer.updates` 是优化步数，分段续训会让第二段的 β=0.1 直接失效"，
并据此把长训改成单段、还写进了 `TRAINING_RECIPE` 与 `PROJECT_STATE`。**这是错的**：
`ppo_trainer.py:381` 的 `self.updates += 1` 在 `update()` 里，而 `update()` **每训练轮只调用一次**，
所以 `trainer.updates` 就是**累计轮数**（铁证：正式作业末行 `[iter 302] ... updates=302`）。
`i` 本来就按累计轮数走 ⇒ 续训时保护期照常延续。**已更正** sbatch 注释、`TRAINING_RECIPE` §2.1、
`PROJECT_STATE` §6；分段续训恢复为安全可用。

**顺带修掉的两个真实缺陷**（都是本轮实测暴露的）：
1. **K4 `--max-updates` 单位错**：按 4,650（当优化步数算）设，302 轮只退火 6.5%，lr 全程 5.94e-06。
   已改成 **300**（= 论文一个 stage 的轮数）。
2. **K1 零更新把流水线永久打回串行**：第 15→16 轮一次 KL=0.091 触发早停 → 该轮零优化步 →
   `--pipeline-zero-step-fallback 1` 关掉执行器且**再没恢复**，19.5 → 31.9 s/轮（−39%）。
   已在 `train_entity_sync.py` 修：**在 pending 批次消费完毕后重新武装重叠采集**，
   零更新回退只影响一轮。`py_compile` 通过，待下一轮实测确认（日志会出现
   `[pipeline] 重新启用重叠采集`）。
## D-066 · 2026-11-16 · local · **提高吃鸡奖励：新增 `--win-bonus`（冠军奖金），v13 用 b=0.7**

**背景**：v12（高压场）名次显著变好（4.837→4.548、前四 42.5%→47.6%、吃鸡 19.8%→22.9%，
配对 CI [+0.088,+0.485]）但 T12–T18 中位战力全线 −1~−5%，判定 SUSPICIOUS。
归因：**奖励只含名次**，在 p85 高压场里"T14 之后叠板面的边际名次收益≈0"，
模型于是把资源全投前期（T4–T8 +4~10%）。用户确认：v12 的 Top5 阵容"很有高手风采"
（袋中旋风/扩音机肖像+蜡油长枪/玛佐克萨斯匕首/餐具肖像，全部非加重护手），要求从 v12 续训并**提高吃鸡奖励**。

**决策**：名次效用表 `u(1)` 由 `1.0` 改为 `1.0+b`，**其余 7 个名次完全不变**。
选 `b=0.7`（用户要"激进但不变态"）：冠军/亚军 = 1.40× → **2.38×**。
选它的理由：①只动冠军一格，保住已学到的"稳前四"结构；②比 b=0.5 激进一档，
又没到 b=1.0（2.80×）那种纯赌徒线；③b=0 时与论文原值逐位相同，可随时回归。

**实现要点（一致性）**：critic 是名次交叉熵头，`V(s)=Σ p·u`。若只改终局奖励不改 V，
价值基线会与奖励不一致。所以做成**同一张 `RANK_UTILITY` 表同时喂两边**，并用
**原地改写**（`set_win_bonus` 改数组元素）而不是换对象——`model/ppo/train_loop` 都是
`from .common import RANK_UTILITY` 持有同一 ndarray 引用，改元素即全局生效。
CLI：`--win-bonus`（默认 0）；sbatch：`HSAI_ARENA_WIN_BONUS:-0.7`，并写进 `run.env`。

**护栏**：吃鸡率 1024 局标准误 ≈ ±1.5pp，须 **≥+3pp** 才算真涨；
若**前四率掉 >4pp**（现 47.6%），判定"在乱赌"，回退 b=0.4。

**验证**：`set_win_bonus` 单测（b=0 逐位等于论文值、b=0.7 → [1.7,0.714,…]、b=0 可复原）；
`check_flags` 77/77；sbatch `bash -n` 通过。提交 `73222fb`。

## D-067 · 2026-11-16 · local · **节点 xgpi11 有 20 倍以上降速，必须长期排除（本次事故复盘）**

**现象**：v13 的 4 分钟 smoke 卡住 14 分钟不出第一轮，`iter=0`，但 32 核满载（CPU 时间 7h36m）。
看起来像死锁，实际不是：同一份代码在 xgpi14 上 3 分 47 秒跑完 7 轮。

**证据（按节点分界，无一例外）**：

| 作业 | 内容 | 节点 | 结果 |
|---|---|---|---|
| 1150096 | v12 smoke | xgpi14 | 完成 3:47 |
| 1150153 | v12 训练 165min | xgpi14 | 完成 |
| 1151516 | v13 smoke (b=0.7) | xgpi11 | 卡死 |
| 1151627 | 隔离A：b=**0**，v13场 | xgpi11 | 卡死 |
| 1151628 | 隔离B：b=0.7，**v12场** | xgpi11 | 卡死 |
| 1151697 | 隔离D：**起点v11**，b=0 | xgpi11 | 12 分钟只走到 `[net]` |

三个隔离实验分别排除了「奖励」「场地」「起点模型」，只剩节点这一个共同因子。

**历史佐证**：项目早期配置文件本来就写了
`--exclude=xgpi11,xgpi15,xgpi17,xgpi20`（`elo_8.sbatch`、`leaderboard_*.sbatch`、
`ab_1095539.sbatch` 都能查到），但后来新建的 `stage_arena.sbatch` / 评测 sbatch
只写了 `xgpi15,xgpi17` —— **黑名单在配置演进中丢了**。这不是新问题，是回退。

**处置**：
1. `stage_arena.sbatch`、`eval_human_gap_snap.sbatch`、`eval_shard_parallel.sbatch`
   全部改回 `--exclude=xgpi15,xgpi17,xgpi11`（提交 `e36cb89`）。
2. 顺带修掉一个自造故障：用 PowerShell `Set-Content -Encoding utf8` 改 sbatch 会加
   **UTF-8 BOM**，导致 `sbatch: not a batch script`；已用无 BOM 方式重写并验证首字节
   `23 21 2f`（`#! /`），提交 `ed5b056`。（与 D-050 同一类教训：远端文本不经 PowerShell 中转。）
3. **以后新写 sbatch 必须照抄这条 exclude**；怀疑"卡住"时先看 `sacct --format=NodeList`
   与健康节点对比，不要先怀疑自己的代码。

**附：如何快速判断有没有空卡**（本次用到的命令）
`sinfo -p gpu -h -N -o '%N|%t|%C|%G'` 看节点状态与切片型号；
`squeue -p gpu -h -t R -o '%N|%b'` 累加每节点已用切片数（每节点共 4 个）。

## D-068 · 2026-11-17 · local · **v1.3 魔鬼训练：只奖前二 + 纯教师 BC + 封禁加重护手 + T1–T20 高压库 + 三轮链**

用户定的方向："让模型学会玩很强的后期阵容"。为此这一轮同时改了四件结构性的事，
全部由用户拍板、Agent 落地并逐项静态验证。

**1. 奖励改成「只奖励前二」**（新增 `--rank-utility-table`，提交 `bd9398a`）
整张名次效用表直接指定为 `[1.00, 0.35, -1.00, -1.15, -1.30, -1.45, -1.60, -1.75]`：
- 1 名→2 名差 **0.65**（论文 0.286 的 2.3 倍）
- **2 名→3 名断崖 1.35**（论文 0.286 的 4.7 倍）← 主设计点
- 3–8 名之间只留 0.15 梯度（保留"别死太早"的微信号）
`--win-bonus` 归 0（整表优先，两者同给时以整表为准并告警）。
**同一张表同时喂终局奖励与 `V(s)=Σp·u`**，两边恒等。

**2. 升本：从「强制」改成「纯教师 BC」**（提交 `b290c45`）
`--forced-tier-prob 0`（一次都不强制）+ `--forced-teacher-until 1B`（全程）+ `--forced-teacher-coef 0.1`，
教师计划 `2:2,5:3,7:4,8:5,10:6`（6 本里程碑从 T12 提前到 **T10**，对齐真人：真人 T10 已有 59.9% 到 6 本）。
教师只在"到点 + Upgrade 合法"时每回合写**最多一条** `teacher_action=3`；金币不够不写、模型自己提前升不惩罚。
静态验证 `ALL_STATIC_OK`（`tests/test_teacher_bc_static.py`）。

**3. 封禁「加重护手」BG36_MagicItem_414**（提交 `f23cd33`）
用户判断"这个饰品让模型后期无脑变大、学不到操作"。**不改模拟器**：在策略取掩码之后、采样之前，
把该选项（`A_CHOOSE+i`）置为非法（`hsbg/az/train/action_ban.py`），训练与评测共用同一份代码，
采样/logprob/PPO 损失三条路径同用一份 mask ⇒ 不伪造 on-policy，`illegal_actions` 仍为 0。
只在"真有合法待选动作"时才解析 player_json（零开销守卫）。10 项静态验证 `BAN_STATIC_OK`。
**边界**：只能拦"摆在你面前让你选"的饰品；随机白给类（如未知宝珠）拦不住，属引擎随机性。

**4. 战团库重建成 T1–T20 高压场**（提交 `aaf16f7f`）
目标 `5/5/16/25 / 40/65/100/160 / 350/900/1600/3000 / 4500/6000/7000/8000 / 9000/10000/11000/12000`。
训练场 `arena-cal-v14`（16,658 阵容）/ 考试场 `arena-cal-v14hold`（2,355 阵容），目标逐位一致。
每档 ±10% 带内 448–1,721 条真人战团。**T19/T20 首次覆盖**（450/448 条）——
之前只建到 T18，`mirror_pool.get()` 找不到帧就跳过注入，模型活到 T19 时对面会退回弱 bot。

**过程**：三轮链（`deploy/slurm/chain_3rounds.sh`）v1.2 → r1 → r2 → r3，每轮 165 分钟，全部 `afterany`，
**中间不评测**（用户定），三轮后一次跑四模型 × 1024 局统一评测（v14hold 场、无强制升本、3 节点 × 8 进程）。
模型改名：`arena-v1.2`（= `arena-v13-20261116` 的硬链接别名，同 inode 不占双倍空间）。

**评测提速的三次实测**（写进来备查）：
- 1 卡 8 进程：0.58 s/局（聚合 458 决策/s）← 最优
- 1 卡 1 进程：1.29 s/局
- 纯 CPU 8 进程：1.88 s/局（慢 3.3 倍，**否决**）
⇒ 提速只能加卡，每卡仍配 8 进程。当前集群只有 3 块 h100-47 空闲，所以用 3 节点。

---

## D-069 · 2026-11-17 · local · **全项目审计 + P0/P1 修复：评测可信性、统计配对、架构兼容、门禁代码化**

用户授权："按审计报告自主制定修复计划并执行；优先 P0 与影响正确性的 P1；修复必须有测试；
尽量修根因；不破坏现有模型与检查点；训练策略/场地/奖励等重大方向只出研究说明，不擅改。"

**审计结论（详见 `docs/analysis/AUDIT_20261117.md`）**：工程骨架比文档更扎实（PPO 数学、
GAE、掩码一致性、注入 RNG 配对、续训状态、战斗规则主体均经复核正确），但当时处于
"测量系统失灵 + 训练信号枯竭"状态：
1. **主评测入口在 HEAD 上崩溃**（`2acfebf` 删掉了 `PPOEvaluator` 构造却留下
   `if evaluator is None:` ⇒ 自家 checkpoint 全 `UnboundLocalError`）；
2. **判胜链路没有 `protocol_sha256`**、面板没记场地（`opponent` 恒为 `"engine_bot x7"`）；
3. **多模型分片互相覆盖** `rows.jsonl.gz`，补救工具让 `illegal/truncated` 只覆盖 1/24 分片；
4. **配对按数组下标**（缺中间分片 ⇒ CI 假性跨 0），M1 是存活条件中位数（会把"活得更久"
   读成"M1 退化"）；
5. **`pytest tests` 直接 INTERNALERROR**（3 个测试文件模块级 `raise SystemExit`）；
6. **借鉴的架构开关不可加载**（z-token/属性/磁力查询改变键集却不写 meta、也不推断），
   且 `z_global` 是**死代码**（算了从未加进 global token）；
7. **`TRAINING_RECIPE §5` 的红线一条都没实现**（非法动作/截断/非有限损失不会停机）。

**已修（全部带测试，见 AUDIT_20261117 §2 的表）**：`eval_policy.py` 成为唯一策略构造入口；
面板写全 protocol + 指纹 + 注入统计；按模型名落盘 + 两个 merge 工具跨分片求和/拒绝缺分片；
`seed_sequence` + 按 seed 配对 + 共同存活子集 M1；`accept_gap_panel` fail-closed；
三个脚本式测试改成真 pytest；ckpt 架构开关持久化 + 中立槽归零；教师 CE 归一化口径
（默认不变、可选 `valid`、写 `teacher_label_share`）；`health_gate.py`；
根目录白名单/大文件门禁 + 三个 `.pt` 移入 `data/thirdparty/`；文档断链与 `--help` 崩溃。

**明确没做（等用户决定）**：训练场强度与重建、λ/信用分配等超参、对手是否"会行动"
（静态快照 vs 自博弈 vs 联赛）、引擎侧随机待选与单流 RNG、`next_is_ghost` 等观测位、
`mirror_hard_gates`/`quality_monitor` 接进 operative 链。

**因 Bug 而失效的历史结论（已登记，不改原文）**：
- **D-068 第 3 项在 v1.3 三轮里完全没生效**：`player_json['pending']` 是 list + 外部标记枚举，
  旧代码按 dict 判定 ⇒ 恒 False（`action_ban.py` 自述 "r3 仍选中 496 次 / 3.81%"）。
  修复提交 `5a67150`(10:16) 晚于三轮链结束(09:39)与评测(10:03) ⇒
  **v1.3 三轮不能用来回答"封禁加重护手有没有用"**。
- **跨代名次对比不可用**：每代场地都按当时模型重建，且当时面板无协议指纹 ⇒ 不同榜。

**下一阶段建议（用户已同意先研究后决定）**：先跑 10 分钟 `R²(板面→名次)` 门禁 →
单变量实验（T12–T20 靶 ×1.15）→ 之后再把"后期运营"变成可学信号（金卡 0.97/局 vs 真人 2.55、
留存增重 +87 vs +690、每格 atk 123 vs 598）。研究报告：
`docs/analysis/TRAINING_OPPONENT_DESIGN_20261117.md`。

## D-070 · 2026-11-17 · local · **收紧/放宽三个动作上限：max-actions 80→100、换位 cap 5→2、Freeze cap 2→1**

用户决定（原话："回合动作上限到100；换位每回合最多2次；冻结每回合最多一次"），要求
**立即应用到所有训练、评测**。这属于 `AGENTS.md` §4 的**冻结契约**变更，用户已明确授权。

**1. 实测依据**（`docs/analysis/ACTION_CAP_20261117.md`）
- `max_actions_per_turn = 80` **几乎从不触发**：评测 r3（1032 局 = 13,031 个回合实例）
  只有 **3 个回合**撞到（**0.023%**，3 局 / 0.29%）；训练三轮（2.098 亿决策）`forced_end_turns`
  合计 **1,967**（≈0.015% 的回合）。逐回合动作数 p50=12 / p90=34 / p99=52 / p99.9=67 / max=81。
  ⇒ 放宽到 100 只是给尾部留余量，**不改变统计行为**（但会换协议指纹）。
- 真正饱和的是另外两个上限：`move` 每回合**打满 5 次**（评测使用率 100%；训练累计被 cap
  掩掉 26.8 亿个合法动作位），`freeze` 使用率 **99%**（cap=2 时均值 1.9 次/回合）。
  ⇒ 收紧它们才是"治动作空转"的方向。

**2. 具体改动**
| 项 | 旧 | 新 |
|---|---:|---:|
| `max_actions_per_turn` | 80 | **100** |
| `move_cap`（每回合换位次数） | 5 | **2** |
| `freeze_cap`（每回合按 Freeze 次数） | 2 | **1** |
| `move_repeat_limit` / `forbid_cancel` / `forbid_unfreeze` / `ban_trinkets` | 2 / 1 / 1 / 1 | **不变** |

**3. 根因式实现（不是改 40 份字面量）**
- 唯一真值仍是 `hsbg/az/train/eval_policy.py::TRAINING_CONTRACT`；**所有**硬校验改成对着它比：
  `PPOGreedyPolicy`、`PPOSelfPlayPolicy`、`train_entity_sync` 的 `ap.error`、`DuelSeatPolicy`、
  `load_config`（duel / round-robin Elo / 3-model Elo / two-model trinket / gauntlet）、
  `capture_trajectory`、`train_az_ppo`、`build_v1_1_panel`。
  以前这些地方各写一份 `80/5/2` 字面量，契约一变就要同步 8+ 处，漏一处就是"训练用新契约、
  评测用旧契约"的静默错配（审计 P2-2 的根因）。
- 顺手补齐 Elo 路径**完全缺失**的三项（`freeze_cap` / `forbid_unfreeze` / `ban_trinkets` 的掩码），
  以前 `DuelSeatPolicy` 只有 move-guard + 禁用 Cancel。
- 默认值同步：`CollectConfig`、`PPOGreedyPolicy`、`MoveGuard.DEFAULT_MOVE_CAP`、训练/评测 CLI、
  `quality_monitor`（原 40）、`thirdparty_policy`、七个 `config/eval-*.json`。

**4. 影响（必须记住）**
- **换榜**：契约进 `protocol_sha256` ⇒ 所有旧面板（含 `v1.3_r3` 的 1032 局）与新面板**不可混比**。
  下一轮评测必须重跑对照。
- **模型行为会变**：`move`/`freeze` 的动作被掩掉后策略只能改选别的动作，因此 T12 起
  "搬随从/冻店"的空转会显著减少，`refresh`/`buy` 的分配可能跟着变——这是**预期效果**，也是下一轮
  要看的主要行为指标。
- **验证**：全量 `pytest tests` 877 用例通过（含按新契约重写的 move-guard 断言）。

---

## D-071 · 2026-11-17 · local · **lobby 种族排除开关（机械 + 龙）：不改引擎的确定性种子过滤**

用户提出："下一次训练想把机械和龙 ban 了，让模型先学会玩别的阵容。"

**1. 现状（引擎口径）**：每局从 **10 个类型均匀抽 5 个**
（`flow.rs` optional = 9 个部族 + ABERRATION，`cfg.lobby_races = 5`）。
`RulesConfig` 只有**强制加入**（`lobby_types`）与 `aberration_forced`，**没有排除列表**；
`RulesConfig::from_json` 对未知键报错，所以配置层也塞不进去。
另：**NAGA 不在抽签池里**（11 个 `Races::NAMES` 中 NAGA 只经双种族卡出现），
所以本引擎的"10"= 9 部族 + Aberration，与真实游戏的 10 个部族不完全一致。

**2. 为什么不改引擎**：`source_hash` = 两个 crate 全部 `.rs` + `cards.json` 的 FNV 指纹
（当前 `c4c71014e3d41760`）。动一行 Rust 就换指纹 ⇒ 所有模型 manifest 的引擎锚点失效，
且需要计算节点重编（本机无 MSVC）。**收益不值这个代价**。

**3. 实现**：`hsbg/az/train/lobby_filter.py` —— **确定性种子过滤**
1. `Game(seed)` 建局（很便宜，不跑模拟）；
2. 从**观测** global 块最后 11 位读 lobby 种族（`f[42:53]`；已实测与
   `state_json()["lobby_races"]` 的 bitmask 逐位一致）；
3. 含被禁种族 → 用 `derive_seed(base, k)`（splitmix64 混合）取下一个候选，重来。
接受率 = C(8,5)/C(10,5) = **22.2%**（禁 2 个）⇒ 平均建 4.5 局；实测 50 局过滤 < 2 s。
候选种子只由 `(base, k)` 决定 ⇒ **同一 base 下所有模型面对同一批局**（配对统计仍成立），
且与其他分片不撞。

**4. 接线（训练 + 评测同口径）**
- `CollectConfig.lobby_exclude` / `EvalConfig.lobby_exclude`（空 = 不过滤，逐位等价旧行为）；
- 三处建局全部改走 `make_filtered_game`：串行采集、lockstep 采集、`evaluate.py`；
- CLI：`--lobby-exclude MECHANICAL,DRAGON`（训练入口与 gap 评测入口都有）；
- **`seed_sequence` 现在记录实际使用的种子**（过滤会换种子，不能再按 `base+gi` 推）。
- 测试：`tests/test_lobby_filter.py`（12 条，含真实引擎验证"lobby 里真的没有机械/龙"）、
  `tests/test_lobby_wiring.py`（5 条，含一次真实评测端到端）。

**5. 风险（用户已知悉并接受为"课程"）**
- **商店池变浓**：5 抽 8 而不是 5 抽 10 ⇒ 同族卡更容易凑齐（难度感知会变）；
- **删掉的正是模型现在最强的两条线**：r3 的终局阵容 Top5 里 4 个是龙/机械
  （希望信使 33.9%、蓄能女沙皇 30.0%、赤红守备巨龙 28.9%、坠落的飞天魔像 27.8%）
  ⇒ 下一轮**预计会出现性能回落**，这是换取"学习其它阵容"的代价；
- **与真实分布不一致**（真实是 10 部族全开）⇒ 该轮结果不能当作"实机强度"，
  只作课程内部对照；恢复时把 `--lobby-exclude` 去掉即可，其余配方不变。
- **换协议**：环境变了 ⇒ 与 r3 的面板不可混比。

**6. 默认关闭**：当前所有入口的默认值都是"不过滤"。启用只需在提交作业时加
`HSAI_ARENA_LOBBY_EXCLUDE=MECHANICAL,DRAGON`（`stage_arena.sbatch` 已透传，见下一条）。

---

## D-072 · 2026-11-17 · local · **补上"借鉴结构"的 CLI 入口 + 逐项判断要不要做**

用户给了一张"参考第三方训练模型的 5 项设计"表，问本项目有没有对应改动、有没有必要做。

**1. 现状核对（结论：4/5 在代码里，但 3 个打不开；2 个是空转）**

| # | 第三方设计 | 我们有没有 | 实际状态 |
|---|---|---|---|
| ① | 46 阵容条件化 token `z` | 有 `use_z` / `n_z=46` / `zcard` / `z_global` / `z_tok` | **默认关**；`zcard` 是**全零 buffer 且全仓无生产者** ⇒ `z_tok` 半边恒等；`z_global` 原本是死代码（D-069 已修）；**中立槽 45 已强制归零** |
| ② | 311 维卡牌属性 + id 双路 | 有 `use_attr` / `attr_f=311` / `attr_proj` | **默认关**；`attr` 也是**全零 buffer 无生产者** ⇒ 投影输出恒 0，等于没加 |
| ③ | magnetize 双线性头 | 有 `mag_query`（零初始化） | **默认关**，但**唯一真正可训**的一项（它有梯度、起点等价旧版） |
| ④ | 奖励 `[1, .2, 0, 0, 0, 0, -.1, -.2]` | 有 `--rank-utility-table` | **方向一致、形状不同**：现役是 `[1.00, 0.35, −1.00, −1.15, −1.30, −1.45, −1.60, −1.75]`。差别在 3–6 名：他们给 **0**（无梯度），我们保留 0.15 的小梯度（还留着"别死太早"） |
| ⑤ | `no_cancel_choose` | 有 `pending_guard.no_cancel_choose` | **默认关**，且被我们更强的 `forbid_cancel=1`（全局禁 Cancel）**完全覆盖** ⇒ 加了也等于没加 |

另外发现：这三项结构开关**在训练入口没有任何 CLI**（只有 `build_entity_net` 的参数），
意味着"实现了"等于"永远打不开"。本轮补上 `--use-z/--n-z/--use-attr/--attr-f/--mag-query`，
默认全关（逐位等价旧模型），并加 `--help` 回归。

**2. 逐项判断（要不要做 / 有没有条件）**
- **② 311 维卡牌属性 → 最值得做，且有条件**：不需要上游数据，我们自己能从
  `engine/crates/hsbg-engine/data/cards.json`（含 `atk/hp/tier/races/keywords/text/kind`）
  生成 311 维表并灌进 `attr` buffer（写一个 CPU 工具 + `--use-attr 1`）。收益：新卡/冷门卡
  的泛化更稳。成本：一天内可完成。
- **③ 磁力双线性 → 零成本可开**：代码已在、零初始化、可训。建议作为下一轮的**第二变量**单独测
  （不要与场地/奖励改动同轮，否则无法归因）。
- **① 46 阵容 z → 有条件但**暂不建议**：`zcard` 需要真表（上游权重里才有），而且训练时必须
  有一个"当前目标阵容"的输入源（课程）。我们现在的瓶颈是"后期乘法成长"，而 z 解决的是
  "打指定阵容"，**前提是先会玩多个阵容**。等 ②③ 与场地问题解决后再谈。
- **④ 奖励表 → 不建议改成它**：3–6 名零梯度会让"保前四"完全失去信号；我们保留小梯度更适合
  "分代比较 + 名次只作参考"的口径。若要更狠的"只奖前二"，把 3–6 的负值加深即可，不必归零。
- **⑤ no_cancel_choose → 不需要**：`forbid_cancel=1` 更强（全局禁），做了也没有行为差异。

**3. 改动**：`train_entity_sync.py` 新增 5 个默认关的架构开关并透传给 `build_net`；
文档记录判断；全量测试 877+ 通过。

---

## D-073 · 2026-11-17 · server · **新契约下重测 r3（512 局）：名次显著变差 ⇒ move/freeze 不是纯空转**

用户要求"用当前环境再评一次 r3，512 局"，用来验证 D-070 的三个上限改动。

**1. 运行**：作业 1156278，节点 xgpi13，8 分片 × 64 局，场地 `arena-cal-v14hold-snap`，
`protocol_sha256 = 10e0f9f0…`，`illegal=0 / truncated=0 / 注入 99.99%`（45,409/45,415）。
墙钟 **194 s**（8 进程 / 1 张 MIG 切片）⇒ **0.38 s/局、2.6 局/s**。

**2. 结果（新契约 100/2/1）**：mean_rank **4.9141**、前四 **40.2%**、吃鸡 **8.4%**、
决策/局 **182.2**；M1 T12 **1287** / T14 **2959** / T16 **5609**；T12 存活 70.3%、T14 27.1%。

**3. 同种子配对对比**（旧契约 r3 的前 512 局，seed 930000..930511，n=512，按下标配对）

| 指标 | 旧 80/5/2 | 新 100/2/1 | Δ |
|---|---:|---:|---|
| mean_rank | 4.4902 | 4.9141 | **+0.424（变差）**，95% CI [+0.545, +0.703] |
| 前四 | 48.4% | 40.2% | −8.2pp |
| 吃鸡 | 14.3% | 8.4% | **−5.9pp（腰斩）** |
| 决策/局 | 207.5 | 182.2 | −25.3（−12%） |
| M1 T12 / T14 / T16 | 1376 / 2999 / 5669 | 1287 / 2959 / 5609 | −6.5% / −1.3% / −1.1% |
| T14 存活 | 33.1% | 27.1% | −6.0pp |

T12 动作混合：`move 5.02 → 2.00`、`freeze 1.91 → 1.00`（**精确等于新 cap ⇒ 策略仍想按更多，被夹住**），
而 `buy 3.84 → 3.77`、`play 7.58 → 7.68` **几乎不变** ⇒ 省下的动作**没有**变成更多买牌/打牌。

**4. 结论（两条必须分开）**
- ✅ "move 打满 cap 是纯空转"这个假设**被证伪**：move 承担战斗站位（攻击顺序/嘲讽保护），
  砍到 2 次/回合后吃鸡率腰斩；`freeze 2→1` 也让"冻住关键牌凑三连"变难。
- ⚠️ **不能**据此断言"cap2/1 本身更差"：r3 是在旧契约（5/2）下训练的，本次是"旧模型 + 新契约"的
  **OOD 评测**。要分离"特化"与"本质"，必须按新契约重训一轮再看。
- ✅ `max_actions 80→100` 无害（评测里只有 0.023% 的回合触发原上限）。

**5. 建议（等用户定）**：不要再用"混合改动 + 旧模型"下结论。做**单变量消融**，每格成本
= 一轮 3h 训练 + 一次 3min 评测：A `100/5/2`（只放宽上限）｜B `100/5/1`（只收 freeze）｜
C `100/2/2`（只收换位）｜D `100/2/1`（当前）。在拿到"同契约训练"的结果前，保留还是回退由用户定；
评测侧两种都能跑（契约写在 panel 的 `protocol` 里）。

**6. 顺带修掉的两个真实缺陷**
- `merge_panels` 以前只把 `protocol_sha256` 带进合并面板，`contract/field/engine` 全丢 ⇒
  这次的面板一度无法自证"比的是什么"。已改为读分片 `panel.json` 的**完整 protocol**
  （`tests/test_merge_panel_protocol.py`）。
- `eval_shard_parallel.sbatch` 的 `--expect-models "${#specs[@]}"` 数的是数组元素
  （每个模型 = `--checkpoint` + `name=path` 两个）⇒ 单模型报 2、merge 直接拒绝写盘。
  改成 `${#pairs[@]}`。

---

## D-074 · 2026-11-17 · server · **r3 在"新契约 + 完整协议"下的正式基线（512 局，可复现）**

D-073 那份面板是"旧协议定义"算出来的（`protocol_sha256` 不含 `lobby_exclude`），
不能与新评测混榜。用户批准重跑一次，产出一份**协议完整、可长期比**的基线。

**运行**：作业 1156519，节点 xgpi12，8 分片 × 64 局，墙钟 **209 s**。
产物：`artifacts/runs/gap-r3-baseline-512-20261117/`

```
name              v1.3_r3
checkpoint        artifacts/models/arena-v1.3-r3/model.pt
checkpoint_sha256 1808f178…c206472
arch              entity            seeds 930000..930511   shard_count 8
protocol_sha256   7e15f114…b5b56c
contract          max_actions 100 / move_cap 2 / freeze_cap 1 / forbid_cancel / forbid_unfreeze
                  / ban_trinkets；lobby_exclude []（本次不 ban 种族）
engine            source_hash c4c71014e3d41760（未改引擎）
outcome           mean_rank 4.9141 · top4 40.23% · win 8.40% ·
                  illegal 0 · truncated 0 · 182.2 决策/局
```

**逐条验证（从 6,336 行逐回合记录实测，不看配置）**

| 检查 | 实测 | 判 |
|---|---|---|
| 每回合动作数 ≤100 | max **87**（0 回合达 100） | ✅ |
| 每回合换位 ≤2 | max **2** | ✅ |
| 每回合冻结 ≤1 | max **1** | ✅ |
| Cancel(7) | **0** | ✅ |
| 每回合 end_turn 恰 1 | 6,336/6,336 | ✅ |
| 动作 id 上界 | 243（无 245+ 扩展段） | ✅ |
| 加重护手（3037）持有 | **0** | ✅ |
| 注入落地 | 99.99%（45,409/45,415） | ✅ |

**可复现性**：同一模型 / 同一 seed 集 / 同一契约 / 同一场地重跑，`rank_sequence`
**512/512 逐局相同**、每回合 M1 全部相同 ⇒ 评测链路是**确定性**的。

**与 D-073 那份的关系**：数值完全一致，只有 `protocol_sha256` 不同
（`10e0f9f0…` → `7e15f114…`，因为协议定义多了 `lobby_exclude` 字段）。
**正式基线用这一份**；D-073 那份只作"同一模型的重复计算"参考。

---

## D-075 · 2026-11-17 · server · **种族 ban 三个驱动脚本全部漏传（静默失效）+ 禁龙机械实测**

用户质问"512 局到底 ban 没 ban"。核查结果：**没有 ban**，而且原因比"忘了开"严重。

**1. 根因（我的失误，三个都漏）**
`eval_human_gap.py` 有 `--lobby-exclude`，但调用它的 sbatch 里**有三个没传**：
`eval_shard_parallel.sbatch`（两次 512 局评测用的就是它）、`eval_cpu_bench.sbatch`、
`eval_human_gap.sbatch`。⇒ 即使设了 `HSAI_GAP_LOBBY_EXCLUDE`，也会被**静默忽略**，
跑出来照旧 10 抽 5、照旧带龙/机械。这与 D-069 修的"契约不闭合"是同一类错误：
**开关存在 ≠ 开关生效**。

**2. 已修**
- 三个 sbatch 全部补 `--lobby-exclude "${HSAI_GAP_LOBBY_EXCLUDE:-}"`；
- `run.env` 现在写死实际生效值（`lobby_exclude=… / freeze_cap=… / max_actions=100 move_cap=2`）；
- **静态测试** `tests/test_lobby_wiring.py::test_every_eval_driver_passes_the_flag`：
  任何调用 `eval_human_gap.py` 的 sbatch 若不含 `--lobby-exclude`，测试直接红。
  （正是它抓出了第二、第三个漏传的文件。）
- 注意：`MECHANICAL,DRAGON` **自带逗号**，不能走 `sbatch --export=K=V`（逗号是分隔符）；
  必须 `export` 进环境后 `--export=ALL`。

**3. 禁龙机械的实测（作业 1156628，8 分片 × 64 局，墙钟 219 s）**
`protocol_sha256 = daee3710…`，`contract` 不变（100/2/1），`lobby_exclude=['MECHANICAL','DRAGON']`。
- **lobby 层 100% 生效**：385/512（**75.2%**）局被替换；用引擎复算实际使用的 512 个 seed，
  **0 个** lobby 含机械或龙；剩下 8 个部族各占 59.6%–65.0%（理论 5/8=62.5%）。
- 其他限制照旧：max actions/turn 82（≤100）、move 2（≤2）、freeze 1（≤1）、cancel 0、
  加重护手 0、`illegal 0 / truncated 0 / 注入 100%`。
- 数值：mean_rank **5.3789**、前四 **27.73%**、吃鸡 **3.91%**、决策/局 **186.3**、
  M1 T12 **1379** / T16 **5938**。

**4. 但"ban 种族"只覆盖**商店池**，两个残留在实测中被抓到**（不是 bug，是定义边界）
- **双种族卡仍会出现**：`BG27_005`（龙+海盗）×92、`BG32_821`（恶魔+龙）×51、
  `BG_DEEP_015`（亡灵+机械）×9、`BG36_764`（机械+鱼人）×6 —— 它们靠"另一个部族"
  留在池子里，身上仍带龙/机械标签。学习席终局 147 个"龙"卡位里 143 个是这类。
- **赠送路径不受管**：`BGS_034`×2、`BG24_500`、`BG21_015`、`BG35_342`、`BG34_171`
  这类单种族残渣是饰品/奖励给的，lobby 过滤拦不到。
- **对手侧完全没变**：注入库（真人战团）里 **43.6% 的板面**仍带龙/机械
  （卡位占比 龙 13.6% / 机械 15.3%）。⇒ 这一轮对模型是"自己不能用这两条线，对手照样有"。

**5. 三档强度（后续由用户选）**
- **L1（已实现）**：lobby 层过滤（商店不出这两个部族）。
- **L2**：L1 + 用引擎已有的 `excluded_cards` 禁掉全部 **474** 张带龙/机械标签的卡
  （含 30/43 张双种族卡，会连带削掉海盗/恶魔/鱼人/亡灵的部分卡）。不需要改 Rust。
- **L3**：L2 + 拦"赠送"路径（饰品/奖励指定随从）⇒ 需要新增掩码或改引擎。

---

## D-076 · 2026-11-17 · local+server · **v1.4-r1 开训前：修 4 处审核隐患 + smoke 逐参数审计（又发现 2 个真问题）**

用户逐项审核配置后要求："全部修改，完成后审计检查一遍，定下目标，开始训练；
注意 smoke 时要着重检查所有参数。"

**1. 审核发现的 4 处隐患（已修）**

| # | 问题 | 修法 |
|---|---|---|
| 1 | `HSAI_ARENA_CKPT` 默认 `arena-v1.2/model.pt` ⇒ 忘覆盖就**从旧模型起训** | 改成**必填**（`${VAR:?}`），缺参数直接报错 |
| 2 | 头部注释写 `MINUTES` 默认 150，实际 165 | 注释改成 165 |
| 3 | `stage_arena.sbatch` 注释写"12 -> 6"，实际值是 `10:6` | 注释改成实际四档，并**去掉 10→6**（用户决定） |
| 4 | `forced_tier.py` 默认计划 `2:2,5:3,6:4,8:5` 与脚本传的 `2:2,5:3,7:4,8:5` 不一致 | 代码默认对齐为 `2:2,5:3,7:4,8:5`（同步改测试） |

**2. smoke 三轮抓到的 2 个真问题（都已修 + 回归测试）**
- **健康门阈值对本游戏是错的**：`entropy_warn_below=0.8` 几乎恒真——早期只有 2 个合法
  动作，最大熵就是 `ln2=0.693`，于是**每一轮都报** `entropy<0.8`（告警疲劳）；
  `throughput_warn_below=1500` 让任何小规模 smoke 都报"吞吐异常"。
  改为 **0.05（只抓崩塌）** 与 **0（默认关闭，需要时显式打开）**。修后 smoke：**0 告警**。
- **动作契约不进训练产物**：`freeze_cap / ban_trinkets / forbid_unfreeze / lobby_exclude /
  max_actions_per_turn` 只存在于命令行，`history.jsonl` 查不到 ⇒ 跑完无法自证用的是哪套限制。
  更糟的是 meta 有**四处**组装点（串行 / parallel 逐局 / lockstep / process_pool 汇总），
  逐处手写必然漂移——第一版只改到串行路径，pool 路径仍然丢。
  ⇒ 新增 `hsbg/az/train/contract_meta.py` 作**单一来源**，四处共用；
  并在 `run.env` 增加一行 `contract ...` 让它自证。

**3. smoke 逐参数审计（每项都是"预期 vs 实际"，取自 run.env + history meta）**

全部 **33/33 通过**，关键项：`ckpt sha256 1808f178…`、`arch entity / 16,436,034`、
`engine c4c71014e3d41760`、`move_cap 2 / repeat 2 / freeze 1 / max_actions 100 /
ban_trinkets 1 / forbid_* 1 / lobby_exclude [] / sims 0`、`rank utility
1.00,0.35,-1.00,-1.15,-1.30,-1.45,-1.60,-1.75`、`win_bonus 0`、`tier coef 0`、
`teacher plan 2:2,5:3,7:4,8:5`（targets 无 '6'）、`forced_actions 0`、`teacher_labels>0`、
`mirror 7 席 p1 p50band 落地 100%`、`illegal/truncated/invalid 0`、`forced_end_turns 0`。

**4. 正式训练（作业 1156986，节点 xgpi14，16:09 起，165 分钟 1 段）**
`run.env` 与 `history.jsonl` 双证：契约行 `max_actions=100 move_cap=2 freeze_cap=1
ban_trinkets=1 lobby_exclude=<none> sims=0`；教师计划四档；镜像注入 46,180/46,182；
`move_guard_blocked_cap ≈ 155 万/轮`（cap 2 真实生效）；零非法动作、零截断。

---

## D-077 · 2026-11-17 · local · 双环境工程审计与隔离修复，保护现役长训

用户要求全面深入审计，并明确工程修复自主执行。原始目标仍是保护长训、不擅改训练策略。
本次从 `8d6d823` 创建本地 `codex/engineering-audit-20261117`，未向服务器部署运行时改动。

**实测状态（as of 2026-11-17 18:27 +08:00）**：作业 `1156986` 仍 RUNNING，619 轮 /
57,799,833 决策，非法/无效/截断/引擎停止/注入错误/丢弃决策为 0；动作预算强制结束回合 27 次。
服务器 HEAD 仍 `8d6d823`，原有 5 个第三方参考文件未跟踪，原样保留；不宣称 IN SYNC。

**修复**：推送保护漏识别 arena 且查询失败放行、登录节点复合命令守卫、下载校验后置、最终
检查点保存失败被吞、非 8 头模型元数据错误、健康计数校验、worker P50/均值聚合、分片协议/
权重/seed/行数据完整性、评测输出与旧模型导出覆盖、测试临时文件竞争。四份现役文档按代码与
实测校正。完整根因、证据、测试结果和部署边界见
[`analysis/ENGINEERING_AUDIT_20261117.md`](analysis/ENGINEERING_AUDIT_20261117.md)。

**验证**：完整工作区检查 966 通过 / 7 跳过 / 0 失败；后补两合并器输出保护的 36 项回归通过；
67 个 shell 脚本语法通过；2 局 CPU 引擎 smoke 与真实产物合并通过。CUDA、Rust 重编译和真实
游戏验收未执行，不能用 CPU 结果替代。

**版本锚点**：引擎 `c4c71014e3d41760`；线上 pin SHA `ba7e8bbc…bd8e7` 未变；起点
`arena-v1.3-r3/manifest.json` 声明 SHA `1808f178…c206472`；D-074 完整协议重算仍为
`7e15f114d0f9290a80e64d8c2e2706c424594adc5ef1a9c44e8e3d59dcb5b56c`。本地两局 bot smoke
使用独立协议，不能当作 held-out 棋力结论。

**历史解释勘误（不重写旧决议）**：D-073 同时改多个 cap 且使用旧契约模型，不能单独归因
move/freeze，也不能仅因旧上限触发少就证明提高上限无害；其中点估计 Δ+0.424 与列出的
CI `[+0.545,+0.703]` 不一致，正式引用前待同 seed 原始证据重算。平均名次接近 4.5 不足以
证明奖励无梯度。论文是否跨 stage 恢复 Adam 状态待确认，不再作为既定事实。

**未做**：没有新增 GPU 作业、修改奖励/超参/网络/训练目标/对手/动作契约、重跑 E1、改线上 pin、
操作游戏或删除服务器证据。审计改动会影响训练源码 digest，部署必须保留旧版本恢复入口，
不得绕过旧 trainer 的源码一致性检查；远端 CUDA 验证仍待独立安排。

## D-078 · 2026-11-17 · local+server · 长训成果保护与工程现代化验收

用户明确授权：先确认现役长训结束，再自主完成双端工程重构/验证与一次正式评测；不启动新长训，
不改变奖励、模型、超参数、对手、游戏规则或判胜标准。本次必要 CPU/GPU 验证与正式评测已获授权。
科学参数继续沿用 D-076 与现役脚本，本决议不创建新训练配方。

**长训已正常完成**：1156986，18:53:59 COMPLETED / 0:0，745 轮 / 69,691,656 决策。
模型 SHA `5058768bf90e989a943c6e21800a0767095a81a2034f4d4bf8d3d603028896a1`，Trainer
SHA `2e6073f7a80fbb57c4cca0ede148145addb4541029083f6fe35048a2d51141dc`。
同代 tensor、有限性、Adam/RNG/调度器/anchor、summary/history 和零工程错误计数均已核验。
登记为 `artifacts/models/arena-v1.4-r1/` candidate；原始训练产物、旧源码与旧环境保留，线上 pin 未变。
历史 latest=741、最终平面文件=745 的差异原样保留；未来最终保存采用同代目录与原子 latest 指针。

**恢复事实勘误**：D-077 对源码 digest 守卫的表述过宽。本次原始 Trainer 的 extra 没有源码 digest；
外围恢复工作流的守卫不能等同于原始 `--trainer-state` 自带校验。保留启动提交 81022069，
并通过新源码上的实际 CUDA 固定 batch 恢复与真实两段单轮训练入口验证。固定 batch 的模型/Adam
逐位相等、调度器相等；真实采集入口从 745 到 746 再到 747，探针模型不发布、不作为新训练起点。
不宣称分布式 worker/在途 pipeline 的整个下一段轨迹可逐位恢复。

**工程**：训练参数解析、评测纯聚合、证据写入/校验、原子资产、来源身份拆分为模块；训练参数
接口 263 项等价，18 个聚合函数 AST 等价。旧分片入口统一为计划执行器，精确局数、失败传播、
逐 seed/逐回合/逐动作证据与原生回放门禁齐备。不可变源码发布与共享资产分离，可回滚；
第三方参考资产按 SHA 登记并原样归档。Windows 不再通过 .pth 引入 Anaconda 全局包，旧环境保留。
详细架构与运维见 `docs/ENGINEERING.md`。

**验收证据**：Windows 989 pass / 8 skip；Rust 942 pass / 1 ignored；CUDA 验收 977 pass / 13 skip；
Linux CPU 全量验收及最后一轮发布检查另有原始 JUnit。真实双模型 4 局、849 动作全部原生回放
一致；71 个 shell 脚本与 2 个 PowerShell 脚本语法通过；隧道临时端口自愈与独立探活通过。
失败的发布验证、单进程 IPC 探针现场保留，修复后新目录验证通过。

**正式评测**：预先提交 `config/evaluation/v14-r1-20261117.json`，起点 v1.3-r3 与最终 v1.4-r1
各 512 局、8 分片、相同 seed 930000..930511，D-074 held-out 场和冻结契约不变。协议
`7e15f114d0f9290a80e64d8c2e2706c424594adc5ef1a9c44e8e3d59dcb5b56c`。
1157979 仅在全部先行验收成功后启动（as of 2026-11-17 20:04，RUNNING），源码发布 r6/b59630b。
完整结果待该轮完成后记录；不按成绩追加评测，不重跑 E1，不自动晋升模型。

## D-079 · 2026-11-17 · local+server · 现代化正式评测完成，排名改善但后期成长未证实

**授权与科学边界**：沿用本次用户对工程重构、验证、正式评测的授权。科学配置仍为 D-076 与
现役脚本，本条不创建训练配方、不改变评测判定标准、不授权下一轮长训。

**正式结果（as of 2026-11-17）**：1157979 COMPLETED / 0:0，31:05；1158028 封存/分析
COMPLETED / 0:0，2:20。v1.3-r3 与 v1.4-r1 各 512 局，相同 seeds 930000–930511，
D-074 held-out 场与 100/2/1 契约。旧基线 512 局的 seed、名次、动作统计、逐回合战力与 trace
和整理前完全一致。全部 1,024 终局、12,777 回合、190,884 个学习席动作的证据验收通过。

平均名次 **4.9141 → 4.6895**，配对差 −0.2246、95% bootstrap CI [−0.4414, −0.0078]；
前四 **40.23% → 43.75%**，差的 CI 跨零；夺冠 **8.40% → 12.70%**，差 +4.2969 个百分点、
CI [+0.5859, +8.0078] 个百分点。逐项区间为探索性分析，不是多重比较校正后的统一胜负结论。
T12 总攻血 P50 **1,287（n=360）→ 1,352（n=345）**，共同存活 n=270 时仅 +0.8%；
T16 **5,609（n=35）→ 4,826（n=54）**，共同存活只有 7 对。排名与夺冠有改善，后期
成长没有证明改善，不能宣布全面变强。**继续保持 candidate，不晋升冠军，不改线上 pin。**

**四个版本锚点**：引擎锁 `config/engine-upstream-lock.json` / `c4c71014e3d41760`；线上 pin
`config/live/model.json` / `ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7`；
候选 `artifacts/models/arena-v1.4-r1/manifest.json` / 模型 SHA
`5058768bf90e989a943c6e21800a0767095a81a2034f4d4bf8d3d603028896a1`；协议
`7e15f114d0f9290a80e64d8c2e2706c424594adc5ef1a9c44e8e3d59dcb5b56c`。
正式 panel SHA `f076b9883bdad127b1e453981ebc654f04005885ad3cadde81cb6c72e7625e66`，
目录封存清单已生成，候选新增只读评测引用 `reports/20261117-heldout-v1.json`，原 manifest 未改。

**补充工程验收**：Windows 完整快照 990 pass / 8 skip，最后运维回归 40 pass；Linux CPU
最终完整快照 981 pass / 16 skip；Rust 942 pass / 1 ignored；CUDA 977 pass / 13 skip。
参数接口、聚合实现和实际旧基线行为等价；固定 batch 精确恢复与真实两段采集/保存/恢复通过。
完整逐动作证据有独立校验、抽取与原生回放入口，不把纯测试通过替代实际运行验收。
隧道真实故障暴露重复超时累计 71.9 秒，改为单一 25 秒截止时间，故障回归与临时端口自愈通过。

**策略观察与记录勘误**：seed 930006 T12 的连续相同移动将两个相邻单位交换后又换回，
已保存可定位证据；没有擅改动作限制。旧运行记录“首轮 tel_tier_10=6.0”与原始日志不符，
首轮实为 5.66125；训练回合开始遥测不能与评测 EOT 指标混用，当前运行文档已更正，历史保留。

完整改造清单、环境职责、恢复资产、曲线/样本量、具体策略例子、同步验收和遗留风险见
[`analysis/MODERNIZATION_20261117.md`](analysis/MODERNIZATION_20261117.md)。恢复资产保留，
没有删除重要历史数据，没有把原生 bot 内部动作未暴露或分布式在途恢复限制写成已解决。

## D-080 · 2026-11-17 · local+server · 转向模型能力研究，冻结配方不变

**用户方向**：实机自动化暂停；唯一核心目标是当前版本模型能力突破。旧模型与旧战团库保留，
不再作为主要研究对象。本地以管理脚本、轻量材料和未来实际验收为主，服务器承载模型、训练数据、
完整评测证据。本轮只研究与必要低成本验证，不启动新长训、不修改核心科学设计、不整理迁移资产。

**资料身份**：唯一现役HSReplay攻略为 `docs/battlegrounds_strategy_guide.md`，SHA
`f03150e215ab5a52d2e2f8f7cfd2d7e02b8aac3d00e8da8886257463860bbb5a`；33流派，141个核心引用、
84张不同卡均与现役卡表唯一匹配。正文没有独立发布日期/网址，不把Git日期冒充网页版本。
AutoTavern本地PDF SHA `1dbebcb9cb67aba30bb0328b98fd6b7a19a6811880cfd7f64c8babea89583c89`。

**新增证据**：两端现存history完整行去重采样3,374,644,031；包含分支/重试，不能算作单一模型
继承量。现役链账面采样约2,380,039,369，早期缺完整消费账；当前轮69,691,656全部进入PPO
processed_samples，约论文后期stage预算22.5%。历史D-065所谓v9三轮5400轮/600M的口径更正为
1675轮/210,690,977；旧决议不改写。论文训练效果与本项目版本/考试不同，不能直接混榜。

完整正式轨迹16个gzip SHA核验后扫描190,884动作；CPU作业1158396完成。最终聚合1158513、
与正式panel逐局/逐回合/动作分类对账1158518通过。初次解包失败与读错EndTurn时点的分析现场保留，
有效经济统计以 `artifacts/state/research-20261117/trace-v2/research-summary-v2.json` 为准。
候选T9+铜须合法购买机会221局、买入2局；罗杰斯91/0；希望信使122/110。此为行为偏好，
未按局面机会成本做反事实证明，不能把所有未买都标为错误。T12支出中位13、P90为18金币，
高周转经济仍少。已有自动成长能力，但没有证明形成普遍的经济运营突破。

CPU合成状态对照证明罗杰斯counter=0/8时模型观测与合法mask一致，而刷新产生不同资源收益；
这是可见机制状态被遗漏的直接证据，不等于已量化其胜率损失。对手压力、奖励、探索、anchor和
监督覆盖的贡献仍待控制实验。平均名次接近4.5不能证明无梯度；老BC负结果不能泛化为所有经过
机制验证的示范都无效。撤销“对手校准是唯一有效杠杆”的当前宪章绝对表述，不改既往实验事实。

**建议而非生效方案**：先验证可见状态输入，再用模拟器核验的经济循环/场位管理示范与学生状态补标签，
最后接回自然整局PPO。先做输入×标签2×2小实验，成功后再研究分级对手、critic和信用分配。
不得把本轮正式held-out轨迹用于训练标签；不得依据研究建议直接启动作业。

四个锚点沿用D-079：引擎 `c4c71014e3d41760`；线上pin
`ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7`；候选manifest/model SHA
`5058768bf90e989a943c6e21800a0767095a81a2034f4d4bf8d3d603028896a1`；正式protocol
`7e15f114d0f9290a80e64d8c2e2706c424594adc5ef1a9c44e8e3d59dcb5b56c`。均未改。
报告：[MODEL_CAPABILITY_RESEARCH_20261117.md](analysis/MODEL_CAPABILITY_RESEARCH_20261117.md)。

## D-081 · 2026-11-17 · local+server · 自主研究训练授权与机制输入实验

用户正式授权Agent自主负责下一阶段研究、开发与强化学习训练，包括架构、观测、奖励、PPO、
教师/BC、超参、对手/库、课程和评测方法；GPU作业与科学变更不再逐项审批。目标是尽快突破
中后期运营，不以实验数量或训练时长为目标。重要成果、恢复能力与实验可信度必须保留。
本轮主动研究/修改/提交新作业截止 **2026-11-18 10:30 Asia/Singapore（02:30 UTC）**。
届时正常作业继续运行，交接编号、状态、预计结束及查看方法；实机自动化仍暂停。

选择首个可验证方向：独立版本 `hsai.mechanisms/v1`，旧3227 floats/124 cards保持完整前缀，
追加55个float（30项己方公开历史/持续增益统计，己方场上7/手牌10/商店8槽计数）。计数白名单
只含动态牌面已确认公开的罗杰斯、希望信使及金色版本；不把任意内部counter假定为公开。
输入另启用旧cards[4]待选来源的零初始化投影。模型增加零初始化残差投影，原卡序/动作不变；
规则引擎与源码指纹不变。不同schema严格绑定checkpoint与动作记录，不允许缺字段静默补零。
该扩展尚未覆盖附件身份、所有机制或历史记忆，不宣称“观测已经完备”。

本地真实v1.4权重迁移：随机输入输出逐位一致，保存/加载逐位一致；旧/扩展模型各2局真实对局
轨迹一致，合计1000动作全部原生回放通过。初始状态观测开销约2.3us→52.4us，须在服务器实际
采集复测吞吐。针对机制/批次/回放测试通过；完整回归发现两处旧接口兼容问题（legacy metadata
键集、测试替身observe透传），已修正并通过相关28项回归。GPU链路验证之后才开展能力训练。

首个受控比较建议落地为同起点、同配方、同训练场的旧输入PPO与扩展输入PPO，各约2M采样，
只作可行性和方向诊断，不据短训宣布晋升；后续经济示范需新训练来源与原生多步核验。
配置与保留seed段见 `config/research/overnight-20261118.json`。旧正式930000..930511仅诊断，
不用于训练标签；新的正式保留段1119000..1119511直到候选确定后才使用。

基线锚点：引擎锁 `config/engine-upstream-lock.json` / `c4c71014e3d41760`；线上pin
`config/live/model.json` / `ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7`；
父模型 `artifacts/models/arena-v1.4-r1/manifest.json` / SHA
`5058768bf90e989a943c6e21800a0767095a81a2034f4d4bf8d3d603028896a1`；可比较正式协议
`7e15f114d0f9290a80e64d8c2e2706c424594adc5ef1a9c44e8e3d59dcb5b56c`。当前不改线上pin。








## D-082 · 2026-11-17 · local+server · 验证经济过程教学与固定双seed迁移预算

依据D-081自主授权，新增只用于独立研究分支的原生引擎核验过程BC：270个私掠者/悬赏令合成任务，
源于独立训练场96局中的45个海盗房间。源游戏拆分不跨训练/开发；正式旧新保留seed未作训练来源。
200更新后，10个独立开发源游戏的60个任务变体中，完整经济收益达成率20%升至83.33%/86.67%。
标签精度不能等同整局棋力；64局自然开发评测未确认突破。暂不扩大奖励、改对手或晋升线上模型。

选择旧输入过程BC分支验证整局迁移：两seed1181811/1181812，各匹配原2M权重对照，
每臂追加10M采样，合计40M；保持原终局奖励/训练对手/PPO配方，按配方重建Adam，anchor到各起点。
作业1158831..1158834，冻结发布research-transfer-r7-20261118（01672a4）；不能只挑最好seed汇报。
扩展输入解决观测混叠但短训未见能力收益，暂不增加其训练预算。局部教学若不能迁移，优先查自然
状态覆盖、满场腾位与长期投资链条，不靠无限增加训练量。详见本轮AUTONOMOUS_RESEARCH报告。

工程失败1158790/1158791为同名history.json防覆盖拒绝，保留现场后修复独立检查点日志，
新增真实训练/保存/重载测试；1158794/1158795从相同原权重新目录重跑正常完成，并非挑成绩重跑。
引擎/线上pin/正式协议四锚点沿用D-081；新研究权重身份完整登记在各run.json/result.json。

## D-083 · 2026-11-18 · 训练重复的环境种子分段纠正

发现train_entity_sync每轮以base_seed+iteration*1000生成连续512个游戏，D-082所选1181811/1181812
环境种子大量重叠，不能作为充分独立的训练重复。1158833/1158834核实仍PENDING后取消（未运行），
保留Slurm账本；同代码、同配置、同10M预算、同起始权重重提第二组到seed2181811：
1158907对照，1158908过程教学。第一组1158831/1158832保持原样。此改动由采样实现证据触发，
不依据第二组成绩（尚无成绩）；最终仍须汇报两组，核对实际seed范围。第一组与先前2M训练环境
种子也有复用，这是共享热启动数据，不宣称全部新环境；正式保留seed不受影响。

自然开发动作扫描证明窄经济教学题的前提在旧输入各64局中均未出现，满场诊断收益达成仅33.33%。
因此另建成对自然/经济核心赋予的完整起点库（31个训练源游戏，T8/T9，金币/血量/等级不变），
只改变episode初始分布，首条learner记录即从该状态开始，隐藏RNG重新抽取；不伪造前缀回报。
默认关闭，完整身份门禁和计算节点测试通过；先提交2迭代链路探针，能力预算待探针和机制分析。

## D-084 · 2026-11-18 · 配对经济起点课程与完整恢复验收

D-081授权下，512局独立自然源采集1158927完成2764状态，1158971生成自然/赋予经济核心两库，
各207训练源游戏；49个满足相同筛选条件的开发源游戏只用于机制诊断，不进起点库或标签。
初版31游戏库及全部现场保留。1158952真实CPU多进程采集/PPO/完整Trainer恢复再更新通过：
updates1→2、决策1473→2551，Adam继续、anchor不变；不声称异步预取流水线不中断逐位等价。

新固定预算计划`config/research/recruit-curriculum-20261118.json`：同过程BC起点SHA
`26225c90da0d2b10442f8635360917d90944834f2afbfc2ee295a18d30dc820a`，seed5182001，
每臂2M采样，episode起点课程p=0.3，只比较自然起点与赋予三经济核心的起点。其余沿用原stage配方；
不改终局奖励、对手或正式评测，不能由该比较单独推导“中局课程比无课程有效”。
冻结发布`research-curriculum-r15-20261118`（fd1b627），1158986/1158987使用Slurm
`afterok:1158909`，GPU链路探针未成功则不会执行能力实验；绝不使用探针更新过的权重作为父模型。
工程门失败即停；完成2M后先看真实技能与自然整局结果，再决定是否延长。

1158977第一组10M后的局部闭环保留：未教学20%，教学63.33%，后者从刚教学83.33%回落。
1158972在49独立开发源的赋予核心首回合中，对照37局/教学39局卖出铜须，三核心全部保留仅12/10局。
这说明窄过程题没有覆盖完整组合，不把“卖铜须”逐项直接判错。1159023仅作禁止卖核心的因果诊断，
干预掩码和原生掩码都保存；不是学习所得能力，不用于正式评测或部署。
引擎、原父模型、线上pin及正式protocol锚点沿用D-081；所有新模型、库与实验均单独保存身份。

## D-085 · 2026-11-18 · 课程预算CLI表示修复，未发生训练后重选

GPU完整起点探针1158909正常完成，2轮/24,166决策，原生采集/PPO/最终检查点均成功。
之后1158986/1158987在5秒/4秒的参数解析阶段失败：`--stop-at-decisions`的类型是float，
提交变量写为`2M`被拒；尚未开始采集或优化。原现场保留。预算在D-084和JSON计划中本来就是
2,000,000，改成CLI接受的`2000000`，两臂在新`5182001-2`目录、同父权重/seed/课程p/源码配方重跑。
不放宽解析器契约、不改变科学参数；本地真实parser预检通过后提交。
依赖失败的评测1159031、审计1159045核实尚未运行后取消；重新登记新路径和依赖，不覆盖评测证据。


## D-086 · 2026-11-18 · 窄过程BC与核心课程未获自然局收益，停止追加

在D-081自主授权下，依据1159164统一开发评测和1159169完整账本/Trainer审计，停止给现有两条
过程BC后10M续训、自然/赋予核心2M课程追加预算。两训练种子的过程干预名次均较匹配对照差，
共同存活成长CI跨零；课程对比亦未获益。保留全部结果与失败现场，正式保留评测未使用，不换冠军或live pin。
详见 `docs/analysis/AUTONOMOUS_RESEARCH_20261118.md` 的02:12结论。
后续研究转向更完整的当前版本经济操作教学，并检验冻结表征/分离Activate查询以控制策略干扰；
这些是研究候选，未宣称能力进步。所有新架构默认关闭、旧权重路径不变。


## D-087 · 2026-11-18 · 当前版本完整经济过程与保守教学比较

D-081自主授权内执行 `config/research/cycle-learning-20261118.json`：v1.4-r1精确中立分叉，
可选分离Activate查询，五种原生可证实的短程资源/成长任务；训练/开发按源游戏分开，正式保留集不用于教学。
新增32局自然整局KL锚点，比较全网学习与冻结表征动作头学习。先检查闭环技能和自然开发对局，
再决定是否分配PPO续训；不预定扩大训练，不直接改变冠军。全量本地1021通过/8跳过。
具体科学参数以JSON计划及新冻结发布脚本为准，旧发布/旧模型/旧训练契约继续保留。


## D-088 · 2026-11-18 · 先验证自然核心购入收益，不盲目追加局部技能PPO

D-087五类技能全网教学改善明显，但1159338自然64局/模型评测无可靠收益。1159316旧正式证据审计
显示核心购入极少，几乎没有循环技能自然触发机会；局部刷新目标也不能直接代表真实失误。
D-081授权内执行自然训练源条件分叉诊断：`config/research/core-acquisition-20261118.json`。
只干预一次合法购买，保留初始对手，配对未来RNG；既测父模型也测已学局部技能的全网候选。
先做CPU工程探针，再做至多768条件续局；不是正式模型评测，也不自动改奖励或产生训练标签。
正式保留seed仍未使用，live pin/冠军不变。默认评测新局路径保持原样，条件续局入口仅研究代码调用。

## D-089 · 2026-11-18 · 整回合可执行流程与训练信号诊断优先于再次扩训

1159407的768个条件续局没有证明单次购买核心有收益，不能据合法购买机会制造专家标签。
1159505完成128局、23,548动作的真实随机策略采集，无优化更新；三个全回合/三个T9+微批诊断中，
GAE(.995)与MC(1.0)策略梯度余弦0.958–0.985。升本教师加权梯度范数仅0.00032–0.00042，
PPO约1.18–1.77；本批没有证据支持“教师压倒PPO”或“GAE把后期方向完全弄反”。
这些是起点微批诊断，不等于整个训练历程或精确Adam更新的归因；MC也不是无噪声真优势。

在D-081授权内，先对49个既有开发源局面验证公开信息的整回合经济操作流程，比较普通三核心、
四核心、金色倍增及缺一组件四种显式合成起点，父策略与流程使用同样未来RNG和100动作预算。
全过程只读己方公开视图；原生合法性、状态回放、合成编辑均保存。仅为机制诊断，非自然棋力评测。
必须先证实整回合收益与可学习性，才考虑用它生成训练来源的示范；正式保留集继续不触碰。
初步11个bot来源探针中完整金色倍增状态8/11触及100动作，但不能据此认定自然模型当前受限于上限；
1159505自然采样没有动作上限强制结束。暂不改变正式动作契约。

## D-090 · 2026-11-18 · 在明确整回合收益后开展一次完整流程教学

1159518 COMPLETED / 0:0，49自然开发源×4显式合成变体×2政策的配对首回合诊断完成。
公开流程相对父模型EOT总攻血均值差：普通三核心+127.96 [90.75,172.49]、四核心+72.02 [34.88,110.74]、
金倍增四核心+255.08 [169.63,344.06]、缺私掠者需要购买+143.96 [110.18,179.82]，每项n=49源游戏。
这些是同资源起点的条件性流程收益，不能宣传为模型进步或自然组阵已解决。

执行`config/research/whole-turn-learning-20261118.json`：未修改架构的v1.4父权重，全网1000更新、
lr3e-6、每变体32标签/步、自然锚点256/步，策略KL系数10、rank KL系数1、全锚点KL>0.03停止。
原始自然库按源游戏隔离，训练使用新的未来RNG；开发精确复用1159518的状态，父策略不重新刷成绩。
完整公开流程是启发式行为示范，不是逐动作最优证明；教师记忆自己刚打出的单位，学生前馈，
可模仿程度须用完整闭环实测，不能由标签准确率替代。动作上限仍100，强制结束不作为BC标签。
仅在完整闭环能力提升且自然64局开发面板可接受后，再决定PPO预算；没有预先启动新长训。

## D-091 · 2026-11-18 · 完整示范暴露自伤缺陷，修订教师后重新验证

1159544在源11802146/complete失败；独立CPU1159582于26秒正常完成复现。
第89次合法动作是奇瑰打击乐手的发现选择，原生诊断全零；英雄从1血降至-3并死亡，游戏正常推进到下一回合。
完整轨迹显示30起始血量被多次打击乐手战吼/铜须倍增消耗。故这是研究控制器的错误策略，
不是引擎损坏，也不能过滤死亡源例后把剩余示范称为可靠。旧298份轨迹和故障原始证据保留，教学未启动。

在D-081授权下调整这个**专用海盗经济教师**：不循环自伤战吼或引入持续自伤/健康支付组件，
先出售已有此类风险随从；不买公开视图标为健康支付的商品。此保守规则不声称恶魔体系中这些卡普遍错误，
不修改游戏规则、原生合法掩码、现役模型或正式评测。失败源局面本地完整重放后60动作正常结束、30血存活。

所有原定源游戏仍进入新版本数据，使用新目录；CPU时限由30分钟改为60分钟。
先将修订教师对照1159518已封存的同起点父策略重新计算配对收益，逐项验证父权重、轨迹SHA和初始状态。
教师指标允许变化并显式记录数量，不能伪称原教师没变；父策略无需重复执行。
只有新示范完整校验和整回合收益检查通过后才提交BC，学习参数仍D-090。自然64局开发候选路径改为`-3`。

## D-092 · 2026-11-18 · 全网完整教学触发KL保护，改用有明确上下文的经济专家

1159647于100次更新触发预设停止：2697自然锚点KL=0.039964>0.03，贪心动作改变6.266%，
开发标签匹配52.749%。未保存候选、不放宽停止阈值、不从失败过程挑检查点。
这表明本次全网配置发生了较大策略干扰；不等于证明所有KL系数或全网学习均无效。

按`config/research/conditional-economy-20261118.json`进行一次结构性隔离验证：
两个相同v1.4组件，中立初始化；海盗在局内且场上/手中持有铜须、罗杰斯、私掠者中至少两种时，
使用经济专家策略，其余策略沿用基础模型。金/普通视为同一类，商店不算持有；只读原3227/124观测。
rank/value始终来自基础模型，避免BC顺带替换成未校准专家critic。BC期间基础模型完全冻结并逐位核验，
专家固定1000更新，lr3e-6、seed1193801、原数据/采样/锚点预算；KL阈值仍0.03。
必须报告门控覆盖率：若自然锚点没有命中，低KL只表示门外不变，不是门内策略已安全。

新checkpoint显式记录组合架构、组件meta、词表SHA和门控规则，严格重载；不伪装成原单网络。
预期存储约两倍参数，门内推理两次前向；这不是免费的优化，也没有自动解决前两核心的自然购入。
先完成实际权重CPU中立/保存重载、多worker真实PPO与完整Trainer恢复，再教学；探针训练权重不用于能力起点。
通过完整闭环后做一轮64局自然开发筛查，尚未安排PPO能力续训或模型晋升。

## D-093 · 2026-11-18 · 检查经济教师的多回合收益，不把首回合身材当长期最优

条件模型真实CPU预检1159706于05:06:36通过：两次真实PPO更新，完整Trainer恢复、Adam步数与原anchor核验。
此前1159683的单组件卡表审计兼容失败现场保留；依赖GPU1159684未启动即取消。
自然覆盖CPU1159709完成：旧正式512局只有3局、9个回合、368个动作在两核心门内；
28局出现已有一核心且第二种核心合法可买的机会。这只是到达率与可购买性，不证明应当购买。

按`config/research/economy-suffix-20261118.json`准备多回合条件检验：49个既有开发源、
普通三核心与金倍增两种显式起点、每源2组新未来RNG，父模型/公开经济控制器/固定终点学生三臂，最多588续局。
公开控制器仅在同一两核心门控内接管，所有动作经过原评测守卫和共享计数器，门外用父策略；
不偷看RNG、不提供额外动作预算、不在自然游戏里自动买入核心。此为手工政策干预，不能当作神经模型成绩。

先做2条CPU真实续局和原生轨迹验证；只有固定学生完成并经检查才提交完整三臂研究。
报告源游戏聚类的配对名次、共同存活T9–T12成长、各回合中位数/样本数、金币支出和触顶次数。
合成起点与生存者选择限制必须保留；不由条件结果自动生产最优标签，不使用正式保留seed，不宣布晋升。

## D-094 · 2026-11-18 · 用目标语义保护和明确成长收益组件检验经济教师

1159742的公开经济控制器完整续局保留了一个可定位错误：seed11960000，T10第88号动作，
稳健异变BG30_804按通用“选最大身材目标”命中骄傲的私掠者，将其变成魔鳍真菌学家。
轨迹SHA `8378eba68e2ba7de584de42f49114c7f243a3a62772a8889b6732dfd1430b4e6`；
原生诊断没有异常。公开`me.casting`及手牌`hid/id`足以识别这是变形法术，不能把它当作普通增益。
原教师/学生多回合研究1159754继续按冻结r34执行，旧数据和权重均不改。

新模式按`config/research/economic-payoff-20261118.json`单独验证，默认legacy行为保留：
1. target_safe：按自己公开的施法卡身份/最近打出记忆识别破坏性目标，保护经济组件；稳健异变优先合法商店目标，
   无安全公开目标时不主动开始该施法。不是给模型或游戏新增非法动作屏蔽。
2. growth：在target_safe上加入现役攻略“海盗-手速金色”的掠夺大师钩牙BG36_344，购买/发现/打出/保留该收益组件；
   罗杰斯+私掠者已在场且升本后仍有6金币时解锁六本。此臂同时改变收益组件处理和储备式升本规则，不能说只识别单个变量。

不额外赠送钩牙；仍用与D-093相同49源、两变体、两未来RNG、100动作预算。
先2条CPU实际续局探针，再392条CPU政策续局，暂不训练神经模型。完成后与原三臂证据按精确起点SHA配对。
已验证默认旧教师四类原生动作序列哈希不变；原生变形目标回归、无安全目标不施法、收益组件获取和升本储备测试通过。
这两项是有明确错误现场/攻略依据的启发式修订，仍须证明实际多回合收益，不把经济循环自动等同强阵容。

## D-095 · 2026-11-18 · 条件专家接入公开机制，保留中立初始化和基础策略

原生状态复现证明：宝石没收BG28_698与稳健异变BG30_804等待目标时，3227浮点与合法掩码完全相同，
124卡数组仅cards[4]的待选来源不同，而旧Entity网络没有读取这一位。旧网络输出逐位相同。
这是有效网络输入混叠；不是声称124卡数组完全相同，先前全观测零重复检查也不能排除它。

条件架构可选接入D-081的公开机制schema：完整保留旧前缀，3282浮点及待选来源零初始化投影。
基础策略和专家都中立扩展；BC仍冻结基础组件及critic，专家才学习新增表示。checkpoint明确绑定组件和外层
schema/哈希/宽度，旧条件权重保持兼容，非法组合拒绝。原生双状态单测验证初始输出等价、专家能够分辨来源、
基础策略/critic不变及保存重载一致；随后单独CPU实际权重/PPO/Trainer恢复预检，探针权重不作为能力起点。

自然开发唯一触发案例1118115在T10打出手中罗杰斯后立即卖出，门控共4动作；当时场上有私掠者但没有铜须。
原示范主要覆盖三核心齐全或铜须+罗杰斯缺私掠者，与实际门控覆盖范围不匹配。
因此后续示范需检验不同缺件/手牌位置和多回合分布，不能仅延长同一单回合BC。
本条尚未宣称扩展输入或新数据带来能力提升；多回合教师比较完成前不提交新能力PPO。

## D-096 · 2026-11-18 · 多回合否定旧教师，限定学习有长期收益的金倍增策略

1159754的588续局完整通过。旧教师普通起点比父模型差1.857名，旧学生差1.286名，95%源聚类区间均不含零；
金倍增学生虽身材更大仍差0.551名。旧示范不追加教学/PPO；首回合收益不能替代长期能力。
1159803的392修订教师续局通过；1159824按相同初始SHA/未来seed/库/引擎/契约合并所有980续局。
growth在金倍增起点相对父模型改善1.184名 [0.714,1.643]，共同T9–T12 log成长差+0.504 [0.364,0.644]；
普通起点仍差1.316名 [0.724,1.888]。单独目标保护基本不改变整体结果，不能把具体目标Bug当全部瓶颈。

执行`config/research/payoff-learning-20261118.json`：只在明确金倍增上下文学习完整多回合growth示范。
新可选门控v2要求海盗在局内且己方持有金铜须、金私掠者、罗杰斯、逃兵；门外保持中立基础策略。
旧v1门控/模型继续保留，词表/门控/schema严格绑定；不把普通起点的负面示范也交给专家。
全256源仍按207训练/49开发隔离；训练使用新未来RNG，开发直接重放封存教师replica0，避免重刷结果。
不按胜负过滤源游戏；只保留门内实际未强制动作及真实守卫掩码。训练不含价值标签，强制动作不作示范。

先CPU两源完整采集/开发全原生重放，再2000固定更新、lr3e-6、每阶段32标签/步、三阶段均匀采样，
保留原KL停止阈值和基础策略/critic逐位检查。输入使用D-095公开机制；BC状态不是PPO Trainer。
之后独立完整续局检验；未预先承诺追加PPO、使用正式保留集或模型晋升。
这是验证长期成长技能可学习性，尚不解决自然购买/组三连；学生丢失组件后回父策略，与教师广门控区别需明示。

## D-097 · 2026-11-18 · 补齐研究续局快照质量审计，异常臂保留并精确定位

D-096“392续局通过”需补充限定：此前只查非法动作/截断/原生回放，未逐项拒绝非严格快照缺失和回退。
1159881新门禁发现问题后，尚未开始的BC1159862立即取消，0运行时间；没有用未审数据更新权重。
1159886检查旧588、新392以及当时已生成数据：旧588全部干净；新392仅target_safe/complete_gold/
case019/replica1出现missing=1/fallback=1，其余指标为零。growth196、parent196和旧学生196均干净。

CPU1159891在新冻结r44精确重放原523动作及每步前后状态和最终状态：唯一缺失发生在T21/seat3，
快照库末回合20，学习席已死亡（-2血、名次7；最后EOT13）。该回退不能改变已固定的学习席名次或既有EOT，
但仍不把此局改写为严格质量合格，不覆盖旧结果、不删异常源。诊断明确ability_eligible=false。
原轨迹SHA e460566933c965a2d1a2eded990da0edaaecaaf244f4d268f002180bce6d8e8c，
诊断complete SHA 5e83c3f3cf5c0f8b51b0d71a57e36cbafc55ed631e9560a8946d4f69ea8ffc9c。

后续比较可明确选择完整parent/growth/新学生臂，仍逐源逐副本完整配对；不按异常或输赢删个别结果。
金倍增growth相对parent的收益不依赖target_safe臂。正式评测/训练使用严格库路径，不据本次研究路径
异常推断其已受影响。新研究续局统一质量门禁，不静默放宽。

1159847全256源/151345实际决策完成；1159887在CPU核对全部评测质量和文件身份后通过。
只有其afterok的1159889启动D-096固定BC，冻结r43=d21ff64、新输出nus-payoff-learning-20261118-2。
准入complete SHA 963de4729476c881f5f9f175f1429a005a9f13cb90b459042dda7faf6b7efd45。
教师/策略/学习参数不因质量修订改变，正式保留集仍未使用。

## D-098 · 2026-11-18 · 固定多回合学生终点验收，拆开缺组件与金色倍率

1159889于06:37:46正常完成2000步，模型89b6dacd44be3d650ec9534a29c93c1545428d8364f094e975ab8f0f1fd26952；
基础策略/critic逐位保持、重载一致。149806门内标签，开发30271标签匹配85.745%，不是最优动作准确率。
执行payoff-student-suffix计划，49源×2起点×2未来共196 CPU续局；只用固定终点，不挑中间权重。

复核fixture发现此前normal是三核心，complete_gold还额外具有逃兵；故两种结果不能单独证明金色倍率必要。
按complete-economy-payoff计划，用既有complete普通四核心起点比较父模型与growth，同49源、2未来、196 CPU续局。
这是区分组件完整性与倍率的最小补充，不穷举所有组合。无额外资源赠送、无新标签、无PPO；
若普通四核心也有长期收益，再决定是否扩展教学范围，当前金门控和运行中的候选不改。

## D-099 · 2026-11-18 · 更正严格库表述并检查全部尾部回退

D-097关于“正式评测/训练使用严格库路径”的一句不准确，现明确更正：实际stage_arena脚本和正式protocol
均为mirror_strict=0。严格EOT载荷合法性、全部尝试成功的结果检查，与StrictMirror策略不是同一个概念。
不能从名称或AGENTS旧文字替代实际配置，也不能据此直接宣称旧证据受影响或不受影响。

已读封存面板：正式v1.3/v1.4各512局的missing/fallback分别6/6与5/5；
本轮process-development中的legacy_2m_bc为1/1，其余已查12个开发模型条目均零。
新CPU1159918读取六份封存面板、相关完整轨迹与八个能力PPO history，检查末库回合20之后的学习席存活/行动
及训练回退范围；不重跑评测、不删异常局、不改正式protocol或模型。异常影响须以具体证据继续限定。
当前196新学生/196普通四核心研究保持冻结，不因这项盘点修改中途规则。

## D-100 · 2026-11-18 · 双网络续局评测改用GPU执行，样本与终点保持

1159899的8核CPU执行吞吐低于预算：约15分钟仍仅完成数十个续局，预计超过45分钟Slurm时限。
尝试仅延长时限被集群拒绝，未发生变更。未据部分名次/成长进行决策；原CPU全部完整/部分轨迹均保留。
改用单H100-47、4CPU/4worker、45分钟的新目录执行相同固定终点/196样本/种子/守卫/快照库，
GPU实际开始后取消旧CPU作业及其尚未开始的依赖分析，避免无效重算继续占用资源。
旧CPU部分不并入能力统计；新GPU全量为本次唯一候选比较。CPU/GPU浮点实现可能有微小差异，
不声称两次执行逐位等价。普通四核心教师CPU研究1159900按原计划继续，未改任何科学参数。

## D-101 · 2026-11-18 · 修复分段入口在Trainer恢复后重置anchor

stage_arena.sbatch原先把--reset-anchor 1放在所有段共有参数，包含--trainer-state的续段也会重建anchor。
这与完整Trainer恢复承诺冲突；它不同于旧K5的“累计轮数/优化步数”误判，累计轮数本身仍然正确。
现在fresh first_args显式reset=1，cont_args显式reset=0，移除公共覆盖参数；run.env记录fresh_only。
真实Bash执行完整段参数拼接（仅替换python可执行体）验证fresh两段为1/0、resume两段为0/0，
各段仅一个reset参数，Trainer参数存在性正确。既有真实CPU PPO恢复探针已验证reset=0下anchor保留。
未运行新的能力训练，未改变旧冻结发布/Trainer；本轮八个能力PPO均fresh单段，不将它们的负结果归因此Bug。

## D-102 · 2026-11-18 · 尾部回退完整定位与未来按学习席状态计数

1159918 CPU2:13完成六面板/八PPO历史盘点；1159941 CPU1:10以封存原动作重放5个相关游戏，
649动作/逐步原生状态/终局完全一致，逐事件解释全部12次面板缺帧：均在T21–T22且学习席已在T9–T12死亡。
故不改变这些学习席的名次、已发生决策或EOT成长；不删除任何局，不重算成“零回退”。
诊断ability_eligible=false，不是新评测样本。complete SHA
975ca1d66a32d4760ff353990076f436817de8366ddf00c1bb443709283239cc。
最初1159935因误传EvalConfig不支持的keep_start_board参数而失败，未进行原生重放；现场保留。
修复后增加真实完整游戏/状态差异拒绝的本地测试，再在新目录完成1159941。

八个能力PPO历史的缺帧共1624次；只有聚合计数，不能宣称全部发生在死亡后或对训练无影响。
这些作业没有被强行改判成无效，也不把极低频回退直接当能力停滞原因。后续MirrorStats新增学习席
alive/dead/unknown三类计数和有界seed/turn/target示例，经多worker汇总及history保留。
只读诊断不改变动作、引擎、随机数、库抽样或既有质量阈值；旧冻结作业不变，旧历史不伪造新字段。

## D-103 · 2026-11-18 · 普通四核心仍判负，验证更可达的单金私掠者

1159900 CPU22:31完成196续局，49源/98配对：普通四核心growth相对父模型名次差+1.173
[0.612,1.714]，共同T9–T12 log成长差+0.441 [0.290,0.605]（39源/64共同存活重复）。
T12中位攻血2927/n68对1832/n91，幸存人数明显不同；不能用更高身材掩盖更差名次。
不把这批普通教师加入教学，不扩大已封存金门控。

single-golden-payoff计划只补充金私掠者+普通铜须/罗杰斯/逃兵，其他起点资源、教师流程及49源/2未来保持。
私掠者可由海盗定向生成，铜须是中立卡，因此单金私掠者是更有实际组建意义的下一条件，不穷举金色排列。
先完成196 CPU续局，不直接生成购买标签或提交PPO；即使条件收益成立，也仍需原生组建/学生闭环验证。
新fixture仅显式计划调用，原VARIANTS默认四项不变，现有数据/旧发布不变。

## D-104 · 2026-11-18 · 多回合学生仍判负，固定预算检验关键事件稀释

1159939 GPU14:48完成196续局、1159940 CPU11秒严格合并588完整parent/growth/新学生臂。
金倍增学生相对父模型名次差+0.551 [0.061,1.041]，共同成长差+0.029 [-0.110,0.166]；
T12存活90重复中只有66个六本，对照教师97/97。普通三核心门外行为的名次与成长均与父模型相同。
不追加这版学生PPO。固定模型标签分类1159952仅16秒：总体85.745%，升本16/84，恢复核心104/183，
三连49/102，收尾填场14/197；动作不匹配本身不等于错误，但实际升本停滞与关键标签不足相伴。

payoff-event-learning计划是一次同预算对照：同fresh父权重/架构/数据/seed1203801、2000更新、lr3e-6、
每步96示范/192锚点、KL系数10/停止0.03。只改变示范采样：一半仍按三个回合阶段（各16），
一半按六类过程事件（各8）；事件映射固定，不采新示范、不使用开发标签优化、不延长原2000步终点。
固定终点再做相同196完整续局，不靠总体标签准确率晋升。若关键决策提高却无长期收益，不继续堆训练时长。
单金私掠者CPU1159958同时独立运行，结果不会中途改变本次教学门控或数据。

## D-105 · 2026-11-18 · 只读定位闭环偏移及教师私有循环记忆

新增封存轨迹旁路审计：对全部49开发来源的replica0金倍增教师及固定学生轨迹，使用原动作推进
同一PPO动作守卫，并从公开历史重建教师记忆，不执行教师的新建议、不创建新能力样本或教学标签。
检查教师可完全复现、学生访问状态中的建议差异、招募期死亡及清除cycled记忆后的动作变化。
原生完整回合测试证明守卫/教师重建一致，并确实存在同一公开场面下由cycled私有记忆改变的建议。
这是控制器记忆消融，不证明存在两个自然可达、观测混叠的环境历史，也不把所有教师分歧标为模型错误。
若没有此类限定，很容易把不完整可观察的教师标签直接当作无记忆学生应该100%学会的最优目标。
本次只运行CPU分析，不因诊断预先更改模型架构或追加PPO；D-104采样对照保持冻结。

## D-106 · 2026-11-18 · 单金条件收益边界与旁路教师拒答保留

1159958完整196续局20:25结束：单金私掠者growth−parent名次-0.337 [-0.867,0.194]，
共同T9–T12对数成长+0.476 [0.370,0.583]（46源/87重复）。成长可信，名次收益仍不确定；
不能据此把普通铜须组合宣布为已验证全局强教师，也不修改D-104运行中模型门控。

只读1159977在7:08因教师对破坏性待选状态无安全续步而退出。这是影子教师适用域限制，
不是原生动作非法或新对局失败。新诊断将这一确切异常计为显式拒答，保留路径/回合/动作位置，
不将拒答替换为标签或跳过游戏。若封存教师自己的轨迹发生拒答则仍硬失败；其他异常也仍硬失败。
新增原生“两步施法后只能破坏核心”测试确认拒答被计数。完整教师/学生旁路审计将在新目录重新运行。

D-104事件采样1159967完成8:55，模型SHA4b7958edea4584a3dd0ceca0c59c930ed8486487d08f09200bbd6c2fbed1a610。
基础/critic不变，开发动作匹配83.601%；1159992只读标签检查：升本49/84、恢复核心128/183、
三连80/102、填场76/197、结束160/170，部分常见动作匹配降低。1159990固定196续局运行中，
1159991依赖完成后合并全部完整臂；不根据分类分数提前判断棋力。

## D-107 · 2026-11-18 · 停止单纯重采样，固定一轮学生访问状态聚合对照

1159990于07:40:03完成196续局，1159991于07:40:17完成严格四臂合并。
事件学生−parent名次+0.490 [0.010,1.000]；相对阶段学生仅-0.061 [-0.510,0.388]。
共同T9–T12成长相对阶段学生+0.108 [-0.011,0.228]；没有可靠的整体能力改善，不追加该配置PPO。
T12六本75/84，虽比阶段学生66/90高，但存活84低于90，不能只报升本率宣称成功。

1159996和1160025只读审计：教师30271有效动作全部从公开历史精确复现，3010次受cycled私有记忆影响。
阶段学生自身访问状态精确建议匹配8686/21359=40.67%，事件学生9439/19564=48.25%，
各有2/49及3/49源在招募期死亡（仅replica0范围）。这与示范分布匹配85.745%/83.601%区别很大。
精确动作分歧可能仅为位置不同，因此1160029额外按动作类型/卡牌分解，不把全部分歧当错误。
阶段学生有1次破坏性目标无安全教师建议；原失败审计1159977保留，1159996新目录显式拒答而非伪标签。

参考[DAgger](https://proceedings.mlr.press/v15/ross11a.html)：模型自身动作改变后续访问分布，
仅拟合教师轨迹不足；可以聚合学习者所访问状态上的专家查询。但本项目教师只是局部启发式，
且依赖未完整编码的历史，不能直接援引论文保证HSAI棋力或假定所有查询有最优答案。

执行`payoff-query-refinement-20261118.json`，本窗口仅一次聚合轮：
从207原训练源中预先等距选择64个，不看结果、不用49开发源；事件学生固定执行全部动作，
同complete_gold初态，未来seed12300000+原全源索引*100。只在金门控内查询公开教师，
实际动作与标签分开保留；强制动作/门外/不安全破坏目标/自伤pending序列明确拒绝标签。
合法死亡完整保留，所有源须通过原生和注入质量，不按胜负筛选。

先两训练源GPU完整采集探针，再64源采集；原开发标签与原始证据保持不变。
之后两臂从同一事件学生固定终点、新Adam、seed12303801、lr3e-6、各2000步、96示范/192锚点出发。
对照只重放原数据；干预每批一半原示范、一半学生状态查询，二者内部均按同一阶段/事件混合方案。
只专家可训练，基础策略/critic逐位保留，原KL停止门禁不变。各固定终点再196完整续局，不选中间模型。
这区分新增访问状态与单纯多训练2000步；不改变输入、奖励、对手、动作契约，不解决自然组件获取，
不使用正式保留集、不预先承诺PPO。若仍无闭环收益，停止这次聚合路线并报告限制。

## D-108 · 2026-11-18 · 隔离“已有钩牙未部署”的条件因果贡献

1160029分解后确认不是主要由位置不一致造成：阶段学生普通钩牙建议3994次仅11次同卡打出，
事件学生2033次仅39次同卡打出。它们是相关重复访问，不是独立游戏；不能只凭此频率断定全部名次差由钩牙造成。

固定一次98续局诊断：全部49开发源/两未来/complete_gold，从封存事件学生出发，只有公开教师建议
将手中钩牙打到已有空位且学生没有正在打任意钩牙时，才执行这一步合法部署。学生已选择打钩牙时保留其位置。
不强制购入、不保护出售、不替换其他决策、不送牌送金币；动作仍经普通守卫与100上限，实际干预数逐局记录。
与封存事件学生的98同初态同未来完整续局配对，包括没有触发干预的所有源，不按结果/触发率筛选。
这是手写策略干预的条件因果诊断，不是新的神经模型能力或自然棋力晋升；不加入任何训练标签。
D-107两臂训练/查询数据保持已冻结方案，不据本诊断中途改配方。

## D-109 · 2026-11-18 · 隔离循环记忆对教师长期收益的贡献

只读D-105证明同一公开场面清除cycled集合会改变3010/30271个教师建议；并未证明自然可达观测混叠，
更没有证明这项历史依赖导致学生失败。为了决定是否值得引入记忆型架构，先检验教师是否需要此偏好。

固定98 CPU续局，全部49开发源/两未来/complete_gold，与已封存growth臂精确配对。
唯一变化是每次启用教师前清空cycled实体集合；保留公开场面、守卫、核心保护/获取/升本、最后打牌来源记忆及其他规则。
因此它是“循环实体记忆”消融，**不是无任何记忆的策略**，也不是神经架构消融。
新模式名growth_no_cycle_memory显式登记，默认legacy/growth分支行为保持；原生测试核验只有清空此集合的效果。

八CPU/16GB/30分钟预算，不占GPU、不生成教学标签；一次完整对照，不自动替换教师或重做数据。
若不能保持长期收益，不以“更容易拟合”为由降级教师；若收益保持，也仅为后续可学习性研究提供依据，
不能据此假定无记忆学生就会成功。D-107两臂固定终点评测继续保持原方案，正式保留seed不用。

## D-110 · 2026-11-18 · 停止一轮聚合，孤立检验必要组件售出导致的门控退出

D-107完整两终点196+196续局与六臂1176严格合并完成。聚合减重放名次差+0.0816
[-0.4184,0.5510]、共同成长差−0.0565 [-0.2076,0.0993]，没有确认聚合更优。
两者相对父模型共同成长均为正，但名次点估计仍差且区间跨零，不追加长训、不晋升、不再聚合一轮。

1160141固定分区拟合：聚合在旧学生查询上的总体匹配83.65%、恢复核心85.98%；
1160144实际自身状态复核：聚合14110/20341=69.37%，重放12753/22355=57.05%。
普通钩牙恢复建议从重放2386次降到聚合118次，同卡打出42/2386→56/118；大量反复漏部署得到改善。
但聚合49来源replica0观察到21次门控退出，16次紧跟出售；重放只有6次/5次。是相关事件，不是独立局数。
这支持检验错误是否转移到组件保留/退出边界，尚不能直接当作因果结论或宣布永久禁卖。

payoff-gate-retention计划固定98完整续局：同聚合权重、49开发源/两未来/complete_gold。
仅在金门控激活时，移除“卖掉场上该组最后一个已拥有成员，从而立即破坏门控”的Sell概率，再归一化。
其余动作分数排序、价值、购买/打出/法术/资源、原生规则和100动作守卫不改；手中或场上另有同组成员则仍可卖。
不强制部署钩牙，不用教师替代其他动作；完整记录被阻止的动作和实际替代次数，包含所有零触发局。
单H100-47/4CPU/45分钟上限，完成后与封存聚合臂严格配对。明确是手写策略约束的因果诊断，
不是新神经模型、不是自然棋力晋升；不自动把该限制部署或用于教学。

成本敏感模仿研究可作为后续方向：[AggreVaTe](https://arxiv.org/abs/1406.5979) /
[AggreVaTeD](https://proceedings.mlr.press/v70/sun17d.html)强调使用后续代价，而不只计数教师动作分歧。
本轮尚未实现这些算法，也没有可靠的每动作专家Q估计；不能套用论文保证HSAI收益。

## D-111 · 2026-11-18 · 循环记忆有条件因果收益，并完成原生历史混叠证书

D-109的1160126/1160127完成：98续局/294严格合并，清空cycled后相对原growth教师平均名次差
+0.7041 [0.3265,1.1023]，49源/98配对；夺冠75/98→59/98，前四95/98→82/98。
共同T9–T12成长差−0.0410 [-0.1772,0.0937]，48源/92共同存活重复。
不能因为幸存者成长接近就认为记忆无用；不将教师替换为该简化版，不据此生成新示范。

新增`tools/prove_play_history_alias.py`构造同一明确编辑初态后的两段合法原生动作：
两边各打出A/B一次、各出售A/B一次，最终可见卡牌顺序/属性、旧3227及扩展3282浮点、124卡牌与合法掩码逐字节相同。
一边保留新A/旧B，另一边保留旧A/新B；教师从公开历史重建的cycled槽位分别[4]/[5]，建议Sell20/Sell21。
因此当前确定性前馈输入无法精确表示这个教师的两种选择，已比“直接清空内部记忆”更强地验证了标签可表示性限制。

限制必须同时保留：起点有显式token/组件/等级/金币编辑，不是自然游戏访问频率证明；
也没有证明两种出售的最优性或最优游戏策略一定需要记忆。教师身份记忆可能是有用启发式，不能等同完整环境状态。
后续优先评估明确的公开动作历史/循环实体标记或小型序列状态，同时保留成本敏感恢复验证；
不把“加RNN”当已证实的万能解，不在现役权重/正式协议中偷偷追加输入。

## D-112 · 2026-11-18 · 公开动作历史旁注，为下一次输入消融准备对齐证据

实现独立`PublicTurnHistory`：只读取己方公开实体标识/手牌类型和**实际执行**的Play/其他动作，
记录本回合跟踪以来由Play引入且仍在场的实体，以及已执行决策数；换回合/显式reset清空，状态带schema身份可保存恢复。
Play触发的新token也会标记，不能声称只是手牌本体；不补造controller reset前的历史，不调用教师或读取隐藏RNG。
原生D-111两段输入混叠历史得到不同槽位标记；完整原生教师回合的在线公开视图与封存记录投影旁注逐行一致。

按`006175b0310401ff3d2f3e1d6b0aad025d8c0a9fe2c1a27faddda5ab63e769ea`锁定原示范manifest。
先一训练源/一开发源CPU探针，成功后256源完整旁注；每源核验原轨迹/labels文件SHA、全部动作与标签行数、
动作/理由/回合顺序，最终核对原总149806行及seed/dev/turn逐数组对应。
只在新目录写history.npz，保留原original_row索引（探针也映射原全量索引），不重写任何原标签或数据划分。
八CPU/16GB/35分钟上限；没有新增对局、没有在本步骤启用新模型输入或提交新学习。
这是后续公开历史输入对照的可复用工程准备，不是已验证的能力提升或完整PPO记忆恢复。

## D-113 · 2026-11-18 · 固定底座的小型公开历史残差学习对照

D-110的1160159/1160160完成。禁止出售最后门控组件共干预150动作、23重复、20源；
相对原聚合模型名次差−0.0204 [-0.2755,0.2449]，共同成长差+0.0342 [-0.0040,0.0796]。
T12金引擎在场从65/92提高到82/91，但整体收益仍不明确，不继续叠加手写禁令。
D-112完整旁注1160195已产出149806行；原119535训练/30271开发划分逐行保持。

基于D-111有收益的教师记忆及原生观测混叠，执行一次隔离学习对照，而非临时全量重写PPO。
冻结D-107重放终点242430b6...，在已有金引擎门内加565237参数的动作logit修正网络，
输入为现有己方场面/手牌/商店及机制特征、卡牌嵌入、底座logit与7个公开历史标记。
两臂参数量、初始张量、训练行、采样序列、Adam及预算完全相同；唯一差别是历史标记真实输入或全置零。
修正幅度固定6*tanh(raw/6)，门外精确回退底座，critic全程不改。历史只由实际执行动作更新，不读取教师内部记忆。

计划`history-residual-20261118.json`：原始教师数据，无新增聚合；seed12403801、lr0.001、
每步96条既有阶段/事件混采、4000更新、梯度裁剪1。一次GPU缓存冻结底座logit，随后仅学习小网络；
不按开发分类分数挑中间checkpoint。每臂固定终点按49开发源×两未来×normal/complete_gold做196完整续局，
比较history−no_history与固定底座，正式保留集继续不用；不追加参数扫描或自动PPO。
一GPU/4CPU/32GB/30分钟学习预算，两个单GPU/45分钟上限评测。允许正常作业跨10:30交接继续运行。

资产明确为“冻结base＋独立residual”，分别核验SHA、词表、输入/历史schema并写入逐动作证据；
保存各自Adam/RNG恢复状态，但不是PPO Trainer或可独立上线模型。局部启发式示范、单种初始化、反复使用开发集、
人工金组件和自然获取缺失等限制保留，历史标记有效性必须由闭环结果判断。

## D-114 · 2026-11-18 · 原生三连前置投资与后续运营的四臂配对诊断

此前单买经济核心反事实没有确认收益，但其后续仍由不会完整运营的策略执行；
D-094已经验证金倍增growth完整续局有优势。下一步只检验一步原生投资的收益是否依赖后续策略，
不因为攻略叫核心就直接写成正标签，不新增训练或改变正在运行的历史输入对照。

使用已单测的privateer_triple前置局面：明确赋予金铜须/罗杰斯/逃兵，普通私掠者场上/手中各一张，
商店第三张可合法购买；保留来源金币、健康、等级与对手，原生Buy实际付费并完成三连，不直接把结果写成金卡。
从既有49开发起点预先取等级5或6且金币>=3的全部来源，逐项登记不合格原因；不按胜负/教师成功筛选。

四臂为parent、parent首动作Buy、growth、growth首动作Buy；购买只替换第一步且服从普通245动作守卫，
100动作计数不重置、教师记忆按实际选择推进，之后完全使用对应的同一后续控制器。对照允许随后自然购买。
每源两个相同未来RNG，最多392完整续局；先一源四臂工程探针，成功后完整一次。
分别比较两种后续下的首步购买效果，再用源聚类bootstrap估计名次效果之差；不把它混为更换整个教师的收益。

一H100-47/4CPU/32GB/45分钟上限，不抢占现运行作业、不变更资源配额。
冻结计划`native-triple-bridge-20261118.json`。这是人工前置条件中的一步衔接，不是自然获取概率、
最优Q值或新模型能力；不自动加入标签、不自动提交PPO。正常作业若跨截止，保留并交接。

## D-115 · 2026-11-18 · 残差候选出现条件收益，固定终点做一次独立来源验证

1160237/1160238于09:26:42/09:28:12完成17:44/19:14；1160239严格合并980续局，
329114实际动作、29564快照尝试全部成功，公开历史全部从实际动作复算一致。
金起点无历史/有历史平均名次2.7041/2.5102，夺冠45/98与49/98；原重放3.0204、30/98。
有历史减重放名次−0.5102 [-0.9388,-0.0816]，49源/98配对；对父模型−0.2245 [-0.7143,0.2755]。
共同T9–T12成长对父模型+0.3793 [0.2269,0.5347]，46源/86配对。
但是有历史减无历史名次−0.1939 [-0.6224,0.2449]、成长+0.0460 [-0.0914,0.1866]，
不能将全部收益归因7个标记；这是额外小网络/优化加历史的组合候选，反复使用开发源且仅一个训练初始化。

只进行一次新来源验证，不再调参或挑checkpoint：v1.4父模型从seed12602000..12602127采集128自然源游戏，
在独立bank明确assessment_only，原有训练数据生成入口拒绝该标记。候选均已固定、所有新来源永不用于本轮学习。
选全部海盗房间T8/T9、健康>=15的最早可用状态，每个源仅一个；这次全部来源均为评测用途，不再取seed%5。
每源两个共同未来RNG、complete_gold显式组件起点，比较parent、无历史残差、有历史残差三固定臂。
不追加教师臂、不使用正式保留1119000..1119511、不把此条件验证包装为自然形成或正式晋升。
这是对本轮示范来源独立的验证，不声称早期缺失历史中的全部PPO种子可完全追溯。

CPU16核/32GB/25分钟新bank；每个固定模型单GPU/4CPU/45分钟上限，最多128×2×3=768续局，
实际合格来源预先由同一无结果条件选定。所有作业均有冻结源码、模型/数据身份和自动严格合并；
如到10:30仍正常运行，则交接继续，无新长训或自动晋升。

另外增加显式冻结名次效用表的只读补充统计（1,.35,-1,-1.15,-1.3,-1.45,-1.6,-1.75），
不改训练奖励/正式判据：有历史减重放效用+0.3694 [0.1041,0.6265]，减父模型+0.2990 [-0.0189,0.6225]；
有历史减无历史+0.1352 [-0.1260,0.3995]。不能把import时默认线性效用误当成实际冻结表。

## D-116 · 2026-11-18 · 新来源采集失败修复与三连诊断结果

09:33的1160307在5秒内被子采集器旧训练种子范围检查拒绝；没有产生评测结果。
1160308/1160309/1160310/1160311依赖链尚未启动，已取消，旧现场完整保留。
修复为父、子采集器均显式传递assessment-only，校验独立12602000..12602127范围、
各分片用途一致；默认训练范围不变。新增边界测试覆盖错用途、越界、空范围。
在新冻结release、新输出序号2重新采集，绝不覆盖序号1。
在任何新验证结果产生前，同时将未来RNG固定为12660000+case*100+replica，
与旧开发续局11960000命名空间分开；所有三臂保持相同规则，不改模型/样本筛选/预算。

三连诊断1160266探针、1160267主运行、1160268严格分析均完成。
49来源中仅14符合事先的5/6本、金币>=3条件，35个4本来源预先排除；共112完整续局、
43422实际动作、3329快照注入全部成功。parent/parent_buy/growth/growth_buy名次均值
3.3571/3.3571/3.1786/2.0000；同growth后续首买收益−1.1786 [-2.2500,-0.1786]，
同parent后续首买收益0.0000 [-0.7857,0.7143]。但两种首买效果之差
−1.1786 [-2.5000,0.0366]仍跨零：这是支持“先学会后续兑现，再学习前置投资”的小样本线索，
不是已证明的交互效应、自然成型或普适购买标签。保留人工其他金组件与重复开发源限制。

## D-117 · 2026-11-18 · 只读核对新循环策略的动作上限现场

D-116重跑1160328采集正常运行，1160329/1160330/1160331固定三臂、1160332合并已提交依赖链，
冻结r74=38eb047。来源与未来种子均独立于旧开发续局，模型不再调整。

D-113的T12父模型仅2/94回合触及100动作，growth教师59/97、重放41/87、
无历史34/90、有历史43/90触顶。这是已有循环能力后的新条件现象，不能倒推成旧自然模型的根因。
提交一次CPU8核/16GB/20分钟只读扫描五完整臂中全部T12触顶回合，不抽取看起来最差的案例；
核验封存分析、完整结果与逐动作SHA，记录强制EOT前最后20次实际动作种类、金币净流出、
场面总攻血变化、结束前金币/手牌/场位。排除EOT后战斗变化，不把金币净流出冒充总花费。
这是描述性生存子集，不估计加大动作上限的因果收益；本轮不修改100动作契约、不重开评测榜。

## D-118 · 2026-11-18 · 触顶现场结果与来源隔离验收

1160347于09:48:24完成4:19，五臂179条T12触顶轨迹SHA全部核验。
有历史43个触顶回合/35来源的最后860动作：Play349、Refresh233、Sell184、Buy57、Choose34、
Activate2、Magnetize1，无Move。最后20步场面总攻血增量中位+420，结束时金币15/手牌4/场位6；
无历史34回合对应+173、16.5/5/6，教师59回合+383、15/4/6。所有强制结束前均无pending。
支持继续研究固定预算内的循环收尾，而不是把它误判为反复移动；仍非追加动作收益的因果估计。

独立来源1160328在09:44:10完成4:06，128局/696状态、65合格来源。
三固定模型各130完整续局，1160329/1160330已启动，1160331受两GPU正常配额等待；不抢占、不改预算。
用途隔离新增六个真实构造入口测试（初始测试调用形参错误已更正），均在读取状态/权重前拒绝评估库。
本地r20全量1127项，1119通过/8跳过/零失败，109.567秒；之后新增触顶统计与六入口隔离7项通过。
本轮阶段综合报告为docs/analysis/RESEARCH_STAGE_REPORT_20261118.md，仍以最终交接节为截止状态。

## D-119 · 2026-11-18 · 截止前跨平台CPU回归与交接

科学方案与三个独立来源固定终点保持不变。新增代码已在Windows通过，另在NUS计算节点冻结发布上
执行完整Python测试和pip依赖一致性检查，8CPU/32GB/15分钟上限，不使用GPU、不安装依赖、不重编译未改引擎。
输出新目录、JUnit及verify摘要和源码SHA，覆盖服务器Python3.12与本地3.13的差异。
仍按10:30停止主动工作，正常在途验证作业若越过截止则保留交接。

## D-120 · 2026-11-18 · 服务器回归的精确浮点断言失败定位

1160375完整NUS CPU验收失败：1134项中1115通过、17跳过、2失败，213.467秒。
失败为条件专家冻结后回退输出逐位相等、同批两行仅待选来源不同的旧模型输出逐位相等。
依赖、卡表、引擎源码与目录检查通过；不能把完整服务器测试写成通过。
先用独立CPU探针分离冻结参数标志、学习前后、同批位置、重复调用与存盘重载，记录最大误差及argmax。
不修改当前GPU作业、环境、推理核开关或科学模型；不直接放宽断言。
Windows初探已复现关闭MHA fastpath时仅切换requires_grad就出现约7.2e-7误差，
固定冻结状态后的实际专家学习不改变base权重/回退；NUS原因和幅度仍待探针确认。

## D-121 · 2026-11-18 · 数值路径已定位，保持关键精确语义的测试修正

1160400于10:00:51完成12秒。NUS Torch2.10.0+cu128 CPU原设置下，切换requires_grad最大差2.38e-7，
重复调用相同位置逐位一致；两个完全相同输入的不同batch行可差1.91e-6，argmax在探针中一致。
固定冻结状态后，专家更新前后base权重、回退输出、critic全部逐位相等；省略的pending来源卡
在相同batch位置改写前后也逐位相等。Windows Torch2.13.0+cpu默认路径没有该差异，关闭fastpath
才复现冻结舍入差。未修改任何环境、推理核默认设置、模型或运行评测源码。

仅修正测试比较条件：学习前后和重载都匹配冻结标志，仍严格torch.equal；忽略来源字段检验保持batch
位置不变并严格torch.equal。冻结标志切换本身另按实测量级2e-6绝对/相对容差检查，而非所有断言放宽。
本地14项定向通过；从新冻结版本、新序号2目录重新运行NUS完整Python回归，并做本地最后一次全量回归。
跨平台/跨算子设置“所有logit逐位一致”不成立，不将之写成恢复保证；本轮实际冻结评测不受测试修正影响。

## D-122 · 2026-11-18 · 新来源结果的元数据与图表口径

1160404已从新r78=0d1976e、序号2目录重跑服务器完整CPU回归，旧失败证据保留。
新来源三臂仍使用原r74模型和运行配置。汇总器原有“重复开发来源”通用说明不适用于这次新bank；
图表原固定49/98标注也不适用于65/130。新增可选assessment-bank身份核对后明确标记来源用途、
128采集来源/65实际合格来源，图表从封存完整结果推导来源数/重复数/纵轴范围。
不改排名、成长、bootstrap或选择任何结果。仅取消尚未启动的1160332分析依赖，以新冻结发布
提交同三完整臂的分析后继；原GPU评测不变、不中断。身份不匹配和未知来源的拒绝测试通过。

## D-123 · 2026-11-18 · 独立来源完成：成长泛化，不晋升、不继续加训

1160331于10:09:48完成18:59；新分析1160425于10:09:57完成9秒。
三固定臂65新来源/各130完整续局，共390局157854实际动作，11305快照尝试全部应用。
新bank用途/身份/来源覆盖核验通过；正式保留seed未使用。

父模型/无历史/有历史平均名次2.7308/2.7769/2.6000，冠军45/54/62，前四111/104/107（分母各130）。
T12总攻血P50为2224/n123、3782/n117、3797.5/n118。
有历史减父模型名次−0.1308 [-0.5385,0.2769]，65源/130对；
共同T9–T12 log成长+0.2630 [0.1367,0.3925]，62源/113对。
冠军率+13.08pp [1.52,23.85]，前四−3.08pp [-13.08,6.92]；
冻结名次效用+0.1773 [-0.0554,0.4012]仍跨零，副指标未做多重比较校正。
有历史减无历史名次−0.1769 [-0.5387,0.1846]、共同成长−0.0632 [-0.1715,0.0444]。
因此确认候选有可泛化的条件成长，不确认平均名次/效用优势，更不确认历史输入单独获胜或自然棋力突破。
不晋升、不换pin、不追加训练；下一轮优先固定100预算内收尾和高后果动作价值，再向前置获取扩展。

1160332未启动即取消，首次新依赖提交因已完成作业不再处于活动依赖窗口被Slurm拒绝、未分配作业；
先核验父/对照完整130局结果，再依赖尚运行1160331提交1160425。旧现场均保留。
汇总器和图表仅改来源元数据，主统计未变；Windows按全部封存终点重算排名、成长、bootstrap和曲线，
最大差1.11e-16（只有5个浮点字段），其余一致。补充效用v2仅继承新来源说明，所有估计与v1完全相同。

最终全量Python回归：Windows r21 1134项/1126通过/8跳过/零失败，108.893秒；
NUS 1160404 1134项/1117通过/17跳过/零失败，199.068秒，pip check和工作区门禁通过。
新增来源元数据拒绝测试另行通过，实际NUS三臂汇总也通过。跳过项含CUDA、Windows及不随Git发布的历史资产。
10:12队列已无运行或等待作业；本轮科学实验全部完成，接下来仅整理证据、同步和截止交接。

## D-124 · 2026-11-18 · 自然整局的一回合对手延迟短对照

用户开启无旧截止的新研究轮，正式长训单独等待确认。D-123 的条件残差没有自然整局晋升依据。
本次检验训练中后期存活窗口是否限制自然成长。仅在训练场 T12–T20 使用原 v14 前一档完整对手帧，
T1–T11 原样，T21 后仍按原缺帧行为。训练不读取考试库。采用原 1158709 的发布、父权重、
超参和 200 万决策终点。seed 1181801 复用该已封存对照；第二个 seed 14818001 新跑两臂。
两组都在原难度 v14hold 的新开发种子 13600000–13600191 上配对，附父模型。训练和评测不挑 checkpoint。
主要看 T9–T12 共同幸存成长、逐回合总攻血 P50 与幸存数，同时看全局名次、终局效用和前四。
两训练种子自然成长同向且整体无明显退化，才考虑把延迟场用于长训。试验结果另记。
四锚点沿用 D-079；正式保留集不用。

## D-125 · 2026-11-18 · 只用开发轨迹检验条件循环的预算末段收尾

D-123 已证明触顶时仍在增长，但没有测量保留当前完整战团能否更好地过战斗。
在旧开发金起点有历史残差的全部 49 源 × 2 续局上，T12 已用 95–99 次动作后，
首次满场、EndTurn 合法而策略准备出售时，对比当场结束和沿封存轨迹运行到本回合本来的结束状态。
两端从各自原生状态执行 EndTurn，按来源/续局配对 32 个新随机流，自然执行结束回合效果与战斗。
未触发的局在整组差中贡献零；另报触发子集的来源聚类区间。只解释本次战斗分、存活和战后生命；
控制臂沿已记录的招募随机路径，不能外推全部商店未来或整局名次。若收益不清，不增加提前结束教师。
版本、资产及隔离同 D-123；对局动作契约与评估保留集均不动。

## D-126 · 2026-11-18 · 收尾战斗结果与自然整局卡类计数修正

1160693 在 98 续局中有 32 次触发，涉及 25 个来源。每次 32 个配对未来，原生 EndTurn 共
2,048 次。触发来源提前收尾的战斗分差 −0.0075，95% 区间 [−0.0247,+0.0034]，战后生命差
−0.020 [−0.205,+0.123]。没有采用满场提前结束或禁售教师。该诊断保留随机商店路径的限制见 D-125。

核查普通局动作分布时发现，evaluate 的卡类细分读取公开视图的 type，但采集器实际拿的是
player_json 的 item.Spell / item.Minion，旧报表把法术计成随从。修正观测性子计数并从封存动作逐条复核。
不修改已封存评测、动作选择、奖励或名次。D-124 的运行发布固定，卡类细分须以后处理复核为准。

## D-127 · 2026-11-18 · 保留原场，检验终局奖励的第二至第三名断崖

D-124 三项新短训 1160650–52 完成。两个种子延迟场减原场共同幸存成长 +0.0193/+0.0373，
95% 区间均跨零；平均名次 +0.266/+0.156，原效用 −0.127/−0.098。新开发 192 局/策略。
原场两控制比父模型多 9/8 个冠军，前四少 12/0 个。迟一档练习不能凭成长点估计进入长训。
1160743 对旧父模型开发 64 局逐条复算 12,904 动作，实买法术 913 次、打出 2,243 次；
旧计数全为零源自封存评测的遥测类型错误，不能诊断为没有买法术。

限定最后一个奖励机制对照。训练效用按名次取 1,.35,0,-.35,-.70,-1.05,-1.40,-1.75，
保持冠军、亚军和末名，第二至第八各相差 .35。原表把 2→3 差设为 1.35、3→8 各差 .15。
两组沿用 D-124 原场、固定 seed、父模型、原训练发布和预算，与各自既有原场控制比较。
新开发种子 15600000–15600255，每个模型一次 256 局，父模型及两组固定候选/对照同测。
按原效用、名次、胜率、前四与共同 T9–T12 成长评估，不以新奖励衡量自身。没有改变正式协议。
若主要结果冲突则不宣传棋力提升，不按单个较好种子定长训；不再扫描中间奖励系数。

## D-128 · 2026-11-18 · 自然整局对照结案，冻结每组 1200 万决策的下一轮提案

1160825/26 平滑奖励各完成 22 轮，1160833/34 在 256 个新开发种子完成五个固定终点评测并通过回放校验。
平滑减原奖励的平均名次差 +0.199/+0.238，原效用差 −0.083/−0.114，共同 T9–T12 对数成长差
−0.017/−0.099。维持原奖励。D-124 的延迟对手场、D-125 的提前结束规则和此前稀有条件教师均不扩训。

原场原奖励两组与父模型在两个开发面板合计各 448 局，冠军 61/68 对父模型 50，T12 存活 315/312 对
294；共同 T9–T12 对数成长差 +0.003/+0.011，两区间均跨零。只支持有限预算的时程检验。
下一轮提案为 v1.4-r1 权重起步、原场原奖励自然 PPO、新 Adam、原升本辅助标签，
两个新采样种子各 1200 万决策。一次新开发面板共 3 × 512 局，按两组共同成长、T12 存活、原效用与
名次的事前门槛决定是否值得进一步投入。没有自动延长、按较好种子挑冠军或改变实机 pin 的步骤。
本轮共五条新短训，实际 10,274,755 决策；另复用旧控制 1158709 一次。两次自然开发面板共 2240 局。
本轮研究停止提交作业，正式训练须待用户确认。参数、资产 SHA 和训练/评测入口见
[冻结配置](../config/research/natural12m-20261118.json)、[完整结果与决策依据](analysis/LONG_TRAIN_PREPARATION_20261118.md)。

## D-129 · 2026-11-18 · 用户确认 D-128，提交两条正式训练及固定面板

发布 `natural12m-ready-20261118` 的源码为 `50a8ee1181c86482b0e55d7b9754b2c2495f7cfd`。
1160982 在计算节点验证了冻结模型、模型清单及训练/评测战团库清单与数据的六项 SHA。
正式训练 1160983 / 1160984 分别使用预定种子 31818001 / 47818001，目标各 1200 万决策。
1160989 验证终点的累计预算、逐轮工程指标与模型/Trainer 文件身份；通过后 1160990 对父模型及
两组固定终点运行 73600000–73600511 的 512 个相同游戏种子。采用 D-128 的五项扩训门槛，
权重固定后只进行一次有效评测；完成即停，进一步投入需用户确认。

## D-130 · 2026-11-18 · 1200 万决策的两个固定终点未复现中后期成长，停止追加算力

1160983 / 1160984 各在首次达到预算的完整轮次停止，分别为 12,077,494 决策 / 128 轮、
12,024,025 决策 / 127 轮。1160989 核实最终模型和 Trainer、累计决策、逐轮工程计数。
1160990 使用 D-128 固定自然开发面板完成三模型 × 512 局，24 份轨迹通过核验。共同 T9–T12
成长差 A +0.0262（272 局）、B −0.0403（298 局），等权 −0.0071，95% 区间 [−0.0573,+0.0414]。
A/B 的 T12 到达率差为 −5.66 / +0.78 个百分点，原效用差 −0.0596 / −0.0443；等权名次差上界
+0.2607。五项扩训条件均未满足。本轮结束，保留两个终点和完整证据，继续以 v1.4-r1 为参考。
两端状态、引擎/模型/评测四个锚点及复算入口见[固定终点结果](analysis/D128_NATURAL12M_RESULTS_20261118.md)。
