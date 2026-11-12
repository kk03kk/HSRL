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
