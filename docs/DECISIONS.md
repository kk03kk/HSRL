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
