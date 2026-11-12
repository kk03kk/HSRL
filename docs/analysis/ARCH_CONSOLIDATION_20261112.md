# HSAI 双端统一管理架构 —— 整理与落地报告

报告时间：2026-11-12 14:20 +0800
执行者：本地 Agent（单 Agent，同时管理 `D:\HSAI` 与 `~/projects/HSAI`）
标记提交：`arch-single-agent-20261112`（annotated tag `059b81e`）
同步状态：本地 / NUS / GitHub 三方同处 `3d25ce6`，两端工作区干净

---

## 0. 一句话结论

**HSAI 已经进入单 Agent 双环境管理状态：代码用 git over SSH 同步（服务器脏就硬拒推送），
大文件用 `ssh cat` 流式传输（实测 17.7 MB/s，逐文件 SHA256），GitHub 降级为里程碑异地归档，
旧的双 Agent 散文协作协议整体退休并完整归档。你以后只提目标。**

---

## 1. 两端现在各有什么、各承担什么

### 1.1 职责（按"什么只能在这里产生"划分，不再按"谁有权写"）

| | 本地 Windows `D:\HSAI` | NUS `~/projects/HSAI` |
|---|---|---|
| 不可替代的能力 | **真实游戏**：读屏、推理接入、自动操作、回执验证 | **GPU**：H100/H200/A100、Slurm、大库、长训 |
| 独有产物 | 实机对局证据 `artifacts/runs/live-*`、`records/evidence/` | 权重、战团库、评测报告、Slurm 日志 |
| 代码 | 全量副本（能读能测） | 全量副本（能读能跑） |
| 不能做 | 训练、建库、大评测、编译 Rust | 实机操作 |
| Python | `.venv` 3.13.5，torch **cpu** | `.venv` 3.12.3，torch CUDA |

### 1.2 实测清单

**本地**：`client/` 6.2 MB / 256 文件，`config/` 33，`docs/` 26，`ops/` 6，`tests/` 101，
`tools/` 41，`engine/` 249 MB / 550，`hsbg/` 137，`workflows/` 129，`deploy/` 43，`data/` 20；
证据与产物 `artifacts/` **42.8 GB**、`records/` **5.3 GB**（都不进 Git）。

**服务器**：仓库 **37 GB**，Git 跟踪 744 文件；`artifacts/` 11.7 GB、`records/` 18.8 GB、
`data/` 758 MB、`.venv/` 5.7 GB、`engine/` 639 MB。
**10 个模型**：`league-final-20261111-step014858`、`league-latest-2026111{0-step012311,1-step013806,1-step014556}`、
`v1.1`、`stage2-proposal-20261111`、`e1-four-20261111{,-prechallenge,-r1-preview}`、`leaderboard-v2`。
**10 个战团库**（`league-eot-v1.1-20261109` 为当前主库）。
登录节点 **4 核 / 15 GB**，`/home` 已用 164 TB、剩 15 TB（92%）。
队列里只有 `1117116 no-shell`（xcnc23 上的分配，非训练），**没有任何正式训练或评测在跑**。

### 1.3 什么放哪（最终规则）

| 资产 | 服务器 | 本地 | 进 Git |
|---|:--:|:--:|:--:|
| 源码（`engine/ hsbg/ workflows/ deploy/ client/ tests/ tools/ config/ docs/ ops/`） | ✅ | ✅ | ✅ |
| `model.pt` | 全部 | 只放线上 pin 的那一个 | ❌ |
| `trainer.pt` | ✅ | ❌ | ❌ |
| 战团库 | ✅ | ❌ | ❌ |
| 训练产物 `artifacts/runs/` | ✅ | ❌ | ❌ |
| 实机证据 | 可导入 | ✅ | ❌ |
| `data/firestone_replays/`（741 MB）、`data/real_library/`（15 MB） | ✅ | ❌ | ❌ |
| `.venv` | 各机自建 | 各机自建 | ❌ |

**判据**：读代码就能跑 → 放两边；读数据才能跑 → 只在服务器；必须在真实游戏边上 → 只在本地。

---

## 2. GitHub 的最终定位（你的问题 1）

### 2.1 结论：保留，但从"同步总线"降级为"异地归档"

| 作用 | 两个 Agent 时代 | 现在 |
|---|---|---|
| Agent 之间的代码同步 | GitHub `main` | **git over SSH** |
| Agent 之间的沟通 | `docs/sync/*.md` 散文 | **废止**（只有一个 Agent） |
| 版本历史 | GitHub | **两端仓库各有完整历史** |
| 异地容灾 | GitHub | **保留 GitHub**（里程碑推送） |
| 大文件传输 | 从不用 | `ssh cat`，17.7 MB/s |

### 2.2 为什么不能整个去掉

本地副本在 `D:\HSAI`，服务器副本在 NUS `/home`。**两份都在会坏的硬盘上。** 一次磁盘故障
或账号问题就两处同时没了——丢的不只是代码，是全部决策历史。GitHub 是唯一一个不在那两台
机器上的副本。代价几乎为零：凭据已在 Git Credential Manager 里，实测推送通过。

### 2.3 为什么不再当日常通道

1. **更慢**：绕第三方一圈，对比 17.7 MB/s 的直连；
2. **多一个故障点**：两端都要能连 github.com；
3. **语义不对**：`push`/`pull` 的隐式合并正是当年两个 Agent 互相覆盖的根源。现在
   `updateInstead` 给的是**硬拒绝**，不是"尝试合并"。

### 2.4 推送节奏与内容边界

- **里程碑推送**：一次工作会话结束时、结构性改动前后、训练出正式结论后。日常改代码不发。
  这样 `origin/main` 是一条有意义的里程碑序列，不是噪音。
- 放：源码、配置、卡表、文档、清单（单文件 ≤ 5 MB）。
- 不放：权重、库、回放、`artifacts/`、`records/`、日志、压缩包（`.gitignore` 已覆盖）。

---

## 3. 代码与大型文件怎么同步（你的问题 2）

### 3.1 代码：git over SSH + 硬拒绝漂移

```powershell
python ops/hsai_ops.py push   # 本地 main -> 服务器
python ops/hsai_ops.py pull   # 服务器 -> 本地
```

本地新增远端 `server ssh://hsai-nus/home/h/hengzhi/projects/HSAI`，双向 fetch/push。
服务器仓库设 `receive.denyCurrentBranch=updateInstead`：

| 服务器工作区 | 结果 |
|---|---|
| 干净 | 接受，**自动更新工作站文件** |
| 有未提交改动 | **拒绝**（`Working directory has unstaged changes`） |

**真实仓库实测（非沙箱，零残留）**：

```
--- 1. dirty a tracked file ---            M README.md
--- 2. attempt a push (expect REFUSAL) ---
 ! [remote rejected] main -> main (Working directory has unstaged changes)
--- 3. server still at old HEAD ---        5683aa0
--- 4. revert + push again ---
   5683aa0..d4a733c  main -> main          (worktree updated, status clean)
```

> 这就是 2026-11-12 早上"服务器 12 个文件飘了 24 小时"的直接解药：**漂移不可能再被一次
> 看起来成功的推送掩盖。**

### 3.2 大文件：`ssh cat` 流式 + 逐文件 SHA256

```powershell
python ops/hsai_ops.py model-pull <name>          # 默认不拉 trainer.pt
python ops/hsai_ops.py model-pull <name> --with-trainer
```

先在服务器算 `sha256sum`，再流式写到本地 `.partial`，落地后本地重算比对，不一致报错并删除；
已存在且 SHA 相同则跳过（可重复运行）。**不需要你手动操作，不需要网盘。**

**实测（经跳板隧道）**：

| 传输 | 体积 | 耗时 | 速率 |
|---|---:|---:|---:|
| `model.pt` + `checkpoint.json` | 67.9 MB | 6.8 s | **10.0 MB/s** |
| `trainer.pt`（单文件，空写入测速） | 243 MB | 13.7 s | **17.7 MB/s** |
| 服务器侧本地磁盘读 | 559 MB | 1.06 s | 527 MB/s（非网络） |

链路约 140 Mbit/s：1 GB ≈ 1 分钟，完整模型目录（≈310 MB）≈ 20 秒，15 GB 战团库 ≈ 15 分钟。

### 3.3 为什么不是 scp / rsync

- **scp**：Windows 路径在 `host:path` 语法里易出错，且不校验完整性；
- **rsync**：本地没装；
- **双向目录同步**：不需要——两端职责不同，真正需要一致的只有代码，那是 git 的活。

---

## 4. 统一管理体系（你的问题 3）

### 4.1 唯一入口 `ops/hsai_ops.py`

| 命令 | 作用 |
|---|---|
| `probe` / `tunnel` | 隧道连通性 / 打印或建立隧道 |
| `state` | 双端状态快照（隧道、两端 HEAD、脏文件、作业、GPU、磁盘、模型、库） |
| `doctor` | 七点自检：ssh 配置、隧道、两端仓库、`updateInstead`、作业、远端配置 |
| `sh` / `sh -f` | 远端执行（带登录节点守卫） |
| `push` / `pull` | git over SSH |
| `jobs` / `models` | Slurm 概览 / 模型清单 |
| `model-pull` | 拉模型并校验 SHA256 |

**三条硬约束（都写进代码，不是文档劝告）**：

1. **脚本走 stdin**（`ssh bash -s`）。已踩两个坑并修掉：PowerShell here-string 吞 shell
   语法；`ssh host bash -c "cd X && cat f"` 实际只把 `cd` 交给 bash（导致 `model-pull` 全失败）。
2. **登录节点守卫**。pytest / torch / cargo / pip install / 训练工作流 / `srun` 默认拒绝。
   守卫剥掉 heredoc 正文与反斜杠续行、只看真正执行的行：
   `git add workflows/train_entity_sync.py` 放行，`.venv/bin/python workflows/train_entity_sync.py` 拒绝。
3. **输出默认截断 12 KB**（`--full` 关闭）。远端输出会进 Agent 上下文。

每条远端命令写进 `artifacts/state/ssh-audit.jsonl`（时间、rc、耗时、字节数）。

### 4.2 状态与文档

| 内容 | 位置 |
|---|---|
| 实时双端状态 | `artifacts/state/latest.json`（`hsai_ops.py state` 生成） |
| 远端命令审计 | `artifacts/state/ssh-audit.jsonl` |
| 当前作业与结论 | `docs/PROJECT_STATE.md` |
| 决策留痕 | `docs/DECISIONS.md`（只追加，本轮新增 D-028…D-032） |
| 一次性分析 | `docs/analysis/<TOPIC>_<YYYYMMDD>.md` |
| 历史文档 | `records/archive/<topic>-<YYYYMMDD>/` |

### 4.3 文档不再静默腐坏

`ops/check_docs.py`：检查内部链接、已退休路径的活引用、以及**指向被 `.gitignore` 忽略目录
的链接**（那种链接换台机器就断）。已接入 `tests/test_ops_guard.py`。

### 4.4 一次开工的标准动作

```powershell
python ops/hsai_ops.py state     # 必须看到 IN SYNC 才动手
```

判据写进 `docs/OPERATIONS.md` §1：`dirty>0` 先读 diff、`heads_equal=false` 先 pull、
有 RUNNING 作业先确认、隧道断先重建。

---

## 5. 旧架构遗留问题的处置

| 遗留问题 | 处置 |
|---|---|
| 服务器 **12 个未提交文件飘了 24 小时** | 逐个读 diff，定性为**真实工作**（1525 行），验证后提交 `304cb0b`。含 `turnband` 回合分位带采样（6 个测试）、`--forbid-unfreeze`、`league_health.py` 健康门禁、48h 作业脚本 |
| 本地 **25 个未提交文件** | 分 3 个提交归档：实机恢复/trace v2、4159 编码修复、ops CLI |
| `docs/battlegrounds_strategy_guide.md` **两端各有一份**（字节相同） | 只跟踪一处（服务器侧提交），本地 fast-forward 合入，SHA 前后均为 `f03150e2…` |
| `data/firestone_replays/`（741 MB）、`data/real_library/`（15 MB）未忽略 | 加入 `.gitignore`，留在服务器 |
| `docs/sync/` **84 KB 散文协议** | 整体退休，归档到 `records/archive/sync-protocol-20261112/`（11 个文件 + 说明 README） |
| `DECISIONS.md` / `ENGINE_RELEASES.md` 内容仍然有效 | 上移到 `docs/DECISIONS.md` / `docs/nus/ENGINE_RELEASES.md`，历史 evidence 路径按"只追加"规则不改，加迁移说明 |
| `check_ownership.py` 的所有权门禁与新架构矛盾 | 语义改为**卫生门禁**：`--strict` 只拦大文件/证据目录/密钥；分类列表降级为"这个文件平时在哪端维护"的提示。本轮它自身跑 `--strict` 退出码 0 |
| **登录节点规格文档写"约 1 GiB"** | 实测 **4 核 / 15–16 GB**。数字更正，结论方向不变（仍不许跑重活），并加守卫强制 |
| `test_4159_long.py` 在 Windows 长期失败 | 真因：子进程 cp936 输出被按 UTF-8 解码，异常让 `stdout` 变 `None`，报成误导性的 "JSON object must be NoneType"。脚本入口钉死 UTF-8（`eec4896`） |
| 文档链接腐坏无检测 | 新增 `ops/check_docs.py` + 24 个回归测试 |
| 两支 `.gitattributes` 与实际工作区 CRLF 不一致 | 审计：索引 729 文件全为 LF（跨机无 diff 噪音），工作区部分 CRLF 属历史遗留，**不做 729 文件的批量重写**（风险高于收益），只记录 |
| 服务器 `records/archive/` **18.3 GB** 只存在于服务器 | 属证据，不删不搬，本报告记录在案 |

---

## 6. 今后只跟我对话，项目怎么运转（你的问题 5）

### 6.1 你只做两件事

1. **提目标**（"把模型拉回本地""跑一轮长训""看看为什么棋力没提升"）；
2. **隧道断了跑一个脚本**（要输一次密码）：

```powershell
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1
```

脚本会先探测、通了就什么都不做，否则建立并**验证**（不信 ssh 的退出码）。`-Force` 精确
清理陈旧转发，只杀 `-L 2222:xlogin2...:22` 那一条，不碰你其它 ssh 会话。

### 6.2 我怎么做

- **开工**：`state` → 有漂移先对账 → 确认运行中作业 → 然后动手。
- **改代码**：在能验证的那一端改，改完**必须提交**，然后 `push`/`pull` 双向同步。
  **不允许服务器留未提交改动过夜。**
- **跑训练**：`sbatch`（作业内不包 `srun`）、新目录、jobid 记账、轮询不 hold。
- **拉模型**：`model-pull`，SHA256 自动校验。
- **实机**：本地跑，每局必须有 `terminal: true` 的 `result.json` 才算有效。
- **里程碑**：更新 `PROJECT_STATE`、追加 `D-###`、推 GitHub。

### 6.3 自主权边界

**自主可做**：读、分析、审计、改代码、写测试、跑测试、建目录、短程 smoke（≤2h）、
提交/取消自己的作业、拉权重、写文档、整理归档、推 GitHub 里程碑。

**必须先问**：删资产/数据/证据、force-push、正式长训（>6h）、大规模评测（>1000 局）、
改游戏目录、改 `config/live/` 的模型 pin。

### 6.4 我做不到的（不粉饰）

- **不是 7×24 常驻进程**。你不说话、也没设定时任务时，我不会自主巡检。
- **不能自己输密码**。隧道重建要你按一次回车。
- **没有交互式 TTY**。`vim`/`top`/`less`/REPL 用不了，必须非交互。

---

## 7. 我实际完成的工作

### 7.1 提交（`b8847ab` → `3d25ce6`，11 个）

```
304cb0b feat(train): turnband mirror curriculum + eval freeze parity + segment health gate   [服务器侧]
8c83703 feat(live): reconnect/quest recovery, unified trace v2, terminal proof
eec4896 fix(4159): make --check-inputs-only output locale-independent
5683aa0 feat(ops): single-agent dual-end ops CLI
d4a733c docs: single-agent dual-environment management; retire the two-agent protocol
796b8e7 test(ops): pin the login-node guard and the doc checker
3eaa106 fix(docs): do not link from tracked docs into gitignored trees
5d0cd4f fix(ops): only flag links into *untracked* machine-local paths
5f028c9 feat(ops): ops/tunnel.ps1 - the one manual step, as a script
15a3873 docs(state): record the architecture verification evidence
3d25ce6 docs(ops): correct the Slurm free-node figures (measured, not aggregated)
tag: arch-single-agent-20261112
```

### 7.2 文件

**新增**：`ops/hsai_ops.py`、`ops/check_docs.py`、`ops/tunnel.ps1`、`ops/README.md`、
`docs/OPERATIONS.md`、`docs/architecture/TRANSPORT.md`、`tests/test_ops_guard.py`、
`records/archive/sync-protocol-20261112/README.md`。
**重写**：`AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/README.md`、`README.md`、
`client/tools/check_ownership.py`。
**上移**：`docs/DECISIONS.md`（+D-028…D-032）、`docs/nus/ENGINE_RELEASES.md`。
**归档**：11 个文件到 `records/archive/sync-protocol-20261112/`。
**更新**：`docs/PROJECT_STATE.md`、`docs/LEADERBOARD.md`、`docs/E_PHASE_RUNBOOK.md`、
`workflows/run_4159_long.py`、`docs/battlegrounds_strategy_guide.md`（去重）。

### 7.3 仓库外

`~/.ssh/config` 新增 `hsai-nus`（`127.0.0.1:2222` + `HostKeyAlias`，BatchMode）与
`hsai-bastion`（`sjump.comp.nus.edu.sg`，保留密码交互）两个别名。

---

## 8. 验证证据

| 项 | 结果 |
|---|---|
| 三方同步 | 本地 / NUS / GitHub 全为 `3d25ce6`，两端 `dirty=0` |
| `doctor` 七点自检 | **全部通过** |
| 本地全量验证 | `tools/verify_workspace.py --skip-rust` → cards / engine-source / python 全 PASS |
| 服务器验证 | `ops/check_docs.py` 干净（29 份 md）；`test_mirror_band.py` + `test_ops_guard.py` **30 passed** |
| 守卫回归 | `tests/test_ops_guard.py` **24 passed**（含两次真实误报的复现） |
| 漂移守卫 | 真实仓库：脏 → `[remote rejected] (Working directory has unstaged changes)`；干净 → 接受并更新工作区 |
| 大文件传输 | 67.9 MB / 6.8 s（10.0 MB/s）、243 MB / 13.7 s（17.7 MB/s），逐文件 SHA256 一致 |
| 新 clone 一致性 | `git archive HEAD` → 743 文件 / 23.9 MB，其中 `ops/check_docs.py` 全绿 |
| SSH 配置解析 | `ssh -G hsai-nus` → `127.0.0.1:2222`、`hostkeyalias xlogin2…`、`batchmode yes`；`hsai-bastion` → `sjump:22`、`batchmode no` |
| 未干扰训练 | 全程未碰 `1117116 no-shell`；未启动任何新训练 |

---

## 9. 限制与遗留风险

| # | 项 | 影响 | 现状 |
|---|---|---|---|
| 1 | **隧道需每天一次人工密码** | 唯一需要你的时刻 | 已给 `ops/tunnel.ps1`；可选升级（Windows 凭据管理器 + askpass 自愈）已设计**未启用** |
| 2 | **我不是常驻进程** | 我不在线时不会自主发现问题 | 需要时用定时任务唤醒；不假装 7×24 |
| 3 | **`1117116 no-shell` 仍 RUNNING（约 15 h）** | 占一个分配 | 未碰。是否取消需你决定 |
| 4 | **D-026 未结** | 实机观测是 `public-approx-v1` 有损投影，训练用完整引擎观测 | 待取证。14858 两局（第 6 / 第 5 名）未见相对 12311 的优势 |
| 5 | **`turnband` + 48h 脚本已提交但未使用** | 无运行依赖它 | 启动需你批准（D-030） |
| 6 | 工作区部分文件 CRLF 而索引为 LF | 纯外观；跨机无 diff 噪音 | 不做批量重写（风险 > 收益），已记录 |
| 7 | 服务器 `records/archive/` 18.3 GB 只在服务器 | 本地无副本 | 属证据，不删不搬 |
| 8 | 本地多出 `artifacts/models/stage2-proposal-20261111`（68 MB） | 传输测试残留 | 可随时 `model-pull` 重取，保留无害 |
| 9 | 单文件传输已验证，未测多 GB 目录树 | 15 GB 库按 17.7 MB/s 估 ≈15 分钟 | 尚未需要；需要时再加 `dir-pull` |

---

## 10. 需要你决定的三件事

1. **是否启动 stage-2 长训**（`config/league-stage2.json` + `deploy/slurm/league_stage2_48h.sbatch`，
   2×H100、94×30min、`turnband` + `forbid-unfreeze`）。脚本与配置已验证可用，但**没有任何
   运行在使用它**。
2. **实机模型是否回退**。`league-final-14858` 在 E1 是点估计第一，但两局实机未见相对 12311
   的优势，且与 D-026 的有损投影解释一致。选项：(a) 先做 D-026 取证再决定；(b) 回退 pin 到 12311；
   (c) 继续用 14858 多跑几局攒样本。
3. **是否启用隧道自愈**（Windows 凭据管理器 + `SSH_ASKPASS`）。收益是把"每天一次密码"降到零，
   代价是凭据材料常驻本机。

---

## 11. 附：一页速查

```powershell
python ops/hsai_ops.py state                     # 开工必跑
python ops/hsai_ops.py doctor                    # 七点自检
python ops/hsai_ops.py push | pull               # 双向同步
python ops/hsai_ops.py sh "squeue -u hengzhi"    # 远端命令（带守卫）
python ops/hsai_ops.py jobs | models | model-pull <name>
python ops/check_docs.py                         # 文档一致性
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1   # 隧道断了
```

规则：`AGENTS.md` ｜ 架构：`docs/ARCHITECTURE.md` ｜ 流程：`docs/OPERATIONS.md` ｜
传输：`docs/architecture/TRANSPORT.md` ｜ 决策：`docs/DECISIONS.md`
