# HSAI 架构：单 Agent 统一管理双环境

状态：**当前生效**（2026-11-12 起）。取代 2026-11-10 的"双 Agent + GitHub 同步"架构。
传输层细节见 [`architecture/TRANSPORT.md`](architecture/TRANSPORT.md)；
日常工作流见 [`OPERATIONS.md`](OPERATIONS.md)。

---

## 0. 一句话

**一个决策者，两个执行环境。** 本地 Windows 提供"真实游戏边上"的能力，NUS 提供 GPU
与模拟器；两端由同一个 Agent 通过 SSH 直接管理，代码用 git over SSH 同步，GitHub 只做
异地备份。

---

## 1. 为什么要改

2026-11-10 的架构把项目切成两个"拥有写权"的域，两个 Agent 各自维护，靠 GitHub
`main` 与 `docs/sync/` 里的散文互相沟通（该机制已于 2026-11-12 退休，见 §5）。
它解决了当时的真问题（本地/服务器引擎指纹错位），但也留下了结构性的成本：

| 问题 | 表现 |
|---|---|
| Agent 之间不能对话 | 一切协调要走 `docs/sync/*.md`（已废止）：LOCAL.md 9 KB + SERVER.md 37 KB + manifests 27 KB ≈ **84 KB 散文要在两端轮流更新** |
| 同步是"希望"而不是"保证" | 服务器上 12 个文件未提交、飘了 24 小时才被发现；其中有 1525 行真实工作（turnband 采样、健康门禁、48h 作业脚本） |
| 消息靠人中转 | 用户被迫在两端之间复制粘贴结论 |
| 职责切分催生重复 | 同一个文件在两端都存在，谁改的算谁的成了日常问题 |

现在这两个 Agent 已经能直接连通，没有理由继续付这份成本。

---

## 2. 环境职责

职责不再是"谁能写"，而是"**什么东西只能在这里产生**"。

| | 本地 Windows `D:\HSAI` | NUS `~/projects/HSAI` |
|---|---|---|
| 独有能力 | **真实游戏**：读屏、推理接入、自动操作、回执验证 | **GPU**：H100/H200/A100、Slurm、大库、长训 |
| 独有产物 | 实机对局证据、`artifacts/runs/live-*` | 训练权重、战团库、评测报告、Slurm 日志 |
| 代码 | 全量副本（能读、能测） | 全量副本（能读、能跑） |
| 不能做 | 训练、建库、大评测、编译 Rust 引擎 | 实机操作 |
| Python | `.venv` 3.13.5，torch **cpu** | `.venv` 3.12.3，torch CUDA |

**两端都保留全量源码**，因为两端都要读代码；**大资产按上表单向存放**，因为搬来搬去
没有收益（详见 `architecture/TRANSPORT.md` §4）。

本地仍然保留模拟器的**只读副本**与 `hsbg_engine` 的 Windows 轮子，用于：
实机链路的契约校验、离线回放、以及实机侧的单元测试。**不在本地改模拟器**——不是因为
没权限，而是因为改了就与训练侧不一致，那是 2026-11-09 审计已经付过代价的错误。

---

## 3. 三层结构

```
                        ┌─────────────────────────┐
                        │  GitHub kk03kk/HSRL     │  异地归档
                        │  里程碑推送，不是同步总线 │  只在两端都没了时才用
                        └────────────▲────────────┘
                                     │ git push（偶尔）
        ┌────────────────────────────┴───────────────────────────┐
        │                                                        │
┌───────▼──────────────┐        git over SSH        ┌────────────▼───────────┐
│ 本地 Windows D:\HSAI │ ◄────────────────────────► │ NUS ~/projects/HSAI    │
│ · client/ 实机桥接   │   push / pull（随时）      │ · engine/ 模拟器内核    │
│ · 实机证据           │   updateInstead 硬拒漂移   │ · hsbg/ 训练栈          │
│ · ops/ 运维层        │                            │ · workflows/ deploy/    │
│ · 全量源码副本       │   ssh cat（大文件）        │ · 权重/库/数据/产物     │
└──────────────────────┘   17.7 MB/s + SHA256      └────────────────────────┘
        │
        │ 用户在本地与 Agent 对话
        ▼
   ┌─────────┐
   │  用户   │  只提目标，不当中转站
   └─────────┘
```

### 3.1 谁听谁的

- **用户**只和本地 Agent 对话，不需要知道服务器上发生了什么。
- **本地 Agent** 是唯一决策者：规划、改代码、跑测试、提作业、收结果、写文档。
- **服务器**没有任何自主进程。它不是"另一个 Agent"，是被 `ssh` 驱动的执行环境。

### 3.2 为什么不需要服务器常驻 Agent

曾经考虑保留一个服务器侧 Agent 做长任务看护。没有采纳，理由：

1. **Slurm 本身就是看护者。** 作业提交后由调度器管理，失败会留现场；不需要一个
   进程盯着。
2. **常驻 Agent 会制造第二份状态**，而两份状态必然漂移——这正是要消除的问题。
3. **它救不了的场景它都在**：真正需要"隧道断了还能动"的时候，用户本来就要重连，
   不如直接重连。

**应急保留**：如果隧道长期不可用而服务器上有紧急动作，用户可以直接 SSH 到服务器手动
执行——`docs/OPERATIONS.md` §7 给了最小命令集。这是"应急恢复能力"，不是一个 Agent。

---

## 4. 运维层 `ops/`

```
ops/
├── hsai_ops.py    单入口 CLI：probe / state / doctor / sh / push / pull /
│                  jobs / models / model-pull / tunnel
└── （所有远端动作都经过它）
```

设计要点：

1. **脚本走 stdin。** `ssh host bash -s` 而不是拼命令行。PowerShell 的引号、here-string、
   反斜杠会吞掉 shell 语法，这个坑已经踩掉好几轮调试时间。
2. **登录节点守卫。** 登录节点是 4 核 / 16 GB 的共享 VM。`HEAVY_PATTERNS` 里的命令
   （pytest、torch、cargo、pip、训练工作流、`srun`）默认拒绝执行，必须加
   `--allow-heavy` 显式放行。守卫会剥掉 heredoc 正文与反斜杠续行，只看真正会被执行的
   行——所以 `git add workflows/train_entity_sync.py` 放行，而
   `.venv/bin/python workflows/train_entity_sync.py` 拒绝。
3. **输出默认截断。** 远端输出会进入 Agent 上下文，默认 12 KB 截断，`--full` 才全量。
4. **留痕。** 每条远端命令写进 `artifacts/state/ssh-audit.jsonl`（时间、rc、耗时、字节数）。

---

## 5. 与旧架构的对应关系

| 旧机制 | 现状 |
|---|---|
| `docs/sync/LOCAL.md`、`SERVER.md`、`README.md` | **归档**到 `records/archive/sync-protocol-20261112/` |
| `docs/sync/SERVER_HANDOFF_20261110.md`、`SERVER_TO_LOCAL_20261111.md` | **归档**（历史交接记录） |
| `docs/sync/manifests/{local,server}.json` + `tools/sync_manifest.py` | **归档**；实时状态改由 `hsai_ops.py state` 生成 |
| `docs/sync/DECISIONS.md` | **保留并上移**为 `docs/DECISIONS.md`（决策留痕仍然有效） |
| `docs/sync/ENGINE_RELEASES.md` | **上移**为 `docs/nus/ENGINE_RELEASES.md`（引擎发版史仍有效） |
| `docs/SYNC_MODEL.md` | **归档**，内容由 `docs/OPERATIONS.md` 取代 |
| `client/tools/check_ownership.py` | **保留**，语义改为"提示这个文件通常由哪端维护"，不再阻止本地修改 |
| GitHub 作为同步总线 | **降级**为异地归档（§5.4 of TRANSPORT） |

归档目录只作证据，不是启动命令。归档后不再更新，也不参与任何流程。

---

## 6. 不变的东西

架构调整**不改变**任何技术口径。以下与旧文档完全一致：

- 模拟器 `engine/hsbg_engine`，酒馆 **37.0.2.256589**，源码指纹 `c4c71014e3d41760`。
- 动作原生 **245**；观测 **3227 floats + 124 cards**。
- 训练栈 `hsbg/az`；一个学习席 + 七个严格 EOT 快照席；`sims=0`。
- 冻结训练契约：`max-actions-per-turn=80`、`move-guard=1`（cap 5 / repeat 2）、
  `forbid-cancel=1`。
- 评测口径：E1 单轮、`protocol_sha256` 不同不混榜、P50 口径。
- 安全红线：不 `reset --hard`、不 `git clean`、不 force-push、不删资产/证据、
  不改游戏目录、不伪造 SHA/清单、不写密钥进日志。

这些规则现在写在 [`../AGENTS.md`](../AGENTS.md)，仍然是唯一权威。

---

## 7. 已知限制

| 限制 | 影响 | 缓解 |
|---|---|---|
| 隧道依赖一次人工密码 | 每天约一次需要用户执行一条命令 | `ops/hsai_ops.py tunnel` 打印现成命令；可选升级见 TRANSPORT §6 |
| Agent 不是常驻进程 | 用户不说话时不会自主巡检 | 需要时用定时任务唤醒；不假装 7×24 |
| 无交互式 TTY | 不能用 `vim`/`top`/`less`/REPL | 批量命令 + `bash -s`；`tail`/`grep` 代替分页 |
| 远端输出占上下文 | 大输出会挤掉工作记忆 | 默认截断 + 子 Agent 做探索 |
| 登录节点 4 核共享 | 不能在那儿跑测试/训练 | `sbatch`；守卫强制 |
| 本地无 GPU | 不能在本地验证训练行为 | 训练侧验证一律在服务器算 |

> 旧文档写"登录节点约 1 GiB 虚拟内存"。2026-11-12 实测是 **4 核 / 16 GB**。数字错了，
> 但结论不变：4 核且多人共享，仍然不该在上面跑重活。
