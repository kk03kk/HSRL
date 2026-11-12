# ops —— 跨端运维层

一个入口同时管本地 Windows 与 NUS 集群。所有远端动作都经过 `hsai_ops.py`。

```powershell
python ops/hsai_ops.py <command>
```

设计理由与传输原理见 [`../docs/architecture/TRANSPORT.md`](../docs/architecture/TRANSPORT.md)，
日常流程见 [`../docs/OPERATIONS.md`](../docs/OPERATIONS.md)。

## 前置

隧道在跑。不在跑时：

```powershell
python ops/hsai_ops.py tunnel    # 有登记凭据 → 自动重建；否则打印手动命令
```

`hsai_ops.py tunnel` 会自动调用 `ops/tunnel.ps1`。未登记凭据时也可以手动建：

```powershell
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Forget
```

隧道原理见 TRANSPORT §1。

## 凭据（隧道自愈）

```powershell
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Setup
```

一次性登记跳板机密码。密码用 Windows DPAPI（`CurrentUser`）加密存在
`%LOCALAPPDATA%\HSAI\tunnel.cred`，只有当前 Windows 账户能解开；**不进对话、不进命令行、
不进进程列表、不进日志、不进 Git**。Setup 会当场做一次真实认证验证。

| 文件 | 作用 |
|---|---|
| `ops/tunnel-auth.ps1` | 凭据库：DPAPI 存取 + 编译 askpass 助手（被点源引用，无独立进程） |
| `ops/tunnel.ps1` | 用户入口：`-Setup` / `-Forget` / `-Status` / `-Force` / `-Port` |
| `C:\ProgramData\HSAI\hsai-askpass.exe` | 运行时产物，Setup 时编译，**必须在 ASCII 路径** |

两个实测踩出来的硬约束（细节见 TRANSPORT §6.3）：

1. **`SSH_ASKPASS_REQUIRE=force` 必须设**，否则 Win32-OpenSSH 不会用 askpass；
2. **askpass 的 exe 路径必须全是 ASCII**，否则 `CreateProcessW failed error:2`
   —— 所以 exe 在 `ProgramData`，密文留在用户配置目录（.NET 读 Unicode 路径没问题）。

还有一条：**含中文的 `.ps1` 必须带 UTF-8 BOM**，否则 Windows PowerShell 5.1 按 ANSI
解码会打乱行结构。`tests/test_ops_guard.py` 有守卫。

## 命令

| 命令 | 作用 |
|---|---|
| `probe` | 隧道连通性（JSON） |
| `tunnel` | 确保隧道在跑：有凭据则无人值守自愈，否则打印手动命令（`--print` 只打印） |
| `state` | 双端状态快照：隧道、两端 HEAD、脏文件数、作业、GPU、磁盘、模型、库 |
| `doctor` | 七点自检：ssh 配置、隧道、两端仓库、`updateInstead`、作业、远端配置 |
| `sh "<cmd>"` / `sh -f script.sh` | 远端执行（带登录节点守卫） |
| `git-status` | 双端 git 状态 |
| `push` | 本地 `main` → 服务器（服务器脏则被拒） |
| `pull` | 服务器 → 本地（`--check` 只看不合并） |
| `jobs` | Slurm 队列 + 最近 7 天记账 |
| `models` | 服务器模型目录清单 |
| `model-pull <name>` | 拉模型到本地，逐文件 SHA256 校验 |

## 三条不能违反的设计约束

### 1. 脚本走 stdin，不走命令行

`remote_run()` 用 `ssh hsai-nus bash -s`，脚本从 stdin 进入。**不要改回
`ssh host "命令"`**：PowerShell 的引号、here-string、反斜杠会吞掉 shell 语法。

同理 `ssh` 会把多余参数用空格拼接 —— `ssh host bash -c "cd X && cat f"` 实际只把
`cd` 交给 bash，其余变成独立词。这个坑踩过一次（`model-pull` 全部失败）。

### 2. 登录节点守卫

登录节点是 **4 核 / 16 GB 的共享 VM**。`HEAVY_PATTERNS` 命中的命令（pytest、torch、
cargo、pip install、训练工作流、`srun`）默认拒绝执行。

守卫会先剥掉 heredoc 正文与反斜杠续行，只看真正会被执行的行，所以：

| 命令 | 结果 |
|---|---|
| `git add workflows/train_entity_sync.py` | ✅ 放行（只是提到文件名） |
| `.venv/bin/python workflows/train_entity_sync.py` | ⛔ 拒绝（真的要跑） |
| `sbatch deploy/slurm/x.sbatch` | ✅ 放行（提交作业是对的） |
| `srun ...` | ⛔ 拒绝（本站 srun 会把双卡缩成单卡） |

确需在登录节点直跑时加 `--allow-heavy`，并在脚本注释里写清理由。

守卫是**启发式**，不是沙箱。它拦的是"手滑"，不是恶意。

### 3. 输出会进 Agent 上下文

远端输出默认截断到 12 KB（`--max-bytes`，`--full` 关闭）。写远端脚本时自己先
`tail` / `grep` / `jq` 收窄，不要 `cat` 大文件。

## 审计

每条远端命令追加一行到 `artifacts/state/ssh-audit.jsonl`：

```json
{"at": "...", "label": "sh", "rc": 0, "secs": 1.2, "bytes": 340, "heavy": null, "cmd": "squeue -u hengzhi"}
```

`state` 的历史快照在 `artifacts/state/state-<timestamp>.json`，最新一份是 `latest.json`。
两者都在 `artifacts/` 下，不进 Git。

## 为什么 `push` 会被拒

服务器仓库设了 `receive.denyCurrentBranch=updateInstead`：

- 服务器工作区**干净** → 推送被接受，工作区文件自动更新；
- 服务器工作区**有未提交改动** → 推送被拒（`Working directory has unstaged changes`）。

这是**故意的**。2026-11-12 发现服务器上 12 个文件飘了 24 小时，其中 1525 行是真实工作。
现在漂移不可能再被一次"看起来成功"的推送掩盖。

处理顺序永远是：`git-status` 看两端 → 决定服务器上的改动是提交还是丢弃（读 diff，
不要猜）→ 需要时 `pull` 合并 → 再 `push`。**永不强推。**

## 扩展方式

新增子命令：在 `hsai_ops.py` 里写一个 `cmd_xxx(args)`，在 `main()` 里
`sub.add_parser('xxx')` 挂上 `set_defaults(func=cmd_xxx)`。远端逻辑写进
`remote_run()` 的脚本文本（或 `sh -f` 的文件），不要在 PowerShell 里拼。
