# 跨端传输：SSH 隧道 · Git over SSH · GitHub · 大文件

状态：**当前生效**（2026-11-12 建立，单 Agent 架构的传输层）
适用：本地 Windows `D:\HSAI` ↔ NUS `~/projects/HSAI`
工具：`ops/hsai_ops.py`（所有远端动作的唯一入口）

---

## 0. 一句话

**代码走 git over SSH，大文件走 `ssh cat` 流式传输，GitHub 只做异地备份。**

两个 Agent 的时代，GitHub 是本地与服务器之间**唯一**能过的桥——因为两个 Agent 无法
直接对话。现在只有一个决策者，桥就没有必要了：本地直接用 `ssh` 连服务器，比绕道
GitHub 更快、更可靠，而且可以在配置层面**强制**拒绝覆盖。

---

## 1. 为什么连不直连，要套一层隧道

这不是设计选择，是 NUS 的网络与认证决定了只有一条路。

| 主机 | 地址 | 认证方式 | 从公网可达 |
|---|---|---|---|
| `sjump.comp.nus.edu.sg`（跳板） | `137.132.80.26/.27` | **只接受密码**（OpenSSH 9.6p1，无 publickey） | ✅ |
| `xlogin` / `xlogin2`（登录节点） | `192.168.5x.x` / `192.168.49.225` | 公钥可用 | ❌ 校园网内 |

已实测排除的两条替代路：

- **NUS VPN 走不通。** "NUS Student"（Cisco Secure Client）**已连接**，适配器 `以太网 2`
  拿到 `10.195.32.104/19`，但**不路由 `192.168.51.x`**。用户只有 Student 授权，
  用不了 `NUSStaff` 组。
- **Windows OpenSSH 的 ControlMaster 不可用。** 报 `getsockname failed: Not a socket`，
  连接复用无法开启。因此不能"一次登录、多路复用"，只能用端口转发。

**结论：一条持久端口转发隧道。**

```powershell
ssh -N -f -L 2222:xlogin2.comp.nus.edu.sg:22 `
    -o ServerAliveInterval=30 -o ServerAliveCountMax=10 `
    -o ExitOnForwardFailure=yes `
    hengzhi@sjump.comp.nus.edu.sg
```

- 只在建隧道时输**一次**跳板机密码；隧道存活期间所有操作免密。
- 隧道把本机 `127.0.0.1:2222` 转发到 `xlogin2:22`，而 xlogin2 **接受已有的
  ed25519 公钥**（`~/.ssh/id_ed25519`，已写入 xlogin2 的 `authorized_keys`），
  所以后续全是非交互的。
- `~/.ssh/config` 里定义了 `hsai-nus` 别名（`HostName 127.0.0.1` / `Port 2222` /
  `HostKeyAlias xlogin2.comp.nus.edu.sg`），主机密钥校验依然有效。

> `HostKeyAlias` 是关键：socket 虽然在 127.0.0.1:2222，但用它冒充
> `xlogin2.comp.nus.edu.sg`，已 pin 的主机密钥才继续匹配。没有它，每次连接都会
> 报 host key mismatch。

### 1.1 隧道是单点故障——以及怎么恢复

隧道断了的表现：`ops/hsai_ops.py probe` 返回 `ok: false`，错误是
`Connection refused`（2222 端口没人监听）。

```powershell
python ops/hsai_ops.py tunnel     # 打印需要手动执行的那一条命令
```

然后**由用户在本地 PowerShell 里执行一次**（需要输密码），再 `python ops/hsai_ops.py doctor`。
这是单 Agent 架构里**唯一**需要人介入的时刻，频率约每天一次。

**为什么不让 Agent 自己输密码：** 密码进对话就进了会话记录；进命令行就进了进程列表。
两条都不可接受。见 §5 的替代方案。

---

## 2. 代码同步：git over SSH

```powershell
python ops/hsai_ops.py push    # 本地 main -> 服务器
python ops/hsai_ops.py pull    # 服务器 -> 本地
```

实现上就是给本地仓库加了一个远端：

```
server  ssh://hsai-nus/home/h/hengzhi/projects/HSAI  (fetch + push)
```

**双向都通**：本地能 `fetch`/`push` 服务器（隧道是本地发起的外连），服务器不能反向
连本地（本地在 NAT 后、没有入站 SSH）。所以**本地是集成点**：

- 在服务器上产生的提交 → 本地 `pull` 取回；
- 在本地产生的提交 → 本地 `push` 推过去。

### 2.1 漂移保护：`receive.denyCurrentBranch=updateInstead`

服务器仓库不是裸库，它同时是工作区（`~/projects/HSAI` 就是跑训练的地方）。默认
git 会拒绝推送到非裸库的当前分支，因为推送会让工作区与 HEAD 不一致。

设成 `updateInstead` 后行为变成：

| 服务器工作区状态 | 推送结果 |
|---|---|
| **干净** | 接受，并**自动更新工作区文件**（等效于在服务器上 `git checkout`） |
| **有未提交改动** | **拒绝**：`Working directory has unstaged changes` |

这正是我们要的。**有未提交改动时推送会失败**，漂移不可能被一次"看起来成功"的
推送悄悄掩盖——这是 2026-11-12 早上在服务器上发现 12 个文件飘了 24 小时的直接教训。

已实测（隔离沙箱 `/tmp`，不碰真实仓库）：

```
-- case 1: src worktree DIRTY --
   push REFUSED as designed:
     ! [remote rejected] HEAD -> main (Working directory has unstaged changes)
-- case 2: src worktree CLEAN --
   push ACCEPTED; src worktree now at two, f.txt=v1v2
```

真实仓库的正向也已验证：`304cb0b..5683aa0` 推送后服务器工作区自动更新，`git status` 干净。

### 2.2 冲突时怎么办

推不进去 = 服务器有新提交或工作区脏。**不要**强推。顺序永远是：

1. `python ops/hsai_ops.py git-status` — 看两端 HEAD 和脏文件
2. 服务器有脏文件 → 先决定是提交还是丢弃（读 diff，不要猜）
3. 服务器有新提交 → `python ops/hsai_ops.py pull` 合并到本地
4. 再 `push`

---

## 3. 大文件：`ssh cat` 流式传输

**权重、战团库、回放数据一律不进 Git。** 它们通过隧道直接流式传输，逐文件校验
SHA256。

```powershell
python ops/hsai_ops.py model-pull league-final-20261111-step014858
python ops/hsai_ops.py model-pull some-model --with-trainer    # 连 trainer.pt
```

工具行为：先在服务器算 `sha256sum`，再 `ssh cat` 流式写到本地 `.partial`，落地后
本地重算 SHA256 比对，不一致就报错并删掉。已存在的文件若 SHA 相同则跳过（可重复运行）。

### 3.1 实测吞吐（2026-11-12，经跳板隧道）

| 传输 | 体积 | 耗时 | 速率 |
|---|---:|---:|---:|
| `model.pt` + `checkpoint.json` | 67.9 MB | 6.8 s | **10.0 MB/s** |
| `trainer.pt`（单文件，空写入测速） | 243 MB | 13.7 s | **17.7 MB/s** |
| 服务器侧本地磁盘读 559 MB | 559 MB | 1.06 s | 527 MB/s（非网络） |

**结论**：链路约 140 Mbit/s，1 GB 约 1 分钟。完整模型目录（`model.pt` + `trainer.pt`
+ json ≈ 310 MB）约 20 秒。因此"训练完把模型拉回本地"完全不需要人工介入，也不需要
任何第三方网盘。

### 3.2 为什么不用 scp / rsync 目录同步

- **scp**：Windows 路径在 scp 的 `host:path` 语法里容易出问题；且 `scp -r` 不会校验完整性。
  `ssh cat` + SHA256 更明确。
- **rsync**：本地没装。
- **双向目录同步**：不需要。两端职责不同（§4），不存在"目录要一致"的诉求；
  真需要一致的只有代码，那是 git 的活。

---

## 4. 什么东西放哪

| 资产 | 服务器 | 本地 | 进 Git | 怎么过来 |
|---|:--:|:--:|:--:|---|
| 模拟器 / 训练 / 评测 / 工作流源码 | ✅ 权威 | ✅ 副本 | ✅ | git |
| `client/` 实机桥接 | ✅ 副本 | ✅ 权威（唯一能运行） | ✅ | git |
| `config/`、`docs/`、`tests/` | ✅ | ✅ | ✅ | git |
| 模型 `model.pt` | ✅ 全部 | 只放线上 pin 的那一个 | ❌ | `model-pull` |
| 完整 Trainer `trainer.pt` | ✅ 权威 | ❌ 不需要 | ❌ | 续训只在服务器 |
| 战团库 `artifacts/libraries/` | ✅ 权威 | ❌ | ❌ | 只训练用 |
| 训练产物 `artifacts/runs/` | ✅ 权威 | ❌ | ❌ | 看日志用 `sh` + `tail` |
| 实机证据 `artifacts/runs/live-*`、`records/evidence/` | 可导入 | ✅ 权威 | ❌ | 需要时打包上传 |
| `data/firestone_replays/`（741 MB） | ✅ | ❌ | ❌ | 服务器独有原始数据 |
| `data/real_library/`（15 MB） | ✅ | ❌ | ❌ | 同上 |
| `.venv` / 依赖 | 各机自建 | 各机自建 | ❌ | 永不跨端 |
| `artifacts/state/`（运维台账） | — | ✅ 生成 | ❌ | 本地生成 |

**判据**：需要**读代码**才跑得起来的东西放两边；需要**读数据**才跑得起来的东西只在
服务器；**必须在真实游戏边上**才能产生的只在本地。

---

## 5. GitHub 的最终定位

### 5.1 保留，但从"同步总线"降级为"异地归档"

| 作用 | 两个 Agent 时代 | 现在 |
|---|---|---|
| Agent 之间的代码同步 | GitHub `main` | **git over SSH**（更快、可强制拒绝） |
| Agent 之间的沟通 | `docs/sync/*.md` 散文 | **废止**（只有一个 Agent） |
| 版本历史 | GitHub | **两端仓库各自都有完整历史** |
| 异地容灾 | GitHub | **保留 GitHub** |
| 大文件传输 | 从不用 | `ssh cat`（17.7 MB/s） |

### 5.2 为什么不能整个去掉

本地工作副本在 `D:\HSAI`，服务器副本在 NUS `/home`。**两份都在会坏的硬盘上。**
一次磁盘故障或账号问题就两处同时没了——那不只是代码，是全部决策历史。GitHub 是唯一
一个不在那两台机器上的副本。

代价几乎为零：`git push` 一条命令，凭据已经在 Git Credential Manager 里（实测
`git push --dry-run` 通过）。

### 5.3 为什么不再当日常通道

1. **更慢**：绕第三方一圈，对比 17.7 MB/s 的直连。
2. **多一个故障点**：两端都要能连 github.com。
3. **语义不对**：`push`/`pull` 的隐式合并正是当年两个 Agent 互相覆盖的根源。
   现在 `updateInstead` 给的是**硬拒绝**，不是"尝试合并"。

### 5.4 推送节奏

**里程碑推送，不是每次提交推送。** 具体触发点：

- 一次工作会话结束时（本轮工作已提交、已双向同步、已通过验证）；
- 结构性改动前后（比如这次架构整理）；
- 训练的正式结论产生后（新榜、新模型入库）。

日常改代码不发生 GitHub 推送。这样 `origin/main` 的历史是**有意义的里程碑序列**，
而不是噪音。

### 5.5 GitHub 上放什么、不放什么

放：源码、配置、卡表、文档、清单（单文件 ≤ 5 MB）。
不放：权重、战团库、回放、`artifacts/`、`records/`、日志、压缩包。
这与 `.gitignore` 一致（`artifacts/`、`records/*`、`*.pt`、`data/firestone_replays/`、
`data/real_library/`）。

---

## 6. 可选的进一步升级（未实施）

**让隧道自愈，把"每天输一次密码"降到零。**

做法：把跳板机密码存进 Windows 凭据管理器（DPAPI 加密、绑定当前 Windows 账户、
不落明文、不进对话、不进进程列表），再配一个 `SSH_ASKPASS` 帮助程序，让
`ssh -N -f` 可以无人值守重建隧道。

```powershell
cmdkey /generic:hsai-nus /user:hengzhi /pass     # 交互式输入，不进 shell 历史
```

**没有实施的原因**：它把"必须有人在场"变成"密钥材料常驻本机"。当前每轮工作本来就
需要用户在场（要下目标、要看报告），每天输一次密码的成本很低；而常驻凭据的收益
要等"Agent 长时间无人值守"才体现。等真有夜间无人值守需求时再开。

---

## 7. 快速排障

| 现象 | 原因 | 处理 |
|---|---|---|
| `Connection refused` on 2222 | 隧道断了 | `python ops/hsai_ops.py tunnel`，用户执行一次 |
| `Permission denied (password)` on sjump | 跳板机只收密码，公钥无效 | 正常，输密码即可 |
| `Host key verification failed` | `~/.ssh/config` 缺 `HostKeyAlias` | 恢复 `hsai-nus` 段的 `HostKeyAlias xlogin2.comp.nus.edu.sg` |
| `getsockname failed: Not a socket` | 试图用 ControlMaster | Windows OpenSSH 不支持，改用端口转发 |
| push 被拒：`Working directory has unstaged changes` | 服务器有未提交改动 | `git-status` → 提交或丢弃 → 再 push |
| push 被拒：`non-fast-forward` | 服务器有新提交 | `pull` 合并后再 push |
| 远端命令被守卫拒绝 | 命中 `HEAVY_PATTERNS` | 改走 `sbatch`；确需直跑加 `--allow-heavy` |
| 下载 SHA256 不匹配 | 传输截断或源变了 | 重跑；持续失败则查服务器侧文件 |
