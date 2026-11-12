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
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1
```

这个脚本会先探测、已通就什么都不做，否则建立隧道并验证；已存在陈旧转发时用
`-Force` 精确清理（只杀 `-L 2222:xlogin2...:22` 那一条，不碰其它 ssh 会话）。

只想看命令而不执行：`python ops/hsai_ops.py tunnel`。

**由用户在本地 PowerShell 里执行一次**（需要输密码），然后 `python ops/hsai_ops.py doctor`。
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

## 6. 隧道自愈（已实施，2026-11-12）

**目标：把"每天输一次密码"降到"一辈子输一次"。**

### 6.1 用法

```powershell
# 一次性：登记跳板机密码（安全提示输入，不进命令行/进程列表/历史/对话）
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Setup

# 以后任何时候（包括 Agent 自己调用），无人值守重建
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status   # 只报告
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Forget   # 删除凭据
```

`-Setup` 会做完整闭环验证：保存 → 回读解密 → 真实认证一次 → 报告 OK。密码错误当场就
知道，而不是等下一次断线才发现。

### 6.2 组成

| 部件 | 位置 | 说明 |
|---|---|---|
| 凭据密文 | `%LOCALAPPDATA%\HSAI\tunnel.cred` | DPAPI `CurrentUser` 作用域 |
| askpass 助手 | `C:\ProgramData\HSAI\hsai-askpass.exe` | 由 `ops/tunnel-auth.ps1` 的 C# 源码在 Setup 时编译 |
| 逻辑 | `ops/tunnel-auth.ps1` + `ops/tunnel.ps1` | 点源引用，无独立进程 |

### 6.3 实测踩到的两个硬约束

**(a) Win32-OpenSSH 支持 `SSH_ASKPASS`，但必须 `SSH_ASKPASS_REQUIRE=force`。**

不设这个变量时，ssh 只在"没有终端"时才用 askpass，而 Windows 上这个判断不可靠。
实测（`OpenSSH_for_Windows_9.5p2`）设 force 后 askpass 助手确实被调起，参数是提示串：

```
ARGS=hengzhi@sjump.comp.nus.edu.sg's password:
```

**(b) askpass 助手的可执行文件路径必须全是 ASCII。**

放在 `C:\Users\黄\AppData\Local\HSAI\` 下时：

```
CreateProcessW failed error:2
ssh_askpass: posix_spawnp: No such file or directory
```

文件明明存在。换成 `C:\ProgramData\HSAI\` 立刻正常——`CreateProcessW` 过不了非 ASCII
路径。所以 **exe 放 ProgramData，密文留在用户配置目录**（.NET 读 Unicode 路径没问题，
只有 `CreateProcessW` 不行）。`Test-HsaiSecret` 会检查 `HelperAscii`。

> 这同时解释了为什么 `%LOCALAPPDATA%` 下的密文没问题：读文件的是 .NET，不是 ssh。

**(c) 附带发现：含中文的 `.ps1` 必须带 UTF-8 BOM。**

Windows PowerShell 5.1 对无 BOM 脚本按系统 ANSI 代码页读取。中文被错误解码后会**打乱
行结构**——`ops/tunnel-auth.ps1` 里那段 C# here-string 曾被整段当成 PowerShell 代码，
报"缺少 using 指令"。`tests/test_ops_guard.py` 里有守卫钉死这个格式。

### 6.4 验证证据（用假密码走通全链路）

| 步骤 | 结果 |
|---|---|
| 助手编译 | `hsai-askpass.exe` 4,608 B，路径全 ASCII |
| 无凭据时 | 助手返回 rc=2 `credential not found`，不吐任何东西 |
| DPAPI 往返 | 存 32 字节 → 助手解出**完全一致**的字符串 |
| 密文不含明文 | 直接读 `.cred` 找不到密码字面量 |
| ssh 是否真用助手 | 是：`Permission denied (password)`（假密码被拒，说明链路通） |
| 清理 | `-Forget` 后文件消失，助手回到 `credential not found` |

### 6.5 安全边界与代价

- 密文只有**当前 Windows 账户**能解开（DPAPI CurrentUser）。同机其他用户读到文件也
  无效。文件 ACL 额外收紧到仅当前用户。
- 明文只在助手进程内存里出现，写完立即 `Array.Clear`。
- 明文**永不**进：对话、命令行参数、进程列表、shell 历史、日志、Git。
- **代价**：跳板机密码从此常驻本机（加密态）。风险模型从"每次手输"变成"这台机器 +
  这个 Windows 账户被攻破即失守"。若不可接受，`-Forget` 随时退回手动模式。
- **密码轮换**：NUS 改密后自愈会失败并给出 `Permission denied`，重跑 `-Setup` 即可。

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
