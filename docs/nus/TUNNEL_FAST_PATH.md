# 隧道重建快路（TUNNEL FAST PATH）

状态：**2026-11-13 12:55 重写**（含实测数字）。适用于 NUS VPN 重连/断开后隧道必然断开的情形。

## 0. 先记住三个数字（本次实测）

| 动作 | 实测耗时 | 说明 |
|---|---|---|
| `-Status` | 0.4–0.8 s | 只读本地：凭据、helper、2222 是否有人听 |
| **banner 探测（预检）** | **0.34 s**（VPN 好）/ **5.0 s**（VPN 坏） | 决定"VPN 好了没"，**不碰隧道** |
| VPN 正常时的完整重建 | **≈6 s 到 UP**（实测 poll #2） | 之后 `state` 3–5 s |

**VPN 未生效时的完整代价**：`tunnel.ps1 -Force` 的失败路径 = 初始探测 + `Wait-Tunnel`
8 次 ×（0.7 s sleep + ≤8 s 硬杀探测）≈ **50 s**（用组件实测反推：单次 ssh 探测 5.0 s，
`Get-NetTCPConnection` 0.16–0.52 s，`Get-CimInstance` 0.08 s）。
所以"卡很久"通常是**在等这 50 s**，不是脚本有问题。

## 1. 为什么会"卡几分钟"——三个不同的坑，别再混为一谈

1. **成功之后包装进程不退出**（2026-11-12 实测）：日志已打印
   `Tunnel on port 2222 is up.`，但调用方一直不返回；**接管道、`*> 文件` 都一样挂**。
   → 不要在同一个调用里等它返回；在 Agent 里用 `run_in_background`，
   或者用 `Start-Process` 彻底脱离（见 §3 的 B）。
2. **杀掉包装进程会连隧道一起杀掉**（实测两次）：`ssh -N` 起的转发是包装进程的**子进程**，
   `job_kill`/`Stop-Process` 包装进程 → 隧道 DOWN。**它是隧道的父进程，必须留着。**
3. **VPN 没生效时 TCP 会"假通"**：`Test-NetConnection sjump:22` 返回 True（校园边缘接了 SYN），
   但 SSH banner 永远不来 → 每次探测都要等满超时。这就是"看起来有网、实际不通"。

## 2. VPN 断开时的指纹（2026-11-13 12:52 实测）

```
Test-NetConnection sjump.comp.nus.edu.sg -Port 22   -> True (0.47 s)     ← 假通
ssh -o ConnectTimeout=5 ... hengzhi@sjump... true   -> 5.04 s 退出
  stderr: Connection timed out during banner exchange
          Connection to 137.132.80.27 port 22 timed out
```

**VPN 正常时的指纹**是立刻返回 `Permission denied (...)`（因为故意不给任何认证方式），
**不是超时**。这条差异就是预检的判据。

## 3. 标准动作：一步步、每步都有上界

### A. 预检（0–6 s，不碰隧道）

```powershell
ssh -o ConnectTimeout=5 -o BatchMode=yes -o PubkeyAuthentication=no `
    -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no `
    -o ConnectionAttempts=1 hengzhi@sjump.comp.nus.edu.sg true
```

- 输出 **`Permission denied`** → 路已通 → 进 **B**。
- 输出 **`Connection timed out during banner exchange`**（或 `port 22 timed out`）
  → **VPN 还没生效，立刻停**，回报用户；不要重建、不要重试、不要等。

### B. 起隧道（返回 ~0.2 s，绝不等待）

```powershell
Start-Process -FilePath powershell -WorkingDirectory 'D:\HSAI' -ArgumentList `
  '-NoProfile','-ExecutionPolicy','Bypass','-File','ops\tunnel.ps1','-Force' -WindowStyle Hidden
```

实测这条命令 **0.07 s 返回**（用 `Start-Sleep 3` 的等价调用验证过），
包装进程完全脱离本次调用：既不会被调用方等住，也不会因为工具超时被"顺带"杀掉
（同时避开坑 1 和坑 2）。`-WorkingDirectory` 必须给，否则相对路径 `ops\tunnel.ps1` 可能解析不到。
（Agent 里如果非要用 `run_in_background`，那就**永远不要 job_kill 它**。）

### C. 轮询（每 1.5 s 一次，最多 25 s）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status
```

看到 `tunnel 2222: UP` → **立刻**进 D；**不要**去等 B 里那个进程，也不要去查它的退出码。

### D. 立刻探活（3–5 s，证明不只是端口通了）

```powershell
python ops/hsai_ops.py probe        # 纯连通性
python ops/hsai_ops.py state        # 顺便看双端同步与作业（推荐）
```

### E. 25 s 仍 DOWN：读日志分类，不要盲等

```powershell
Get-Content C:\ProgramData\HSAI\tunnel-2222.log -Tail 5
```

| 日志特征 | 含义 | 动作 |
|---|---|---|
| `banner exchange` / `Connection timed out` | VPN/校园侧未生效 | **停**，请用户确认 VPN |
| `Permission denied` | 跳板机密码变了 | 请用户跑 `ops\tunnel.ps1 -Setup` |
| `bind: Address already in use` | 有陈旧监听 | 再跑一次 `-Force`（脚本会先清） |
| 本地 2222 `Connection refused` | 转发没建起来 | 再跑一次 `-Force` |

## 3.5 端到端实测时间线（2026-11-13 12:55，用户重连 VPN 后第一次跑）

```
A  预检 banner      0.34 s   stderr = "Permission denied (password)"   → 路已通
B  Start-Process    0.07 s   立即返回，包装进程脱离
C  轮询 -Status     6.0 s    poll #2 报 tunnel 2222: UP
D  probe + state    3–5 s    host=xlogin2，drift: IN SYNC
                                    ─────────────
                        合计 ≈ 7 s
```

**对比过去**：同一件事以前要 5 分钟以上，且一半时间在"等一个其实已经成功的调用返回"。
现在**没有一步是盲等**：每步都有硬上界，失败也会在 6 s 内被预检拦住。

## 4. 禁止清单

- ✗ 在同一个调用里等 `tunnel.ps1 -Force` 返回（坑 1）
- ✗ 给 `tunnel.ps1` 接管道 / `*> 文件`（坑 1 的另一半）
- ✗ `job_kill` / `Stop-Process` 任何隧道相关后台进程（坑 2）
- ✗ VPN 没确认前就重建隧道（坑 3），然后在 50 s 的失败路径里等
- ✗ 任何一步超过 ~25–30 s 没有结果还继续等——掐掉看现场

## 5. 拓扑备忘（为什么必须端口转发）
- 跳板机 `sjump.comp.nus.edu.sg` 只接受**密码**认证；`xlogin2` 在校园网内不可直连；
  NUS VPN 不路由 `192.168.5x`。
- 所以本机 `2222` → `xlogin2:22`；`~/.ssh/config` 里 `hsai-nus` = `127.0.0.1:2222`（`BatchMode`）。
- 密码只以 DPAPI 加密形式存于 `%LOCALAPPDATA%\HSAI\tunnel.cred`，由
  `C:\ProgramData\HSAI\hsai-askpass.exe` 读取；**永不进对话/命令行/日志**。

## 6. 顺手记一条：`state` 报 `DRIFT` 时先看是不是"本地领先"

2026-11-13 12:55 实测：隧道刚建好，`state` 报
`drift: DRIFT (heads_equal=False, local_dirty=0, server_dirty=0)`
—— 两端**都干净**，只是**本地比服务器多一个还没 push 的提交**
（`git log --oneline -4` 一看就知道：服务器的 HEAD 是本地 HEAD 的父提交）。
push 之后立刻 `IN SYNC`。

**判据**：`DRIFT` + 两端 `dirty=0` ≈ 本地领先（未推送）；`DRIFT` + 任一端 `dirty>0`
才是真的有未提交改动。**不要**看到 DRIFT 就去做 `reset`/`checkout` 之类的危险动作。
