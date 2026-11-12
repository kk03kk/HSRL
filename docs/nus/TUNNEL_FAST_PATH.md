# 隧道重建快路（TUNNEL FAST PATH）

状态：**已实测**（2026-11-12 22:20）。适用于 NUS VPN 重连后隧道必然断开的情形。

## 1. 真正的根因（2026-11-12 22:30 更正）：**脚本成功之后不退出**

实测证据：日志里已经打印 `Tunnel on port 2222 is up.`，但调用进程一直不返回；
**不接任何管道、用 `*> 文件` 重定向，同样会挂**。所以问题不在调用方的管道。

更重要的是：`ssh -f -N` 起的转发是**那个包装进程的子进程**，用 `job_kill` / `Stop-Process`
杀掉包装进程会**连隧道一起杀掉**（实测两次都是 DOWN）。之前把这种后台任务当"幽灵"杀掉
是错的——**它就是隧道的父进程，必须留着**。

结论：**不要在同一个调用里等 `-Force` 返回**。正确动作是
「后台起 → 只轮询 `-Status` → 保留那个后台任务」。

```powershell
# ✗ 都会挂住（管道不是原因，但别接）
python ops/hsai_ops.py tunnel 2>&1 | Select-Object -Last 8
powershell -File ops\tunnel.ps1 -Force -Port 2222 *> file

# ✓ Status 是有界的（实测 0.8 s），用它判断成没成
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status
# ✓ 重建：后台起，别等它返回，别杀它
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Force -Port 2222
```

**待修**（下次改代码时做，需要真实环境验证）：让 `ops/tunnel.ps1` 把 ssh 子进程**彻底脱离**
包装进程（独立 job object / 独立 stdio），使脚本能在建好后正常 exit 而不带走转发。

## 1b. 附带教训：管道确实也会挂（独立问题）

`ssh -f -N` fork 的子进程继承 stdout，只要脚本 stdout 是管道，管道就等不到 EOF——
这与上面的"包装进程不退出"是两个不同的坑，都会表现为"卡住"。**两种都别做**。

## 2. 标准动作（有界、不会卡）

```powershell
# ① 先看状态：这条路径不 fork 任何东西，实测 0.8 s 返回
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status

# ② 需要时重建：后台起，然后轮询状态（不要在同一个调用里等它）
#    （Agent 里用 run_in_background，人手动跑就直接跑，别加管道）
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Force -Port 2222

# ③ 约 12 s 后复查
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status   # 期望 tunnel 2222: UP
```

实测时间线（2026-11-12 22:20，VPN 刚重连）：`-Status` 0.8 s 报 DOWN → 后台 `-Force`
→ 12 s 后 `-Status` 报 **UP**，重建本身约 2 s。

## 3. 排查顺序（不要盲目等）

1. `-Status`（0.8 s）：DOWN 才需要重建；它同时会打印凭据与 helper 状态。
2. 无 ssh 进程 + 2222 无监听 = 隧道完全没起，直接重建。
3. 有 ssh 进程但 2222 无监听 = 半死状态，`-Force` 会先清理再建。
4. **任何一步超过 ~30 s 没有结果，就掐掉看现场**，不要等它"自己好"。

## 4. 重建后确认

```powershell
python ops/hsai_ops.py state      # 期望 drift: IN SYNC
```
