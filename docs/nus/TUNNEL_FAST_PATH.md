# 隧道重建快路（TUNNEL FAST PATH）

状态：**已实测**（2026-11-12 22:20）。适用于 NUS VPN 重连后隧道必然断开的情形。

## 1. 结论先讲：**绝对不要给 `ops\tunnel.ps1` 接管道**

```powershell
# ✗ 会挂几分钟，然后"1 秒就成功"——看起来像死锁
python ops/hsai_ops.py tunnel 2>&1 | Select-Object -Last 8
powershell -File ops\tunnel.ps1 -Force *>&1 | Out-String

# ✓ 直接跑，不加任何管道
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Force -Port 2222
```

**机理**：脚本内部用 `ssh -f -N -L` 起转发，ssh fork 出的子进程**继承父进程 stdout**。
一旦脚本的 stdout 是管道，管道写端就被那个长期存活的 ssh 子进程一直持有，管道**永远
等不到 EOF**，`Select-Object` / `Out-String` / `Tee-Object` 就一直挂着——而隧道其实
在第一秒内就建好了。这与历史记录里"`ssh -f` 配管道在 Windows 上死锁（实测挂满 120 s）"
是同一个坑，只是这次是**调用方**踩上去的。

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
