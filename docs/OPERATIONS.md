# HSAI 工作规范：单 Agent 双环境日常运作

状态：**当前生效**（2026-11-12）。取代 `docs/SYNC_MODEL.md` 与 `docs/sync/` 的散文协作协议。
架构见 [`ARCHITECTURE.md`](ARCHITECTURE.md)，传输层见 [`architecture/TRANSPORT.md`](architecture/TRANSPORT.md)。

---

## 0. 用户只需要做一件事（一次性设置）

**登记一次跳板机密码**，之后隧道可以自愈，你连这一步都不用再做了：

```powershell
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Setup
```

密码用 Windows DPAPI 加密存在本机（只有你的 Windows 账户能解开），**不进对话、不进
命令行、不进进程列表、不进日志**。Setup 会当场做一次真实认证验证，密码错了立刻知道。

之后你只需要**提目标**（"把模型拉回本地""跑一轮长训""看看为什么棋力没提升"），
其余全部由 Agent 完成，包括判断先做什么。

> 还没做 Setup 时，隧道断了仍然可以手动建：`ops\tunnel.ps1` 会在当前终端提示输密码。
> 想撤销：`ops\tunnel.ps1 -Forget`。

---

## 1. 每次开工的固定开场

不允许跳过。**这是唯一能防止"漂移飘一天"的机制。**

```powershell
python ops/hsai_ops.py state
```

输出一屏：

```
HSAI dual-end state  2026-11-12T13:33:21+08:00
  tunnel : OK  xlogin2
  local  : 5683aa0 main dirty=0
  server : 5683aa0 main dirty=0 size=37G disk=... 92%
           job 1117116 no-shell RUNNING 14:28:55 xcnc23
           models=10 libs=11
  drift  : IN SYNC (heads_equal=True local_dirty=0 server_dirty=0)
```

判据：

| 看到的 | 含义 | 动作 |
|---|---|---|
| `drift: IN SYNC` | 两端同 HEAD、都干净 | 直接干活 |
| `dirty > 0` 任一端 | 有未提交改动 | **先读 diff 决定提交还是丢弃，不写新代码** |
| `heads_equal=False` | 两端分叉 | `pull` 合并，再决定 |
| 有 RUNNING 作业 | 有活在跑 | 先确认它是否该被我动 |
| `tunnel: DOWN` | 隧道断 | **Agent 自己跑** `python ops/hsai_ops.py tunnel`（流程见 §1.3），不要等用户 |

`python ops/hsai_ops.py doctor` 是更严格的七点自检（ssh 配置、隧道、两端仓库、
`updateInstead`、作业、远端配置），结构性改动前后各跑一次。

---

### 1.3 隧道断了怎么办（唯一正确流程）

**本节是权威流程。** 实测数字、日志指纹、历史事故只记在
[`nus/TUNNEL_FAST_PATH.md`](nus/TUNNEL_FAST_PATH.md)，那里**不再**是独立流程。
每一步都有硬上界：任何一步超过 ~30 s 没结果就掐掉看现场，禁止盲等。

断了会表现成：`state` 里 `tunnel: DOWN`；`sh/push/pull/jobs/models` 报
`Connection refused`（2222 没人监听）或 `远端命令超时`。

#### A. 预检 VPN（0–6 s，不碰隧道）

```powershell
ssh -o ConnectTimeout=5 -o BatchMode=yes -o PubkeyAuthentication=no `
    -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no `
    -o ConnectionAttempts=1 hengzhi@sjump.comp.nus.edu.sg true
```

| 输出 | 含义 | 动作 |
|---|---|---|
| `Permission denied` | 路已通（故意不给任何认证方式） | 进 B |
| `Connection timed out during banner exchange` / `port 22 timed out` | **VPN 还没生效**（此时 `Test-NetConnection` 仍返回 True，是假通） | **停**，只回报"需要用户连 VPN"，不重建、不重试 |

#### B. 重建（两个等价入口，都能在一次工具调用里安全跑完）

**B1 默认（推荐）**

```powershell
python ops/hsai_ops.py tunnel
```

凭据已登记时完全无人值守，实测 **6–10 s 返回**（2026-11-13 23:08，VPN 刚重连）：

```
隧道已断，尝试自愈……（脚本输出写 D:\HSAI\artifacts\state\tunnel-selfheal.log）
  自愈脚本 rc=0 用时 6.3s
  --- 自愈日志尾部 ---
  | Port 2222 does not answer; clearing any stale forward first.
  | Creating tunnel on port 2222 using the stored credential (unattended)...
  | Tunnel on port 2222 is up.
  |   ssh pid 8788   log C:\ProgramData\HSAI\tunnel-2222.log
自愈成功，隧道已通（xlogin2）。
```

它之所以可以"直接等"，是因为 2026-11-13 起它把脚本输出**写日志文件**而不是接管道
（原因见下面的禁令 1），并且对子进程设了有界超时。

**B2 脱离式（脚本本身可疑、或要强制重建）**

```powershell
Start-Process -FilePath powershell -WorkingDirectory 'D:\HSAI' -ArgumentList `
  '-NoProfile','-ExecutionPolicy','Bypass','-File','ops\tunnel.ps1','-Force' -WindowStyle Hidden
```

**0.07 s 返回**，包装进程完全脱离本次调用（同时避开"等包装进程"和"工具超时连带杀掉"）。
随后每 1.5 s 轮询一次、最多 25 s：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status   # 只读，安全
```

看到 `tunnel 2222: UP` 立刻进 C。**不要**去等 B2 那个进程、不要查它的退出码，
更**不要 kill 它**——`ssh -N` 是它的子进程，杀包装进程隧道会一起消失。

#### C. 独立探活（3–5 s，证明不只是端口通了）

```powershell
python ops/hsai_ops.py probe        # 期望 host=xlogin2
python ops/hsai_ops.py state        # 首行 tunnel : OK  xlogin2；顺带看同步与作业（推荐）
```

#### D. 25–30 s 仍 DOWN：读日志分类，不要盲等

```powershell
Get-Content C:\ProgramData\HSAI\tunnel-2222.log -Tail 5
```

| 日志特征 | 原因 | 处理 |
|---|---|---|
| `banner exchange` / `Connection timed out` | VPN/校园侧未生效 | 停，请用户确认 VPN |
| `Permission denied (password)` | 跳板机密码轮换或输错 | 用户跑一次 `ops\tunnel.ps1 -Setup` |
| `No stored credential` / rc=3 | 从没登记过密码 | 同上 |
| `bind: Address already in use` | 有陈旧监听 | 再跑一次 `-Force`（脚本会先精确清理） |
| 端口 UP 但 `probe` 失败 | ssh key / `HostKeyAlias` 问题 | `python ops/hsai_ops.py doctor` |

#### 禁令（每条都是实测事故换来的）

1. **不要给建隧道脚本接管道**：`stdout=PIPE` / `| Tee-Object` / `*> 文件` / `$(...)` 都算。
   隧道是**常驻** ssh，它继承调用者的 stdout 写端且永不关闭，调用者永远等不到 EOF。
   2026-11-13 两次实测：① `python ops/hsai_ops.py tunnel` 内部用管道 → 卡 4 分钟被用户打断；
   ② `powershell -File ops\tunnel.ps1` 被工具管道接住 → 不返回，**直到人手动杀掉那条 ssh**
   才返回（53.8 s，正好是杀它的时刻）。`hsai_ops.py tunnel` 因此改成写
   `artifacts/state/tunnel-selfheal.log` 再读文件尾；`tunnel.ps1` 把 ssh 的
   stdin/stdout/stderr 三路全部重定向到文件。`tests/test_ops_guard.py` 有两条测试钉死。
2. **不要手工执行 `ssh -N -f -L 2222:...`**：`-f` 只是把 ssh 丢到后台，输出仍挂在你那条
   管子上，同样挂死。它只作为**原理**记在 [`architecture/TRANSPORT.md`](architecture/TRANSPORT.md)。
3. **不要 kill 隧道相关进程**，也不要对后台起的隧道作业 `job_kill`。

封装脚本的其余开关：

```powershell
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1            # 无人值守（凭据已登记）
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Force     # 强制重建（只杀 2222 那条转发）
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status    # 只报告，不动手
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Verify    # 临时端口演练，不碰 2222
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Setup     # 登记/更新密码（唯一需要人的一步）
```

#### 改 `ops\*.ps1` 的人注意（编码，会静默毁掉脚本）

`ops/` 与 `client/` 下含中文的 `.ps1` **必须带 UTF-8 BOM**：Windows PowerShell 5.1 读无 BOM
脚本时用系统 ANSI 代码页，中文被错解并**吞掉换行**，整段结构失效。2026-11-13 真实发生过：
用编辑器改写 `ops\tunnel.ps1` 后 BOM 丢了，注释行 `# …refused。` 把下一行的
`if (-not $Force) {` 吞进注释，PowerShell 5.1 报"意外的标记 }"，
于是自愈脚本变成必定失败而没人察觉。
`tests/test_ops_guard.py::test_ps1_files_with_non_ascii_must_be_utf8_with_bom` 钉住了这条；
丢了就补回来（不改内容，只补三个字节，本机实测可用）：

```powershell
$p='ops\tunnel.ps1'; $b=[IO.File]::ReadAllBytes($p)
if (-not ($b[0] -eq 0xEF -and $b[1] -eq 0xBB -and $b[2] -eq 0xBF)) {
  $out = New-Object byte[] ($b.Length + 3)
  [Array]::Copy([byte[]](0xEF,0xBB,0xBF),0,$out,0,3)
  [Array]::Copy($b,0,$out,3,$b.Length)
  [IO.File]::WriteAllBytes($p,$out)
}
```

---

## 2. 改代码

**在哪端改都一样，规则只有一条：改完必须提交。**

```powershell
# 1. 改（本地或服务器，取决于哪端能验证）
# 2. 本地验证
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.venv\Scripts\python.exe tools\verify_workspace.py --skip-rust
# 3. 提交
# 4. 双向同步
python ops/hsai_ops.py push        # 本地 -> 服务器
python ops/hsai_ops.py pull        # 服务器 -> 本地
```

### 2.1 什么时候在服务器上改

只有一种情况：**需要 GPU 才能验证的改动**（训练栈、模拟器行为、评测器）。
其余一律在本地改，因为本地有测试环境、有编辑器、有 Agent。

在服务器上改的流程：

```powershell
python ops/hsai_ops.py sh -f some_script.py     # 或用编辑器经 sftp
```

`sh -f` 本身是**字节透明**的（`remote_run` 用 `input=script.encode("utf-8")` 走 ssh stdin），
但**本地这一段不透明**：Windows PowerShell 5.1 的 `Get-Content` / `Set-Content` 默认按
ANSI 代码页（本机 `ACP=936`）解码，BOM-less UTF-8 中文会先被当成 GBK 解出乱码再存回去，
多字节字符被吃成 `?` 时还可能**吃掉行尾引号** —— 远端于是报
`syntax error near unexpected token '('`，而报错行号指向无辜的一行（2026-11-14 凌晨踩过，
见 `docs/DECISIONS.md` D-050）。所以：

- **文件内容只走 `write` / `edit` 工具**（全程 UTF-8 无损）；非要用 PowerShell 读写就显式
  写 `-Encoding UTF8`（或用 `[IO.File]::ReadAllText($p, [Text.Encoding]::UTF8)`）。
- 交给 `hsai_ops.py sh -f` 的脚本**一律纯 ASCII**（注释、`echo` 都写英文）：编码环节哪天再出
  问题也只是乱码，不会把引号吃掉、不会把脚本变成语法错误。
- 新写的远端脚本先 `bash -n` 过一遍再执行（`sh -f` 里 `cat > /tmp/x.sh <<'EOF' … EOF` 再
  `bash -n /tmp/x.sh`）。

改完**当轮提交**，然后 `pull` 回本地。**不允许在服务器上留未提交改动过夜**——
这是 11-12 早上 12 个文件飘了 24 小时的直接教训。

### 2.2 提交粒度

一次提交一件事，信息写清"为什么"。参考已有风格：`feat(live):` / `fix(4159):` /
`docs:` / `sync(server):`。

### 2.3 禁止

`git reset --hard`、`git clean`、force-push、跨端直接 scp 覆盖源码、
在服务器上 rebase 别人（历史）的提交。

---

## 3. 训练作业

### 3.1 提交

```bash
sbatch deploy/slurm/<job>.sbatch
```

- **作业内不要包 `srun`**（本站会把双卡缩成单卡）。
- **新作业用新输出目录**；已存在且非空的目录一律拒绝覆盖（脚本里 `[[ -e "$out" ]] && exit 2`）。
- `squeue` 拿 jobid → **立刻记进台账**（见 §6）。
- 不改 `CUDA_VISIBLE_DEVICES`，GPU 身份以原始 Driver UUID / Torch 实际计算为准。

### 3.2 分区与时限（2026-11-12 实测）

| 分区 | 时限 | GPU | 节点数 | 当时空闲 |
|---|---|---|---:|---|
| `gpu` | 3:00:00 | h100-47×4, h200-141×4, a100-80, a100-40×2, nv | 74 | 7 全闲 + 18 部分 |
| `gpu-long` | 3-00:00:00 | 同上 | 39 | 2 全闲 + 11 部分 |
| `long` | 3-00:00:00 | — | 130 | — |
| `normal` | 3:00:00 | — | 167 | — |

机器负载随时变。**以 `sinfo -p gpu,gpu-long` 的实时输出为准**，不要引用这里的历史数字。

**排队教训（2026-11-12 实测）**：`deploy/slurm/league_stage2_48h.sbatch` 里写死了
`--gres=gpu:h100-47:2`，结果排队预计 **2 小时 15 分**；改成命令行覆盖
`--gres=gpu:2`（不限型号）后**立刻开跑**。`nv:2`、`a100-40:2`、`a100-80:1`、
`h100-96:2`、`h200-141:4` 都在池子里，而训练代码只用 cuda:0 / cuda:1，**不关心型号**。
除非明确要复现某个型号的吞吐数字，否则不要钉死 GPU 型号。

### 3.3 监控

```powershell
python ops/hsai_ops.py jobs                      # 队列 + 最近 7 天记账
python ops/hsai_ops.py sh "tail -30 artifacts/runs/<run>/history.jsonl | jq -c '{...}'"
```

**轮询，不 hold 会话。** 不要写 `while` 循环盯着。

### 3.4 失败

**用新目录重跑，保留旧现场。** 不覆盖、不删除。工程错误（非法动作、截断、引擎异常、
注入失败、非有限损失、状态损坏）严格停止并保现场；正常棋力波动只记录，不自动止损。

一轮训练跑完后，先过健康门禁再决定下一步：

```bash
python workflows/league_health.py --run artifacts/runs/<run>
```

### 3.5 多段训练期间**冻结训练代码**（2026-11-12 实测教训）

`personal.py` 在 `--resume` 时会校验训练代码 SHA 是否与本轮开始时一致，不一致直接拒绝：

```
ValueError: 训练代码已变更；请另开实验，不能伪装为精确续训
```

这是**正确的 fail-closed 行为**（防止把"改了代码的续训"伪装成精确复现），但它意味着：

> **一段多段实验在跑的时候，不要提交任何改动到 `hsbg/`、`workflows/`。**

实测代价：实验 `1122109` 跑完第 1 段（健康门禁通过、版本 14871）后，因为我在它运行期间
把新的配置键 commit/push 进了 `personal.py`，第 2 段续训被拒，整个 4 段实验在第 1 段
（16 次迭代）就结束了。

**正确做法**（按优先级）：

1. 多段实验开始前把要做的代码改动全部做完并同步；
2. 实验期间只改 `docs/`、`config/`、`tools/`（不参与训练代码 SHA），或干脆不改；
3. 需要中途改训练代码 → 让当前实验跑完，或显式放弃它另开新实验（新目录）。

---

## 4. 把模型拿到本地

线上模型只有一个需要本地存在：

```powershell
python ops/hsai_ops.py model-pull league-final-20261111-step014858
```

逐文件 SHA256 校验，已是最新则跳过。约 68 MB / 7 秒。

**完整 Trainer（`trainer.pt`，243 MB）默认不拉** —— 续训只在服务器发生，本地不需要。

拉完要同时更新 `config/live/model.json` 的 pin 与 `client/hsrl_agent/live/policy.py`
里的 SHA 注册（两者必须一致，`tests/test_live_*` 会检查）。

---

## 5. 实机

实机只能在本地跑。

```powershell
python client/live_245.py doctor                 # 环境自检
python client/live_245.py autoplay --out artifacts/runs/live-<purpose>-<date>-<seq>
python client/tools/live_status.py               # 一行状态
```

- **新输出目录**，不复用。
- 每局终局必须有 `result.json`（含 `terminal: true` 与 `terminal_proof`）才算有效对局。
- 长时间无人值守用 `client/tools/live_overnight.py`（进程外监管 + 停滞检测）。
- 跑完生成报告：`client/tools/live_night_report.py`。

实机证据不进 Git。需要给服务器做规则验证时打包（`pack_evidence.py`，tar.gz + sha256），
用 `ssh cat` 上传到 `artifacts/imports/`。

---

## 6. 状态与文档

| 内容 | 放哪 | 谁写 |
|---|---|---|
| 实时双端状态 | `artifacts/state/latest.json`（生成） | `hsai_ops.py state` |
| 远端命令审计 | `artifacts/state/ssh-audit.jsonl`（生成） | `hsai_ops.py` 自动 |
| 当前作业与结论 | `docs/PROJECT_STATE.md` | Agent，阶段性更新 |
| 决策留痕 | `docs/DECISIONS.md`（只追加） | Agent，一条一个编号 |
| 一次性分析 | `docs/analysis/<TOPIC>_<YYYYMMDD>.md` | Agent |
| 历史文档 | `records/archive/<topic>-<YYYYMMDD>/` | 不删除，只移动 |

**规则**：`docs/` 只放当前有效文档；被取代的移进归档并保留。运行日志、旧排行榜、
已完成复盘不进 `docs/`。

---

## 7. Agent 不在时的手动应急

如果隧道长期不可用、又有紧急动作必须做，用户可以直接 SSH 到服务器：

```powershell
# 1. 建隧道（输一次密码；不要自己拼 ssh -N -f -L，理由见 §1.3）
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1
# 2. 进服务器
ssh -p 2222 -o HostKeyAlias=xlogin2.comp.nus.edu.sg hengzhi@127.0.0.1
```

进来之后的**最小命令集**：

```bash
cd ~/projects/HSAI
squeue -u $USER                      # 看作业
tail -50 artifacts/runs/<run>/history.jsonl
scancel <jobid>                      # 取消作业（不会删产物）
git log --oneline -5 ; git status    # 看代码状态
```

**不要**手改代码后不提交；**不要** `git reset --hard`；**不要**删 `artifacts/`。
任何"修复"都应该留给下一次 Agent 回合，除非是"继续跑/停止"这类开关。

---

## 8. 自主权边界

Agent 自主决定做什么、什么时候做。只有下面这些必须先问：

| 必须先问 | 理由 |
|---|---|
| 删除资产 / 数据 / 证据 | 不可逆 |
| force-push / 重写历史 | 不可逆，且会毁掉异地备份的意义 |
| 正式长训（> 6 小时） | 占卡、改模型走向 |
| 大规模评测（> 1000 局） | 占卡 |
| 改游戏目录 | 会破坏用户资产 |
| 改 `config/live/` 的模型 pin | 直接影响线上实机行为 |

可以自主做：读、分析、审计、改代码、写测试、跑测试、建目录、短程 smoke（≤ 2 h）、
提交/取消**自己**的作业、拉权重、写文档、整理归档。

---

## 9. 红线（永不变）

- 不 `reset --hard`、不 `git clean`、不 force-push、不全仓格式化。
- 不删除历史证据、模型、日志、库、原始数据；不删旧项目；不覆盖非空输出目录。
- 不修改游戏目录。
- 不伪造 `plan.json`、checkpoint、SHA、卡表或恢复状态。
- 不绕过源码锁、卡表、注入、数据质量与引擎异常门禁。
- 不在日志或文档写密钥、token、私密路径。
- 登录节点不跑重活。

---

## 10. 一小时速查

```powershell
# 开工
python ops/hsai_ops.py state
python ops/hsai_ops.py doctor

# 同步
python ops/hsai_ops.py push
python ops/hsai_ops.py pull

# 服务器
python ops/hsai_ops.py sh "squeue -u hengzhi"
python ops/hsai_ops.py jobs
python ops/hsai_ops.py models
python ops/hsai_ops.py model-pull <name>

# 本地验证
.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.venv\Scripts\python.exe tools\verify_workspace.py --skip-rust

# 隧道断了（凭据已登记时 Agent 自己就能做，不需要问用户；详见 §1.3）
python ops/hsai_ops.py tunnel                               # 唯一正确入口，6–10s，禁止接管道
python ops/hsai_ops.py state                                # 独立确认：tunnel : OK  xlogin2
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Status
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1 -Verify   # 临时端口演练
```
