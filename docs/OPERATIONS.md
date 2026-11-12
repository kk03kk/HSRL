# HSAI 工作规范：单 Agent 双环境日常运作

状态：**当前生效**（2026-11-12）。取代 `docs/SYNC_MODEL.md` 与 `docs/sync/` 的散文协作协议。
架构见 [`ARCHITECTURE.md`](ARCHITECTURE.md)，传输层见 [`architecture/TRANSPORT.md`](architecture/TRANSPORT.md)。

---

## 0. 用户只需要做两件事

1. **提目标**（"把模型拉回本地""跑一轮长训""看看为什么棋力没提升"）。
2. **隧道断了时跑一个脚本**（要输一次密码）：

```powershell
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1
```

其余全部由 Agent 完成，包括判断先做什么。

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
| `tunnel: DOWN` | 隧道断 | 让用户执行 `hsai_ops.py tunnel`，等重连再继续 |

`python ops/hsai_ops.py doctor` 是更严格的七点自检（ssh 配置、隧道、两端仓库、
`updateInstead`、作业、远端配置），结构性改动前后各跑一次。

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

| 分区 | 时限 | GPU | 空闲/总 |
|---|---|---|---|
| `gpu` | 3:00:00 | h100-47×4, h200-141×4, a100-80, a100-40×2, nv | 64 / 74 |
| `gpu-long` | 3-00:00:00 | 同上 | 37 / 39 |
| `long` | 3-00:00:00 | — | 49 / 130 |
| `normal` | 3:00:00 | — | 54 / 167 |

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
# 1. 建隧道（输一次密码）
ssh -N -f -L 2222:xlogin2.comp.nus.edu.sg:22 -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=10 -o ExitOnForwardFailure=yes hengzhi@sjump.comp.nus.edu.sg
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

# 隧道断了
python ops/hsai_ops.py tunnel                         # 打印命令
powershell -ExecutionPolicy Bypass -File ops\tunnel.ps1   # 或直接跑脚本
```
