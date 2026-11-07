# 强制升本长训（p50-forced-tier）：NUS 部署与提交

本页所有命令由**用户手工执行**。代理不登录服务器、不提交作业。
本地已完成：机制实现、37 项单测、CPU 端到端短跑、配方与入口脚本；**未在 NUS 运行过**。

与上一轮（`NEXT_P50_DEPLOYMENT.md`）的关系：同一 clone、同一 `.venv`、同一套库，
**无需新建战团库**。差别是配置、入口脚本、Slurm 脚本与环境变量 `HSAI_LONG_OUT`。

## 0. 前置：确认服务器上的东西没变

```bash
cd ~/projects/HSAI || exit
git log --oneline -3          # 应当能看到本轮 commit（含 forced_tier.py）
git status --short            # 应为空；有本地改动先处理，不要带病提交
ls artifacts/libraries/bot-p50-3702-v1/{train-start,train-ramp1,train-ramp2,train,dev,test}
ls ../HSRL/personal/runs/nus-4159-economy-long-002/milestones/decisions-558819917
```

若路径不同，提交前 export（与上一轮同名）：

```bash
export HSAI_SOURCE_7357=.../milestones/decisions-558819917
export HSAI_SOURCE_RUN=.../nus-4159-economy-long-002
export HSAI_P50_BANK=artifacts/libraries/bot-p50-3702-v1
export HSAI_P50_AUDIT=artifacts/runs/p50-forced-tier-preflight   # 新 GPU 验收目录
export HSAI_LONG_OUT=artifacts/runs/p50-forced-tier-long-001     # 新长训目录
```

## 1. 登录节点：只读预检（不上 GPU）

**第一次部署必须先跑 `forced-gpu`**：`--phase forced` 会校验"GPU 验收报告与当前源码一致"，
在验收作业跑完之前它必然报 `GPU验收后训练源码变化`（这是守卫在正常工作，不是配置错误）。

```bash
cd ~/projects/HSAI || exit
export HSAI_P50_AUDIT=artifacts/runs/p50-forced-tier-preflight   # 本次验收目录，全程不变
/usr/bin/python3 -S tools/nus_submit.py --phase forced-gpu --check-only
```

通过了再去第 2 步。等第 2 步的 `report.json` 落地后，`--phase forced --check-only` 才有意义：

```bash
/usr/bin/python3 -S tools/nus_submit.py --phase forced --check-only   # 验收后、正式提交前
```

`--phase forced-gpu` 检查：7357 元数据/SHA/文件长度、父实验三件套、`bot-p50-3702-v1` 六套库
清单与快照存在、`config/p50-forced-tier.json` 的强制计划与**顺延/教师/`reset-anchor` 约束**
（缺 `reset-anchor=1` 会直接拒绝，避免又继承旧 anchor）。它**不要求**已有 GPU 验收报告。

`--phase forced` 额外要求：存在 `profile=forced` 的 GPU 验收报告，且
`training_source_sha256` / `runner_sha256` 与当前代码一致。

## 2. GPU 工程验收（1 小时上限，必做）

```bash
# 用强制升本专用的预检任务（会真的跑一遍强制升本并验收执行统计）
/usr/bin/python3 -S tools/nus_submit.py --phase forced-gpu --submit
sacct -j <GPU作业号> -X --format=JobID,State,Elapsed,ExitCode
/usr/bin/python3 -S - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('artifacts/runs/p50-forced-tier-preflight/report.json').read_text())
print('passed:',r['passed'],'profile:',r.get('profile'),'versions:',r['versions'])
print('source Adam:',r['source_adam_step_min'],'final:',r['final_restore']['adam_step_min'])
print('forced_tier:',json.dumps(r.get('forced_tier'),ensure_ascii=False))
PY
```

`--phase forced-gpu` 与 `--phase gpu` 的区别只有两点：提交的是
`deploy/slurm/p50_forced_preflight.sbatch`（task=`p50-forced-preflight`），且预检用
**新 runner 的 `inputs()`**——上一轮的 `run_7357_preflight.py` 复用了旧 runner，里面有
"配方必须等于上一轮"的硬断言，会直接拒绝本轮配方（已修：加 `--profile forced`）。

预检额外验收（失败会直接 `raise`，不会写 `passed:true`）：

| 检查 | 判据 |
|---|---|
| `forced_tier` 段存在 | `history.jsonl` 最后一条记录里有该段 |
| 强制执行过 | `forced_actions > 0` |
| 5 本样本足够 | `tier5_samples ≥ 0.25 × games`（否则说明多数局在 T8 前就结束） |
| 5 本到达率 | `on_time + deferred ≥ 0.60` |
| 未达比例 | `missed ≤ 0.25` |

通过后会打印 `forced_tier: {...}`，其中的 `tier5_on_time / tier5_deferred / tier5_missed`
就是长训最该盯的三个数。若失败：改 `HSAI_P50_AUDIT` 用新目录重试，**不要绕过**。

## 3. 正式长训（24 小时）

```bash
cd ~/projects/HSAI || exit
export HSAI_LONG_OUT=artifacts/runs/p50-forced-tier-long-001
/usr/bin/python3 -S tools/nus_submit.py --phase forced --submit
```

本轮实际配方（`workflows/run_forced_tier_long.py` + `config/p50-forced-tier.json`）：

| 项 | 值 |
|---|---|
| 起点 | 7357 完整 Trainer（558,873,732 / Adam 545,990），冻结到 `initial-state/` |
| 训练场 | `train-start`(0) → `ramp1`(90M) → `ramp2`(180M) → `train`(260M) |
| 强制升本 | `2:2,5:3,6:4,8:5`，carry 1 |
| 解除强制 | 概率 1.0 → 0.6(90M) → 0.3(180M) → 0.0(260M)；教师标签到 260M，coef 0.05 |
| PPO | lr 2e-6（`lr-min-ratio 1.0` = 不退火）、entropy 0.008、target-kl 0.03、clip 0.2、minibatch 1024、单 epoch |
| anchor | `anchor-beta 0.01` + **`reset-anchor 1`**（修正继承的旧 anchor） |
| 预算 | 24h Slurm、1410 分钟控制、45 分钟收尾、最多 10 亿新决策 |
| 评测 | 起点 96 局；每 30M 一次 96 局配对；终评 512 局；**Dev/Test 不带强制计划** |

**注意**：正式 Dev/Test 评测必须**不带** `--forced-tier-plan`（`personal eval` 默认为空）。
"规则内"探针另建目录并显式加参数，且 `protocol_sha256` 会不同——不要和正式评测混榜。

## 4. 监测（三个数决定要不要人工介入）

```bash
squeue -u "$USER" -o '%.12i %.24j %.12T %.12M %R'
cat artifacts/runs/p50-forced-tier-long-001/train/latest.json
python3 - <<'PY'
import json, pathlib
p=pathlib.Path('artifacts/runs/p50-forced-tier-long-001/train/history.jsonl')
last=None
for line in p.open():
    o=json.loads(line)
    if o.get('type')=='header': continue
    last=o
ft=last.get('forced_tier') or {}
print('iter',last['iter'],'decisions',last['collect']['decisions'])
print('prob',ft.get('prob'),'forced',ft.get('forced_actions'),'teacher',ft.get('teacher_labels'),
      'blocked',ft.get('blocked'))
print('targets',json.dumps(ft.get('targets'),ensure_ascii=False))
PY
```

判读：

1. `targets` 里 5 本的 `on_time` 比例应 ≥0.8；`deferred` 高是 T6 小饰品的预期现象；
   `missed` 持续出现说明计划与引擎规则冲突，必须先分析再继续。
2. `blocked` 每局约 0–2 属正常（待选阶段）；若远高，检查是否到达上限或金币异常。
3. 260M 之后 `prob=0.0`：此时看 `dev` 的 T8–T12 升本执行率是否 >0.05，作为"自主性"判据。

## 5. 安全暂停与恢复

```bash
# 暂停：等当前完整更新提交后停止，保留最后状态（不因名次波动触发）
touch artifacts/runs/p50-forced-tier-long-001/STOP_REQUESTED
```

恢复（先确认写入者已退出）：

```bash
cd ~/projects/HSAI || exit
mv artifacts/runs/p50-forced-tier-long-001/STOP_REQUESTED \
   artifacts/runs/p50-forced-tier-long-001/STOP_REQUESTED.saved-$(date +%Y%m%d-%H%M%S)
export HSAI_RESUME=1
export HSAI_LONG_OUT=artifacts/runs/p50-forced-tier-long-001
/usr/bin/python3 -S tools/nus_submit.py --phase forced --submit
```

恢复会继承 latest 的完整 Trainer 与课程位置（含强制概率所在的累计决策），不会倒回 7357。
若改了源码/配置/库，**必须换新目录**（预检会因输出目录非空而拒绝）。

## 6. 若 GPU 验收不通过

不要跳过。按顺序排查：

1. `--phase forced --check-only` 的报错文本（缺文件/库/配置约束）；
2. `artifacts/runs/p50-forced-tier-preflight/report.json` 的 `passed`/`versions`/`rates`；
3. `[anchor] 已按 --reset-anchor 重建为当前策略` 是否出现在日志；
4. `history.jsonl` 里 `collect.forced_tier` 是否存在、`targets` 是否非空。
