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

```bash
cd ~/projects/HSAI || exit
/usr/bin/python3 -S tools/nus_submit.py --phase forced --check-only
```

`--phase forced` 检查：7357 元数据/SHA/文件长度、父实验三件套、`bot-p50-3702-v1` 六套库
清单与快照存在、`config/p50-forced-tier.json` 的强制计划与**顺延/教师/`reset-anchor` 约束**
（缺 `reset-anchor=1` 会直接拒绝，避免又继承旧 anchor）。

## 2. GPU 工程验收（1 小时上限，必做）

```bash
# 先跑一次工程验收：真实 7357 → 1 轮 PPO → 保存/恢复 → 换库 → 1 轮
/usr/bin/python3 -S tools/nus_submit.py --phase gpu --submit
sacct -j <GPU作业号> -X --format=JobID,State,Elapsed,ExitCode
/usr/bin/python3 -S - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('artifacts/runs/p50-forced-tier-preflight/report.json').read_text())
print('passed:',r['passed'],'versions:',r['versions'])
print('source Adam:',r['source_adam_step_min'],'final:',r['final_restore']['adam_step_min'])
PY
```

要额外确认（本轮新增）：

```bash
# 强制升本的统计必须出现在 history 的 collect 段
grep -o '"forced_tier":{[^}]*' artifacts/runs/p50-forced-tier-preflight/*/history.jsonl | head
# 被强制的样本必须被策略损失排除：trainer 里 forced 计数 > 0 且 policy_loss 非 NaN
```

验收不过就改 `HSAI_P50_AUDIT` 用新目录重试，**不要绕过**。注意 GPU 验收用的是上一轮的
`run_7357_preflight.py`；如果它不覆盖强制升本路径，就以第 3 步的前 10M 作为实际验收。

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
