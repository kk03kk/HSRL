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

预检额外验收（**先把全部检查算完、写 `report.json` 再抛错**，失败也能看到四类指标）：

| 检查项 | 判据 |
|---|---|
| `injection` | `mirror_attempt==mirror_applied==eligible_live_seats`，非法/截断/缺失/回退全 0 |
| `efficiency` | `optimizer_steps>0` 且 `processed_samples/decisions ≥ 0.5`（早停丢批的容忍线）；同时记录 `kl_early_stop_iterations` |
| `forced_tier` | 段存在、`forced_actions>0`、`tier5_samples ≥ 0.25×games`、`on_time+deferred ≥ 0.60`、`missed ≤ 0.25` |

通过后会打印 `forced_tier: {...}`，其中的 `tier5_on_time / tier5_deferred / tier5_missed`
就是长训最该盯的三个数。失败时 `report.json` 里有 `passed:false`、`failed:[...]`、`checks:[...]`；
改 `HSAI_P50_AUDIT` 用**新目录**重试，**不要绕过**、不要把阈值调低。

> 已知一次真实失败（2026-11-07，作业 1090732）：配置里的强制计划是"本轮新增决策"语义，
> 而训练器用累计决策比较，阈值全在过去 ⇒ `prob=0`、一次没强制。已由
> `run_forced_tier_long.shift_forced_schedule()` 平移为绝对时间轴修复，详见
> `docs/research/FORCED_TIER_LOCAL.md` §6。若再次出现 `forced_actions=0`，先查
> `train.config.json` 里的 `forced-tier-schedule` 是不是绝对累计值。

## 2b. 性能事故与防复发（2026-11-07，作业 1091647）

**实测事实（来自 1091647 的 history.jsonl + .out）**：

| 阶段 | 实测 | 基线（1085131 长训 / 1085078 验收） |
|---|---|---|
| 单轮训练循环（1 iteration，512 局） | **73 s / 53 s**（含 8 卡池 collect 23 s + PPO 3.9–4.5 s + 落盘 0.7 s） | collect 中位 11.5 s、PPO 9 s；1085078 三轮共 294 s |
| 采样速率 | 4,129 / 3,555 dec·s⁻¹（per-worker 600 dec/s） | 8,594 dec·s⁻¹（per-worker 1,127 dec/s） |
| 每轮**进程启动到训练开始** | **~850–980 s**（两轮 iter 时间戳相差 15m07s，而循环本身只有 53 s） | ~40–60 s |

**结论**：33m48s 不是"训练变慢"，而是**验收把训练器重启了 3 次，每次付一份 ~15 分钟的冷启动**；
12 小时长训只在开头付一次，稳态仍是每轮几十秒。验收结构本身（3 × 进程冷启动）才是墙钟大头。

**同时暴露的真实 bug（更严重）**：`--lockstep 1` 生产路径（`personal train` 唯一使用的路径）
的 `collect_ppo_lockstep` **从来没有汇总强制升本统计**——`forced_tier_game` 只写进了
`_one_game` / `collect_ppo_records_parallel`，lockstep 的 meta 里既没有逐局统计也没有汇总，
于是 history 里 `prob=0.0 / forced_actions=0 / targets={}` 全是缺省值。后果：

1. 验收的 `check_forced_tier` 永远拿不到证据（这次作业本来就过不了）；
2. 12 小时长训会在"机制不可观测"的状态下跑完，事后无法判断升本是否执行。

已修：`collect_ppo_lockstep` 逐局写 `forced_tier_game`，并按 `aggregate_forced_tier(metas)`
（注意该函数吃**逐局 meta**，传内层 dict 会得到空汇总）写 `meta["forced_tier"]`；
`tests/test_parallel_collect.py::test_lockstep_reports_forced_tier_meta` 锁定
`forced_actions>0`、`targets["2"]` 存在、记录里有 `forced=True` 样本。

**已核对（可复现证据）**：

| 项 | 结论 | 证据 |
|---|---|---|
| 配方漂移 | 相对 `config/p50-7357-train.json` **只有 10 个已批准键不同** | `tests/test_config_parity.py` |
| 战团库 | 与 1085131 长训**同一个库**（train 04359ca4…、dev d493ef78…） | closeout `plan.json` |
| Slurm 资源 | 32 CPU / 96 GiB / `h100-47:2` / 同一 `--exclude` | 两个 sbatch 头 |
| 强制升本机制开销 | **1.89 µs/决策**（观测快路径 0.53 µs；`player_json` 慢路径 213 µs 未触发） | `artifacts/local/bench_forced_tier_overhead.py` |
| 采集/更新热路径 | 新代码全部在 `_forced is None` 之后短路 | `git show 9786df4` |

**另修**：`train_entity_sync` 曾把 `forced-teacher-coef` 合并进 `teacher_bc_coef`
（偷偷打开"在线缩放教师"的 CE 通路、两条 DAgger 标签共用同一字段），现为两个独立系数，
同时 >0 时拒绝启动。

**新增两道门禁**：

- **配置漂移**：`--phase forced/forced-gpu` 与 `run_forced_tier_long.config()` 双侧逐键比对
  快速基线，只允许 `FORCED_ALLOWED_CONFIG_DIFF`；改 `games/processes/pipeline/minibatch/…` 直接拒绝。
- **吞吐**：每轮打印并写 `report.json.timings`（墙钟 / collect / dec·s⁻¹ / ppo / samples），
  阈值 `MAX_ROUND_SECONDS=300`、`MIN_DECISIONS_PER_S=3000`；每轮 `--run-minutes 8` 封顶单轮最坏耗时。



## 3. 正式长训（12 小时）

```bash
cd ~/projects/HSAI || exit
export HSAI_LONG_OUT=artifacts/runs/p50-forced-tier-long-001
/usr/bin/python3 -S tools/nus_submit.py --phase forced --submit
```

本轮实际配方（`workflows/run_forced_tier_long.py` + `config/p50-forced-tier.json`）：

| 项 | 值 |
|---|---|
| 起点 | 7357 完整 Trainer（558,873,732 / Adam 545,990），冻结到 `initial-state/` |
| 训练场 | `train-start`(0) → `ramp1`(40M) → `ramp2`(80M) → `train`(120M) |
| 强制升本 | `2:2,5:3,6:4,8:5`，carry 1 |
| 解除强制 | 概率 1.0 → 0.6(40M) → 0.3(80M) → 0.0(120M)；教师标签到 120M，coef 0.05 |
| PPO | base lr 2e-6，**实际 1.7548e-6 → 1.6667e-6（iteration 10000 触底）后恒定**；entropy 0.008、target-kl 0.05、clip 0.2、minibatch 1024、单 epoch |
| anchor | `anchor-beta 0.01` + **`reset-anchor 1`**（修正继承的旧 anchor） |
| 预算 | **12h Slurm、690 分钟控制、45 分钟终评预留、4 亿决策上限**（时间绑定） |
| 评测 | 起点 96 局；每 30M 一次 96 局配对；终评 192 局 ×2（参考/候选）；**Dev/Test 不带强制计划** |

12h 的时间账：690 min 控制预算 − 45 min 终评 = 645 min 训练窗口；扣除每 30M 一次的
96 局评测开销，按上一轮 7,095 dec/s 估算约 **230–270M** 新决策 ⇒ 120M 之后仍有
**45–55% 的完全自主训练**。若实测吞吐明显低于 6,000 dec/s，milestone 会自然减少，
但 120M 节点与自主段比例不变（由调度节点而非时间决定）。

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
3. **120M 之后** `prob=0.0`：此时看 `dev` 的 T8–T12 升本执行率是否 >0.05，作为"自主性"判据；
   同时看逐回合自身/场地 P50 比值是否在后续里程碑继续上升（自主段仍在成长才算成功）。

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
具体地：`cum_decisions` 存在 trainer state 的 `extra` 里随快照一起保存/恢复，
`train.config.json` 里是**绝对**的 `forced-tier-schedule`（`558873732:1,598873732:0.6,…`），
两者相乘决定恢复后第一步的强制概率；`--phase forced` 的预检会重新核对源码/入口/库摘要，
所以**任何代码或配置改动都会要求重跑 GPU 验收**，这是设计而非故障。
若改了源码/配置/库，**必须换新目录**（预检会因输出目录非空而拒绝）。

## 6. 若 GPU 验收不通过

不要跳过。按顺序排查：

1. `--phase forced --check-only` 的报错文本（缺文件/库/配置约束）；
2. `artifacts/runs/p50-forced-tier-preflight/report.json` 的 `passed`/`versions`/`rates`；
3. `[anchor] 已按 --reset-anchor 重建为当前策略` 是否出现在日志；
4. `history.jsonl` 里 `collect.forced_tier` 是否存在、`targets` 是否非空。
