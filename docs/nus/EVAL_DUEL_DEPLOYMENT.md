# NUS 部署与提交：双模型 128 局评测（eval-duel）

本页是「7357 + 10004 对 6 个引擎 bot、128 局、每回合 80 动作 / 5 换位」这套评测在 NUS 的
**执行手册**。协议与口径见 [docs/EVAL_DUEL.md](../EVAL_DUEL.md)。

**状态（2026-11-07）**：本地已实现、单测通过、CPU 冒烟/16 局验证通过；
**尚未在 NUS 运行**，本页所有命令由用户手工执行，代理不登录、不提交作业。
本轮协议 SHA 与报告以服务器实际产物为准。

## 0. 先看规则（本仓库已确认的硬约束）

| # | 规则 | 出处 |
|---|---|---|
| 1 | 登录节点 1 GiB 虚拟内存；只做 git / sbatch / sacct / squeue / tail / 小 JSON | `docs/nus/ENVIRONMENT.md:11,18` |
| 2 | 该限制**由代码强制**：无 `SLURM_JOB_ID`、host 含 login/sjump、非 x86_64、`ulimit -v ≤ 1 GiB` 都直接返回 2 | `deploy/slurm/common.sh:3-24` |
| 3 | venv / Torch / Rust 编译 / pytest / 模型加载只在计算节点 | `deploy/bootstrap_nus.sh:5-6,19-26`；`deploy/slurm/p50_cpu.sbatch:19` |
| 4 | 活动目录 `~/projects/HSAI`（独立 clone/.venv/artifacts）；旧 `~/projects/HSRL` **只读** | `docs/nus/ENVIRONMENT.md:3,9`；`docs/nus/NEXT_P50_DEPLOYMENT.md:3` |
| 5 | 提交只用 `sbatch`；作业内**不包 srun**（本站 srun 子 step 曾把双 MIG 缩成一卡） | `deploy/slurm/common.sh:26-27`；`docs/nus/INCIDENTS.md:15` |
| 6 | 不 unset / 不改 `CUDA_VISIBLE_DEVICES`；设备身份由原始 Driver UUID 证明 | `deploy/slurm/common.sh:27`；`workflows/launch_nus_refinement.py:92,171-175` |
| 7 | 提交流程固定：`bash -n` → `sbatch --test-only` → `sbatch --parsable` | `tools/nus_submit.py` `main()` |
| 8 | 输出目录按作业号隔离；**非空即拒绝**，失败证据保留 | `deploy/slurm/evaluate.sbatch:21-23`；`tools/nus_submit.py` `eval_precheck()` |
| 9 | 排队/运行期间冻结代码、依赖、引擎：改了就跑新目录、新作业号 | `docs/nus/ENVIRONMENT.md:16` |
| 10 | 不进 PPO 的工程错误严格拒绝入榜（非法动作/截断/引擎异常） | `hsbg/az/train/engine_quality.py`；本评测 `report.json.valid` |

本作业的资源：`gpu` 分区、`gpu:h100-47:1`、16 CPU、64 GB、1 小时、`--exclude=xgpi15,xgpi17`。
**只要 1 张 MIG**：评测是 2 个 16M 模型的前向，单卡足够；请求 2 卡只会让排队更久，
也不会改变结果（greedy argmax，两个模型顺序决策）。

## 1. 本机（Windows）：提交代码 + 准备权重

```powershell
cd D:\HSAI
# 1) 复制两份权重到上传目录并核对 SHA256（配置里的期望值来自 model.json）
python tools/stage_duel_models.py
# 产物：artifacts/models/duel/7357/model.pt、artifacts/models/duel/10004/model.pt、
#      artifacts/models/duel/MANIFEST.json

# 2) 提交代码（权重不进 Git：.gitignore 已忽略 artifacts/ 与 *.pt）
git add -A
git commit -m "Add 2-model + 6-bot duel evaluation (128 games, 80 actions / 5 moves)"
git push origin main
```

期望值（`config/eval-duel-7357-vs-10004.json`）：

```
A entity16m-nus-7357        60594db00a627884d47afd4a326639ab9b1fa7c48eb88618399051d971dcff82
B entity16m-p50-10004-820m  32ec7a5c249fd8591ccda4d40100cbb76664f2f20ffe1ebda6011ea02f428a4a
（两份都是 67,864,188 字节）
```

## 2. 把权重放到 NUS（二选一）

**A. 服务器上已有原件**（1080772 的 7357、P50 长训的 10004 之前都在服务器上）：
先在登录节点找到候选文件并用 `sha256sum` 核对，**只有 SHA 一致**才复制成新路径：

```bash
cd ~/projects/HSAI || exit
mkdir -p artifacts/models/duel/7357 artifacts/models/duel/10004
# 列出候选（只读历史目录，不改它）
ls -l ../HSRL/personal/runs/nus-4159-economy-long-002/milestones/decisions-558819917/model.pt 2>/dev/null
find ../HSRL -path '*p50-7357-long-001*' -name model.pt -size 67864188c 2>/dev/null | head
# 逐个核对（期望 60594db0… / 32ec7a5c…）
sha256sum <候选路径>
# 一致后再复制到评测目录（不要用硬链接，避免和旧实验互相影响）
cp <7357候选> artifacts/models/duel/7357/model.pt
cp <10004候选> artifacts/models/duel/10004/model.pt
```

**B. 本机上传**（服务器没有原件或 SHA 不符）：

```powershell
scp D:\HSAI\artifacts\models\duel\7357\model.pt  <user>@<nus-host>:~/projects/HSAI/artifacts/models/duel/7357/model.pt
scp D:\HSAI\artifacts\models\duel\10004\model.pt <user>@<nus-host>:~/projects/HSAI/artifacts/models/duel/10004/model.pt
scp D:\HSAI\artifacts\models\duel\MANIFEST.json  <user>@<nus-host>:~/projects/HSAI/artifacts/models/duel/MANIFEST.json
```

## 3. 登录节点：拉代码 + 三重预检（纯标准库）

```bash
cd ~/projects
test ! -e HSAI || true          # 目录已存在时不要覆盖
cd ~/projects/HSAI || exit
git status --short               # 应为空；有本地改动先处理，不要带着脏树提交作业
git log --oneline -3             # 确认看到本轮 "duel evaluation" 提交
git pull --ff-only               # 首次 clone 用：git clone https://github.com/kk03kk/HSRL.git HSAI
mkdir -p artifacts/runs/slurm

# 1) 权重文件：字节数 + SHA256 与配置逐项比对（只读，登录节点安全）
/usr/bin/python3 -S tools/stage_duel_models.py --check-only
# 2) 协议与部署文件：128 局 / 80 动作 / 5 换位 / 去抖 2 被写死检查
export HSAI_EVAL_OUT=artifacts/runs/duel-eval-001
/usr/bin/python3 -S tools/nus_submit.py --phase eval --check-only
# 3) 环境存在性（venv 缺失说明还没跑过 --phase cpu，不能提交评测）
test -x .venv/bin/python && echo venv-ok || echo 'venv 缺失：先跑 --phase cpu'
```

`--phase eval --check-only` 的通过输出包含 `passed: true`、`games: 128`、`seed: 73000000`、
两份权重的 SHA 与输出目录；任何一项不符都会以 `PRECHECK FAILED` 退出，**不要绕过**。

## 4. 提交评测作业

```bash
cd ~/projects/HSAI || exit
export HSAI_EVAL_OUT=artifacts/runs/duel-eval-001     # 每次必须是新目录！
/usr/bin/python3 -S tools/nus_submit.py --phase eval --submit
# --submit 内部：bash -n → sbatch --test-only → sbatch --parsable；记录打印出的作业号
```

作业里做的事（`deploy/slurm/p50_eval.sbatch`）：

1. `hsrl_require_compute`：确认在计算节点、有 Slurm 作业号；
2. 拒绝已存在的 `$HSAI_EVAL_OUT`（保留旧基线）；
3. `python -S tools/nus_submit.py --phase eval --check-only`：协议/权重再核一次；
4. `python tools/stage_duel_models.py --check-only`：计算节点上再核 SHA（防上传截断）；
5. `hsrl_gpu_job python -u workflows/launch_nus_refinement.py --task eval-duel --out "$out"`：
   启动器先写 `visibility-<job>.json`（`nvidia-smi -L`、拓扑、原始 Driver UUID、cgroup），
   再把 `CUDA_VISIBLE_DEVICES` 换成已核验的 MIG UUID 并 `exec` 评测入口。

## 5. 监测与取回产物

```bash
squeue -u "$USER" -o '%.12i %.24j %.12T %.12M %R'
sacct -j <作业号> -X --format=JobID,State,Elapsed,ExitCode
tail -f artifacts/runs/slurm/hsai-duel-eval-<作业号>.out     # 每 16 局打印一次平均名次
ls -l artifacts/runs/duel-eval-001/                          # report.json / report.md / visibility-<job>.json
```

只看摘要（登录节点不要整份解析大 JSON）：

```bash
/usr/bin/python3 -S - <<'PY'
import json
from pathlib import Path
out = Path('artifacts/runs/duel-eval-001')
r = json.loads((out / 'report.json').read_text(encoding='utf-8'))
print('valid:', r['valid'], r['reasons'])
print('protocol:', r['protocol_sha256'])
print('totals:', {k: r['totals'][k] for k in ('games','ranked','decisions','illegal','truncated','engine_bad')})
for label in ('A','B'):
    m = r['results'][label]
    print(label, m['model'], 'mean_rank', m['rank']['mean_rank'],
          'top4', m['rank']['top4_rate'], 'win', m['rank']['win_rate'],
          'final_body_p50', m['final_state']['body_p50'])
print('paired:', r['paired'])
PY
```

取回：

```bash
scp -r <user>@<nus-host>:~/projects/HSAI/artifacts/runs/duel-eval-001 ./duel-eval-001
```

## 6. 失败与重跑

- 作业失败（`ExitCode` 非 0）时**先读 `report.json.reasons`**：`illegal>0` / `truncated>0` /
  `engine_bad>0` 是工程错误，必须保留现场、定位后重跑，**不许**放宽门禁；
- 重跑一律**新目录**：`export HSAI_EVAL_OUT=artifacts/runs/duel-eval-002`；旧目录留证；
- 同一 `HSAI_EVAL_OUT` 不允许并发提交（sbatch 的第 2 步会直接拒绝已存在目录）；
- 代码/依赖/权重任一项变化都要重新生成协议与报告，**不同 `protocol_sha256` 禁止同榜**；
- 作业只是评测：不写 checkpoint、不碰 `STOP_REQUESTED`、不与长训写同一目录。

## 7. 与训练作业的关系

- 评测作业**只读**权重，可以和长训共存（不同输出目录、不同作业号），但**不要**在长训
  正在写同一个 `artifacts/runs/*` 目录时提交；
- 若服务器上还没有 `.venv`，先按 `docs/nus/NEXT_P50_DEPLOYMENT.md` §1 跑 `--phase cpu`
  （bootstrap 编译 Rust/Torch 需要计算节点，且远超 1 GiB 登录节点限制）；
- 引擎必须是从当前 Rust 源码构建的扩展：评测入口在 `fingerprint()` 里调用
  `hsbg_engine.build_is_current(engine/)`，不匹配会直接失败而不是给出可疑报告。
