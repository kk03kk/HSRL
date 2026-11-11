# NUS 评测提交与监控

状态：当前活动文档。代理直接运行在 NUS，不需要本地传权重或回传结果。

## 1. 工作目录

```bash
cd ~/projects/HSAI
```

活动代码、模型、库、评测结果都在服务器：

```text
artifacts/models/      模型与排行榜
records/archive/libraries-2026-11-09/  已退役的旧战团库（不再使用）
artifacts/runs/        评测/训练输出
records/               原始证据
docs/                  当前文档
```

不要使用旧的 `D:\HSAI` 或旧 HSRL 目录。

## 2. 登录节点只做轻量操作

登录节点只有约 1 GiB 虚拟内存：

- 可以：`git`、`sbatch/sacct/squeue`、`tail`、小 JSON 标准库读取。
- 不可以：`pytest`、Torch、模型加载、Rust 编译、pip 构建、建库、大型 JSON 解析。

所有重操作在 Slurm 计算节点执行。

## 3. 提交原则

- 只用 `sbatch`；作业内不要包 `srun`。
- 输出目录必须是新目录；已存在且非空的输出目录拒绝提交。
- 排队/运行期间冻结代码、依赖、引擎和权重。
- GPU 以真实 Driver UUID / `nvidia-smi -L` 证据为准；不 unset `CUDA_VISIBLE_DEVICES`。
- 工程错误（非法动作、截断、引擎异常、模型/卡表不一致）直接失败，不绕过。

## 4. 当前评测入口

当前正式入口是 **E1 四模型校准**（联合 Bradley-Terry，6 对 × 448 局，ordered56）：

- 规则与当前分数：[LEADERBOARD.md](../LEADERBOARD.md)
- 入口：`config/eval-e1-four.json` + `workflows/eval_round_robin_elo.py`
- r1 报告：`artifacts/runs/e1-four-20261111/report/report.json`
  （协议 `acd7eff7…`，含 `code_sha256`）
- 旧 v2 gauntlet：`config/eval-latest-gauntlet.json` +
  `workflows/eval_latest_gauntlet.py`，只作历史/候选筛选，不与 E1 混榜。

旧的 2 model + 6 bot 固定 128 局协议仍保留为历史参考（[EVAL_DUEL.md](../EVAL_DUEL.md)），
不能与 Elo 混榜。旧战团库已归档，Elo 评测只用 engine bot。

正式启动前必须确认：engine `c4c71014e3d41760`、4 个模型 SHA、E1 config SHA、
报告里的 `code_sha256`、`protocol_sha256`，以及 `valid/leaderboard_eligible=true`。
`shared` 种子必须使用 seed-block bootstrap。
**E1 只跑一轮**：6 对 ×448 局，报告 valid 后立即停止；不跑第二轮确认、不做冠军挑战赛；
冠军按点估计第一。只有 invalid/工程错误/协议缺陷才重跑并先请示。
历史 v2 的 12311 1067.68 与 E1 的 1055.92 不能直接比较（协议不同）。

## 5. 监控与产物

```bash
squeue -u "$USER" -o '%.12i %.24j %.12T %.12M %R'
sacct -j <job> -X --format=JobID,State,Elapsed,ExitCode
tail -f artifacts/runs/slurm/<job>.out
```

评测报告必须写入自己的输出目录，至少保留：

```text
report.json
report.md
index.json（如适用）
visibility-<job>.json
```

报告先落盘，再按 `valid` 门禁退出；失败现场保留。

## 6. 失败与重跑

- 先读 `report.json.reasons`，不要先改门禁。
- 重跑使用新目录、新作业号。
- 不覆盖旧结果；不把不同 `protocol_sha256` 的结果合并。
