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

当前唯一排行榜是 **leaderboard-v2（gauntlet-anchored）**：

- 规则与当前分数：[LEADERBOARD.md](../LEADERBOARD.md)
- 当前榜：`artifacts/models/leaderboard-v2/leaderboard.json`
- 新模型 gauntlet：`config/eval-latest-gauntlet.json` + `workflows/eval_latest_gauntlet.py`
- 旧 8 模型流程（历史）：`config/eval-elo-8.json` + `workflows/eval_round_robin_elo.py`

旧的 2 model + 6 bot 固定 128 局协议仍保留为历史参考（[EVAL_DUEL.md](../EVAL_DUEL.md)），
但不能与 Elo 混榜。旧战团库已归档，Elo 评测只用 engine bot。

正式启动前必须确认：engine `c4c71014e3d41760`、8 个模型 SHA、协议 SHA
`837231e0d2f6e583549b095581e60e2986ac899511872b6e0798091b4dc85e94`。

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
