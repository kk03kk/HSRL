# 下一作业：entity-library-v2

下列连接命令中的 `YOUR_NUS_USER` 请替换为自己的NUS用户名。

本地已完成整链检查。与旧v1分开发布，复用服务器现有Python/PyTorch/Rust环境；CPU作业只需构建本版原生模块，不重装环境。旧源码、实验与模型保留。

## 两段作业

| 作业 | 分区与资源 | 工作 | 时限 |
|---|---|---|---|
|CPU准备|normal，16CPU，64GB|新原生构建、全量测试、1796万模型CPU前向、生成三份完整上下文库|2:45|
|GPU资格|gpu，a100-40×1，16CPU，64GB|真实显存/反向预检→BC16波→行为门槛→5万样本库PPO→BC/latest/best独立考试|2:45|

A100资源可能是MIG，以doctor和resource_preflight的实际名称/可见显存为准。CPU生成至少6,216局（每个split覆盖2,072种合法上下文），不是在CPU上正式训练大模型。不能把小规模Windows速度直接当服务器ETA；两阶段预计为几十分钟到数小时，排队另计，实际速度由本次报告确认。

规则保留已确认的36.6.3+两补丁修复、随机大厅、200次训练预算，无8Move限制。数据主要是机器人源战团；真人单局只进入匹配的训练上下文。详细流程与限制见TRAINING_PIPELINE_CURRENT.md。

## 1. 本机PowerShell上传

```powershell
cd D:\HearthstoneRL\AutoTavern
scp -J YOUR_NUS_USER@sjump.comp.nus.edu.sg runs/dist/AutoTavern-entity-library-v2.tar.gz runs/dist/AutoTavern-entity-library-v2.tar.gz.sha256 YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/
```

## 2. 服务器解包到独立目录

```bash
(
  set -e
  cd "$HOME"
  sha256sum -c AutoTavern-entity-library-v2.tar.gz.sha256
  base="$HOME/projects/hearthstone-rl/AutoTavern"
  parent="$base/runs/releases/entity-library-v2"
  if [ -e "$parent/AutoTavern" ]; then
    printf '发布目录已经存在；停止，不覆盖。\n'
    exit 1
  fi
  mkdir -p "$parent"
  tar --no-same-owner -xzf "$HOME/AutoTavern-entity-library-v2.tar.gz" -C "$parent"
  cd "$parent/AutoTavern"
  python3 tools/package_nus.py --verify .
  python3 tools/prepare_paper_release.py --environment "$base"
)
```

## 3. 提交CPU→GPU依赖链

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/entity-library-v2/AutoTavern"
prepare_job=$(sbatch --parsable deploy/nus/library_prepare.sbatch) &&
pilot_job=$(sbatch --parsable --dependency=afterok:"$prepare_job" deploy/nus/library_pilot.sbatch) &&
printf 'prepare=%s pilot=%s\n' "$prepare_job" "$pilot_job"
squeue -u "$(id -un)"
```

CPU准备完整通过才启动GPU；源对局失败会留诊断、缺失上下文会补采，覆盖仍不足就阻止训练。

## 4. 看状态与结果

替换两个作业编号：

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/entity-library-v2/AutoTavern"
sacct -j PREPARE编号,PILOT编号 --format=JobID,JobName%30,State,Elapsed,ExitCode,MaxRSS -P
tail -n 60 slurm-hsrl-lib-prepare-PREPARE编号.out
tail -n 60 slurm-hsrl-entity18m-pilot-PILOT编号.out
cat runs/entity-library-pilot-PILOT编号/report.json
```

CPU库报告：runs/library-data-v2/report.json。GPU报告中的resource_preflight记录实际微批量、累计规模、显存与资源探针耗时；effective-config.json保存实际采用配置。

- `status=bc_not_ready`：BC门槛未通过，保留bootstrap/candidate.pt与gate.json，没有启动PPO；Slurm仍可能正常COMPLETED。
- `status=completed`：5万样本资格训练与BC/latest/best保留考试全部完成。
- `status=interrupted/failed`：按日志和报告处理，不能当成资格通过，不盲目重提或覆盖输出。

本次不自动追加正式长训练，不开始真实游戏。服务器报告回来后再决定下一预算。
