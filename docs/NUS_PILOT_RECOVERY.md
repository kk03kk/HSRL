# 首模型恢复训练：recovery-v1

本版复用已有 Python/CUDA 环境，**重新构建一次修改后的 Rust 扩展**。不重新安装环境，不覆盖 paper-stage-v1，不直接 resume 旧身份，不再提交四架构对照。提交两个作业：CPU 构建验收 → 单 A100 补强 BC / 条件性 PPO。

## 1. 本地 PowerShell 上传

```powershell
cd D:\HearthstoneRL\AutoTavern
scp -J YOUR_NUS_USER@sjump.comp.nus.edu.sg runs/dist/AutoTavern-recovery-v1.tar.gz runs/dist/AutoTavern-recovery-v1.tar.gz.sha256 YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/
```

## 2. 已登录服务器的终端：独立解包

```bash
(
  set -e
  cd "$HOME"
  sha256sum -c AutoTavern-recovery-v1.tar.gz.sha256
  base="$HOME/projects/hearthstone-rl/AutoTavern"
  parent="$base/runs/releases/recovery-v1"
  if [ -e "$parent/AutoTavern" ]; then
    printf '发布目录已存在，停止；不要叠加覆盖。\n'
    exit 1
  fi
  mkdir -p "$parent"
  tar --no-same-owner -xzf "$HOME/AutoTavern-recovery-v1.tar.gz" -C "$parent"
  cd "$parent/AutoTavern"
  python3 tools/package_nus.py --verify .
  python3 tools/prepare_paper_release.py --environment "$base"
)
```

准备工具先复制旧扩展；构建作业会将新 wheel 中的扩展原子替换到**新发布目录**。旧环境、旧扩展和历史模型不变。

## 3. 提交

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/recovery-v1/AutoTavern"
build_job=$(sbatch --parsable deploy/nus/recovery_build.sbatch) &&
train_job=$(sbatch --parsable --dependency=afterok:"$build_job" deploy/nus/recovery_train.sbatch) &&
printf 'build=%s train=%s\n' "$build_job" "$train_job"
squeue -u "$(id -un)"
```

构建申请 normal / 8 CPU / 16GB / 最长 2:45；训练申请 gpu / A100-40GB ×1 / 16 CPU / 64GB / 最长 2:45。最长时限不是预计实际时长。构建参考此前约 20–25 分钟，缓存与负载会影响；BC 与验收预计数分钟至数十分钟，若进入 PPO，20 万有效决策按此前吞吐粗估约 25–50 分钟，另外计评测；排队时间不包含在内。新方案的耗时尚未实测，不能保证。

## 实际执行内容

1. 校验源码；内容隔离的 Cargo 缓存重建；Rust/Python 全量测试；doctor。
2. 从原服务器 `paper-stage-v1/AutoTavern/runs/nus-paper-pilot-20261027-132048/compact_gru128-s701/latest.pt` 兼容初始化权重。新身份、新优化器，不恢复旧 PPO 进度。
3. 追加 16 波 BC，每波 16 大厅、每批 4 个 epoch；监督学习率 3e-4，去掉 BC 名次辅助损失。保留 compact_gru128。
4. 独立教师动作准确率 ≥50%（至少128个非强制决策），且 greedy/sample 固定机器人评测平均名次均 ≤6.5、无失败截断，才生成 `bc-ready.pt`。每模式16个大厅；这只是基础行为门槛。
5. 通过后才开始新 PPO，预算20万有效决策；随后分别考试 BC 与 PPO 模型。未通过保留 candidate 与 gate 报告，停止 PPO，不掩盖结果。

输出为 `runs/recovery-训练JOBID/`。bootstrap 的进度与 PPO 分开；PPO 文件里 bc_done=0 不代表前面的 BC 未运行，初始化来源记录在 manifest。

## 查看结果

将下面 JOBID 换成提交返回的两个编号：

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/recovery-v1/AutoTavern"
sacct -j 构建JOBID,训练JOBID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS -P
tail -n 100 slurm-hsrl-recovery-训练JOBID.out
cat runs/recovery-训练JOBID/bootstrap/gate.json
ls -lh runs/recovery-训练JOBID/bootstrap runs/recovery-训练JOBID/ppo
```

`BOOTSTRAP_GATE_NOT_READY` 时作业会正常结束、Slurm 可显示 COMPLETED；**应以 gate.json 的 ready_for_ppo 为准**，此时没有 ppo 目录是预期结果。失败、超时或未通过先返回日志，不盲目重复提交。训练收到预警会保存阶段可用产物，但不保证一次作业完成全部阶段。

请返回 build/train 的 sacct、训练日志末尾及 gate.json；通过后再返回 independent-bc.json / independent-final.json。登录节点不导入 torch。

## 规则范围

36.6.3 已实现内容、排除纳迦与艾萨拉、repair2 因果批处理与先攻修正，加三条官方37.0.1黑暗之赐组合禁选；不是完整当前实机规则。官方依据和故障证据见 [研究报告](research/首模型实证排查与恢复训练_2026-10-27.md)。真人战团库仍在采集/战斗代理阶段，尚未成为本次 PPO 训练对手；本次是机器人大厅及历史模型混合机制中的 bots 主线，不能称论文完整复现。
