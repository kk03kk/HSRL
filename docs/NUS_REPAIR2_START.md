# 第二版训练：修复与重新提交

主项目：`D:\HearthstoneRL\AutoTavern`。本发布保留服务器旧 `runs/nus-ppo`；新运行进入 `runs/nus-ppo-v2`。原 job 1020969 已失败，checkpoint 可读至 BC 8/PPO 25。

> 增量构建修复：原215文件包在已有旧target的服务器上可能复用旧引擎缓存。当前恢复命令与永久补丁见 [构建缓存修复](research/NUS增量构建缓存修复_2026-10-27.md)。仅对原215文件包重新提交普通rebuild不能保证刷新缓存；使用文档中的全新target命令。后续永久补丁的env.sh会自动按源码内容选择target。

## 发布包含的改变

- 等数量战团随机先攻，去除酒馆等级分支。
- 批量交互发现决策上下文过时时延期，返回 applied；采集只学习实际执行动作并回退延期席位的记忆。跨席位新增选择不再被当非法动作处理。
- 真正失败保存准确大厅、原生错误、初始状态和动作序列，可用 `python tools/replay_training_failure.py 文件` 回放。不会把失败轨迹强行当自然第八。
- PPO 在超 KL 阈值时跳过本次更新，日志分开记录均值、最大值、触发值与已执行次数。
- 明确每 128 个已执行决策重置 GRU；32 步训练窗口与采样对齐，前缀重建有界。该选择改变记忆语义，不冒称保留无限整局记忆。
- 默认完整自然对局 gamma=1/lambda=1，减少早期终局信号衰减；BC 名次辅助权重 .1。二者是待比较的训练设计，不承诺必然棋力更强。
- 对手组权重固定为 bot .4、latest .2、BC .2、history .2；无历史时其 .2 分配给 latest。有界历史组内均分，不再随历史数量改变机器人占比。
- BC 后第 0 波固定评测；固定 greedy 对手，分别报告 learner greedy/sample；两组验证种子，每组每种模式共 64 局。best 以 sample 模式两种对手组的均名次平均选取。独立 test 使用不同种子，不用于挑 best。
- 验证保留/恢复训练 RNG；严格 resume 不能跨版本；显式 --init-from 只导入兼容模型权重，重置优化器、进度、历史对手和 best，记录来源 SHA256。

## 1. 在本地 PowerShell 上传

```powershell
scp -J YOUR_NUS_USER@sjump.comp.nus.edu.sg "D:\HearthstoneRL\AutoTavern\runs\dist\AutoTavern-nus-repair2.tar.gz" "D:\HearthstoneRL\AutoTavern\runs\dist\AutoTavern-nus-repair2.tar.gz.sha256" YOUR_NUS_USER@xlogin.comp.nus.edu.sg:/home/h/YOUR_NUS_USER/
```

## 2. 登录服务器后校验与解压

```bash
cd "$HOME"
sha256sum -c AutoTavern-nus-repair2.tar.gz.sha256 &&
cd "$HOME/projects/hearthstone-rl/AutoTavern" &&
python3 tools/package_nus.py --verify . &&
tar --no-same-owner -xzf "$HOME/AutoTavern-nus-repair2.tar.gz" -C .. &&
python3 tools/package_nus.py --verify .
```

这是完整源码发布，覆盖项目源文件，不包含 .venv、.toolchains、二进制或 runs，不删除旧模型。新 Rust 源码必须重建原生扩展，不能仅复用旧 verify 作业。不要在 xlogin 上 import torch 或编译。

## 3. 提交依赖链

先进行 CPU 重建/全量检查/256 大厅随机合法动作压力测试，再 GPU 冒烟，再完整模型 40 波验收，最后正式训练。GPU 验收不满 40 波视为失败；不会放行正式作业。

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern"
rebuild_job=$(sbatch --parsable deploy/nus/rebuild.sbatch) &&
smoke_job=$(sbatch --parsable --dependency=afterok:"$rebuild_job" deploy/nus/smoke.sbatch) &&
qualification_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" deploy/nus/qualification.sbatch) &&
train_job=$(sbatch --parsable --dependency=afterok:"$qualification_job" deploy/nus/train.sbatch) &&
printf 'rebuild=%s smoke=%s qualification=%s train=%s\n' "$rebuild_job" "$smoke_job" "$qualification_job" "$train_job"
```

资源：重建 normal/8CPU/16GB/3h；smoke gpu/通用 GPU/4CPU/16GB/30min；资格 gpu/a100-40/16CPU/64GB/2h45min；正式 gpu-long/a100-40/16CPU/64GB/71h。a100-40 可能是 MIG 切片，以实际 doctor 为准。

验收与正式分别在 `runs/nus-qualification-v2`、`runs/nus-ppo-v2`，正式默认从新初始化开始。这样旧、新规则基线可明确比较；旧模型仍保留，且之后有显式权重迁移入口。40 波资格作业是稳定性门槛，不能证明高棋力或完全无 bug。若任一失败，先发日志，不重复提交依赖链。

## 4. 后续评测与恢复

训练期间日志 `slurm-hsrl-ppo-作业号.out` 与 `runs/nus-ppo-v2/metrics.jsonl`。服务器作业与模型状态需通过用户输出确认，不假定已启动。

```bash
# 在 GPU 作业中执行：复核与训练一致的验证协议（相同局数、种子、BC 锚点）
python -m taverngym.rl.train evaluate --checkpoint runs/nus-ppo-v2/best.pt --device cuda --split validation --out runs/nus-ppo-v2/recheck.json
# 独立测试作业，默认两组 test seeds，不参与 best 选择
sbatch deploy/nus/evaluate.sbatch
```

同版本中断后 `train.sbatch` 自动 --resume；不自动创建新作业，不改变旧 run。跨版本迁移必须新输出目录，使用 `train --config ... --out 新目录 --init-from 旧latest.pt`，绝不修改旧文件身份或强行 --resume。当前推荐先完成新基线，因此默认链没有使用旧模型热启动。

## 已知范围

依旧是 36.6.3 Implemented 无纳迦单人基线；未声称完整当前版本实机复刻。未公开的概率、提供池条件与极端触发长尾仍需规则资料/用户审阅。教师是弱启发式，尚未引入用户示范；本轮改进不能替代真实学习曲线、多 seed 与实机一致性验证。
