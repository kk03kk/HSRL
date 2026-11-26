# NUS 无人值守 PPO

这个入口复现 37.0.3 第 288 轮后的九小时实验。新补丁按 [研究流程](WORKFLOW.md)重新批准初始化及引擎后，更新控制器的冻结身份和种子计划。历史作业继续使用各自的冻结检出。

## 文件和角色

- [准备器](../tools/ppo_night_prepare.py)生成原版、混合、增压、后期真人以及回退配方，另外生成开发和独立评测计划。需要已生成的 HSRL 战团库和 BC 环境。
- [控制器](../tools/ppo_night.py)验证输入、保存候选、选择下一阶段和写最后报告。[单作业入口](../deploy/slurm/ppo_night.sh)负责 Slurm 环境。
- [新供体采集](../deploy/slurm/ppo_night_fresh.sh)构建独立合成库。[整链提交](../deploy/slurm/ppo_night_submit.sh)检查预检回执后，提交三段训练、对照、封存、评测和汇总。
- `preflight-receipt.json`、`gpu-preflight.json`、`state.json`、`events.jsonl` 和 `jobs.txt` 提供版本、启动及中间决策证据。分支的 `latest.pt` 含模型、Adam、随机数、迭代与下一局种子。

## 已验收的运行配方

每轮 1,024 局。50% 为七个原版 Bot，25% 为五个原版和两个 V2.0 教师，25% 在 T8 后用三个强档模拟战团替代三个原版席。使用整卡 H100 NVL、10 CPU、38 GiB 主机内存，每 2,048 个样本更新，微批 1,024，固定 512 行 BF16 前向及 FP32 概率。

从第 288 轮继承权重和 Adam。固定学习率 7.5e-6、温度 1.0、熵 0.002、clip 0.20、γ 1.0、GAE λ 0.995、价值损失 0.50、名次辅助损失 0.05、梯度裁剪 1.0。KL 早停 0.02，轮末拒绝 0.05，每四个优化步骤检查一次。保留前四表现和原终局奖励。只用当前策略采样的动作更新一遍。

## 从冻结资产开始

以下变量由 [模拟器、BC 和战团验收](README.md)提供。操作在 NUS 登录节点发起，编译、采样、训练及评测在相应计算节点运行。以下命令重放历史实验，冻结检出 `5b3fa60` 对应指纹 `b37403fe50d5b72b`。若改用当前源码，应重新构建 wheel 并通过预检。

```bash
cd ~/projects/HSRL
root="$PWD"
run="$root/artifacts/ppo/37.0.3-warband-replication"
assets="$root/artifacts/warband/37.0.3-research-01"
bc="$root/artifacts/bc/37.0.3-69011f7d5c74"
parent="$root/artifacts/ppo/37.0.3-first-training-assessment/selected.pt"
commit=$(git rev-parse 5b3fa60)
python3 tools/ppo_night_prepare.py "$run" "$assets"
git worktree add --detach "$run/source" "$commit"
printf '%s\n' "$commit" > "$run/source_commit"
entry="$run/source/deploy/slurm/ppo_night.sh"

verify=$(sbatch --parsable -p normal -c 4 --mem=12G -t 00:06:00 \
    --chdir="$run/source" --output="$run/logs/check-%j.log" \
    "$entry" "$run" "$assets" "$bc" "$parent" verify)
smoke=$(sbatch --parsable -p gpu --gres=gpu:h100-96:1 -c 10 --mem=38G -t 00:09:00 \
    -d "afterok:$verify" --kill-on-invalid-dep=yes --chdir="$run/source" \
    --output="$run/logs/gpu-check-%j.log" "$entry" "$run" "$assets" "$bc" "$parent" smoke)
fresh=$(sbatch --parsable -p normal -c 2 --mem=9G -t 00:12:00 --chdir="$run/source" \
    --output="$run/logs/fresh-%j.log" "$root/deploy/slurm/ppo_night_fresh.sh" "$run" "$assets")
printf 'verify=%s\nsmoke=%s\nfresh=%s\n' "$verify" "$smoke" "$fresh" > "$run/jobs.txt"
```

等 CPU 和 GPU 预检完成后再提交全链。正式发布的源码必须与安装的 wheel 通过 `build_is_current()` 一致性检查。若同一实验需要统计独立的重复训练，在运行前另行封存种子计划。

```bash
sacct -j "$verify,$smoke,$fresh" --format=JobID,State,ExitCode,Elapsed
"$root/deploy/slurm/ppo_night_submit.sh" "$run" "$assets" "$bc" "$parent"
```

## 调度和晋级

三段使用 `gpu` 分区，各限时 2 小时 59 分。主线启动后有九小时窗口，监测、恢复和段间排队计入窗口，最终评测另排资源。使用 `afterany` 让后段在前段异常退出后仍能获得调度，并读取最后一次原子保存。两条对照共用一组独立于主线的训练种子，并以相同预算并行运行。登录端和 Windows 都可以退出。

每 32 轮开发评测。连续两次真人效用有可信增益、保住脚本场，且独立训练种子的混合复验优于纯脚本时才试增压。连续停滞允许一次 T10 真人场探索。明显退化或识别到稳定复现的数值、引擎故障时，用最近通过开发检查的完整权重和 Adam 试低学习率纯脚本回退。持续严重故障或无可用开发结果时，保留已验收候选并进入终评。

五个独立面板每个 2,048 局。比较冻结候选、第 288 轮及已完成的对照。真人盘面复用封存来源，全新的独立种子用于对局。新合成库另外隔离原始供体种子。报告按全局种子配对给 95% 区间，保留不可用的面板和工程错误。真人效用相对两个基线均通过，且脚本平均名次和前四非劣，才有正式晋级证据。`REPORT.md` 和 `RESULTS.json` 由汇总作业写回运行目录。

运行号、状态读法和日志位置见 [NUS 操作手册](NUS.md)。 [本轮结果记录](reports/37.0.3.md)
