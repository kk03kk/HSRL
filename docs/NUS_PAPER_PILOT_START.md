# NUS 下一步：两模型、两种子、等预算试训

2026-10-27。依据用户回传的1023445/1023446结果，两作业成功，四候选均完成BC2+PPO3、6次PPO更新，未触发KL提前停止。下一步不再重复短测，进行等预算试训。

## 为什么选这两种

| 模型 | 有效样本/秒 | 峰值 allocated GiB | 峰值 reserved GiB |
|---|---:|---:|---:|
| legacy128 | 140.1 | 7.15 | 11.25 |
| compact_gru128 | 145.1 | 2.63 | 3.66 |
| compact_ff128 | 138.1 | 2.62 | 3.67 |
| compact_ff192 | 133.6 | 3.93 | 5.42 |

紧凑表示节省显存，但目前没有前馈吞吐明显更快的证据。GRU前缀重算3波共0.47秒，远不是当前端到端主要耗时。先比较compact_gru128与compact_ff128，保持其他配置相同，检验记忆是否有学习价值；暂不增加大模型和旧长token模型的预算。

四候选对bots的sample评测在短测前后都是平均第8，每项仅2局，不能宣布已经学会或以6.75等混合BC对手分数挑赢家。每个候选只收集约3000个PPO样本，结果主要证明完整性与成本。

## 下一轮配置

- 两模型：compact_gru128、compact_ff128；两个独立训练seed：701、1701，共4个run。
- 每个run：BC16波（替代短测的BC2），PPO至少20万有效learner样本，整波停止略有超出；10000波只是保护上限。
- 其他保持已测配置：16大厅、每厅1 learner、lr3e-5、单epoch、32段×32步、脚本mixed对手、无额外anchor正则。
- 每25波及预算结束时验证。完成预算后，分别对BC anchor和末期latest运行现有独立四分布考试，每profile每模式2seed×16大厅。
- 检查完成日志中的有效样本数；若提前结束，作业报未完成，不把信号退出当成达到完整预算。相同配置的残缺checkpoint保留，可后续显式恢复。
- 此轮仍用已验收的合法脚本对手。人工八回合已转换为战斗代理，但尚未接入PPO的library席位采样；不把此轮称为真人库训练。

每项申请1张A100-40、16CPU、64GB、gpu分区2小时45分。4项通过Slurm afterany依赖串行，最多同时用1张GPU；不同seed彼此独立，一个失败不把其他试验堵死。按短测吞吐，纯PPO采集+更新每run约23–25分钟，另加BC/验证/独立考试与排队；不是保证墙钟时间。

## 上传一个启动脚本与校验文件

在本机PowerShell：

```powershell
scp -J YOUR_NUS_USER@sjump.comp.nus.edu.sg "D:\HearthstoneRL\AutoTavern\deploy\nus\paper_pilot.sh" "D:\HearthstoneRL\AutoTavern\runs\dist\paper_pilot.sh.sha256" YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/
```

在服务器：

```bash
cd "$HOME"
sha256sum -c paper_pilot.sh.sha256 && bash "$HOME/paper_pilot.sh"
```

脚本默认进入已验收副本：
`$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/paper-stage-v1/AutoTavern`。

它先验证原271项源码清单，再在 `runs/nus-paper-pilot-日期时间/` 下生成配置、提交4个任务，并把job ID写入 `jobs.tsv`。不覆盖原SOURCE_MANIFEST、原生扩展、训练源码、benchmark输出或旧run；无需重建Rust。所有新配置和运行目录独立，4项都从各自相同预算的BC开始，不续接三波短测模型。

这是一次性用户提交脚本，不是驻留服务或自建调度器。若中途sbatch失败，已提交的ID仍在jobs.tsv，先核查，不要反复运行制造重复任务。

## 跑完回传什么

脚本会打印本轮目录。用实际目录名替换下方 `PILOT_DIR`：

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/paper-stage-v1/AutoTavern"
pilot="runs/PILOT_DIR"
cat "$pilot/jobs.tsv"
ids=$(cut -f1 "$pilot/jobs.tsv" | paste -sd, -)
sacct -j "$ids" --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS -P
```

每个run应有 `metrics.jsonl`、`bc-anchor.pt`、`latest.pt`、`independent-bc.json`、`independent-final.json`。后续比较固定20万预算的latest与BC，不用短测名次或单一best选胜者。先确认是否学会、两个seed是否一致，以及独立脚本分布上是否都改善，再决定百万级/更大预算与library课程。

当前没有正式最终棋力结论，所有独立考试仍是模拟器/脚本代理指标。人工库的后续工作是回合采样与训练接入、更多阶段覆盖，以及版本/特殊效果适配；采集过程不重新要求来源、截图或外部核查。
