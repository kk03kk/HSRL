# 论文启发实验：下一次 NUS 作业

2026-10-27。本次本地交付的终点是 **CPU 验收 → 单 A100 测量四个候选**。两个作业都成功后，先读取测量报告，再定正式训练配置；脚本不会自动追加长训练。论文分析见 `docs/research/paper-2610.33845/论文分析与训练路线调整.md`，本次实现说明见 `docs/research/论文启发实验阶段交付_2026-10-27.md`。

## 本次选择

先回答训练最需要的两个问题：现有循环结构的代价有多大；紧凑表示和无循环策略在 NUS 上是否能稳定训练。当前没有作者的私有战团库、训练器或权重，不能直接复现其最终成绩。四个配置是我们自己的受控候选：

| 配置 | 模型 | 参数量 | Transformer 实体 token |
|---|---|---:|---:|
| legacy128 | 原有表示 + GRU | 1,755,273 | 125 + 可变额外效果 |
| compact_gru128 | 紧凑表示 + GRU + 独立价值头 | 1,778,442 | 49 |
| compact_ff128 | 紧凑表示 + 前馈 + 独立价值头 | 1,515,018 | 49 |
| compact_ff192 | 更宽/深的紧凑前馈 | 2,900,042 | 49 |

紧凑表示包含卡牌 ID、66 维静态属性、64 维确定性规则文本词形 hash，以及归属实体聚合。词形 hash **不是预训练语言语义 embedding**；本模型也不是论文的 1800 万参数模型。

首次 GPU 测量每个候选 BC 2 波、PPO 3 波，16 大厅、每厅 1 learner、学习率 3e-5、单 epoch、1024 个序列位置/更新批次上限（32 段 × 32 步，包含 padding）。原生/节奏/经济/混合脚本的独立考试使用另一组固定种子，分别报告 greedy/sample。此规模只验吞吐、显存和完整性，不能宣布某候选棋力最强。脚本名称不代表已证实的强弱层级。

## 一、上传独立源码包

本地交付：

```text
D:\HearthstoneRL\AutoTavern\runs\dist\AutoTavern-paper-stage-v1.tar.gz
D:\HearthstoneRL\AutoTavern\runs\dist\AutoTavern-paper-stage-v1.tar.gz.sha256
```

将这两个文件上传到服务器 `$HOME`。可在本地 PowerShell 使用：

```powershell
scp -J YOUR_NUS_USER@sjump.comp.nus.edu.sg "D:\HearthstoneRL\AutoTavern\runs\dist\AutoTavern-paper-stage-v1.tar.gz" "D:\HearthstoneRL\AutoTavern\runs\dist\AutoTavern-paper-stage-v1.tar.gz.sha256" YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/
```

不覆盖原项目代码。新执行副本位于原项目 `runs/releases/paper-stage-v1/AutoTavern`，复用原来已安装的 Python 环境和同版本原生扩展。所有新 run 都留在该独立副本内。

## 二、在登录终端解包、绑定环境

下面只运行标准库工具，不在登录节点导入 torch：

```bash
(
  set -e
  cd "$HOME"
  sha256sum -c AutoTavern-paper-stage-v1.tar.gz.sha256
  base="$HOME/projects/hearthstone-rl/AutoTavern"
  release_parent="$base/runs/releases/paper-stage-v1"
  if [ -e "$release_parent/AutoTavern" ]; then
    printf '该独立发布目录已存在；停止解包，请核查既有版本，不叠加覆盖。\n'
    exit 1
  fi
  mkdir -p "$release_parent"
  tar --no-same-owner -xzf "$HOME/AutoTavern-paper-stage-v1.tar.gz" -C "$release_parent"
  cd "$release_parent/AutoTavern"
  python3 tools/package_nus.py --verify .
  python3 tools/prepare_paper_release.py --environment "$base"
)
```

准备工具只复制 `_core.abi3.so`，并写 `EXECUTION_ENV.json`；不会安装包、改共享 `.venv` 或改原目录源码。原生源码指纹仍须为 `35275a3e7711a471`，计算节点 doctor 会再次检查其内容与源码一致。Linux/CUDA 兼容性由下面作业验收，Windows 本测不能替代它。

如果上述块失败，先处理报错，勿继续提交。若只是终端断开、目录已经成功准备，可进入该目录重新 `--verify` 和 `prepare_paper_release.py`（绑定同环境可重复执行），无需再次解包。

## 三、提交两个依赖作业

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/paper-stage-v1/AutoTavern"
verify_job=$(sbatch --parsable deploy/nus/paper_verify.sbatch) &&
bench_job=$(sbatch --parsable --dependency=afterok:"$verify_job" deploy/nus/paper_benchmark.sbatch) &&
printf 'verify=%s benchmark=%s\n' "$verify_job" "$bench_job"
squeue -u "$(id -un)"
```

| 作业 | 分区 / GPU | CPU | 内存 | 最大运行时限 | 内容 |
|---|---|---:|---:|---|---|
| hsrl-paper-check | normal / 无 | 4 | 12 GB | 45 分钟 | 源码清单、完整 Python 测试、CPU doctor |
| hsrl-paper-bench | gpu / A100-40 × 1 | 16 | 64 GB | 2 小时 45 分钟 | 四候选真实尺寸短训练、严格 checkpoint 加载、分布考试、吞吐/显存报告 |

时限是申请上限，不是预计耗时；排队时间另计。GPU 作业使用 job ID 命名输出目录，不覆盖历史训练。收到提前结束信号时保留已完成迭代；未完成候选会报错退出，不继续跑下一个。重新提交测量会生成新目录，从头测量各候选；不会把残缺报告当完整结果。

## 四、结果与下一决策

将 `VERIFY_ID`、`BENCH_ID` 替换为提交返回的数字：

```bash
cd "$HOME/projects/hearthstone-rl/AutoTavern/runs/releases/paper-stage-v1/AutoTavern"
sacct -j VERIFY_ID,BENCH_ID --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS -P
tail -n 80 slurm-hsrl-paper-check-VERIFY_ID.out
tail -n 80 slurm-hsrl-paper-bench-BENCH_ID.out
python3 - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('runs/nus-paper-benchmark-v1').glob('job-*/report.json')):
    r = json.loads(p.read_text())
    print(p, 'complete=', r['benchmark_complete'], 'device=', r['device'])
    for c in r['results']:
        print(c['candidate'], 'params=', c['parameters'],
              'unique PPO samples=', c['unique_ppo_decisions'],
              'samples/s=', round(c['effective_decisions_per_second'], 2),
              'peak allocated MiB=', round(c['peak_cuda_allocated_bytes']/1024**2, 1),
              'updates=', c['optimizer_updates'], 'KL stops=', c['kl_early_stopped_waves'])
PY
```

请保留每个候选的 `manifest.json`、`metrics.jsonl`、`distribution-exam.json`、checkpoint 和顶层 `report.json`。吞吐分母只含 PPO 采集与更新时间；整项耗时另计 BC、评测等，不能直接用采样吞吐估算全部墙钟时间。每个候选的峰值显存也包含独立考试。

接下来的选择顺序：

1. 排除原生/源码不符、动作拒绝、训练截断、非有限梯度、没有实际更新或异常 KL 提前停止等问题。评测截断单列，失败不记为正常第八。
2. 根据吞吐、显存和前缀重算代价，选择可承担的批量与 1–2 个候选；不按这几局名次选赢家。
3. 使用按有效 learner 决策数控制的等预算 pilot、更多独立大厅与多个训练 seed 比较是否真正学会；四个预备配置是最多 200 波或至少 20 万样本的整波停止预算，可能超出样本目标，也可能先触及迭代上限。它们没有自动提交。
4. 同步补齐来源可追溯、版本可映射的强战团库与强度标定，再研究论文式 library opponents 和分阶段训练。当前自然 bot 库及旧自动化战团只作资料和工具验证，禁止直接称作现实强度校准库。

旧 server repair2 BC 模型尚未下载本机。本次四候选统一从各自新 BC 起步，公平比较结构成本。旧 checkpoint 仍在独立冻结环境解释；严格身份校验保留，不能跨架构 resume 或忽略身份硬加载。

## 本地复验入口

在主项目执行（输出目录须不存在）：

```powershell
python tools/validate_paper_stage.py --out runs/paper-stage/recheck-01
```

包含完整 Python 测试、四个正式候选尺寸前后向、四个缩小尺寸 BC/PPO/独立考试闭环。原生 Rust 本轮未改；866 项历史原生回归属于前一交付的证据。本次不会操作实机、输入鼠标键盘、启动搜索或依赖参考仓库代码。
