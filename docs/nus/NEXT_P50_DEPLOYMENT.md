# 7357→P50课程：新布局NUS部署与提交

本地完成不等于NUS完成；本页所有命令由用户手工执行。原~/projects/HSRL是历史实验与只读资源来源，保留原Git、环境、模型、raw和库；禁止在旧目录git pull覆盖布局。本轮新clone放~/projects/HSAI，独立.venv、artifacts、实验目录。

## 1. 登录节点：独立clone与轻量输入预检

```bash
cd ~/projects
# 仅首次执行；目录已存在时停止核实，不要覆盖。
test ! -e HSAI && git clone https://github.com/kk03kk/HSRL.git HSAI
cd ~/projects/HSAI || exit
mkdir -p artifacts/runs/slurm
/usr/bin/python3 -S tools/nus_submit.py --phase cpu --check-only
/usr/bin/python3 -S tools/nus_submit.py --phase cpu --submit
```

--submit先做纯标准库文件/元数据检查、bash语法检查和sbatch --test-only，再真正提交。test-only失败不算通过，脚本不会继续提交。记录CPU作业号。

默认只读来源：
- ../HSRL/personal/runs/nus-4159-economy-long-002/milestones/decisions-558819917：7357的model/trainer/checkpoint.json。
- ../HSRL/personal/runs/nus-4159-economy-long-002：真实旧plan/report/train.config。
- ../HSRL/personal/artifacts/libraries/bot-firestone-3702-economy-v2-live-v1：完整旧库及raw三分割，不能只有本地元数据。

路径不同可在提交前export HSAI_SOURCE_7357、HSAI_SOURCE_RUN、HSAI_OLD_BANK；全部用真实目录，禁止伪造plan.json。新库位置可export HSAI_P50_BANK=artifacts/libraries/自定义新名。

CPU作业8核/32GB/3小时：首次新建环境（Python3.12、Torch2.10+cu128、现有Rust1.98.1），编译当前engine，完整Rust/Python/卡表/源码门禁，7357CPU恢复审计，然后制作六套正式P50库和严格注入审计。Rust独立超时7200秒，避免过去600秒误杀。已有旧.venv不复制、不修改。

```bash
sacct -j CPU作业号 -X --format=JobID,State,Elapsed,ExitCode
# 只读小摘要，不在登录节点运行pytest/Torch/建库。
/usr/bin/python3 -S - <<'PY'
import json
from pathlib import Path
v=json.loads(Path('artifacts/local/p50-cpu-validation.json').read_text())
print('CPU gates:',v['passed'])
r=json.loads(Path('artifacts/libraries/bot-p50-3702-v1/report.json').read_text())
print('Bank complete:',r['complete'],'source last:',r['source_last_turns'])
for k,m in r['splits'].items():
 print(k,m['validation'],'source_games',m['source_games'])
PY
```

只有CPU作业COMPLETED/0:0、gates和bank complete为true才进入GPU。不覆盖失败库；若确需重试，使用新HSAI_P50_BANK并保留证据。建库补采上限不是“必跑局数”，优先复用已验证raw；实际样本数以新报告为准。

## 2. 登录节点：一次GPU工程验收

```bash
cd ~/projects/HSAI || exit
/usr/bin/python3 -S tools/nus_submit.py --phase gpu --submit
```

资源2×H100-47、32CPU、96GB，1小时上限。保持已验证batch启动，不包装srun、不取消Slurm隔离；验证真实Driver UUID和Torch可用双设备，再模型加载。

7357原状态→1轮PPO→保存恢复并换到ramp1→1轮→换到final库再1轮。验证完整Adam延续、样本处理、严格注入、数值及8局诊断链路。探针应7357→7358→7359→7360；这些权重不作为正式起点。

```bash
sacct -j GPU作业号 -X --format=JobID,State,Elapsed,ExitCode
/usr/bin/python3 -S - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('artifacts/runs/p50-7357-preflight/report.json').read_text())
print('passed:',r['passed'],'versions:',r['versions'])
print('rates:',r['training']['rates'])
print('source Adam:',r['source_adam_step_min'],'final:',r['final_restore']['adam_step_min'])
PY
```

若GPU节点可见性仍不符，预检会失败并留证，不能保证Slurm站点配置永远正确；不要绕过检查。新失败输出改export HSAI_P50_AUDIT=artifacts/runs/p50-7357-preflight-新编号后重试，长训读取同一环境变量。

## 3. 用户确认验收后：正式24小时长训

无需再扫描性能参数，不以短期名次作停止门。保持CPU/GPU验收之后的代码、环境、库不变。

```bash
cd ~/projects/HSAI || exit
export HSAI_LONG_OUT=artifacts/runs/p50-7357-long-001
/usr/bin/python3 -S tools/nus_submit.py --phase long --submit
```

预检核对真实7357元数据、新库、CPU门禁、GPU来源/库摘要及训练源码，避免缺文件排队后秒失败。计算节点仍核SHA/引擎/实际设备。新长训始终从冻结7357，累计558873732开始；Adam545990、实际lr约1.75477e-6完整继承。24小时Slurm，控制预算1410分钟，45分钟收尾预留，最多新增10亿决策。

课程按新增决策：0–10M=train-start，10–20M=train-ramp1，20–30M=train-ramp2，30M以后=train；第1–8回合在四阶段都为最终目标，第9–19回合分别0/1⁄3/2⁄3/1进度。达到10M边界需等当前PPO轮完成，可能小幅超出，不中途丢batch。每次换库显式记录上一库摘要，不重置完整Trainer。

起点评测96局；每新增30M决策96局同种子配对监测，正常收尾512局；每次保留完整原始阵容和逐回合经营、金流、动作、战斗统计/P50（含点结束回合前与真实EOT结算后两种口径，分别标注）。Test不在长训中反复查询；正常名次波动只记录。

## 4. 监测、暂停、恢复

```bash
squeue -u "$USER" -o '%.12i %.24j %.12T %.12M %R'
cat artifacts/runs/p50-7357-long-001/train/latest.json
cat artifacts/runs/p50-7357-long-001/learning-health.json
# 安全暂停：当前完整更新提交后停止，保留最后状态。
touch artifacts/runs/p50-7357-long-001/STOP_REQUESTED
```

STOP只用于暂停，不因普通名次波动自动创建。需要确认暂停后再恢复；不对正在写入的目录并发提交。保留原暂停标记并设显式resume：

```bash
cd ~/projects/HSAI || exit
# 先用squeue和status确认上次写入者已经退出。
mv artifacts/runs/p50-7357-long-001/STOP_REQUESTED \
   artifacts/runs/p50-7357-long-001/STOP_REQUESTED.saved-$(date +%Y%m%d-%H%M%S)
export HSAI_RESUME=1
export HSAI_LONG_OUT=artifacts/runs/p50-7357-long-001
/usr/bin/python3 -S tools/nus_submit.py --phase long --submit
```

恢复继承latest完整快照与课程位置，不重跑已有阶段评测，不倒回7357。每次重新申请24小时是新的墙钟分配，但总新决策上限相对7357固定10亿。源码/库改变必须独立新目录显式分叉，不能原目录删除provenance绕过。硬scancel只能恢复最后提交点，未提交的采样不保证保存。

原始权重、大库和历史回放不入Git。新仓库可部署代码与配置，但仍需要以上独立管理的资源。新的7357和7710已在本地核验，不要求重新上传它们到NUS。
