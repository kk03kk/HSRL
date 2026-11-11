# NUS 环境与运行方式

更新：2026-11-11（新增 §7 节点/排队/失败经验）

## 1. 基本工作方式

代理直接运行在 NUS 登录节点。所有开发、测试、评测、训练、Slurm 提交和监控都在 NUS 完成；不再需要把权重、日志或结果传回本地。

活动工作区：

```text
~/projects/HSAI
```

用户只给大方向；代理自主规划、执行、测试、提交、监控和归档。

## 2. 资源限制

| 项目 | 约束 |
|---|---|
| 登录节点 | 约 **1 GiB 虚拟内存** |
| 重任务 | 只能在 Slurm 计算节点 |
| GPU | `h100-47` MIG；以实际可见设备为准 |
| 训练参考 | 2×h100-47、32 CPU、96 GiB |
| 评测参考 | 1×h100-47、16–32 CPU、64–96 GiB |

登录节点只做：

- Git
- `sbatch` / `sacct` / `squeue` / `scancel`
- `tail` 日志
- 小 JSON 标准库读取

登录节点不做：

- `pytest`
- Torch / 模型加载 / 推理
- Rust 安装或编译
- pip 构建
- 建库
- 训练
- 大型 JSON 全量解析

## 3. Slurm 规则

- 只用 `sbatch` 提交；不要在作业里再包 `srun`。
- 本站曾出现 `srun` 子 step 把 batch 双卡缩成单卡。
- 不 unset、不手工改 `CUDA_VISIBLE_DEVICES` 来“修复”设备数。
- GPU 身份必须由 `nvidia-smi -L`、原始 Driver UUID 和 Torch 实际计算证明。
- 输出目录按作业号隔离；已存在且非空的目录拒绝覆盖。
- 排队/运行期间冻结代码、依赖、引擎、库和权重；要改就跑新目录、新作业号。

## 4. 提交前检查

登录节点使用轻量预检，不加载 Torch：

```bash
cd ~/projects/HSAI
/usr/bin/python3 -S tools/nus_submit.py --phase ... --check-only
```

提交：

```bash
/usr/bin/python3 -S tools/nus_submit.py --phase ... --submit
```

不要跳过 `bash -n`、`sbatch --test-only` 和输出目录检查。

## 5. 恢复与停止

- 完整状态包括 `model.pt`、`trainer.pt`、Adam、调度器、RNG、anchor 和累计计数。
- 安全停止：在实际运行目录创建 `STOP_REQUESTED`，在提交边界暂停。
- `scancel` 是硬中断，只能恢复最后已提交快照。
- 代码/引擎变化后必须新目录显式分叉，不能原目录盲目 resume。
- 工程错误、非法动作、截断、引擎异常、注入失败：停止并保留现场。
- 正常棋力波动：记录，不自动止损。

## 6. 当前活动路径

```text
engine/                        当前模拟器
hsbg/az/                       训练栈
workflows/                     训练/评测/建库入口
deploy/slurm/                  Slurm 入口
artifacts/models/              模型与排行榜
records/archive/libraries-2026-11-09/  旧战团库（已退役，不再使用）
artifacts/runs/                运行产物
records/                       原始证据与归档
docs/                          当前文档
```

历史实验、旧部署和本地 Windows 说明已归档到 `records/archive/`（按 `<topic>-<YYYYMMDD>/` 组织），不能作为当前命令使用。

---

## 7. 2026-11-11 审计新增：节点、排队与失败经验

### 7.1 节点架构不统一（已实际踩坑）

- `gpu`/`gpu-long` 里同时存在 **x86_64** 与 **aarch64** 节点。`deploy/slurm/common.sh`
  的 `hsrl_require_compute` 会拒绝 aarch64；表现是作业 1 秒内失败，`.err` 只有：
  `此Slurm配方仅验证x86_64，不能复用到aarch64节点。`
- 已确认 aarch64：`xgpj0`（a100-80）。已确认 x86_64 且训练/评测可用：
  `xgpi*`（h100-47 / h100-96）、`xgpg*`（a100-40）、`xgph*`（a100-40）。
- 提交前检查：
  ```bash
  scontrol show node <node> | grep -E 'Arch=|CfgTRES|Gres='
  ```
  不要只看 `sinfo` 的 `idle`；`idle` 不等于可跑本项目。

### 7.2 排队时间与 `-w`/`--nodelist`

- `sbatch --test-only` 会给出预计开始时间；2026-11-11 实测 `gpu-long` 可能排到
  次日，而 `gpu` 分区有空闲节点时可在数分钟内开始。
- 需要立即验证时，先找真正空闲且架构正确的节点，再显式指定：
  ```bash
  sinfo -p gpu -N -o "%N %t %C %m" | grep -w idle
  scontrol show node <node> | grep -E 'Arch=|CfgTRES|Gres=|State='
  sbatch --test-only -w <node> deploy/slurm/<job>.sbatch
  sbatch -w <node> deploy/slurm/<job>.sbatch
  ```
- `sbatch --test-only` 不是提交；确认输出目录和资源后再提交。
- 不要在未确认节点架构时把长作业提交到 `gpu-long`；aarch64 节点会在启动瞬间失败。

### 7.3 内存请求不能超过节点 `RealMemory`

- `xcnd0` 等 normal 节点 `RealMemory=63000` MB；`--mem=64G` 会得到
  `allocation failure: Requested node configuration is not available`。
- 用 `--mem=60G` 或先查：
  ```bash
  scontrol show node <node> | grep RealMemory
  ```

### 7.4 GPU 类型与 Torch 可用性

- `gpu:nv`（MIG/老驱动）节点曾出现：
  `CUDA initialization: CUDA unknown error ... setting available devices to zero`。
  Torch 验证/评测不要用 `nv`；优先 `h100-47`、`a100-40`、`h100-96`。
- 登录节点没有 GPU，`nvidia-smi` 失败是正常的；GPU 检查只能在计算节点做。
- 一个作业申请 `--gres=gpu:h100-47:2` 时，训练器通常占 `cuda:0`，8 个 actor
  进程占 `cuda:1`。`--gpus 1` 指 **actor 使用的 GPU 数**，不是作业总 GPU 数；
  看到 8 个 worker 都在 `cuda:1` 不代表第二张卡空闲。

### 7.5 已有 CPU allocation 上的轻量分析

- 本会话有 `salloc --partition=long --cpus-per-task=4 --mem=16G` 的交互作业
  `hsai-agent`（xcnc30）。登录节点可用：
  ```bash
  srun --jobid=<salloc_jobid> --overlap hostname
  srun --jobid=<salloc_jobid> --overlap .venv/bin/python <script>
  ```
  只用于 CPU/小数据分析；不要在这类分配上跑 GPU 训练。
- 仍然不要在 sbatch 作业内部再包 `srun` 启动 GPU 训练（本站会把双 MIG 缩成单卡）。

### 7.6 输出目录与证据

- 多数入口拒绝“已存在且非空”的输出目录。若先 `mkdir -p "$out"` 再写
  `nvidia-smi.txt`，Python 入口会报 `output exists and is not empty`。
- 正确顺序：让入口创建目录，或只创建空目录并在入口成功后再写附加证据；
  新实验永远用新目录，不覆盖旧现场。
- 失败作业保留 `.out/.err`；用 `sacct -j <id> --format=JobID,State,ExitCode,Elapsed,End`
  复核真实结束状态，不要只看 `squeue`（失败的作业会立即从队列消失）。

### 7.7 常用排查命令

```bash
squeue -u "$USER" -o "%.10i %.30j %.10T %.10M %.10L %R"
sacct -u "$USER" --starttime today --format=JobID,JobName%30,State,ExitCode,Elapsed,End
sinfo -p gpu,gpu-long -o "%20P %6D %10t %30G %20N"
scontrol show job <id>
scontrol show node <node>
tail -n 100 artifacts/runs/slurm/<job>.err
```

### 7.8 本会话尚未解决/待确认

- `gpu:nv` CUDA unknown error 的根因（驱动/MIG/cgroup）未继续深挖；当前策略是避开。
- aarch64 GPU 节点是否会在未来提供可用的本项目环境，待确认。
- `gpu-long` 的实际回填策略会随集群负载变化，`--test-only` 的预计时间只能作参考。
