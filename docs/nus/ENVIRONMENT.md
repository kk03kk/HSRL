# NUS 环境与运行方式

更新：2026-11-09

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

历史实验、旧部署和本地 Windows 说明已归档到 `records/archive/docs-2026-11-09/`，不能作为当前命令使用。
