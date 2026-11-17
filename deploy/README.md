# 部署入口

源码、环境、共享资产和验收规范见 `docs/ENGINEERING.md`。登录节点只做 Git、Slurm 提交/查询
与小文件读取；下面的测试、编译和模型加载均在计算节点运行。

| 用途 | 当前入口 | 备注 |
|---|---|---|
| 正式评测 | `slurm/evaluate_suite.sbatch` | 显式 `HSAI_EVAL_PLAN`、`HSAI_EVAL_OUT`，冻结 release 中提交 |
| 旧评测环境变量兼容 | `slurm/eval_shard_parallel.sbatch` | 转为同一 runner 的计划，不再维护第二套分片算法 |
| 发布验收 | `slurm/validate_release.sbatch` | CUDA/Python、登记最终模型、固定 batch 恢复、对局 smoke |
| 原生与 CPU 验收 | `slurm/validate_cpu.sbatch` | Rust、CPU Python、已有 smoke 的全动作回放 |
| 训练入口验收 | `slurm/validate_training_entry.sbatch` | 两个单轮段，仅工程探针，不发布训练成果 |
| 已冻结训练配方 | `slurm/stage_arena.sbatch` | 仍需训练授权；本轮现代化未启动新长训 |
| 已有建库入口 | `slurm/build_arena_snap.sbatch` | 科学采样/校准规则未改 |

其余脚本保留历史工作流兼容性，不能因文件存在就认定其模型路径、科学配置或资源参数仍适用。
复用前核对对应文档和用户授权；失败保留原输出，新输出目录重做工程验证。
