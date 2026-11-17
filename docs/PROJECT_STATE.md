# HSAI 当前状态

状态：**as of 2026-11-17，工程现代化与正式评测完成，最终交付核验中**。
动态作业以 `python ops/hsai_ops.py state` 实测为准。历史在 [DECISIONS.md](DECISIONS.md)；
完整本轮报告在 [MODERNIZATION_20261117.md](analysis/MODERNIZATION_20261117.md)。
本文件原 18:27 快照保存在 Git 历史及本地 `records/archive/project-state-20261117-1827/`。

## 当前结论

v1.4-r1 长训正常完成并登记 candidate。与起点 v1.3-r3 的同协议 512 对 seed 正式比较：
平均名次 4.9141 → 4.6895，夺冠率 8.40% → 12.70%；T12 总攻血 P50 1,287（n=360）
→ 1,352（n=345），T16 5,609（n=35）→ 4,826（n=54）。
排名和夺冠率改善，但后期成长没有证明改善；共同存活 T16 只有 7 对。
**不宣布新冠军，不更换线上 pin。** 完整配对区间、共同存活曲线及运营观察见本轮报告。

用户继续决定科学方向。本次授权工程重构、计算节点验证与正式评测，不授权下一轮长训或改变
奖励/模型/超参/对手/游戏规则/评测标准。当前配方仍看 D-076 与
`deploy/slurm/stage_arena.sbatch`；评判口径仍为 [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md)。

## 当前资产与版本锚点

| 资产 | 位置与身份 |
|---|---|
| 引擎 | `config/engine-upstream-lock.json`；37.0.2.256589 / `c4c71014e3d41760`；245 动作，3227 floats + 124 cards |
| 当前候选 | `artifacts/models/arena-v1.4-r1/`；745 轮 / 69,691,656 决策；manifest 绑定 model SHA `5058768bf90e989a943c6e21800a0767095a81a2034f4d4bf8d3d603028896a1` |
| 完整 Trainer | 候选目录及原运行均保留；SHA `2e6073f7a80fbb57c4cca0ede148145addb4541029083f6fe35048a2d51141dc` |
| 对照起点 | `artifacts/models/arena-v1.3-r3/`；SHA `1808f178ae38bf5910e3561043a2478c9f559eb7a6cf065a948a8fbc1c206472` |
| 线上 pin | `config/live/model.json`；SHA `ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7`，本次未改 |
| 训练场 / 考试场 | `artifacts/libraries/arena-cal-v14-snap` / `arena-cal-v14hold-snap` |
| 正式协议 | `7e15f114d0f9290a80e64d8c2e2706c424594adc5ef1a9c44e8e3d59dcb5b56c` |
| 正式评测 | `artifacts/runs/nus-eval-v14-r1-20261117-1/`，complete/validation 均 valid |
| 分析与封存 | `artifacts/runs/nus-eval-analysis-v14-r1-20261117-1/`；正式目录另有 evidence-inventory |

## 运行事实（as of 2026-11-17）

- 长训 **1156986** 于 18:53:59 COMPLETED / 0:0，原始资产完整；没有重启。
- 正式评测 **1157979** COMPLETED / 0:0，耗时 31:05，两个模型各 512 局；19.09 万个学习席动作完整验收。
- 封存与分析 **1158028** COMPLETED / 0:0，耗时 2:20。
- 两端 CPU、Rust、CUDA、真实恢复及真实训练入口工程探针均已完成，详细作业及失败修复现场见报告。
- 真实游戏仍 `real_game_accepted=false`；离线通过不能替代 D-026 实机验收。

## 维护入口

[ENGINEERING.md](ENGINEERING.md) 定义模块边界、冻结发布、环境复建、模型登记、完整评测与验收矩阵。
正式新入口为 `workflows/evaluate_suite.py` / `deploy/slurm/evaluate_suite.sbatch`，
旧分片脚本兼容转入同一执行器。每次发布用固定提交的 Git worktree，作业输入/环境在运行期间保持不变。

Windows 使用独立 CPU 环境，旧环境完整保留；NUS 使用原已验证 CUDA 环境。
两端同步源码和语义契约，模型/库/训练资产按职责保留，不强行统一二进制包版本。
训练参数接口与纯聚合实现已做等价校验；旧基线 512 局的 seed、名次、动作统计及逐回合 trace
与重构前完全一致。科学实现没有随工程拆分调整。

## 仍需关注

后期成长、反复移动/冻结和留牌选择值得用已保存动作证据进一步研究，不能直接认定为已证实的错误策略。
分布式在途采集轨迹尚未证明逐位恢复；固定 batch Adam/模型恢复已证明等价。
环境/共享资产通过身份校验和运行纪律冻结，尚非操作系统强制只读。
第三方参考源码 revision/授权范围待确认；Windows 沿用的 NumPy 2.4.0 为索引撤回版本。
历史缺协议的面板仍为 UNVERIFIED，不能与本轮混榜。
