# HSAI 当前状态

状态：**as of 2026-11-17，现代化与正式评测完成；本轮模型能力研究完成，下一轮科学方案待决策**。
动态作业以 `python ops/hsai_ops.py state` 实测为准。历史在 [DECISIONS.md](DECISIONS.md)；
完整本轮报告在 [MODERNIZATION_20261117.md](analysis/MODERNIZATION_20261117.md)。
本文件原 18:27 快照保存在 Git 历史及本地 `records/archive/project-state-20261117-1827/`。

## 当前研究方向（D-080）

真实游戏自动化暂停，核心目标转为现役模型的中后期运营突破。唯一现役攻略为
[battlegrounds_strategy_guide.md](battlegrounds_strategy_guide.md)，33流派/84张不同核心卡，
内容SHA `f03150e215ab5a52d2e2f8f7cfd2d7e02b8aac3d00e8da8886257463860bbb5a`。
旧模型/旧库保留，不再作为主要研究对象；两端存储只做盘点和建议，本轮未迁移或删除。

[研究报告](analysis/MODEL_CAPABILITY_RESEARCH_20261117.md)已复核两端历史与全部190,884个正式学习席动作。
可见历史去重采样3,374,644,031次，不等于一个模型继承量；当前链采样账面约2.38B。
候选已在500/512局到达六本，但T9+铜须有合法购买机会221局、买入仅2局，罗杰斯91/0；
T12支出中位13金币。自动成长组合有效，主动经济循环很少出现。
CPU对照证明罗杰斯counter不同可产生完全相同的模型输入、却有不同下一步收益；其能力损失尚待消融量化。

推荐顺序：可见机制状态覆盖 → 模拟器验证的多步经济示范/学生状态补标签 → 整局PPO迁移；
先小型2×2消融，再考虑分级对手与信用分配。**均为建议，未修改冻结配方、模型输入或评测标准，未启动新训练/GPU作业。**
研究CPU账本/扫描/聚合/对账均已完成，原始证据未改。

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
- 补强掩码校验后的正式记录 CPU 复核 **1158282** COMPLETED / 0:0，11:29；未重新进行对局。
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

本次最终模型、完整 Trainer、启动/训练记录和环境清单另有 Windows 冷备份，11 个文件共
340,375,006 bytes，逐文件 SHA 复算通过。全部旧模型/库已作元数据盘点；旧格式缺登记不等于
资产损坏，也不表示已获现役兼容认证。完整库和正式动作原始证据仍以 NUS 封存资产为准。
