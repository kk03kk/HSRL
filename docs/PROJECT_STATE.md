# HSAI 当前状态

状态：**as of 2026-11-17 18:27 +08:00**（本次工程审计实测；只写"现在是什么"，历史在
[`DECISIONS.md`](DECISIONS.md)，失效结论清单在 [`analysis/AUDIT_20261117.md`](analysis/AUDIT_20261117.md)）。
服务器实时状态用 `python ops/hsai_ops.py state` 取，**不抄本文件**。

> **本文件的三条硬规则（2026-11-17 新增）**
> 1. 任何带作业号/时间戳的段落必须写 `(as of <date>)`，否则视为历史。
> 2. **模型名/冠军/库名只在这里写一次**，且必须与 `DECISIONS.md` 最新条一致；
>    本文与决议冲突时**以决议为准**。
> 3. 参数真值不写在这里，看 `deploy/slurm/stage_arena.sbatch` 与最新 `D-###`。

---

## 1. 一句话

在**校准战团场**上做 PPO（1 学习席 + 7 席严格 EOT 快照）。2026-11-17 审计发现两个
风险：**(a)** 历史面板显示现役训练场在 T12–T16 约为模型的 **2.2–2.6 倍**，可能削弱奖励分辨力；
但“没有梯度”尚无充分证据，不能由平均名次接近 4.5 推出；**(b)** 历史模型逐回合板面增长
落后于 winner-biased 真人参照，提示后期运营不足，但仍需控制共同存活样本与协议差异。
历史分析与建议见
[`analysis/AUDIT_20261117.md`](analysis/AUDIT_20261117.md) §5 与
[`analysis/TRAINING_OPPONENT_DESIGN_20261117.md`](analysis/TRAINING_OPPONENT_DESIGN_20261117.md)。

---

## 2. 目标与研发模式

见 [`CHARTER.md`](CHARTER.md)。2026-11-17 用户口径补充：
- **不设单一判胜指标**。主看**逐回合 M1 与增长曲线**（尤其看中后期会不会滚雪球），
  名次/前四/吃鸡只作参考。
- 但**评测数据必须准确、可复现、可比较**（协议指纹、按 seed 配对、有效性门、共同存活子集）。
- 任何 GPU 作业仍需用户批准；训练配方与场地强度的变更由用户决定。

---

## 3. 最近一次完整对照（同一场地、同 seed、1028 局、4 个模型）

场地 `arena-cal-v14hold-snap`（held-out，与训练库帧签名交集 0），入口
`artifacts/runs/gap-v1.2-plus3-1024-20261117/`（服务器）。**名次只作参考。**

| 模型 | mean_rank | 前四 | 吃鸡 | T12 M1 | T16 M1 |
|---|---:|---:|---:|---:|---:|
| v1.2（本轮起点） | 4.477 | 49.5% | 7.2% | 1,311 | 4,692 |
| v1.3_r1 | 4.615 | 44.2% | 12.3% | 1,348 | 5,634 |
| v1.3_r2 | 4.580 | 44.8% | 14.9% | 1,422 | 5,521 |
| v1.3_r3 | 4.549 | 46.7% | 14.2% | 1,376 | 5,669 |

**怎么读这张表（重要）**：
- 8 人局随机名次期望 4.5；1028 局下 `mean_rank` 的标准误 ≈0.071 ⇒ 4.549 与 4.5 **无法区分**。
  前四 46.7% 略低于 50%；这是描述性比较，**不能据此证明奖励无分辨力或梯度为零**。
- 三轮训练的真实变化在**后期板面**：T16 +21%、吃鸡 7.2%→14.2%；而 T12 只 +5%。
- 该面板**没有** `protocol_sha256`（旧代码产物）⇒ 按 `EVAL_PROTOCOL` §4 规则 5，
  它是 **UNVERIFIED**：只能当"同一轮内多模型对照"，**不能与任何新结果混榜**。

---

## 4. 已确认的事实（不要重复验证）

1. **库不泄漏**：`train-joint.jsonl`（1,783 局）与 `eval-joint.jsonl`（199 局）game_id 交集 **0**；
   `arena-cal-v14`（16,658 帧）与 `v14hold`（2,355 帧）帧签名交集 **0**（2026-11-17 实查）。
2. **镜像注入真的生效**：训练侧 `mirror_applied/attempt = 45,870/45,878 = 99.98%`；
   评测侧按 `(seed, turn, seat)` 派生 RNG，跨模型可配对。
3. **工程洁净**：v1.3 三轮全程 `illegal_actions / invalid_games / truncated_games = 0`。
4. **模型后期结构缺口**（T12，v1.3_r3 vs 真人 winner-biased 基线）：金卡/局 **0.97 vs 2.55**、
   有金卡局占比 61% vs 90%、每格 atk **123 vs 598**、留存随从每回合增重 **+87 vs +690**。
5. **动作空转**：T12 每回合 argmax 行为 `refresh 5.44 / move 5.02 / freeze 1.91`，
   move 与 freeze **都打满上限**（训练日志 `move_guard_blocked_cap` 每轮 142 万次）。
6. **强制升本课程已改为纯教师 BC**：`prob=0`（不强制）+ 教师标签（每回合最多 1 条，
   `--forced-teacher-coef 0.1`）。模型确实在自升：T4 里程碑 `self=418 / on_time=94`。
7. **工程面（已修）**：主评测入口的 `UnboundLocalError`、多模型落盘覆盖、协议指纹缺失、
   按下标配对、`pytest tests` 不可收集、`z_global` 死代码、ckpt 架构开关不可加载——
   见 [`analysis/AUDIT_20261117.md`](analysis/AUDIT_20261117.md) §2。

---

## 5. 当前资产（名字只写这一处）

| 类别 | 位置 | 说明 |
|---|---|---|
| 引擎 | `engine/`（`hsbg_engine` 37.0.2.256589，指纹 `c4c71014e3d41760`） | 245 动作；观测 3227 floats + 124 cards |
| 训练器 | `workflows/train_entity_sync.py` | entity 16,436,034 参数；续训用 `--trainer-state`（**无** `--resume`） |
| 训练入口 | `deploy/slurm/stage_arena.sbatch` | 含 MIG UUID 重定向 + 自动降级 |
| 建库入口 | `deploy/slurm/build_arena_snap.sbatch` | 纯 CPU；校准库 + 快照库同源 |
| 评测入口 | `deploy/slurm/eval_shard_parallel.sbatch`（8 分片）/ `eval_node_job.sbatch` | 现役 1024 局；**现在会写 `protocol_sha256`** |
| 判胜工具 | `tools/compare_panels.py` | 逐回合 M1/M2 + 按 **seed** 配对 bootstrap + **共同存活子集** M1 |
| 合并工具 | `tools/merge_panels.py`（新布局）/ `merge_panels_multi.py`（旧分片） | 有效性计数跨分片求和、协议校验、缺分片拒绝 |
| 健康门 | `hsbg/az/train/health_gate.py` | 硬门 fail-closed；熵/KL/吞吐只警告 |
| 训练场 | `artifacts/libraries/arena-cal-v14-snap` | 校准目标与统计在源库 `artifacts/libraries/arena-cal-v14/summary.json`，snap 目录只有快照与 manifest |
| 考试场 | `artifacts/libraries/arena-cal-v14hold-snap` | held-out，只判胜 |
| 真人库 | `data/real_library/firestone-37.0.2-joint/{train,eval}-joint.jsonl` | 1,783 / 199 局（服务器） |
| **当前候选模型** | `artifacts/models/arena-v1.3-r3/`（2026-11-17 12:55 发布，`status=candidate`） | `model.pt` sha `1808f178…`，v626 / 64,308,028 决策 / 16,436,034 参数；= v1.3 三轮链产物（作业 1153480）。**未宣布为冠军** |
| 候选模型的评测 | [`analysis/EVAL_V13_R3_1024_20261117.md`](analysis/EVAL_V13_R3_1024_20261117.md) | 1032 局 held-out；6 本停留 0.45→**10.03**；T15/T16 M1 **+29.5%/+20.8%**；吃鸡 7.2%→14.2%；名次 Δ+0.073（CI 跨 0）。**无 `protocol_sha256`（legacy 面板），不得与新榜混比** |

---

## 6. 运行与审计状态（as of 2026-11-17 18:27 +08:00）

**D-076 已授权并启动训练**：作业 `1156986` 在 `xgpi14` 运行，16:09 起，165 分钟单段，
Slurm 截止 19:09。产物 `artifacts/runs/nus-arena-v1.4-r1-20261117/`。
18:27 实测 619 轮 / 57,799,833 决策，非法/无效/截断/引擎停止/注入错误/丢弃决策均 0；
27 次动作预算强制结束回合不是整局截断。此为观测快照，不保证未来状态。

**工程审计修复在本地隔离分支** `codex/engineering-audit-20261117`，未部署至服务器。
服务器 HEAD `8d6d823`，保留原有 5 个未跟踪第三方参考文件；因此不能宣称两端 IN SYNC。
完整问题、验证、未解决风险与后续部署条件见
[`analysis/ENGINEERING_AUDIT_20261117.md`](analysis/ENGINEERING_AUDIT_20261117.md)、D-077。

先前“没有排队/运行作业”的文字已被 D-076 取代。历史训练建议仍是建议，不构成新任务授权；
本次审计没有提交新 GPU 作业、改变奖励/模型/对手/超参或模型 pin，也没有中断当前训练。

---

## 7. 硬约束

见 [`../AGENTS.md`](../AGENTS.md)：远端脚本纯 ASCII、登录节点不跑重活、新作业新目录、失败保现场、
不删资产、密钥不进对话。**CVD 唯一允许的例外**：`stage_arena.sbatch` 的 MIG UUID 重定向（D-063）。
