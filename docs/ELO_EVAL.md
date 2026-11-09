# 8 模型 pairwise Elo 评测方案

状态：**runner 已实现、8 模型已选取并待 staging/audit、旧战团库已归档；正式作业等待用户启动确认。**  
目标：用两个焦点模型 + 六个共同 bot 的 8 人局，构建模型之间的 pairwise Elo 强度榜。

## 1. 定义

标准 Elo 是 pairwise 体系。我们的游戏是 8 人酒馆战棋，不能直接变成纯 2 人局；本方案把：

- 2 个焦点模型作为评分对象；
- 6 个引擎 bot 作为共同环境；
- 每局只比较两个焦点模型的最终名次；
- 28 个模型对做全轮转；
- 最后用 pairwise 结果拟合 Elo 风格分数。

因此这里是 **2 model + 6 bot 的 pairwise Elo**，不是“官方 2 人 Elo”。

## 2. 8 个模型

| # | 模型 | 角色 | SHA 前缀 |
|---|---|---|---|
| 1 | `m4159` | 历史基线 | `110e0330` |
| 2 | `m1080772-v7357` | 当前批准起点/主基线 | `60594db0` |
| 3 | `m1080772-v7710` | 1080772 上一轮末段 | `1f63d744` |
| 4 | `m1085131-v9786` | live-v1 点估计最佳 | `988fdc7f` |
| 5 | `m1085131-v10004` | 1085131 上一轮末段 | `32ec7a5c` |
| 6 | `m1095539-v7641` | 当前 run 30M | `52bcddbb` |
| 7 | `m1095539-v9595` | 当前 run 240M，p50 96 局最佳 | `caf0fe56` |
| 8 | `m1095539-v9691` | 当前 run final | `68a8a587` |

`m1095539-v7936`（60M）是已知灾难点，不入主榜；需要时可作为外部控制模型单独加跑。

## 3. 对局安排

- 8 个模型 => 28 个无序模型对。
- 每个模型对 **64 局**。
- 总场数：`28 × 64 = 1,792` 局。
- 每局 8 个座位：2 个焦点模型 + 6 个引擎 bot。
- bot 使用 `hsbg_engine::bot` / `game.bot_play`；不注入 P50、live-v1 或人类战团库。
- 不启用 forced tier、teacher、课程。
- greedy argmax，`sims=0`。
- 每回合 80 动作、5 次换位、重复换位去抖 2。

### 座位轮转

每个模型对 64 局：

- 前 32 局：A 坐 `k % 8`，B 坐 `(k + 1) % 8`；
- 后 32 局：A/B 角色互换；
- 每个焦点模型在每个座位各坐 8 局；
- 其余 6 个座位由 engine bot 填充；
- 每局记录 pair、焦点座位、模型角色和 seed。

目的：降低先手、座位顺序和固定 bot 面板带来的偏差。

## 4. Elo 计算

每局只看两个焦点模型的相对名次：

```text
rank_A < rank_B  -> A 胜
rank_A > rank_B  -> B 胜
rank_A = rank_B  -> 0.5
```

用 28 个模型对、每对 64 局的结果拟合 Bradley-Terry / Plackett-Luce pairwise 模型：

- 锚定 `7357 = 1500`；
- 换算到标准 Elo 标尺：`400 / ln(10) × 强度差`；
- 每局级 bootstrap 2000 次，输出 95% CI；
- 输出两两胜率矩阵 `P(i 排得比 j 高)`；
- CI 重叠时写“分不出”，不强行排名。

不使用简单 K=16 顺序更新；批量 pairwise 最大似然更稳定。

## 5. 记录与门禁

报告至少包含：

- 8 个模型的 SHA、架构、卡表、前向审计；
- 协议 SHA（模型列表、引擎、规则、seed、座位轮转、Elo 方法）；
- 每局 placement、焦点座位、seed；
- 每模型 448 局的名次分布、前四率、冠军率；
- Elo + bootstrap CI；
- 两两胜率矩阵；
- 逐回合 tier / P50 战团强度；
- 非法动作、截断、引擎异常；
- `valid` 门禁。

任何以下情况直接 `valid=false`：

- 非法动作 > 0；
- 截断局 > 0；
- 引擎异常；
- 某模型对局数不足 64；
- 8 个座位未坐满；
- 模型 SHA/架构/卡表不一致；
- 焦点座位轮转不完整。

## 5.1 启动前确认清单（2026-11-09 已准备）

| 项 | 确认值 |
|---|---|
| 配置 | `config/eval-elo-8.json`（config SHA256 `79e6592e...`）；冻结清单 `config/eval-elo-8.lock.json` |
| 引擎 | `37.0.2.256589`，源码指纹 `c4c71014e3d41760` |
| 模型 | 8 个，全部 `entity` / `16,436,034` 参数；文件 SHA 与 `artifacts/models/elo-8/MANIFEST.json` 一致 |
| 模型审计 | 8/8 `forward_ok=true`、`card_table_ok=true`（`artifacts/models/elo-8/AUDIT.json`） |
| 对局 | 28 对 × 64 局 = **1,792** 局；每模型 448 局 |
| 种子 | `88,000,000`–`88,001,791`，按 pair 连续分配 |
| 座位 | 每对前 32 局 A/B 座位错位，后 32 局交换；每模型每座位 8 局/对 |
| bot | 6 个 `hsbg_engine` 内置 bot；不注入任何旧战团库 |
| 规则 | greedy argmax，`sims=0`；80 动作/回合、5 换位/回合、重复换位去抖 2 |
| Elo | Bradley-Terry MLE；7357=1500；`l2=0.01`；2,000 次 game-level bootstrap；`400/ln10` |
| 协议 SHA | **`837231e0d2f6e583549b095581e60e2986ac899511872b6e0798091b4dc85e94`** |
| 正式入口 | `sbatch deploy/slurm/elo_8.sbatch` |
| 资源 | 1 × H100-47、32 CPU、96 GB、`gpu-long` 8 小时 |

正式作业尚未提交；该协议 SHA 只在 `config/eval-elo-8.json`、runner、模型 SHA、engine 指纹均不变时有效。

## 6. 实现入口

已实现：

```
workflows/eval_round_robin_elo.py   # 28 对、pairwise Elo、bootstrap、门禁、report.json/md
config/eval-elo-8.json              # 8 模型、64 局/对、seed 88000000、anchor 7357
tools/stage_elo_models.py           # 复制 8 份权重并核 SHA/bytes
deploy/slurm/elo_8.sbatch           # 正式：gpu-long、H100-47、8h
deploy/slurm/elo_8_smoke.sbatch     # 冒烟：1 对 × 2 局
tests/test_elo_eval.py              # 座位轮转、seed 区间、Elo 拟合
```

复用 `workflows/eval_two_models_six_bots.py` 的对局核心和 `DuelSeatPolicy`；只给 `run_games` 增加可选 `seats_for` 参数，默认仍为旧的 A/B 座位 0/1，旧协议不变。

## 7. 资源与状态

正式配置：`1 × H100-47`、32 CPU、96 GB、**8 小时**（gpu-long）；只评测不训练。
按 live-v1 的约 8 秒/局估算，1,792 局约 4 小时；8 小时为排队/模型审计/报告余量。

当前状态：

- 旧 `artifacts/libraries/*` 三个战团库已整体归档到
  `records/archive/libraries-2026-11-09/`，新评测不再使用任何旧库；
- 8 个模型已 staging 到 `artifacts/models/elo-8/models/`，文件 SHA/bytes 与配置一致；
- GPU/CPU 模型审计已完成：8/8 forward 与卡表一致；16 局单对冒烟 0 非法动作、0 截断、0 引擎异常；
- 尚未提交正式 1,792 局 Elo 作业；
- 正式启动命令：`sbatch deploy/slurm/elo_8.sbatch`；
- 启动前确认：`config/eval-elo-8.json` 的模型 SHA、engine `c4c71014e3d41760`、64 局/对。
