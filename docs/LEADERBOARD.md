# HSAI 排行榜（E1 四模型校准）

状态：**当前（2026-11-11）**。E1 取代 `leaderboard-v2` 作为正式评测口径；
v2 保留为历史/过渡证据，不能与 E1 混榜。

- E1 入口：`config/eval-e1-four.json` + `workflows/eval_round_robin_elo.py`
- r1 报告：`artifacts/runs/e1-four-20261111/report/report.json`
- r2 报告（正式）：`artifacts/runs/e1-four-20261111-r2/report/report.json`
  （与 r1 六对逐对完全一致；r2 使用审计后冻结代码）
- 引擎：`37.0.2.256589` / `c4c71014e3d41760`

## 1. 规则（E1）

1. 四个模型（V1.1、12311、13806、final）做完整 round-robin：6 对 × **448 局/对**。
2. 每局 2 焦点模型 + 6 引擎 bot；`ordered56` 覆盖全部 56 个有序座位对，8 个完整周期。
3. 每对使用同一段 448 个 seed（common random numbers）；bootstrap 按 **seed block**
   联合重采样，不把 6 对当独立样本。
4. 用 Bradley-Terry MLE 拟合唯一 Elo；V1.1 为锚点 1000；5,000/10,000 次 game-level
   bootstrap 给 Elo CI 和排名 CI。
5. **冠军规则（用户 2026-11-11 明确）**：E1 只跑一轮；按点估计 Elo 第一名作为冠军/推荐模型。
   不跑第二轮“确认”，不做冠军挑战赛。排名 CI 只用于标注可信度，不改变冠军。
6. 只有 `valid=true` 且 `leaderboard_eligible=true` 的报告可入榜；0 illegal /
   0 truncated / 0 engine_bad；协议 SHA + `code_sha256` + 模型 SHA 全部核对。

## 2. 当前榜（r2 正式；r1 逐对一致）

| 排名 | 模型 | Elo | 95% CI | 排名 95% | 平均名次 | 前四率 | 冠军率 |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | `league-final-14858` | **1086.53** | [1062.49, 1111.01] | [1, 2] | **1.589** | 99.18% | 51.34% |
| 2 | `league-latest-13806` | 1068.45 | [1045.78, 1091.47] | [1, 3] | 1.644 | 99.11% | 49.48% |
| 3 | `league-latest-12311` | 1055.92 | [1032.16, 1079.91] | [2, 3] | 1.681 | 98.59% | 47.17% |
| 4 | `V1.1` | 1000.00 | [1000.00, 1000.00] | [4, 4] | 1.876 | 97.32% | 36.68% |

冠军：**`league-final-14858`**（E1 点估计第一；按用户规则直接作为冠军/下一轮起点）。
final 对 13806 为 240–208（53.57%，Wilson 95% CI 约 [48.9%, 58.2%]），不显著；
因此只能称“点估计冠军”，不能写成“已证明显著第一”。

## 3. 逐对结果（448 局/对）

| i | j | 结果（i 胜 / j 胜） | i 计分胜率 | 平均名次差 |
|---|---|---:|---:|---:|
| V1.1 | 12311 | 184 / 264 | 41.07% | +0.246 |
| V1.1 | 13806 | 181 / 267 | 40.40% | +0.277 |
| V1.1 | final | 173 / 275 | 38.62% | +0.375 |
| 12311 | 13806 | 211 / 237 | 47.10% | +0.089 |
| 12311 | final | 205 / 243 | 45.76% | +0.147 |
| 13806 | final | 208 / 240 | 46.43% | +0.071 |

## 4. 协议锚点

| 项 | SHA / 路径 |
|---|---|
| E1 r1 协议 | `acd7eff72745ebe7cc591957fb058fd4d0e1fc737c09b505aae05d36ff8d6cef` |
| E1 r2 协议（正式） | `0fbf28d2f82c08062291ef6526275507dfeffa1f5db85159ac7531437f3f723d` |
| E1 r1 报告 | `artifacts/runs/e1-four-20261111/report/report.json` |
| E1 r2 报告 | `artifacts/runs/e1-four-20261111-r2/report/report.json`（正式） |
| 正式榜 | `artifacts/models/e1-four-20261111/leaderboard.json` |
| final model | `ba7e8bbcf69f3f6142041369de3f9930242bae2de5c826c205c6d509b5dbd8e7` |
| 13806 model | `77d3fe54c56ead55bd45c80978dfe30160c2dfe02309c5966005ef92730b4f7c` |
| 12311 model | `5069aa62b8d20c57f2332fb284f5f57b639501dbd9721c3e2fefed41218d29cd` |
| V1.1 model | `68a8a58743617bef004fef556bd263a4f4939e22b5f5f70352dfa697abc0848e` |

## 5. 历史榜（不可混用）

- `artifacts/models/leaderboard-v2/leaderboard.json`：过渡榜，12311 点估计 1067.68，
  协议 `e4182351…`；与 E1 的 1055.92 不可直接比较。
- v2 的 `pairwise_elo` 矩阵由旧代码 `sigmoid(Elo差)` 生成，尺度错误；排名/Elo/CI 不受影响
  （CI 已用原始 pair counts 复核）。见 2026-11-11 审计 §A4。
- 旧 8 模型 `1099277`、4 模型 `1107182`、live-v1 11 模型都只作历史证据；不同引擎/协议
  不能混榜。

## 6. 实机选择

E1 点估计前二是 final 与 13806；两者统计上未分开。实机仍走 `live-v1` / 真机验收，
不能把 E1 第一直接当作 live pin。当前 live pin 仍是 V1.1；新模型上实机前先完成
`docs/sync/DECISIONS.md` D-026 的本地 obs/action 取证。
