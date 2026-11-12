# 真人数据资产清单（唯一口径）

状态：**2026-11-13 01:45 盘点**。任何训练/评测引用真人数据前先看这里，
不要凭路径名字猜强度。

来源：`data/real_library/firestone-20261112-1000/library.jsonl`（995 局，Firestone 解析，
`perspective=self` 带真实 MMR；`opponent` 行继承上传者 MMR，`mmr_source=game_uploader`）。
生成器：`tools/build_human_entry_bank.py`（战团库）、`tools/build_human_chain.py`
（带 `chain_turn/chain_strength/chain_cards` 的入场库）、`tools/build_bc_samples.py`
（动作样本）。

## 1. 战团库（对手 / 入场注入）

| 路径 | MMR 下界 | 行 | 局 | 用途 |
|---|---:|---:|---:|---|
| `firestone-bank-1000/train-m8k.jsonl` | 8000 | 1,168 | ~60 | 原来的**对手**库与入场库 |
| `firestone-bank-1000/eval-m8k.jsonl` | 8000 | 86 | — | 评测 split（与训练不相交） |
| `firestone-bank-6k/train-m6k.jsonl` | 6000 | **12,689** | ~995 | **新对手库**（D1 在用），多样性 16× |
| `firestone-bank-6k/eval-m6k.jsonl` | 6000 | 1,349 | — | 评测 split |

分回合 p50 对比（Σ=atk+hp，招募结束时）：

| turn | 8k | 6k | 6k/8k |
|---:|---:|---:|---:|
| 9 | 342 | 323 | 0.94 |
| 10 | 879 | 784 | 0.89 |
| 11 | 2232 | 1958 | 0.88 |
| 12 | 4983 | 4386 | 0.88 |
| 14 | 17555 | 14353 | 0.82 |

**6k 只比 8k 弱 12–18%，但局数是 16 倍** —— 对"教阵容形状"来说 6k 明显更划算；
对"卡强度档位"来说仍以 8k 为准（评测曲线用的是 8k 摘要）。

## 2. 入场库（带 self-chain，供 `entry-boards` 与 `board-target`）

| 路径 | MMR 下界 | 行 | 局 | 序列长度 | 用途 |
|---|---:|---:|---:|---|---|
| `firestone-bank-1000/train-m8k-selfchain.jsonl` | 8000 | 696 | **56** | 9–16 | 现役入场库（所有 E/G/H/B/C/D 臂） |
| `firestone-bank-6k/train-m6k-selfchain.jsonl` | 6000 | **8,088** | **668** | 5–17 | **新入场库**，局数 11.6×，尚未启用 |

⚠️ `tools/build_human_chain.py` 的 `--min-mmr` 是 2026-11-13 才加的；
**不传就是全部 995 局**（含 3k MMR），等于把弱阵容当专家教学。已修，并在 summary 里
记 `min_mmr` / `dropped_mmr_below_floor`。

## 3. 动作样本库（BC / DPO / board-problems）

`data/firestone_bc/bc_samples.jsonl.gz`：297,884 条（`tools/report_bc_bank.py` 实测）

| MMR | 条数 | 占比 | 局 |
|---|---:|---:|---:|
| 0–6k | 77,904 | 26.2% | — |
| 6–7k | 142,015 | 47.7% | — |
| 7–8k | 54,529 | 18.3% | — |
| **≥8k** | **23,436** | 7.9% | **60** |
| ≥10k | 8,012 | 2.7% | — |

晚局（`turn>=8 & tier>=4`）占 82.6%，动作以 `play_minion`(27%) / `sell`(18%) /
`play_spell`(17%) / `reroll`(14%) 为主。**≥8k 只有 60 局**，只适合当小系数正则
（`bc-coef 0.05`，C1 在用）。

## 4. 不要再用

`humancurve-*` 系列库是**假强度**：把我们自己快照的 atk/hp 重标到真人分位数，
卡牌构成仍是我们的弱阵容，且参考分布没有 MMR 下限（≈0.59× 的 ≥8k p50）。
它的产物不能当"真人级对手"，也不能当验收依据。
