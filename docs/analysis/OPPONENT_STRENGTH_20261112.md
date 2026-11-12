# 对手强度天花板：诊断、干预与验证（2026-11-12）

状态：**进行中**。第 1 次干预已上线（对手库换成真人曲线校准版），训练实验在跑。
诊断依据见 [`HUMAN_GAP_20261112.md`](HUMAN_GAP_20261112.md)。

---

## 1. 根因（实测，不是推测）

上一轮把"模型后期弱"量化成：**5 个模型在 T12 的战力都只有真人胜者的 0.20–0.23 倍，
而多跑 7.4 亿决策只换来 +15%**。这一轮继续往下挖，从最近一次真实训练运行
（`artifacts/runs/stage2-gpu-smoke-20261112-005739/train/history.jsonl` 最后一条）里
直接读到根因：

| 字段 | 值 | 含义 |
|---|---|---|
| `mirror_frames` | **8,872** | 对手板面全部来自**我们自己的** EOT 快照库 |
| `mirror_applied` | 49,741 / `mirror_attempt` 49,741 | 镜像注入 100% 生效，无缺帧 |
| `human_board_p` | **0.0** | **真人强度对手从未启用过** |
| `ppo.config.rank_utility_coef` | 1.0 | 奖励 = 终局名次 |
| `ppo.config.terminal_strength_coef` | **0.0** | 没有任何"变强"的奖励项 |
| `ppo.config.shaping_coef` | **0.0** | 势函数塑形关闭 |

**结论**：训练环境是"我们自己的模型当对手 + 只看名次"。既然模型用 1/4.5 的
真人战力就能稳赢（top4 98–100%、冠军 73–85%），把 T12 战力从 1100 提到 5000
**在奖励上零回报**。梯度指向"别变强"。这是自洽的次优均衡，不是训练不足。

---

## 2. 干预：把对手的后期强度拉到真人水平

### 2.1 工具链（此前全是死代码）

服务器上早就有这套机器，但**从未跑通过**：

| 组件 | 状态 |
|---|---|
| `workflows/build_human_curve_bank.py`（179 行） | 有 `build()`，**无 CLI、无调用方** |
| `workflows/prepare_validated_library.py` | 逐帧过 Rust 引擎验证并写 `validation` |
| `tools/verify_human_curve_bank.py` | **本轮新写**：校验生成的库是否真的落在真人曲线上 |

本轮补上的环节：

1. 给 `build_human_curve_bank.py` 加 CLI（`--sources/--human/--out/--split/--quota`），
   并把真人数据过滤成 train-only 视图（`references()` 拒绝非 train 行）。
2. 发现**整条链断在这里**：`require_calibrated_library` 要求
   `validation.accepted > 0 且 quarantined == 0`，而 `build_human_curve_bank` 与
   `scale_library` 产出的 manifest **都没有 `validation` 字段** → 库会被
   `--mirror-strict 1` 直接拒绝。补上 `deploy/slurm/validate_library.sbatch` 串起
   `prepare_validated_library`。

### 2.2 做法

拿一个已验证的引擎快照库，把每一帧的攻击/生命**按比例重分配**，使该回合总攻血落在
真人 Train 分布的指定分位上；牌面构成与实体引用不变。逐回合还乘一个后期缓和系数
`coefficient(turn)`（T≤7 为 1.0，T12 为 0.8，T≥14 为 0.7）。

### 2.3 产物

`artifacts/libraries/humancurve-20261112/train`
（来源：`league-eot-v1.1-20261109/train` + `league-src-7357-20261109/train`，quota=128）

- **6,144 帧**，24 个回合，256 帧/（回合,座位）键
- 引擎逐帧验证：**6,144 accepted / 0 quarantined**
- `require_calibrated_library` → **ACCEPTED**（`driver=human_curve_full_snapshots`）

### 2.4 验证：库真的变强了吗

`tools/verify_human_curve_bank.py`，同一单位 `Σ(atk+hp)`：

| 回合 | 缓和系数 | 新库 p50 | 真人 p50 | 比值 | 目标比值 | 判定 | 原库 p50 | **增强** |
|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 6 | 1.00 | 64 | 74 | 0.86 | 1.00 | OK | 86 | 0.7× |
| 8 | 0.90 | 134 | 179 | 0.75 | 0.90 | OK | 217 | 0.6× |
| 10 | 0.90 | 597 | 879 | 0.68 | 0.90 | OK | 591 | 1.0× |
| 12 | 0.80 | **2,751** | 4,983 | 0.55 | 0.80 | **OFF** | 949 | **2.9×** |
| 14 | 0.70 | **8,347** | 17,555 | 0.48 | 0.70 | **OFF** | 1,305 | **6.4×** |

**结论（诚实版）**：

- 干预**确实生效**：T12 强度 ×2.9，T14 ×6.4。原库 T12 是 949，模型 1,135，
  所以旧环境下模型本来就比对手强 20%；新库下模型只有对手 p25 的 0.86 倍。
- 但**没打到自己的目标**：T12 想要 0.80 只到 0.55，T14 想要 0.70 只到 0.48。
  原因待查（候选：分位映射与实际帧分布不一致、缓和系数重复施加、
  `min_games=8` 让晚期参考被最近可靠回合替代）。
- T6/T8 反而**低于**原库（0.7×/0.6×），因为真人冠军的早期板面本来就不如我们模型。
  这对训练无害，但意味着"前期变简单了"。

**因此这一轮训练用的是"已经显著更强但尚未达到名义目标"的对手**，不是满配。

---

## 3. 训练实验设计

`config/league-humancurve.json`，相对 `league-stage2.json` 只改三处：

| 键 | stage2 | humancurve | 理由 |
|---|---|---|---|
| 对手库 | `league-eot-v1.1` | `humancurve-20261112` | 本干预本身 |
| `mirror-seats` | 7 | **4** | 模型 T12 只有真人 0.22 倍；7 个座位全换真人级，学习者每局垫底、名次奖励恒定、PPO 拿不到梯度 |
| `mirror-strict` | 1 | **0** | 允许其余 3 个座位回落到 bot，保住胜负可分性 |
| `mirror-sample` | 55/60/70/75/80/75 | **25/30/35/40/40/40** | 新库整体更硬，取低分位让第一轮难度仍可赢 |

`mirror-seats` / `mirror-p` 原本在 `personal.py train` 里**硬编码为 7 / 1**，配置无法
表达"部分注入"，本轮加入 `OPTIONS` 白名单（原有的"后出现的配置键覆盖先前命令行 flag"
机制即可生效）。

启动方式（不动 48h 脚本，用命令行覆盖分区与时限）：

```bash
sbatch --partition=gpu --time=01:30:00 \
  --export=ALL,HSRL_CONFIG=config/league-humancurve.json,\
HSRL_LEAGUE_BANK=artifacts/libraries/humancurve-20261112,\
HSRL_SEGMENTS=2,HSRL_SEGMENT_MINUTES=30,\
HSAI_LEAGUE_SOURCE=artifacts/models/league-final-20261111-step014858,\
HSAI_LEAGUE_OUT=artifacts/runs/league-humancurve-1h-20261112 \
  deploy/slurm/league_stage2_48h.sbatch
```

从 `final-14858` 分叉，2 段 × 30 分钟。

### 验收标准（事先定好，避免事后找理由）

用**同一个** `eval_human_gap` 面板（192 局、同 seed、同对手）对比 fork 前后：

1. **T12 战力 p50** 必须从 1,135 上升（这是干预的直接目标）；
2. 不能以牺牲前期为代价：T6/T8 不得下降超过 15%；
3. top4 / 冠军率在引擎 bot 面板上不得显著恶化（说明没有把基础能力练坏）；
4. 若 T12 上升但 T14 不动，说明只学会了"中期变厚"而没学会后期爆发，需要换招。

---

## 4. 未解决的问题（下一步）

1. **校准未达目标分位**（§2.4）——先定位原因再谈满配对手。
2. **真人数据是胜者偏差**（816/995 第 1 名）——对手强度上限偏高，
   需要用 dev/test 留出面板防止过拟合到"冠军局"。
3. **奖励仍只有名次**。对手变强后名次信号会变，但"变强"本身仍无直接奖励；
   若第一轮证明对手强度是瓶颈，下一步才是加 `terminal_strength_coef` 或
   流派辅助头（`archetype_bias` 现成、默认关闭）。
4. **阵容掌握度**：`tools/analyze_composition_coverage.py` 已就绪，
   待模型侧 `final_boards` 面板产出后与"真人赢家 60.4% 含 S 级核心卡"对照。
