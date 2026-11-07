# HSAI 酒馆战棋模型库 v1（2026-11-07）

本报告盘点 `D:\HSAI` 本地全部酒馆战棋 AI 权重，完成逐文件核验、按权重内容去重、按引擎兼容性分层，并生成可直接用于 NUS 统一测评的标准化模型库：

```
artifacts/models/library-v1/
  manifest.json          机器可读总表（67 个模型、去重、谱系、评测历史、缺口）
  models/<id>/model.pt   硬链接到原始权重（不复制、不新增占用）
  models/<id>/trainer.pt 仅主线 6 个完整 Trainer
  models/<id>/model.json 单模型卡片
  cohorts/cohorts.json   评测分组与禁止同榜说明
  index.csv              一行一个模型，便于表格/脚本读取
  gaps.json              已知缺口及补取 SHA
  SHA256SUMS.txt         45 个权重的文件级 SHA256
```

重算入口：

```powershell
python workflows/build_model_library.py          # 重建 manifest/硬链接/cohort
python tools/verify_model_library.py             # 逐模型复验（SHA+加载+前向+卡表）
python tools/export_model_library.py --cohort mainline-v1 --out artifacts/exports/mainline-v1.tar.gz
```

本次核验没有修改任何原始模型、Trainer、库或引擎；模型库里的 `model.pt` 全是硬链接，删库不会删源文件。

---

## 1. 结论摘要

| 项目 | 数量 | 说明 |
|---|---:|---|
| 扫描 `.pt` 文件 | 128 | 9.63 GB（表观大小，不含 `.venv`） |
| 可读权重文件 | 127 | 1 个 `__MACOSX/._model.pt`（163 B AppleDouble 垃圾）已排除 |
| **去重后模型** | **67** | 按逐张量内容哈希（含卡表 buffer），不是按文件名 |
| 可直接评测（eval-ready） | 39 | 当前 `hsbg.az` 可加载、3227+124→245 前向有限 |
| 其中严格 A 类（37.0.2 卡表） | 15 | 8 个主线 + 7 个早期 tiny |
| B 类（旧卡表、可加载） | 21 | 19×14.8M + 14.9M champion + s14-best |
| C 类（工程冒烟/测试） | 4 | 3 个 forced-tier 冒烟 + 1 个测试夹具，禁止进棋力榜 |
| D 类（旧栈，当前不可加载） | 27 | AutoTavern `entities.*` 与旧客户端 1.24M |
| 主线完整 Trainer | 6 | 4159/5506/6039/7357/7710/10004 |
| 归并掉的多余文件 | 61 | 同权重 `model.pt`/`trainer.pt`、副本、硬链接别名 |

**最重要的三条：**

1. **全部 NUS 主线 Trainer 的 `anchor_model` 都是 s14-3702，不是上一检查点。** 4159、5506、6039、7357、7710、10004 六个 Trainer 内嵌的 anchor 逐张量与 `entity16m-s14-3702` 完全相同（相对 L2=0）；对 7357 而言 anchor 与自身 L2=9.002。下一轮从 7357 续训必须启用 `reset-anchor 1`；本地 forced-tier 冒烟的 anchor 已正确指向 7357，说明修复有效。
2. **早期 14.8M/14.9M/16.4M(s14-best) 的“卡表漂移”是一个特征约定变化，不是词表重排。** 旧表与新表 3984 行中 3898 行文本哈希在同一行号上完全相同；差异集中在第 13 列 cost：旧表 2124 个随从 cost=3，新表 2879 个随从 cost=0。因此旧模型可自洽运行，但只能进 B 类榜，不能与 A 类混榜。
3. **P50 课程 8 个被评测的里程碑里，本地只有 1 个权重；最终 10004 反而没有 96 局棋力评测。** 7 个里程碑（588.9M–798.9M）只有 SHA 与指标，`gaps.json` 已列出补取 SHA 与目录。

---

## 2. 方法

对每个文件：

1. **文件级 SHA256**：`weights-sha256.json`，用于下载/归档完整性。
2. **权重级 SHA256**：按 `state_dict` 排序后的 key + 张量原始 float 字节计算，用于去重、判断 `model.pt` 与 `trainer.pt` 是否同一模型、判断 anchor 身份。
3. **加载与前向**：用项目自身 `hsbg.az.train.model.load_checkpoint` 严格加载；再用真实的 `3227 floats + 124 cards + 245 mask` 做一次 CPU 前向，要求输出 `(1,245)` 且全部有限。
4. **卡表核对**：把 checkpoint 内嵌的 `card_static`/`card_text` 与当前引擎 `card_static()`/`load_card_tables()` 逐元素比较，记录差异行数、最大绝对差和列分布。
5. **谱系核对**：对同架构 16.4M 模型做参数级 L2（排除卡表 buffer）与相对 L2，确认父子关系而不是按文件名推断。
6. **Trainer 核对**：比较 Trainer 内嵌 `model` 与同名 `model.pt`；把 `anchor_model` 与全部已知模型做逐张量匹配，匹配不到再算最近邻。

原始证据在 `artifacts/local/inventory/`：`weight-scan.json`、`canonical-hashes.json`、`verify-az.json`、`lineage.json`。这些是本地派生数据，不是训练输入。

---

## 3. 模型谱系

```
                         (BC 500k / 3753 局)
                                  │
                    entity14m-original 14.8M (旧卡表, cost=3)
                                  │
              ┌───────────────────┴────────────────────┐
              │  AutoTavern 14.8M 期间                 │
              │  baseline5m / main5m×4 / bot50m×6      │
              │  champion-c0-1975 (14.9M, +seq_head)   │
              └───────────────────┬────────────────────┘
                                  │ 加入 seq_head + card_head → 16.4M
                         s14-best (pre-3702 卡表)
                                  │  仅替换卡表 buffer（学习参数逐位相同）
                         s14-3702  ←──────────────────── 所有主线 Trainer 的 anchor
                                  │  +198.8M 决策
                              4159 (198,819,917)
                                  │  +150.0M
                              5506 (348,831,055)
                                  │  +60.0M
                              6039 (408,863,650)   ← 实机 245 桥当前白名单
                                  │  +150.0M
                              7357 (558,873,732)   ← 批准的续训起点
                         ┌────────┴─────────┐
                    +40.9M                +90.0M (P50 课程分叉)
                      7710                 8255/648.9M
                  (599,740,461)                │ +171.4M
                         │                 10004 (820,314,909)
                         └──── 两条支线没有共同后续 ────┘
```

L2 距离（排除卡表 buffer，相对 L2 在括号内）：

| 边 | L2 | 相对 L2 | 解读 |
|---|---:|---:|---|
| s14-best → s14-3702 | 0.000 | 0.0000 | 只替换卡表 buffer，学习参数未变 |
| s14-3702 → 4159 | 5.243 | 1.13% | 198.8M 决策的累计漂移 |
| 4159 → 5506 | 3.668 | 0.79% | |
| 5506 → 6039 | 1.808 | 0.39% | |
| 6039 → 7357 | 2.889 | 0.62% | |
| 7357 → 7710 | 1.487 | 0.32% | 7710 是独立权重，不是 7357 副本 |
| 7357 → 8255 (648.9M) | 2.357 | 0.51% | P50 分叉 |
| 8255 → 10004 | 2.717 | 0.58% | 再加 171.4M 决策 |

这条链与 `PROJECT_STATE.md`/`TRAINING_HISTORY.md` 的版本号、累计决策数逐项一致；7357 的 SHA `60594db0…` 与 1085131 `initial_model` 逐字节相同。7710 的 SHA `1f63d744…` 与 7357 不同，**没有**把 7357 冒充成 7710。

---

## 4. 标准化模型库

### 4.1 ID 与状态

- `production`：当前实机白名单（仅 6039）。
- `approved-resume`：AGENTS.md 批准的下一轮起点（仅 7357）。
- `candidate`：可用于评测/对比的正式历史模型。
- `reference`：谱系参照（s14-3702、s14-best），不计入“训练轮次”。
- `historical`：早期架构，可加载但只作历史对照。
- `local-smoke` / `test-fixture`：工程产物，禁止进棋力榜。
- `archive-only`：旧栈，当前无法加载。

### 4.2 兼容分层（写进 manifest，不允许口头解释）

| League | 判据 | 能否同榜 |
|---|---|---|
| `A-current-3702` | `hsbg.az` 可加载；245 前向；内嵌卡表与当前 37.0.2 **逐字节一致** | 可以，必须用同一 `protocol_sha256` |
| `B-legacy-table` | 可加载；245 前向；内嵌 pre-3702 卡表（cost=3 约定） | 只能与 B 类同榜，报告必须标 `table=legacy` |
| `C-engineering` | 可加载但只是冒烟/测试 | 不评测 |
| `D-archive` | AutoTavern `entities.*` 或旧客户端 | 当前栈不能评测 |

`leaderboard_eligible=true` 只覆盖 A/B 且非 smoke/test；是否真的能同榜还取决于评测协议，见 §6。

---

## 5. 主榜：A 类主线 8 个模型

| ID | 版本 | 累计决策 | model.pt SHA256（前 12） | Trainer | 历史评测（协议） | 均名次 / 前四 / 夺冠 |
|---|---:|---:|---|---|---|---|
| `entity16m-s14-3702` | — | — | `fb0911c8cfd9` | 无 | — | — |
| `entity16m-nus-4159` | 4159 | 198,819,917 | `110e03305bdb` | 有 | 1080772 Dev96 (`e51924…`) | 4.385 / 62.5% / 1.04% |
| `entity16m-nus-5506` | 5506 | 348,831,055 | `3ebbc28b6cd3` | 有 | 1080772 Dev96 (`e51924…`) | 3.781 / 81.25% / 4.17% |
| `entity16m-nus-6039` | 6039 | 408,863,650 | `eb91fd41fa99` | 有 | 1080772 Dev96 (`e51924…`) | 3.833 / 70.83% / 5.21% |
| `entity16m-nus-7357` | 7357 | 558,873,732 | `60594db00a62` | 有 | 1080772 Dev96 (`e51924…`) / P50 Dev96 (`8a02ce…`) | 3.667 / — / — ；4.469 / 59.38% / 1.04% |
| `entity16m-nus-7710` | 7710 | 599,740,461 | `1f63d7446b4e` | 有 | 无 | — |
| `entity16m-p50-8255-648m` | 8255 | 648,873,732 | `b60231fb73e6` | **无** | P50 Dev96 (`8a02ce…`) | 4.271 / 58.33% / 1.04% |
| `entity16m-p50-10004-820m` | 10004 | 820,314,909 | `32ec7a5c249f` | 有 | 无（只有 health） | — |

**禁止**把 4159/5506/6039 的 4.385/3.781/3.833 与 7357/8255 的 4.469/4.271 放在同一张排行榜：前者是 `bot-firestone-3702-economy-v2-live-v1/dev` + `e51924…`，后者是 `bot-p50-3702-v1/dev`（`d493ef78…`）+ `8a02ceb4…`。同一模型 7357 在两个协议下分别是 3.667 和 4.469，差 0.80 名，就是协议差。

主榜 8 个模型已全部通过：文件 SHA、权重 SHA、严格加载、245 前向、卡表逐字节一致。6 个 Trainer 的 SHA 也全部与清单一致。

### 5.1 完整 Trainer 清单

| ID | updates | Adam 步（provenance） | 实际 lr | anchor | 备注 |
|---|---:|---:|---:|---|---|
| `entity16m-nus-4159` | 4159 | 194,178 | 1.86137e-6 | s14-3702 | 迁移自原版 4159 |
| `entity16m-nus-5506` | 5506 | 340,270 | 1.81647e-6 | s14-3702 | |
| `entity16m-nus-6039` | 6039 | 399,159 | 1.79870e-6 | s14-3702 | |
| `entity16m-nus-7357` | 7357 | 545,990 | 1.75477e-6 | s14-3702 | 批准起点 |
| `entity16m-nus-7710` | 7710 | — | — | s14-3702 | |
| `entity16m-p50-10004-820m` | 10004 | — | — | s14-3702 | 1085131 终点 |

4159/5506/6039 的 Adam 步和 lr 来自各自 `provenance.json`；7357 来自 `records/research/long1080772/trainer7357.json`。7710/10004 的完整审计未在本次重算，manifest 只记录 Trainer SHA 与 anchor 身份，不编造 Adam 步。

---

## 6. P50 课程（1085131）的实际情况

固定 Dev 协议（`8a02ceb4…`，库 `d493ef78…`，96 局 / 12 独立 seed，greedy，move_cap 5）：

| 累计决策 | 版本 | 本地权重 | 均名次 | 前四 | 夺冠 | 对 7357 配对名次差 |
|---:|---:|---|---:|---:|---:|---|
| 558.9M（7357 基线） | 7357 | 有 | 4.469 | 59.4% | 1.0% | — |
| 588.9M | 7638 | **缺** | 4.583 | 56.3% | 0.0% | +0.115 [−0.260,+0.448] |
| 618.9M | 7945 | **缺** | 4.615 | 52.1% | 1.0% | +0.146 [−0.135,+0.427] |
| 648.9M | 8255 | 有 | 4.271 | 58.3% | 1.0% | −0.198 [−0.615,+0.271] |
| 678.9M | 8561 | **缺** | 4.531 | 54.2% | 2.1% | +0.063 [−0.260,+0.417] |
| 708.9M | 8865 | **缺** | 4.708 | 49.0% | 0.0% | +0.240 [−0.073,+0.552] |
| 738.9M | 9171 | **缺** | 4.542 | 55.2% | 3.1% | +0.073 [−0.281,+0.458] |
| 768.9M | 9479 | **缺** | 4.427 | 57.3% | 0.0% | −0.042 [−0.292,+0.219] |
| 798.9M | 9786 | **缺** | 4.490 | 54.2% | 1.0% | +0.021 [−0.219,+0.260] |
| 820.3M | 10004 | 有 | 未评测 | — | — | — |

结论沿用 `docs/research/LONG1085131_REVIEW.md`：8 个里程碑对 7357 的配对名次差均值 +0.052，前四率 8/8 为负（均值 −4.8pp），没有可测的棋力提升。本库只负责“哪些权重真的存在、能否评测”，不重复论证训练结论。

---

## 7. 早期模型（B 类 21 个 + tiny A 类 7 个）

### 7.1 14.8M entity（旧卡表）

19 个独立权重，来自 AutoTavern 的 5M/B50M/审计/恢复实验：

- `entity14m-original`：BC 起点（`trainer_updates=201`，`bc_games=3753`），SHA `116eb3df…`。
- `entity14m-baseline5m`、`entity14m-hsbrsim-resume`、`entity14m-friend-ckpt-5023397`、`entity14m-audit-bf16`、`entity14m-audit-fp32`。
- bot50m 快照 7 个（`decisions-10.0M … 45.0M`）+ provisional best。
- main5m 快照 4 个（`decisions-1.04M … 4.02M`）；其中一部分是同一 run 的不同步数，名字已保留在 `model.json` 的 aliases 里。
- `entity14m-champion-policy`：champion 的 policy 导出。

这些模型都能被当前 `hsbg.az` 加载并输出 245 动作；卡表是旧约定，归 B 类。

### 7.2 14.9M champion

`entity14m9-champion-c0-1975`：14.8M 骨干 + `seq_head`（245×385=94,325 参数），比 14.8M 多一个动作序列辅助头；旧卡表。它与 `entity14m-champion-policy` 不是同一权重：前者 14,902,579 参数，后者 14,808,254。

### 7.3 tiny MLP（0.96M，7 个）

`tiny-gpu1079842-seed`、`tiny-gpu1079842-exact`（与 repeat 同权重）、`tiny-opponent-controller-final`、`tiny-opponent-controller-step15`、`tiny-scaling-step24`、`tiny-scaling-step25`、`tiny-scaling-final`。

结构是 3227→256→256→{245 policy, 8 rank}，不编码 cards；属于 AZ 早期 smoke/控制器实验。可评测，但不具备与 16.4M entity 同级的表示能力，建议只作基线。

---

## 8. 归档层（D 类，27 个）

当前 `hsbg.az` **不能**加载，原因是它们的 state_dict 使用 AutoTavern 的 `entities.*` 命名与不同模块结构：

| 家族 | 数量 | 参数 | 来源 |
|---|---:|---:|---|
| `autotavern-entity-18.7m` | 6 | 18,740,981 | entity18m pilot、library-life-5m、proof-final |
| `autotavern-2.3m` | 8 | 2,302,069 | nus-first-model、recovery-1024784 |
| `autotavern-1.05m` | 3 | 1,048,885 | library-stage pilot/proof |
| `autotavern-0.65m` | 3 | 652,341 | paper-stage resume-proof |
| `autotavern-0.27m` | 5 | 269,175 | local-delivery 各阶段 |
| `legacy-client-1.24m` | 2 | 1,241,354 | 旧候选动作策略 + turn-search value 模型（3840 卡词表） |

这些不是“坏文件”，而是旧代码栈的合法历史产物。它们已被 `artifacts/archives/**` 的 ZIP/RAR 覆盖，且当前没有训练/评测路径依赖；保留为只读档案。**不要把 1.24M 旧客户端策略接到 245 动作桥上**：`docs/client/STATUS.md` 已说明两套协议不同。

---

## 9. 去重结果与别名

按权重内容去重后 67 个模型；以下是最容易误判的几组：

| 权重 SHA（前 12） | 文件数 | 说明 |
|---|---:|---|
| `9c7539263c58` | 4 | 7357 的 `model.pt`/`trainer.pt` + closeout bundle 两份别名 |
| `5698588162fa` / `f9f20633ecee` / `e872d32a1a71` | 各 4 | 5506/6039/4159 的 model+trainer ×2 份副本 |
| `a0f03edc4db9` | 4 | 7710 model+trainer ×2 |
| `7dad8115e7d1` | 3 | s14-best 与 publish-representative：**文件 SHA 不同、权重完全相同** |
| `61e1e7e24ce4` | 6 | forced-tier serial 7358 与 smoke2 的全部别名 |
| `fa346628cfa1` | 2 | gpu1079842 exact 与 repeat：文件 SHA 不同、权重相同 |
| `ddf80198d1ff` | 4 | library-life-5m 与 entity18m 的 bc-anchor/best 同权重 |

因此“128 个文件”不等于“128 个模型”；按文件 SHA 去重会得到 111，按权重内容去重才是 67。`manifest.json` 的 `files[]` 保留每个别名及其文件 SHA，`weights_sha256` 才是模型身份。

另外扫描了 `artifacts/archives/` 的 ZIP/TAR：55 个 `.pt` 成员中，8 个大小/路径没有同名 loose 副本的成员已逐一解出并计算权重哈希，全部等于已有 payload（`policy_localpair.pt`→1.24M value、recovery `best/latest`→2.3M baseline、main5m `decisions-5022890`→baseline5m、`selected-model`/`general-reference`→14.8M original、bot50m `selected-bot-model`/`decisions-25022392`→provisional best）；其余成员都有同名同尺寸 loose 副本，另有两个 163 B AppleDouble 垃圾。**没有只存在于压缩包、未进入 manifest 的独立模型。** `.rar`（s14-best）按原包保留，权重已有 loose 副本。

---

## 10. 核验发现（需要写进下一轮操作）

### 10.1 anchor 从未前进（最高优先级）

六个主线 Trainer 的 `anchor_model` 全部逐张量等于 `entity16m-s14-3702`（`fb0911c8…`），包括 4159 自己、5506、6039、7357、7710、10004。7357 的 anchor 与 7357 自身 L2=9.002（相对 1.93%），与 1085131 复盘中的 L2=9.00 完全吻合。

影响：`anchor_beta=0.1/250` 的“锚定上一检查点”设计从未生效，全程都在把策略拉向 s14-3702；这不是“anchor 太强”，而是“anchor 指错对象”。本地 forced-tier 冒烟使用 `reset-anchor 1` 后，anchor 正确等于 7357（逐张量相同），证明修复路径有效。

操作要求：从 7357 续训必须保留 `reset-anchor 1`；任何“沿用旧 Trainer 但不重置 anchor”的 resume 都会复现该缺陷。

### 10.2 卡表差异是 cost 约定，不是词表重排

旧表（`card_static` SHA 前缀 `dacb0c85`）与当前表（`8ea4fd0f`）：

- 3984 行中 2195 行有差异；其中 **2037 行差异在第 13 列 cost**：旧表 2124 行 cost=3、当前表 2879 行 cost=0（随从 cost 约定）。
- 文本表 3898/3984 行在同一行号完全一致，只有 86 行文本变化。
- 因此词表顺序未变（若重排，同序号文本一致行应接近 0），旧模型可以自洽运行；但“随从 cost=3”是旧特征约定，归 B 类。

不建议为了“统一”而把旧模型卡表替换成当前表：那是**另一个策略函数**，会改变被评测对象。若确实需要，`workflows/migrate_checkpoint_3702.py` 可生成显式迁移副本，并保留 `migration.json` 记录；迁移副本必须作为新 ID，不能覆盖原模型。

### 10.3 model/trainer 配对全部一致

6 个主线 Trainer（4159/5506/6039/7357/7710/10004）内嵌 `model` 的权重哈希与各自 `model.pt` 完全相同（0 个不一致）；1085131 的 10004 也一致。其余 43 个 Trainer（旧 AutoTavern / tiny / publish）只做了 SHA/anchor 记录，它们不是当前主线可加载模型，未纳入这条结论。

### 10.4 7710 与 s14-best 的身份

- 7710 与 7357 文件 SHA 不同、L2=1.487，是独立权重；不能把 7357 的结果写成 7710。
- s14-best 与 publish-representative 权重完全相同（canonical `7dad…`），只是容器字节不同；s14-3702 是它们仅替换卡表 buffer 后的迁移副本。

---

## 11. 缺口（`gaps.json` 已机器可读）

| 缺口 | 影响 | 处理 |
|---|---|---|
| 1085131 七个里程碑权重（588.9M/618.9M/678.9M/708.9M/738.9M/768.9M/798.9M） | 无法在统一协议下重测这 7 个点，只能用历史 96 局指标 | 按 `gaps.json` 的 SHA 从 NUS 原 `milestones/decisions-*` 补取；本地不伪造 |
| 7616（1080772） | 只有对 7357 的配对差 +0.156，权重缺失 | 同上；不要用 7710 冒充 |
| 1080772 中间 milestone 228.9M/258.9M/288.8M（v4433/4703/4972） | 有 checkpoint.json 与 SHA，本地无权重 | 按 `gaps.json` 的 `members[].model_sha256` 从 NUS 补取；318.8M–588.8M 只有诊断，不补造 SHA |
| 648.9M 的 `trainer.pt` | 该点不能精确续训 | 补取 `6c0a8a34…`；否则从 10004 续 |
| 原版 4159（255790）完整状态 `c8e…` | 不能研究“原版→3702 迁移”的完整差异 | 可选归档；现有 4159 是 3702 迁移副本，不能冒充原版 |
| 6157/6160、B100k/B10M | 历史分支不可复现 | 可选归档，不阻断 7357 主线 |
| 27 个 AutoTavern/旧客户端模型 | 当前栈不能加载 | 保留只读档案；若要复现，需在旧栈单独评测，不能与现代模型同榜 |

---

## 12. 下一轮 NUS 统一测评建议

1. **先锁定一个协议再开跑。** 推荐 `bot-p50-3702-v1/dev`（`d493ef78…`）+ 96 局/12 seed + greedy + sims=0 + 80 动作保护 + **不带强制升本计划**（`protocol_sha256` 会变）。A 类 8 个模型全部用同一协议重跑；不要引用历史 e519/8a02 成绩当新榜。
2. **只把 `cohorts/mainline-v1` 放进主榜。** 8 个模型：s14-3702、4159、5506、6039、7357、7710、8255、10004。其中 7710/10004 从未有 96 局评测，正是本轮最值得补的两个点。
3. **B 类单独成榜并标 `table=legacy`。** 早期 14.8M/14.9M/s14-best 与 A 类不可同榜；tiny 虽属 A 类（不编码 cards），也只作基线。若要研究“架构/规模”而不是“训练轮次”，可在 B 类内部比较。
4. **打包与上传用 `tools/export_model_library.py`。** 只带 mainline cohort，包含模型、model.json、过滤后的 manifest、SHA256SUMS 和 cohorts；到 NUS 后先跑 `tools/verify_model_library.py`，再提交评测。
5. **续训边界不变。** 从 7357 完整 Trainer 起，`reset-anchor 1`；不要用 7710 末尾或 7359 探针。648.9M 没有 Trainer，不能作为精确 resume 点。
6. **P50 口径不变。** 评测报告仍以“每张战团总攻击+总生命的中位数（P50）+ 样本量”为主参考，均值/P25/P75/P90 为辅；牌面 P50 不等于战斗胜率，棋力仍看配对名次、前四率。

---

## 13. 复现命令

```powershell
# 1) 文件级 SHA：本轮用 PowerShell Get-FileHash 一次性生成
#    结果在 artifacts/local/inventory/weights-sha256.json
# 2) 逐文件结构扫描
python artifacts/local/inventory/scan_weights.py
# 3) 当前栈加载/前向/卡表核对
python artifacts/local/inventory/verify_az.py
# 4) 谱系 L2 与 anchor 审计
python artifacts/local/inventory/lineage.py
# 5) 权重级去重哈希
python artifacts/local/inventory/canon_all.py
# 6) 生成模型库
python workflows/build_model_library.py
# 7) 复验模型库（推荐每次部署前执行）
python tools/verify_model_library.py --cohort mainline-v1
```

`artifacts/local/inventory/` 下的中间 JSON 是本轮派生证据，可重算；`artifacts/models/library-v1/manifest.json` 是交付物。旧的 `artifacts/models/registry.json`（96 条、缺少 7357/7710/1085131）没有被删除，但不再作为评测入口；新库的 `manifest.json` 是唯一评测清单。
