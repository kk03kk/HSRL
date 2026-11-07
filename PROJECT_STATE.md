# 当前状态（2026-11-07）

HSAI 已从代码工作区扩充为代码、模型、数据、研究和服务器经验的统一工作区。旧 D:\HSBG、D:\HearthstoneAI、D:\HearthstoneRL 全程只读。没有访问 NUS、提交作业、打开游戏或操作游戏输入。

## 2026-11-07（下午）：下一轮训练体系已实现并交付

用户批准三项决定（配方可改、训练场降压、强制升本）。本轮**已实现并本地验证**，不是方案讨论：

- **强制升本**（`docs/FORCED_TIER.md`、`hsbg/az/train/forced_tier.py`）：计划 `2:2,5:3,6:4,8:5`，T9+ 自由、六本不强制；判据只用引擎合法掩码的 Upgrade 位（不读/不补金币）；被强制的决策按守卫口径 `forced=True / old_logprob=0`，**不进策略梯度但保留价值链**；`carry=1` 顺延并如实记 `deferred`，超期记 `missed`，每目标都有归类。
- **逐步解除**：`forced-tier-schedule 0:1.0,90M:0.6,180M:0.3,260M:0.0` + `forced-teacher-untile 260M`（DAgger 标签，每回合最多一条，coef 0.05）；260M 之后完全自主，长训后段本身即验收窗口。
- **训练场**：从现成库 `train-start`（T12=1135）起步，按 0/90M/180M/260M 升到 ramp1/ramp2/train，**与解除强制的时间点对齐**——帮助越少、对手越强。不新建库。
- **配方**（`config/p50-forced-tier.json`）：lr 恒定（`lr-min-ratio 1.0`，去掉继承来的地板钉死；量级不变，抬 lr 留作 E3 A/B）、`entropy-coef 0.002→0.008`、`target-kl 0.05→0.03`、`anchor-beta 0.02→0.01` 且 **`reset-anchor 1` 修正"anchor 指向更早旧策略"**（本地实测 L2=9.00 的缺陷已修）。minibatch/lr 量级不动，避免与课程改动混淆。
- **本地证据**：37 项单测通过；CPU 端到端短跑（真实 16.4M 模型，2 轮）T2/T5/T8 全部 on_time 4/4、T6 3/4+1 deferred（小饰品冲突，与 200 局扫描一致）、非法 0、截断 0；lockstep+教师标签通路 `teach_n` 正常。详见 `docs/research/FORCED_TIER_LOCAL.md`。
- **已知限制（写进文档）**：初始策略给 Upgrade 的概率 ≈1e-6（CE≈13 nat），`lr=1.75e-6` 下 6000 轮只能移动 ~1 nat，**教师 CE 补不平差距**；自主性依赖"稀有 on-policy 样本 + 高本价值已学会"的自我强化。若长训最后 100M 的 T8–T12 升本执行率 <0.05，下一轮改用定向机制（动作头单独大 lr / 反事实 DPO）。
- **部署**：`tools/nus_submit.py --phase forced`、`deploy/slurm/p50_forced.sbatch`、`workflows/run_forced_tier_long.py`、launcher task `p50-forced`、配置 `config/p50-forced-tier.json`。**尚未在 NUS 运行**，需用户按 `docs/FORCED_TIER.md` §部署 执行 CPU→GPU→长训三步验收。

## 2026-11-07：1085131 复盘完成（本轮结论，优先级最高）

- 归档 `artifacts/imports/nus-1085131-closeout` 已核：**105/105 文件 SHA256 与 MANIFEST 一致**，单包 tar.gz SHA 一致；`history.jsonl` 2,647 轮重算与服务器 `training-report.json` 逐字段一致（261,441,177 决策 / 261,066,811 样本 / 256,275 Adam 步 / 6 次 KL 早停 / 569 次强制 EndTurn / 22 次注入前物种偏差 / 1,355,264 局）。
- **结论：261M 决策没有产生可测的棋力变化。** 8 个里程碑对 7357 的配对名次差均值 +0.052（范围 −0.198…+0.240），前四率 8/8 为负（均值 −4.8pp），自身 T12 战团 P50 无趋势（929–1183）。评测分辨率 ≈0.34 名，"测不出"与"没改善"必须分开。
- **根因（已证实）**：本轮把对手库 T12–T15 目标从 1135/1528/1992/3044 抬到 **3000/6000/10000/13000**（2.6–4.6×），模型自身曲线不动，自身/对手比值掉到 **0.28/0.19/0.16**，第 12 回合起实战 0–6% 胜率；伤害只由对手棋盘决定（`combat.rs`），必败区里 `∂rank/∂θ≈0`。四阶段课程本身按规格执行，**是课程内容而不是课程机制的问题**。
- **"不愿升本"是理性而非缺陷**：升本价每回合 −1（`config.rs: upgrade_costs [5,7,8,11,11,10]`、`flow.rs` 每回合减 1），T10 只需 5 金、T13 只需 2 金，T10 有 95/96 局可负担却 0 次执行；回合末金币 P50=0、P90=1、从未 ≥8，即**从不跨回合留钱**，升本的期权每回合被自己花掉。
- **两个必须修的工程缺陷**：trainer 的 `anchor_model` **不是 7357**（本地实测 L2=9.00，是更早的旧策略）；`anchor_beta 0.1/250` 与 lr 调度因继承 b10m 的全局迭代号从未生效（全程序列 lr 只降 5% 后钉地板）。
- 详细复盘 `docs/research/LONG1085131_REVIEW.md`（含 6 张图），下一轮方案 `docs/research/NEXT_TRAINING_PLAN_1085131.md`，派生数据与脚本 `records/research/long1085131/`（`README.md` 有一键重算命令）。
- **下一轮建议起点仍是 7357**（与 AGENTS.md 批准一致），但需先修 anchor 与调度计数；训练场回落到现成库 `bot-p50-3702-v1/train-start`（**不必新建库**，四个库共用同一 `source_digest`，可直接当四级阶梯），主指标改为逐回合自身/场地比值 + 等级分布 + 升本执行率。
- **治理提醒**：方案里对探索（ε-mix/entropy）、`minibatch`、`value_coef`、`target_kl`、teacher BC 的改动**改变了 AGENTS.md 当前冻结的 PPO 配方**，必须先经用户批准并逐项 A/B（`NEXT_TRAINING_PLAN` §4，约 9 GPU·小时），不得直接进 24h 长训。

## 先读这些

| 资料 | 内容 |
|---|---|
| `AGENTS.md` | 不可忽略的契约与工作规则 |
| `docs/nus/ENVIRONMENT.md` | 服务器限制、部署与安全恢复 |
| `docs/nus/INCIDENTS.md` | 历史故障、原因和已采用修复 |
| `docs/research/LONG1085131_REVIEW.md` | **最新**：1085131 长训复盘（为什么没有提高，含图表与证据分层） |
| `docs/research/NEXT_TRAINING_PLAN_1085131.md` | **最新**：下一轮方案、5 组低成本对照实验（E0–E4，约 9 个训练臂）、实施路线与风险 |
| `docs/research/TRAINING_HISTORY.md` | 模型演进、实验结果和当前训练口径 |
| `docs/client/STATUS.md` | 实机模型、协议及兼容缺口 |
| `records/migration/2026-11-06/summary.json` | 实际迁移/校验统计 |
| `records/migration/2026-11-06/cleanup.md` | 按来源逐项核对的清理建议 |

## 当前主线与边界

- 1080772已安全暂停，COMPLETED/0:0、17:28:49。末尾7710累计599740461；本轮新增400920544，399364028有效样本、391796 Adam步，153087357/153087357严格活席注入、异常/回退0。71次注入前已有物种偏差（11批观察）根因未定位，不称整个模拟器无Bug。
- 81文件单包已下载逐一核SHA、完整3551轮history与服务器报告精确重算一致。归档在artifacts/imports/nus-1080772-closeout/bundles/bundle-1084568/extracted；派生摘要在records/research/long1080772，详细复盘见docs/research/LONG1080772_REVIEW.md。
- **用户批准下一轮7357完整Trainer**，累计558873732、16.44M参数、Adam545990、实际lr1.7547666666e-6；已核SHA、卡表/Adam/调度器/RNG/CPU前向。真实网络合成小批次更新/保存/恢复/更新通过且源文件不变。探针7359不作为起点。7357与7710规范资源在artifacts/models/nus-1080772。
- 下一轮P50库按config/opponent-p50-next.planned.json制作；本地48局Bot源和六套smoke库1469派生帧通过。正式源复用服务器同引擎原始Bot，缺口才补采。第10回合起至少5怪；目标四阶段0/10M/20M/30M；晚回合source_turn/tail_reserve诚实计数，尾部总量限目标0.25–3倍。正式NUS库尚未制作，不把smoke当正式库。
- 保留原PPO/奖励/网络/80动作保护/5换位/去抖2；当前主要经营问题是T10–12低本刷新、后期样本稀少，暂无数值崩溃或熵塌缩证据。下一轮96起点/每30M96/正常收尾512局详细配对评测，所有报告突出P50与到达样本量。正常棋力波动不停止，Test留候选确认。
- 新NUS布局必须独立~/projects/HSAI clone、独立.venv和artifacts，旧HSRL只读。CPU全门禁/建库→一次GPU更新保存恢复换库验收→用户提交24h长训。入口tools/nus_submit.py会在排队前做轻量路径/来源检查；不伪造plan.json，不取消Slurm隔离、不用srun缩小双卡。详见docs/nus/NEXT_P50_DEPLOYMENT.md。所有NUS验收仍由用户操作，代理未访问服务器。
- Firestone20261106快照1000条feed/999回放，仍998条255790及2条254477，没有256589。主分析777份Train/17666板。强度主参考统一P50；晚期来源58/16/4/2，18/19目标是人为平滑设计，不宣称可靠真人分布。
- 实机链路仍固定离线验收6039，55条公开历史快照dry-run通过；未自动切7357，当前真实游戏未验收。客户端公开近似观测不等于模拟器完整状态。
- Git发布只含代码、配置、必要数据、当前文档和精选研究摘要；模型/库/大历史归档/私密配置全部本地保留并忽略。旧项目和游戏目录未修改；不能据此整目录删除旧环境或工具链。

## 模块关系

`engine` → 原生 Python 扩展 → `hsbg/env` → `hsbg/az` 网络/搜索/BC/PPO/采样 → `workflows` 实验入口 → `deploy` 计算节点执行；`config` 冻结配方；`artifacts/libraries` 提供对手；`hsbg/eval` 与评测入口写独立报告。`client` 是实机桥接，不能直接当成模拟器环境。`records` 只读证据不参与训练导入。

## 后续资源与边界

1080772的完整history、全部13阶段评测、7357完整状态及7710最后完整状态已经下载核验，不再重复要求下载这些文件。原始报告缺少512终评是用户安全暂停的结果，不代表下载丢失。

当前37.0.2原始Bot及完整库帧仍在NUS；本地只有该正式库元数据和本轮48局原生smoke。下一轮CPU任务直接只读复用服务器raw，正式P50新库在新目录生成，不必先把GB级原始数据传回Windows。若资源路径不同，在提交前设置文档列明的HSAI_SOURCE_7357/HSAI_SOURCE_RUN/HSAI_OLD_BANK并运行标准库预检。

原版255790的4159、6157/6160等额外历史完整状态依然是可选归档缺口，不阻断已完整核验的7357续训。新的正式库、下一轮评测和最终checkpoint待服务器实际生成后再独立打包归档。代理不访问NUS，下一步由用户手工CPU→GPU→长训，禁止跳过验收。
