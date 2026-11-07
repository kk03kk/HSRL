# 当前状态（2026-11-07）

HSAI 已从代码工作区扩充为代码、模型、数据、研究和服务器经验的统一工作区。旧 D:\HSBG、D:\HearthstoneAI、D:\HearthstoneRL 全程只读。没有访问 NUS、提交作业、打开游戏或操作游戏输入。

## 2026-11-07（深夜）：GPU 验收 1091647 性能事故审计、双模型评测结论、下一轮配方确认

**128 局双模型评测已在 NUS 完成**（作业 1092254，4m23s，`valid=true`，0 非法/0 截断/0 引擎异常，
协议 `3de3f809…`，runtime cuda12.8/Linux）。报告归档在 `artifacts/imports/nus-duel-eval-1092254/`。
7357 平均名次 1.727 [1.586,1.867]、冠军率 46.1%、**最终等级 P50=4（≥4 本 57.8%、≥5 本 7.0%、≥6 本 1.6%）**；
10004 为 1.867 [1.727,2.016]、36.7%、**P50=3（≥4 本 34.4%）**。配对 Δ=−0.141（95% CI [−0.359,+0.070]，
70:58 局）⇒ **棋力统计上不可分**；差异集中在等级曲线（7357 从 T14 起稳定 4 本，10004 到 T19 仍 P50=3）。
逐回合战团 P50 中前期几乎重合（T12 973 vs 922），T14 后样本降到 92→28 帧，不足以下结论。

**下一轮长训起点与配方（用户已确认）**：起点 **7357**（等级曲线更好 + 已核验完整 Trainer + 真实 lr
1.7548e-6→1.6667e-6 尚未触底；10004 的 trainer 已在地板 1.6667e-6，且其 261M 决策来自过强课程）。
对手库改为"稍弱于模型"的分级校准：**T1–T8 沿用旧值**，T9–T19 = round(ratio × 7357 自身 P50)，
ratio 依次 **0.70 / 0.80 / 0.88 / 0.95**（对应 0/40M/80M/120M），dev/test 取 **1.00**。
曲线由 `tools/derive_opponent_curve.py` 从评测报告生成（`config/opponent-p50-slightly-weaker.json`，
profile=`p50-v2-weaker`），旧库在 T12–T19 是模型的 **3.1–21.6 倍**、真实 bot 池的 8–50 倍。

**性能事故（用户报告）**：强制升本 GPU 验收 1091647 两轮 33m48s，基线 1085078 三轮 4m54s。
审计结论（每条都有可复现证据）：配方相对 `config/p50-7357-train.json` **只有 10 个已批准键不同**；
**战团库与 1085131 长训是同一个**（digest 逐项相同）；Slurm 资源完全相同；**强制升本机制实测
1.89 µs/决策**（`_obs_tier` 走观测快路径 0.53 µs，`player_json` 慢路径 213 µs 未触发）
⇒ 机制不是瓶颈。**已修**：`train_entity_sync` 曾把 `forced-teacher-coef` 合并进
`teacher_bc_coef`（会偷偷打开"在线缩放教师"的 CE 通路、两条 DAgger 标签共用同一字段），
现改为两个独立系数并在同时 >0 时拒绝启动。**新增门禁**：①配置漂移（只允许
`FORCED_ALLOWED_CONFIG_DIFF` 的键与快速基线不同，`--phase forced/forced-gpu` 与 runner 双侧生效）；
②吞吐（每轮墙钟 ≤300 s、采样 ≥3,000 dec/s，并写 `report.json.timings`；每轮 `--run-minutes 8`
把单轮最坏耗时封顶）。**尚未定位** 33m48s 的具体阶段：需要 1091647 的
`train/history.jsonl` 两条记录与 `.out` 计时行；下一轮验收会在 ≤3 轮 × 8 分钟内给出拆解并失败。

## 2026-11-07（夜）：双模型 + 6 bot 的 128 局配对评测已实现（本地验证，待 NUS 运行）

用户指定：用 7357 与 10004 在 NUS 跑一次评测，环境为当前模拟器（37.0.2.256589，**无修改**），
规则每回合 **80 动作 / 5 换位**（保留研究配方的重复换位去抖 2），**2 模型 + 6 个引擎启发式 bot**，
**128 局**，记录排名、酒馆等级、阵容战力与最佳局最终阵容。

- **为什么新写入口**：`hsbg/az/train/evaluate.py` 只支持 1 个 learner 座位；`personal.py evaluate` 把
  `--games` 当**每座位局数**（8×N），且 `valid` 绑定镜像库注入。本评测是 2 个学习座位同时在场 +
  6 个实时 bot、无库注入，故新增 `workflows/eval_two_models_six_bots.py`，**不改**引擎、训练栈、
  网络、奖励、PPO 与既有评测口径。
- **协议**：`config/eval-duel-7357-vs-10004.json`（A=`entity16m-nus-7357`，
  B=`entity16m-p50-10004-820m`，`games=128`、`seed=73000000`、greedy、`max_actions_per_turn=80`、
  `move_cap=5`、`move_repeat_limit=2`、`max_steps=50000`、`forced_tier_plan=""`）。座位**逐局交换**
  （偶 A=seat0 / 奇 A=seat1），两模型同 seed、同 bot 面板 ⇒ 名次差是配对样本。
  `protocol_sha256` 含引擎/词表指纹、`training_source_sha256`、评测入口 SHA、模型 SHA 与三个规则参数，
  任何改动都会变协议 ⇒ 不许与训练、旧 80/14、40 动作评测混榜。
- **规则实现**：复用 `MoveGuard(cap=5, repeat_limit=2)`；80 动作上限沿用 trainer 判据
  （`n_turn > limit and mask[0]`）。唯一偏差是 `hard_cap=False`——cap 后若引擎只给换位动作则退回原掩码
  继续并记 `move_cap_overflow`，不抛错中断整轮；该计数写进报告。
- **记录口径**：每个 (对局, 模型, 回合) **只取一帧**，在 `EndTurn` **之前**取（招募结束、战斗未结算），
  所以 `frames == games_at_turn`；`survivors` 并列。主参考 `P50(Σatk+Σhp)`，`atk_p50`/`hp_p50` 只作辅助
  （`tests/test_duel_eval.py` 用反例锁定不能相加）。另给逐回合等级 P50/≥4/≥5/≥6、金币 P50、达标回合、
  最终等级与战团分布、名次分布/前四率/冠军率/效用/对局级 bootstrap CI、座位拆分、配对 Δ 及 CI、
  最佳 5 局的完整最终战团（卡名/卡表等级/攻血/金色/关键词）与最终战力前 3 局、bot 面板参照。
- **门禁**：完成局数≠128、`illegal>0`、`truncated>0`、引擎诊断非零、名次越界、某模型无决策或名次序列
  不全 ⇒ `valid=false` + 非零退出（报告先落盘保留现场）。
- **NUS 入口**：`tools/nus_submit.py --phase eval`（纯标准库预检：协议未被改、两份权重字节数/SHA、
  不是同一份权重、输出目录非空即拒、部署文件齐备）→ `deploy/slurm/p50_eval.sbatch`
  （`gpu` + `gpu:h100-47:1`、16 CPU、64G、1h、`--exclude=xgpi15,xgpi17`）→
  `launch_nus_refinement.py --task eval-duel`（先留证再 exec）。权重放
  `artifacts/models/duel/{7357,10004}/model.pt`，用 `tools/stage_duel_models.py` 复制并核 SHA。
- **本地证据（只证明管线，不是棋力结论）**：`tests/test_duel_eval.py` 23 项、
  `tests/test_nus_submit_eval.py` 11 项全绿；2 局冒烟与 16 局 CPU 验证
  （`artifacts/runs/duel-local-validation`，116 s、8,307 决策、0 非法/0 截断/0 引擎异常）通过；
  16 局里 A 平均名次 1.50、B 1.88，配对 Δ 均值 −0.375（95% CI [−0.75, 0.125]）。
- **重要观察（需在 NUS 的 128 局上确认）**：引擎 bot 明显弱于两个模型（16 局里 bot 合并平均名次 5.5、
  模型前四率 100%、bot 常在 T16 前被清场），因此**前四率会饱和，主判据是配对 Δ**；这与 P50 库
  （人为抬高 T12–T15 目标）的强对手面板不是一回事。
- **文档**：`docs/EVAL_DUEL.md`（协议/口径/门禁/本地证据）、
  `docs/nus/EVAL_DUEL_DEPLOYMENT.md`（规则审计 + 完整命令序列）。**尚未在 NUS 运行**。
- **已知本地环境缺陷（与本轮无关，未修改）**：`tests/test_az_ppo.py::test_entry_script_help`
  在 Windows 上因 `subprocess.run(text=True)` 用 GBK 解码 UTF-8 `--help` 输出而失败；
  脚本本身 `--help` 正常（返回 0）。

## 2026-11-07（晚）：历史模型盘点完成，`library-v1` 交付

- 全量扫描 128 个 `.pt`（9.63 GB，不含 `.venv`），按**逐张量内容哈希**去重为 **67 个模型**；按文件 SHA 会得到 111，不能再按名字数模型。
- `artifacts/models/library-v1/`：`manifest.json` 总表、39 个可评测 `model.pt`、6 个主线完整 Trainer、单模型卡、`cohorts/`、`gaps.json`、`SHA256SUMS.txt`。模型文件是硬链接，不复制、不改源。
- 兼容分层：A 当前 37.0.2 卡表 15 个（8 主线 + 7 tiny）、B 旧卡表 21 个、C 工程冒烟 4 个、D 旧栈不可加载 27 个。旧卡表差异已定位为 **cost 约定**（旧表随从 cost=3，新表 0；3984 行中 2037 行差异在该列），文本表 3898/3984 行同序号一致，词表顺序未变。
- 核验确认：39 个可评测模型全部严格加载 + 3227/124→245 前向有限；主榜 8 个卡表逐字节一致；6 个主线 model/Trainer 配对 0 不一致；`s14-3702` 与 `s14-best` 学习参数逐位相同（仅换卡表 buffer）；7710 是独立权重（对 7357 L2=1.487），不是副本。
- **发现**：4159/5506/6039/7357/7710/10004 六个主线 Trainer 的 `anchor_model` 全部逐张量等于 `s14-3702`（对 7357 L2=9.002），anchor 从未前进；从 7357 续训必须 `reset-anchor 1`。forced-tier 冒烟已正确 anchor 到 7357。
- **缺口**：1085131 七个里程碑（588.9M–798.9M）只有 SHA 与 96 局指标，本地无权重；7616 无权重；648.9M 无 Trainer；原版 4159（255790）`c8e…` 只有迁移副本。`gaps.json` 给出补取 SHA。
- 报告 `docs/models/MODEL_LIBRARY.md`，库说明 `artifacts/models/library-v1/README.md`；复验 `python tools/verify_model_library.py`，打包 `python tools/export_model_library.py --cohort mainline-v1`。旧 `artifacts/models/registry.json`（96 条、缺 7357/7710/1085131）保留但不再作为评测入口。

## 2026-11-07（下午）：下一轮训练体系已实现并交付

用户批准三项决定（配方可改、训练场降压、强制升本）。本轮**已实现并本地验证**，不是方案讨论：

- **强制升本**（`docs/FORCED_TIER.md`、`hsbg/az/train/forced_tier.py`）：计划 `2:2,5:3,6:4,8:5`，T9+ 自由、六本不强制；判据只用引擎合法掩码的 Upgrade 位（不读/不补金币）；被强制的决策按守卫口径 `forced=True / old_logprob=0`，**不进策略梯度但保留价值链**；`carry=1` 顺延并如实记 `deferred`，超期记 `missed`，每目标都有归类。
- **逐步解除**：`forced-tier-schedule 0:1.0,40M:0.6,80M:0.3,120M:0.0` + `forced-teacher-until 120M`（DAgger 标签，每回合最多一条，coef 0.05）；120M 之后完全自主，长训后段本身即验收窗口。12h 预算下自主段约占 45–55%。
- **训练场**：从现成库 `train-start`（T12=1135）起步，按 0/40M/80M/120M 升到 ramp1/ramp2/train，**与解除强制的时间点对齐**——帮助越少、对手越强。不新建库。强制概率/teacher 截止/换库三条时间轴在启动期断言同界。
- **配方**（`config/p50-forced-tier.json`）：lr 量级不变——base 2e-6，续训实际 **1.7547666666e-6 缓降到 1.6667e-6（iteration 10000 触底）后恒定**（`lr_min_ratio/max_updates` 由 trainer state 决定，配置已改成与 state 一致并在启动时打印/告警）；`entropy-coef 0.002→0.008`、`target-kl 0.05`（曾试 0.03，因早停丢批回退）、`anchor-beta 0.02→0.01` 且 **`reset-anchor 1` 修正"anchor 指向更早旧策略"**（本地实测 L2=9.00 的缺陷已修）。minibatch/lr 量级不动，避免与课程改动混淆。
- **本地证据**：37 项单测通过；CPU 端到端短跑（真实 16.4M 模型，2 轮）T2/T5/T8 全部 on_time 4/4、T6 3/4+1 deferred（小饰品冲突，与 200 局扫描一致）、非法 0、截断 0；lockstep+教师标签通路 `teach_n` 正常。详见 `docs/research/FORCED_TIER_LOCAL.md`。
- **已知限制（写进文档）**：初始策略给 Upgrade 的概率 ≈1e-6（CE≈13 nat），`lr=1.75e-6` 下 6000 轮只能移动 ~1 nat，**教师 CE 补不平差距**；自主性依赖"稀有 on-policy 样本 + 高本价值已学会"的自我强化。若长训最后 100M 的 T8–T12 升本执行率 <0.05，下一轮改用定向机制（动作头单独大 lr / 反事实 DPO）。
- **部署**：`tools/nus_submit.py --phase forced`、`deploy/slurm/p50_forced.sbatch`、`workflows/run_forced_tier_long.py`、launcher task `p50-forced`、配置 `config/p50-forced-tier.json`。**尚未在 NUS 运行**，需用户按 `docs/FORCED_TIER.md` §部署 执行 CPU→GPU→长训三步验收。

## 2026-11-07：1085131 复盘完成（本轮结论，优先级最高）

- 归档 `artifacts/imports/nus-1085131-closeout` 已核：**105/105 文件 SHA256 与 MANIFEST 一致**，单包 tar.gz SHA 一致；`history.jsonl` 2,647 轮重算与服务器 `training-report.json` 逐字段一致（261,441,177 决策 / 261,066,811 样本 / 256,275 Adam 步 / 6 次 KL 早停 / 569 次强制 EndTurn / 22 次注入前物种偏差 / 1,355,264 局）。
- **结论：261M 决策没有产生可测的棋力变化。** 8 个里程碑对 7357 的配对名次差均值 +0.052（范围 −0.198…+0.240），前四率 8/8 为负（均值 −4.8pp），自身 T12 战团 P50 无趋势（929–1183）。评测分辨率 ≈0.34 名，"测不出"与"没改善"必须分开。
- **根因（已证实）**：本轮把对手库 T12–T15 目标从 1135/1528/1992/3044 抬到 **3000/6000/10000/13000**（2.6–4.6×），模型自身曲线不动，自身/对手比值掉到 **0.28/0.19/0.16**，第 12 回合起实战 0–6% 胜率；伤害只由对手棋盘决定（`combat.rs`），必败区里 `∂rank/∂θ≈0`。四阶段课程本身按规格执行，**是课程内容而不是课程机制的问题**。
- **"不愿升本"是理性而非缺陷**：升本价每回合 −1（`config.rs: upgrade_costs [5,7,8,11,11,10]`、`flow.rs` 每回合减 1），T10 只需 5 金、T13 只需 2 金，T10 有 95/96 局可负担却 0 次执行；回合末金币 P50=0、P90=1、从未 ≥8，即**从不跨回合留钱**，升本的期权每回合被自己花掉。
- **两个必须修的工程缺陷**：trainer 的 `anchor_model` **不是 7357**（本地实测 L2=9.00；v1 盘点已确认它逐张量等于 `s14-3702`）；`anchor_beta 0.1/250` 与 lr 调度因继承 b10m 的全局迭代号从未生效（全程序列 lr 只降 5% 后钉地板）。
- 详细复盘 `docs/research/LONG1085131_REVIEW.md`（含 6 张图），下一轮方案 `docs/research/NEXT_TRAINING_PLAN_1085131.md`，派生数据与脚本 `records/research/long1085131/`（`README.md` 有一键重算命令）。
- **下一轮建议起点仍是 7357**（与 AGENTS.md 批准一致），但需先修 anchor 与调度计数；训练场回落到现成库 `bot-p50-3702-v1/train-start`（**不必新建库**，四个库共用同一 `source_digest`，可直接当四级阶梯），主指标改为逐回合自身/场地比值 + 等级分布 + 升本执行率。
- **治理提醒**：方案里对探索（ε-mix/entropy）、`minibatch`、`value_coef`、`target_kl`、teacher BC 的改动**改变了 AGENTS.md 当前冻结的 PPO 配方**，必须先经用户批准并逐项 A/B（`NEXT_TRAINING_PLAN` §4，约 9 GPU·小时），不得直接进长训（本轮为 12h）。

## 先读这些

| 资料 | 内容 |
|---|---|
| `AGENTS.md` | 不可忽略的契约与工作规则 |
| `docs/nus/ENVIRONMENT.md` | 服务器限制、部署与安全恢复 |
| `docs/nus/INCIDENTS.md` | 历史故障、原因和已采用修复 |
| `docs/research/LONG1085131_REVIEW.md` | **最新**：1085131 长训复盘（为什么没有提高，含图表与证据分层） |
| `docs/models/MODEL_LIBRARY.md` | **最新**：全部历史模型盘点、去重、兼容分层与 NUS 测评入口 |
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
- 新NUS布局必须独立~/projects/HSAI clone、独立.venv和artifacts，旧HSRL只读。CPU全门禁/建库→一次GPU更新保存恢复换库验收→用户提交12h长训。入口tools/nus_submit.py会在排队前做轻量路径/来源检查；不伪造plan.json，不取消Slurm隔离、不用srun缩小双卡。详见docs/nus/NEXT_P50_DEPLOYMENT.md。所有NUS验收仍由用户操作，代理未访问服务器。
- Firestone20261106快照1000条feed/999回放，仍998条255790及2条254477，没有256589。主分析777份Train/17666板。强度主参考统一P50；晚期来源58/16/4/2，18/19目标是人为平滑设计，不宣称可靠真人分布。
- 实机链路仍固定离线验收6039，55条公开历史快照dry-run通过；未自动切7357，当前真实游戏未验收。客户端公开近似观测不等于模拟器完整状态。
- Git发布只含代码、配置、必要数据、当前文档和精选研究摘要；模型/库/大历史归档/私密配置全部本地保留并忽略。旧项目和游戏目录未修改；不能据此整目录删除旧环境或工具链。

## 模块关系

`engine` → 原生 Python 扩展 → `hsbg/env` → `hsbg/az` 网络/搜索/BC/PPO/采样 → `workflows` 实验入口 → `deploy` 计算节点执行；`config` 冻结配方；`artifacts/libraries` 提供对手；`hsbg/eval` 与评测入口写独立报告。`client` 是实机桥接，不能直接当成模拟器环境。`records` 只读证据不参与训练导入。

## 后续资源与边界

1080772的完整history、全部13阶段评测、7357完整状态及7710最后完整状态已经下载核验，不再重复要求下载这些文件。原始报告缺少512终评是用户安全暂停的结果，不代表下载丢失。

当前37.0.2原始Bot及完整库帧仍在NUS；本地只有该正式库元数据和本轮48局原生smoke。下一轮CPU任务直接只读复用服务器raw，正式P50新库在新目录生成，不必先把GB级原始数据传回Windows。若资源路径不同，在提交前设置文档列明的HSAI_SOURCE_7357/HSAI_SOURCE_RUN/HSAI_OLD_BANK并运行标准库预检。

原版255790的4159、6157/6160等额外历史完整状态依然是可选归档缺口，不阻断已完整核验的7357续训。新的正式库、下一轮评测和最终checkpoint待服务器实际生成后再独立打包归档。代理不访问NUS，下一步由用户手工CPU→GPU→长训，禁止跳过验收。
