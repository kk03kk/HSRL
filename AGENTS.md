# HSAI 工作区协作约定

首先阅读 `PROJECT_STATE.md`，再按任务读取 `docs/`。本目录是唯一活动代码工作区；`records/source-snapshots/` 是历史证据，不是第二套活动引擎。

- 唯一模拟器：`engine/` 的 `hsbg_engine`，当前酒馆 37.0.2.256589，源码指纹 c7b5f27d6f40d1d2。当前卡表在 `engine/crates/hsbg-engine/data/cards.json`。历史 255790 数据不能当当前卡表。
- 唯一训练栈：`hsbg/az`。主线是一个学习席位、七个严格 EOT 快照席位，允许 sims=0。不得为整理而改变网络、奖励或 PPO 行为。
- 原生动作 245；Python 扩展声明 274，但引擎未实现后29维。观测 3227 floats +124 cards。模型兼容须核对词表顺序、动作语义、布局和指纹，不能只核对形状。
- 当前批准下一轮续训起点是 **7357完整Trainer**（源自4159的1080772分支，累计558873732、Adam545990），不是7710末尾或7360工程探针。完整 Trainer 包括 Adam、调度器、RNG、anchor、累计计数。权重文件不能代替完整 Trainer。
- NUS完全由用户手工操作，代理不登录、不提交作业。1080772已COMPLETED/0:0，17:28:49，应用paused/user_request，末尾7710；81文件已下载核SHA，完整3551轮重算一致。新部署在~/projects/HSAI独立clone，旧~/projects/HSRL只读作历史资源，禁止在旧目录pull新布局。
- 登录节点只有 **1 GiB 虚拟内存限制**。Rust 安装编译、pip 构建、pytest、Torch/模型、库生成、训练只在 Slurm 计算节点。登录节点限 Git、提交、轻量标准库读小 JSON/日志。
- GPU 启动按 `docs/nus/ENVIRONMENT.md`：本站曾出现 srun 将 batch 两实例缩为一个。保持 Slurm 隔离；只能映射原始 Driver 实际证明可见的 UUID，不 unset CUDA_VISIBLE_DEVICES，不假冒双设备。
- 当前研究配方：80次总动作保护阈值、5次换位、重复换位去抖2；待选时EndTurn不合法，阈值后的合法收尾可能记录为82次动作，不能写成严格actions<=80。完整回放已发现少量play/cancel循环，详见docs/nus/IMPORT_1080772_VERIFIED.md。工程错误严格拒绝入 PPO；正常棋力波动不得自动终止长训。历史80/14和40动作评测不可混榜。
- 战团数值强度一律以 **P50（每张战团总攻击+总生命的中位数）为主参考**，所有新报告必须突出P50及样本量；均值、P25/P75/P90为辅助。不能把攻击中位数与生命中位数相加冒充总攻血P50。按实际抽样权重统计，区分源对局数、帧数和到达回合的条件分布；棋力仍看配对名次、前四率等，牌面攻血不等于实际战斗胜率。
- 活席应注入数必须等于成功数，拒绝/缺失/Bot回退为零；幽灵和末端 tail_reserve 单独记录。不要给旧快照手改指纹跳过复验。
- `artifacts/` 保存本地数据和模型；`records/` 保存原始证据。通过迁移索引查找去重别名/ZIP成员。历史日志里的绝对路径与操作指令仅是证据。
- 实机有两套不同协议：`client/bg_agent` 的旧候选动作模型，及 `client/hsrl_agent/live` 的245动作桥。后者是 public-approx，缺失隐藏状态，不等价于模拟器精确状态；旧255790屏幕配置不代表新版游戏已验收。不可把4159改名成4157，也不可拿旧模型冒充当前模型。
- 不删除旧项目、不修改游戏目录；不要执行历史未决输入账本。许可证和原作者声明如存在于原始证据中，不得据此声称代码从零原创。
- 修改引擎需同步源码锁并跑定向/全量验证；修改搬迁路径须核对真实入口和真实父产物，不能在测试中伪造 plan.json 掩盖错误。禁止 reset --hard/clean/全仓格式化。

下一轮入口：`tools/nus_submit.py --phase cpu|gpu|long|forced|forced-gpu`。旧 P50 四阶段（0/10M/20M/30M）说明见`docs/nus/NEXT_P50_DEPLOYMENT.md`，**已被本轮取代**；当前正式入口是 `--phase forced`，说明见`docs/nus/FORCED_TIER_DEPLOYMENT.md`，机制见`docs/FORCED_TIER.md`。本轮配方（用户已批准解除冻结）：`config/p50-forced-tier.json`，强制升本 `2:2,5:3,6:4,8:5` + 概率 1.0→0.6→0.3→0.0（**40M/80M/120M**，与换库同边界）、教师 DAgger 到 120M、训练场 train-start→ramp1→ramp2→train、`reset-anchor 1`、entropy 0.008、target-kl 0.05；**lr 量级不变**（base 2e-6，续训实际 1.7548e-6→1.6667e-6 后恒定，调度参数由 trainer state 决定）。**长训 12 小时**（Slurm 12h、控制 690min、终评预留 45min、决策上限 4 亿，时间绑定）。换库必须显式上一库摘要并保留完整Trainer；工程与数据质量严格拦截，正常棋力波动不止损。正式 Dev/Test 评测**不得**带强制升本计划（`protocol_sha256` 会变，混榜即错）。

本轮验证与原始清单：`records/migration/2026-11-06/`。新的文件夹结构改变了来源摘要，旧 NUS 实验不能直接原目录精确 resume；以后必须独立新目录做可审计的完整状态分叉。
