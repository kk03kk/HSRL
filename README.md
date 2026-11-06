# HSAI：酒馆战棋强化学习与实机适配

基于 AutoTavern 的 Rust 模拟器与 HSBRSIM 的训练栈，开展个人 BC/PPO 训练、固定战团库评测和真实游戏适配。`D:\HSAI` 是唯一活动代码及本地资源工作区；不是一套从零原创的模拟器。上游来源与必要差异见 [engine/FORK.md](engine/FORK.md) 和源码锁。

**当前状态：1080772在17小时28分安全暂停，完整归档已核验；下一轮从7357完整Trainer开始，采用P50校准课程。** 本地代码和smoke已验证，正式NUS建库/GPU验收尚待用户执行。真实游戏仍未验收。

下一轮直接阅读[长训复盘](docs/research/LONG1080772_REVIEW.md)和[NUS完整部署命令](docs/nus/NEXT_P50_DEPLOYMENT.md)。新布局部署至`~/projects/HSAI`，旧`~/projects/HSRL`保留为只读历史资源；不要在旧目录pull新布局。

## 从这里开始

- Agent先读 [AGENTS.md](AGENTS.md) 和 [PROJECT_STATE.md](PROJECT_STATE.md)。
- 模拟器固定为 **37.0.2.256589**，上游基线 `1c2086a`，本地原生源码指纹 `c7b5f27d6f40d1d2`；不声称已经同步未来补丁。
- 原生动作 **245**，观测 **3227 floats＋124 cards**。Python额外声明的29维动作未被引擎实现，不能用于训练或实机输出。
- 当前训练主线：一个学习席位＋七个严格完整快照席位、`sims=0`、PPO。搜索/MCTS与BC代码保留，不代表当前长训在使用MCTS。
- 下一轮训练从**7357完整Trainer**开始（16.44M参数、Adam545990、实际lr约1.75477e-6），保持Adam/调度器/RNG/anchor。实机适配仍固定**离线验收6039**，不会因训练模型更新自动替换。

## 代码与数据分层

```text
engine/      Rust游戏规则、战斗、快照事务、PyO3接口与当前权威卡表
hsbg/        观测/动作、搜索、网络、BC/PPO、并行采集、SHM、评测
workflows/   建库、迁移、训练、评测、实验控制入口
client/      当前245动作实机桥、离线入口、旧候选协议兼容层、C#插件
config/      训练配置、源码锁、真人数值参考、实机模型选择/历史UI配置
data/        历史/兼容静态数据；不能覆盖当前引擎卡表
deploy/      Windows安装与NUS计算节点安装、Slurm入口
tools/       源码/卡表/资源完整性检查、只读服务器审计和下载
tests/       核心契约及离线回归
artifacts/   大模型、数据、库、OCR资源、本地环境构建与运行输出，不入Git
records/     研究/原始回放/源码快照/迁移和服务器证据
docs/        服务器限制、训练历史、实机边界及审计报告
```

`engine → hsbg.env → entity网络 → masked策略/BC/PPO → 并行采集/流水线 → 原子checkpoint → 独立评测`。

实机为另一条适配链：`Power/Scene日志及视觉 → public-approx公开状态 → 3227/124编码 → 6039 → 服务器授权的245动作映射 → 几何规划 → 单笔输入/因果回执`。它不把真人游戏伪装成随机模拟器状态，也不能还原全部隐藏队列和效果。

本地完整文件树可重新生成workspace-tree.txt（本地清单不入Git）。历史代码只在records归档，不是第二套活动引擎。

## 本地运行

当前已建立 `.venv`。请始终使用它，避免误用系统Python中的旧editable安装：

```powershell
cd D:\HSAI
& .\.venv\Scripts\python.exe tools/audit_workspace_readiness.py
& .\client\start.ps1
& .\client\start.ps1 -Mode Rehearsal
```

默认命令只做离线doctor；Rehearsal读取历史255790公开快照，输出到 `artifacts/local/`，不打开游戏、不创建窗口后端、不发送输入。已有55条历史快照的模型/合法动作/几何/Controller dry-run全部通过；这不等价于新版本实时日志与界面验收。

已有离线公开快照可以直接调用：

```powershell
& .\client\start.ps1 -Mode Decide -Snapshot 'D:\HSAI\artifacts\local\snapshot.json'
```

`snapshot.json`必须为桥接模块生成的完整公开快照；此命令不会自动从游戏读取状态。完整命令接口：

```powershell
& .\.venv\Scripts\python.exe client/live_245.py --help
```

底层CLI保留显式 `run/autoplay/calibrate`；本轮不执行。6039需要256589版本经过review的UI配置；默认继承的255790配置会被提前拒绝。不要为了运行而改白名单或把旧坐标标记为新版本已验证。旧 `client/run.py` 使用另一种候选动作协议，不可拿16.44M权重直接替换它的旧模型。

### 环境与安装

本机Python3.13环境仅用于本地；NUS仍是已验证的Python3.12/Torch2.10+cu128。当前本地 `.venv` 复用 `D:\Anaconda` 的外部Torch等通用依赖，NumPy/SciPy和OCR库安装在HSAI环境中；没有旧HSBG/HearthstoneAI/HearthstoneRL导入路径。复用关系记录在 `.venv/Lib/site-packages/external-runtime.pth`，不要删除Anaconda。

首次全量测试暴露共享Anaconda数值库的兼容性崩溃，已使用本环境NumPy2.4.0/Scipy1.17.0的wheel修复，未设置绕过OpenMP错误的开关。运行版本见 `artifacts/local/requirements-windows-resolved.txt`；该文件包含可见的外部通用包，不是推荐全量安装锁。

在另一台机器建立独立环境时，使用外部Python、Rust1.98.1及对应Windows链接工具，然后执行：

```powershell
& .\deploy\setup_windows.ps1 -Python python
```

这是安装/编译操作，不会启动游戏。模型、完整数据另行传输。OCR三个ONNX资源在 `artifacts/client/ocr`，有独立哈希清单；依赖安装/首次准备可重新获取资源。C#插件的BepInEx/Harmony等外部SDK尚未验证齐全，当前245日志/输入链路不依赖启动此插件。

### 本地验证

```powershell
& .\.venv\Scripts\python.exe tools/verify_workspace.py --skip-rust --json artifacts/local/validation.json
& .\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

不带模型/缓存/历史产物的干净发布副本：226项Python测试通过、8项跳过（其中包含未分发权重的检查），源码锁和卡表检查通过。新增EOT诊断另有原生plain/detail同结果回归。Rust机制有改动时另跑 `cargo test --release --manifest-path engine/Cargo.toml -p hsbg-engine`；本轮未改Rust机制，不重复编译冒充新验证。之前941项Rust测试记录仍保留。

## 模型、库与恢复状态

| 本地资源 | 位置/状态 |
|---|---|
| s14 Best | artifacts/models/best，原始模型保留 |
| 4159迁移版 | artifacts/models/nus-1080772/initial-4159-3702，完整Trainer已核验 |
| 5506 / 6039 | 同目录两个阶段子目录，完整Trainer和原始96局评测已归档 |
| 7357 / 7710 | 同目录step-007357-360m / step-007710-paused，完整Trainer已核验；7357为下一轮起点 |
| 当前实机选择 | config/live/model.json；精确SHA固定6039、16,436,034参数 |
| 当前37.0.2库 | artifacts/datasets/library-metadata/... **只有清单/校准/验证，无帧和raw，不可训练** |
| 历史库 | artifacts/libraries，九套不同用途历史库，不能冒充当前库 |
| Firestone原数据/旧模型 | artifacts/datasets、models/historical、archives，查注册表和迁移索引 |

模型索引：`artifacts/models/registry.json`；数据索引：`artifacts/datasets/registry.json`。原始4159模型SHA为c8e...，迁移4159为110e...，两者分开登记。完整来源、CPU恢复、Adam/调度器/RNG检查见 [首批归档核验](docs/nus/IMPORT_1080772_VERIFIED.md)。

分类模型与原始下载之间使用新工作区内部硬链接，节约存储，**不是独立备份**。归档目录不可原地训练；续训应复制到新实验。只拿model.pt不能恢复完整训练。

## NUS：当前限制与后续切换

登录节点只有 **1GiB虚拟内存限制**。模型、Torch、pip构建、Rust安装/编译、pytest、库生成与训练都在Slurm计算节点执行。登录节点只读小JSON/日志、Git及作业管理。详见 [环境档案](docs/nus/ENVIRONMENT.md) 与 [故障手册](docs/nus/INCIDENTS.md)。

1080772已安全暂停；旧目录及c0e4c82代码保留为历史证据和只读资源。NUS仍完全由用户手工操作，代理没有登录或提交作业。

长训规则为80次总动作保护阈值/5次换位/重复换位去抖2。待选状态不能直接EndTurn，合法收尾可能让记录超过80；完整评测发现少量play/cancel循环记录82次，已登记问题。正常棋力波动仅记录，工程错误仍严格保护。安全暂停在实际运行目录创建STOP_REQUESTED，硬中断仅能恢复最后提交点。

`deploy/slurm` 已按新布局整理，但**还没有完成HSAI新布局在NUS的验收**。`long_4159.sbatch`仍是4159历史配方入口，不是此次24h最终模型的自动续训指令。下一轮使用p50_cpu/p50_preflight/p50_long入口，完整分步命令见上方部署文档。

## 完整性结论与缺口

| 方面 | 判定 |
|---|---|
| 模拟器/网络/BC/PPO/采样/评测核心代码 | 已保留，可本地测试 |
| 6039到实机脚本的离线链路 | 已接通；55条历史快照通过 |
| 当前真实游戏闭环 | 未验收；UI版本、时序、真实公开状态近似仍有限制 |
| 当前新版库本地训练 | 缺frames和raw，不能运行 |
| 本轮NUS训练归档 | 81文件已核验，完整3551轮重算一致；安全暂停未进行512终评 |
| Git与新NUS工作区 | 代码发布至kk03kk/HSRL；新NUS部署与CPU/GPU验收仍由用户执行 |
| 规则正确性 | 有回归证据；71次注入前物种账本偏差根因未解，不能声称全部游戏机制完美 |
| 策略质量 | 固定Dev改善；高本转型、无进展play/cancel循环仍待改进，不能保证真实游戏棋力 |
| 历史原始完整状态 | 原版4159、6157/6160、B100k/B10M等仍待核实/补取 |

## 旧目录能否删除

**不能把除HSAI以外的目录统统删除。** 本轮核对了之前库存43532个源文件，未变化；当前运行环境也没有旧项目导入路径。但旧HearthstoneRL/.audit_tools仍含外部Rust/链接工具，HSBG有非项目课程文件；全局Anaconda的旧editable配置也未改动。旧Git历史未迁移，这是此前明确的范围，不等于已经备份Git。

逐项清理判定及其条件见 [完整性与清理审计](docs/WORKSPACE_COMPLETENESS.md)。本轮不删除、不移动旧目录、不清理炉石本体。

## 发布与资源边界

Git包含核心代码、必要静态数据、配置、当前文档和精选研究摘要；不包含模型、完整战团库、个人环境、密钥、原始归档与Git历史。原始资料仍保留在本地records/artifacts。发布采用新的单根提交；旧NUS工作区不做原目录pull，使用独立clone部署。

来源：[AutoTavern](https://github.com/MCG-NJU/AutoTavern)、[HSBRSIM](https://github.com/Gallo13th/HSBRSIM)。当前状态和下一轮操作以本README、PROJECT_STATE及NEXT_P50_DEPLOYMENT为准，历史文档不作为现行启动命令。

## 战团强度报告口径

所有新报告以每张战团总攻击＋总生命的P50为主参考，同时列出来源样本量与P25/P75/P90；均值仅作辅助。P50不等同于战斗胜率。下一轮曲线决策见[研究决策](docs/research/OPPONENT_P50_NEXT.md)，上一轮已安全暂停，新库/新配置仅用于独立下一轮。

## 当前下一轮配方

一个学习席＋七个严格完整Bot EOT快照；37.0.2.256589，245动作、3227+124观测。Train/Dev/Test使用独立源对局，统一以随机抽样总攻血P50校准。Train前8回合立即到目标，中后期用0/10M/20M/30M固定课程到位；Dev/Test全程冻结。第10回合起源阵容至少5怪，晚回合复用标明真实source_turn，全部活席注入必须成功。

保留原PPO配方（单epoch、1024minibatch、BF16、clip0.2、KL0.05、gamma1/lambda0.995、熵0.002）、80动作保护阈值/5换位/去抖2；奖励不新增强制升本或牌面刷分。2×H100-47、32CPU、96GB、8采集进程/512局，24h分配；96局起点，每30M决策96局详细配对监测，正常收尾512局。正常棋力波动不触发止损。

模型、战团库、OCR/游戏回放和原始服务器数据不随Git分发。代码clone可构建，但训练前必须准备文档指定的7357完整状态和原始Bot数据。服务器原有资源直接只读复用，无需从Windows重新上传大文件。所有重任务在计算节点，登录节点只运行`python3 -S tools/nus_submit.py`轻量预检与提交。
