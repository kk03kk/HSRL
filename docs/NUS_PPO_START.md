# NUS：BC + 循环 PPO 启动作业

> 当前推荐：2026-10-27 第二版修复发布，见 [NUS_REPAIR2_START.md](NUS_REPAIR2_START.md)。原 job 1020969 已失败并保存至第 25 波；新代码必须重新构建原生扩展，经 40 波完整规模 GPU 验收后再启动长训练。下文保留首版操作历史。

本地唯一源码根：`D:\HearthstoneRL\AutoTavern`。本轮交付是可提交作业的训练基线，不是已训练的强模型；服务器环境搭建和 CUDA 验收由下述作业自动执行。助手未登录 NUS。

## 1. 版本与纳迦纠正

用户指出纳迦已移除，官方 [36.6.1 公告](https://news.blizzard.com/en-us/article/24302091/aberrations-join-battlegrounds-at-blizzcon) 也说明纳迦及其专属英雄、卡牌退出。本实现保留自己的 AutoTavern 基线，没有引入 HSBRSIM 纳迦回池/艾萨拉内容；默认随机大厅无纳迦，配置明确 ban 艾萨拉，测试检查该约束。

**实际训练引擎为随仓库固定的 36.6.3.253932 Implemented 单人模式基线，不宣称已经精确复刻 37.x 实机。** 当前数据未整体替换为朋友的 37 卡表，避免其错误回池与未验证池标志。这是可运行训练基线；未来实机部署前仍须逐项核对最新平衡、规则、效果覆盖与时间成本。版本、词表、规则与源码指纹进入 checkpoint，变化后禁止静默续训。

## 2. 已实现主线

- 实体卡牌嵌入 + 数值/位置编码 + Transformer + GRU，源/目标感知动作打分，合法动作 masked softmax，8 名次分布价值头。
- 训练观测在上游固定观测之外，增加己方/可见侦察快照的附加效果、实例计数、临时状态、卡牌副本状态、玩家计数、技能进度、待选项和排队效果。不读敌方隐藏手牌/当前战团、共享池真实状态或 RNG。
- 完整自然对局采集，按玩家有效决策形成序列。等待 tick 不进入训练，淘汰时的名次回报归到最后一个实际决策。
- 两种可见信息购买偏好的轻量 bot 教师作 BC 与价值预热；PPO 期间混合原生 bot、BC anchor、当前策略和有界历史池；对手在整局中固定。
- 分块循环 PPO；每次 minibatch 用当前参数从 episode 起点无梯度重放前缀，避免复用过期隐藏状态。实现正确性优先，前缀重放有额外开销，后续按服务器性能决定优化方式。
- 有限局 gamma=1、GAE、clip、entropy、rank CE、梯度裁剪、KL 早停；固定 bot 与 BC 混合评测，轮换学习座位、bootstrap CI。
- 原子 checkpoint：模型、优化器、RNG、迭代与 BC 进度、对手池、词表/规则/源码身份。Linux flock/Windows 锁阻止两个作业同时写同一 run。
- SIGUSR1/TERM 请求在安全边界停止；部分采样丢弃，保留上个完整 checkpoint；意外强杀仍可从上个迭代恢复。没有自动提交新作业的服务。

## 3. 本地生成并传输完整源码包

PowerShell，在本地项目根运行：

```powershell
python tools/package_nus.py
scp -o ProxyJump=YOUR_NUS_USER@sjump.comp.nus.edu.sg runs/dist/AutoTavern-nus.tar.gz runs/dist/AutoTavern-nus.tar.gz.sha256 YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/
```

打包只需 Python 标准库，不依赖本机 PyTorch；包包含本人的未提交修改、Cargo.lock、依赖与作业脚本，没有 `.git`、密码、虚拟环境、模型或朋友仓库。用户仍手动完成 SSH 密码认证。

首次部署，在 NUS 登录节点执行：

```bash
cd "$HOME"
sha256sum -c AutoTavern-nus.tar.gz.sha256
mkdir -p "$HOME/projects/hearthstone-rl"
tar --no-same-owner -xzf AutoTavern-nus.tar.gz -C "$HOME/projects/hearthstone-rl"
cd "$HOME/projects/hearthstone-rl/AutoTavern"
python3 tools/package_nus.py --verify .
```

这是新项目部署目录，后续已有训练成果时不要盲目覆盖源码后接着加载旧权重。源码发生变化，续训指纹校验会拒绝；升级应单独目录和 run。

## 4. 依次提交：安装 → GPU 冒烟 → 正式训练

始终在远端项目根提交：

```bash
setup_job=$(sbatch --parsable deploy/nus/setup.sbatch) &&
smoke_job=$(sbatch --parsable --dependency=afterok:"$setup_job" deploy/nus/smoke.sbatch) &&
train_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" deploy/nus/train.sbatch) &&
printf 'setup=%s smoke=%s train=%s\n' "$setup_job" "$smoke_job" "$train_job"
squeue -u "$(id -un)"
```

`&&` 确保某次 sbatch 本身报错时不提交后续作业。afterok 只保证前置作业成功后执行；失败依赖可能保持 DependencyNeverSatisfied，需用户查看日志并取消不再需要的等待作业，不会自动修复或绕过检查。

| 作业 | 配额与行为 |
|---|---|
| setup | normal，8 CPU，16G，最长 3h；用户项目内安装 Python venv、隔离 Rust 1.98.1、maturin，构建原生扩展并运行测试；不申请 GPU |
| smoke | gpu，1 张任意 GPU，4 CPU，16G，30min；验证 CUDA、一次 BC、两次 PPO、评测、历史快照与 checkpoint；是功能验收，不是棋力验收 |
| train | gpu-long，单 A100-40，16 CPU，64G，最长 2d23h；16 局并行、2 学习席、默认 1000 个完整迭代；已有 latest 自动续训 |

GPU 资源依据已采集的 Slurm 配置与账号限制。未预先占用节点或保证排队时间；实际获批资源以 Slurm 为准。A100-40 暂不可用时可在提交时覆盖 `--gres=gpu:a100-80:1` 或已确认的 H100 型号；改变 GPU 需记入评测记录。

安装使用 [官方 PyTorch 2.10.0 CUDA 12.8 wheel](https://pytorch.org/get-started/previous-versions/)，不需要计算节点系统安装 nvcc；CUDA 能否工作仍由 smoke 实测。Python venv 模块若缺失，setup 会从 PyPA 下载 virtualenv zipapp 创建隔离环境。Rust 安装在项目 `.toolchains/`，不改用户 shell 或系统包。

## 5. 日志、恢复与模型回传

作业日志：提交目录 `slurm-hsrl-setup-<job>.out`、`slurm-hsrl-smoke-<job>.out`、`slurm-hsrl-ppo-<job>.out`。

正式产物：

```text
runs/nus-ppo/manifest.json       词表、规则、源码与训练配置
runs/nus-ppo/metrics.jsonl       BC/PPO/评测指标
runs/nus-ppo/latest.pt           唯一默认续训入口，完整迭代检查点
runs/nus-ppo/best.pt             固定验证集上较好模型，供推理/独立评测
runs/nus-ppo/bc-anchor.pt        BC 冻结对手
runs/nus-ppo/league/             有界历史池
runs/setup/                     依赖清单与 CPU 环境验收
```

latest 与当前 league/anchor 必须一同保留才能完整恢复；best 可用于推理和独立评测，不作为历史池完整恢复入口。再次 `sbatch deploy/nus/train.sbatch` 会自动从 latest 继续；1000 次已完成时会正常退出，增加配置 iterations 可继续。

环境 build/schema/词表、模型结构、种子、奖励等发生变化时不能静默续训。仅 device、线程、总迭代数、eval_games 可调整；如需改变其他训练设置，应开新 run 并明确设计权重初始化流程。

临时放大批量前先测显存与采样吞吐；线程必须服从实际申请 CPU。若显存不足，先降低 chunks_per_batch/n_games；这属于新实验配置，不修改已有 run 的语义。

独立未见种子评测（登录节点提交专用 16 CPU/GPU 作业，避免在登录节点推理或超过已有作业 CPU 分配）：

```bash
sbatch deploy/nus/evaluate.sbatch
```

模型回本机时保留同版本源码与 manifest；尚未实现真实点击脚本接入，不能把本交付当作实机无人值守 bot。先验证推理接口、状态同步、动作语义和耗时。

## 6. 已知限制与失败路径

- 默认每席每回合 60 动作是显式、可见的实验预算，不等于真实 APM/网络时间。到预算后由引擎合法 mask 约束，没有跨座位共享计数器；BC 跳过仅有唯一合法动作的强制标签。
- 模拟器 turn cap、战斗截断、driver guard、无进展、非法动作会拒绝该采样并报错；不把未完赛记为第八。诊断保留 `failed-lobby.json`，需审查后另行修复，不自动放宽规则。
- 完整对局波次等待最慢局；前缀重放计算较多。第一版保证数据语义，吞吐优化应测量后推进。
- 1000 次迭代与默认 BC 数据量是可启动规格，不保证强度或特定训练天数。早期模型可能明显弱于 bot，必须看独立棋力曲线。
- 当前评测覆盖固定 bot 与 BC 混合；真正强对手、真实版本校准、长线联赛反制者和搜索蒸馏为后续增强，不伪称本轮全部完成。
- HOME 用户配额/备份、节点 /tmp 清理规则未获确认。首版成果保存 HOME；不使用临时盘保存唯一成果，不承诺无限空间。
- Windows CPU 本地测试不能替代 NUS Linux/CUDA 验收，因此依赖链必须保留 GPU smoke 门禁。

## 7. 本地验收

- 原生引擎构建通过；Rust 全量测试 **862 项通过**，包含新增的观测别名、隐藏信息、运行时副本和纳迦退出回归。
- 本地 Rust 测试使用 debug test profile；原生 Python 扩展使用 release 构建。`cargo fmt` 与 release/all-targets Clippy（warnings 视为错误）已通过，服务器 setup 还会重跑 release 测试。
- Python 全量测试 **58 项通过**，包括原有 46 项与新增训练/部署检查。Windows 运行需启用 UTF-8（PYTHONUTF8=1）；测试中没有删除原有断言。
- 小规模 BC → 两次循环 PPO → 两组固定验证对手 → 历史模型/最新/最佳 checkpoint，整个流程已在 Windows CPU 执行。
- 同一小实验“连续两次更新”与“一次更新后重启并续训”，模型参数、随机状态和进度一致。该结论限相同本机 CPU 环境，不承诺跨 CPU/GPU 或不同 PyTorch 版本位级一致。
- 独立 seed 的 evaluate CLI 已执行。上述短实验中的棋力仍弱，仅证明功能闭环，不作为强模型成果。
- Bash 脚本语法检查和源码包完整性/篡改检测通过。**NUS 上的 Linux 构建、CUDA 训练及 Slurm 参数获批仍由 setup/smoke 作业实际验收，未伪称已运行。**

## 8. 修复首版源码包漏收 overlay（无需重装）

首个 NUS setup 1020740 已通过 Linux 原生构建、Clippy 与 Rust release 全量测试，但 Python 测试为 57 passed / 1 failed：源码包漏收 `tools/overlays/36.6.3.json`，版本一致性断言找不到该文件。这是打包缺陷；保留测试，补齐文件。

已修复为打包整个 `tools/` 源目录，并将 overlay 与生成器设为必需文件；新增漏包和增量修复回归检查，提供 `tools/validate_nus_bundle.py` 从隔离解压目录执行全套 Python 测试，避免开发目录掩盖缺失文件。

2026-10-27 修复验收：开发目录完整测试 60 passed；全新解压后的发布目录完整测试 60 passed。原 202 文件发布树应用增量后，206 文件源清单校验通过。没有修改原来的版本一致性断言，也没有修改引擎或训练源码。

旧 smoke/train 1020741/1020742 用户已执行取消。修复包只包含变更/新增源码和新源清单，不包含 Rust 引擎变更、模型代码变更或任何 venv/toolchain/target/runs，不需要重新编译。

本地 PowerShell 传输：

```powershell
cd D:\HearthstoneRL\AutoTavern
scp -o ProxyJump=YOUR_NUS_USER@sjump.comp.nus.edu.sg runs/dist/AutoTavern-nus-repair1.tar.gz runs/dist/AutoTavern-nus-repair1.tar.gz.sha256 YOUR_NUS_USER@xlogin.comp.nus.edu.sg:~/
```

服务器应用（任一步失败则停止此命令链，不会继续执行后续项）：

```bash
(cd "$HOME" && sha256sum -c AutoTavern-nus-repair1.tar.gz.sha256) &&
cd "$HOME/projects/hearthstone-rl/AutoTavern" &&
python3 tools/package_nus.py --verify . &&
tar --no-same-owner -xzf "$HOME/AutoTavern-nus-repair1.tar.gz" -C .. &&
python3 tools/package_nus.py --verify .
```

源清单校验成功后，重新建立依赖链；使用新的 verify 作业替代 setup，复用已安装环境和成功的原生构建：

```bash
verify_job=$(sbatch --parsable deploy/nus/verify.sbatch) &&
smoke_job=$(sbatch --parsable --dependency=afterok:"$verify_job" deploy/nus/smoke.sbatch) &&
train_job=$(sbatch --parsable --dependency=afterok:"$smoke_job" deploy/nus/train.sbatch) &&
printf 'verify=%s smoke=%s train=%s\n' "$verify_job" "$smoke_job" "$train_job"
squeue -u "$(id -un)"
```

verify 仅申请 normal、4 CPU、8G、30min，执行源清单校验、Python 全量测试、CPU doctor 和依赖清单记录；不会重复下载依赖或重新编译引擎。后续反馈新的 verify/smoke 作业日志即可。
