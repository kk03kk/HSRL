# 工程架构与验收入口

状态：2026-11-17 现代化后的维护约定。科学设计仍以训练配方、评测协议和用户决议为准。
本文件描述工程机制，不构成新训练或改配方授权。

## 1. 代码、执行与证据分开管理

| 层 | 当前职责 | 主要入口 |
|---|---|---|
| 引擎与规则 | 原生 245 动作、观测布局、卡表、游戏状态 | `engine/`、`hsbg/env/` |
| 训练领域 | 网络、PPO、采集、健康门、检查点；参数解析独立 | `hsbg/az/train/`、`hsbg/az/train/cli.py` |
| 评测领域 | 纯聚合函数、流式动作记录、证据验证 | `hsbg/evaluation/metrics.py`、`trace.py` |
| 运行公共能力 | 原子写入、文件身份、环境身份、受控共享资产路径 | `hsbg/runtime/` |
| 工作流 | 组织上述能力；CLI 不再作为聚合算法唯一宿主 | `workflows/train_entity_sync.py`、`evaluate_suite.py`、`register_completed_model.py` |
| 部署与运维 | SSH、Slurm、不可变源码发布、状态/同步/下载 | `ops/`、`deploy/slurm/` |
| 实机 | 保持独立输入边界、模型 pin、离线/真实验收区分 | `client/hsrl_agent/live/` |
| 大资产 | 模型、Trainer、库、逐局记录、参考资产与历史恢复副本 | `artifacts/`、服务器数据目录 |

Windows 与 NUS 同步的是 Git 源码提交，不是把所有资产或二进制环境复制成相同。
Windows 使用 CPU Torch 与 Windows 引擎轮子；NUS 保留已验证的 CUDA 环境。
两端用引擎源码指纹、卡表顺序、动作/观测语义和模型 SHA 建立兼容性，而不是只看 shape。

## 2. 冻结发布，而不是冻结整个研发目录

`ops/releases.py --workspace . --commit <已提交的SHA> --name <唯一发布名>` 在服务器创建
`artifacts/releases/<name>/` 的 detached Git worktree，不改变主检出。发布目录的 `.venv`
与 `artifacts` 指向受控共享资产；`data/records` 保留提交中的目录文件，仅挂载缺失资产。
发布记录位于 `artifacts/releases/manifests/`。

作业从发布目录提交，排队及运行期间不改其源码、共享环境或输入资产。之后的改动创建新发布。
评测开始/结束核对源码、包版本身份以及权重/输入库内容哈希；环境仍是共享挂载，因此禁止
作业运行期间原地升级依赖。未来升级应新建环境，先验证再为新发布指定，不能改正在使用的环境。
发布工具默认解析主工作区 `.venv` 的真实目标，也可用 `--environment .venvs/<version>`
指定工作区内的已验证环境。后续切换主 `.venv` 不重定向旧发布；工具不自动安装包。

回滚使用保留的分支/发布目录；不用 reset --hard，不覆盖旧输出。失败发布和失败作业保留，
报告明确说明失败原因。发布本身不自动提交作业，不自动改模型 pin。

## 3. 检查点与模型登记

训练结束写入独立 `snapshots/final_<version>/`，同时保存 model/Trainer 并计算内容哈希。
完整生成后更新平面兼容别名，最后原子发布 `latest.json`；**latest 指向的同一代文件是权威组合**。
若别名更新中失败，旧指针仍有效，失败必须令作业报错。不要凭两个平面文件的存在推断事务完成。

`workflows/register_completed_model.py` 在计算节点验证已结束训练的平面最终文件与 summary、
history、Trainer 代数和张量，再复制成不可覆盖的候选模型目录：
`model.pt + trainer.pt + manifest.json + checkpoint.json + validation.json + MODEL_CARD.md`。
原始运行目录保留。登记 candidate 不表示晋升冠军或改变线上 pin。

历史 v1.4-r1 的 `latest.json` 停留在周期版本 741，而最终平面文件是 745。该历史证据不改写；
已验证、登记的最终模型取 745，恢复源码另行保留。这也解释了为何仅检查 latest 不足以验收旧运行。

## 4. 唯一正式评测执行器

新评测先提交 `config/evaluation/*.json` 计划，显式记录模型/SHA、局数、种子、分片、协议选项
与停止规则。入口 `workflows/evaluate_suite.py`，Slurm 包装为 `deploy/slurm/evaluate_suite.sbatch`。
旧 `eval_shard_parallel.sbatch` 只把旧环境变量转换成相同执行器的计划，保留原科学默认值。

执行器用商余数分配局数，覆盖每个计划 seed 恰好一次。所有子进程须成功；缺分片、重复 seed、
不同权重或协议、缺行数据、非法/截断对局、输入在运行中变化，均不能产生 `complete.json`。
每模型保留独立逐回合 rows 与完整动作 trace；合并器从原始行重算统计，不能平均各分片 P50。

完整 trace 校验可以按文件分配到有界 CPU 子进程，保持输出与串行相同；并发不超过 8、分片数
和分配的 CPU 数。`tools/seal_evaluation.py` 在成功完成后给全部运行文件建立内容清单，分析
产物另放目录。模型下载默认只列顶层普通文件，`reports/` 等子目录不会被误当成权重文件下载。

正式评测目录的证据层次：

1. `run.json`：原计划、源码、环境、输入内容哈希、精确分片。
2. `shard*/command.json/process.log/exit.json`：实际命令与退出状态。
3. 每模型 `rows-*.jsonl.gz` 与 `decisions-*.jsonl.gz`（具体文件名由面板引用）。
4. `panel.json`：聚合结果、逐 seed 名次和逐回合数据、trace 清单。
5. `validation.json`：全记录完整性校验与原生步骤回放。
6. `complete.json`：仅在所有验收完成后写入，绑定 panel/validation 文件 SHA。

逐动作记录包括 game/seed/turn/seat/动作序号、实际动作与提议动作、强制标志、完整前后引擎
状态、3227 浮点观测与 124 卡槽（little-endian 字节 base64）、原生合法掩码、策略分布/信息。
状态是即时原生 `step` 前后，后续快照注入可在下一条前状态看到。记录覆盖**学习模型的每次动作**；
原生 `bot_play` 内部 bot 决策未被引擎 API 暴露，不能声称逐个记录了全部对手内部动作。
对手快照身份由输入库哈希和协议绑定。

查询某个模型的某局、某回合/动作：

```bash
.venv/bin/python tools/inspect_decisions.py <panel.json> --model <name> --seed <seed> --turn 12 --out <new.json>
.venv/bin/python tools/validate_eval_evidence.py <panel.json> --out <new-validation.json> --replay 64
```

查询按 manifest 选择分片并校验哈希，直接导出记录，不重新采样模型。统计必须同时给 P50 和
样本量，比较共同存活 seed，不能把 winner-biased 真人面板当作对等对手比赛。

## 5. 环境与可选外部源码

Windows `.venv` 是指向 `.venvs/windows-20261117` 的 junction，旧环境完整保存在
`artifacts/environments/windows-pre-modernization-20261117`。旧环境通过 `.pth` 引入了
整个 Anaconda site-packages；新环境仅共享 Python 标准库，不引入全局第三方包。
`config/environments/windows-cpu.lock.txt` 固定实测包版本；引擎轮子另装，项目与 client 以 editable 安装。
OpenCV 统一为 headless 发行版，避免两个包同时拥有 cv2。项目不使用 OpenCV GUI 窗口接口。
沿用 NumPy 2.4.0 是为保持既有本地数值版本；该版本在索引被标记撤回，后续升级需单独验证，
不能把这份锁当作“所有依赖永远安全”的保证。服务器 NumPy/Torch 与本地不同，保留角色差异。

复建 Windows 环境（先创建新目录，不覆盖现有环境）：

```powershell
python -m venv .venvs/windows-<date>
.venvs/windows-<date>/Scripts/python.exe -m pip install -r config/environments/windows-cpu.lock.txt
.venvs/windows-<date>/Scripts/python.exe -m pip install artifacts/local/wheels/hsbg_engine-1.0.0-cp310-abi3-win_amd64.whl --no-deps
.venvs/windows-<date>/Scripts/python.exe -m pip install -e . -e ./client --no-deps
```

可选第三方 `hsbg-ppo` 源码登记于 `config/reference-assets.json`，只在服务器资产目录保存并
逐文件校验后加载；原目录完整归档。原始仓库 revision 与源码授权范围仍待确认，不随 Git 再分发。
缺少可选资产时明确报错，不自动下载或补齐外部源码；不影响自家训练与本次正式评测。

## 6. 验收矩阵

| 链路 | 验收入口 | 能证明什么 |
|---|---|---|
| 静态/卡表/引擎锁/Python | `tools/verify_workspace.py --skip-rust` | 源码、兼容契约与 CPU 回归 |
| Rust + Linux CPU | `deploy/slurm/validate_cpu.sbatch` | 在计算节点构建和测试原生引擎；全量回放小样本证据 |
| CUDA 与模型登记 | `deploy/slurm/validate_release.sbatch` | GPU 测试、真实模型加载、候选登记、双模型真实对局 smoke |
| Adam/RNG/调度器恢复 | `tools/validate_model_restore.py` | 同固定 batch 保存恢复后模型/Adam 逐位相等 |
| 真实训练入口恢复 | `deploy/slurm/validate_training_entry.sbatch` | 多进程采集、PPO、最终保存、第二进程续训；每段仅一轮 |
| 全量正式评测证据 | `tools/validate_eval_evidence.py` | 每条记录、终局名次/seed/动作数与面板一致；抽样原生重放 |
| 本地推理链路 | `client/live_245.py doctor`、live 离线测试 | 模型 pin、语义/词表兼容；不等于真实游戏验收 |
| 隧道 | `ops/tunnel.ps1 -Verify` 后独立 `probe` | 临时端口的真实重建/清理，不中断现有通道 |

固定 batch 精确恢复不等于分布式采集的整个下一段轨迹逐位一致。当前恢复资产保存 Trainer
RNG，但不宣称所有 worker/在途 pipeline 状态都已保存。真实游戏验收仍由 D-026 状态决定。
