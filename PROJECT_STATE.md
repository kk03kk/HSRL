# 当前状态（2026-11-06）

HSAI 已从代码工作区扩充为代码、模型、数据、研究和服务器经验的统一工作区。旧 D:\HSBG、D:\HearthstoneAI、D:\HearthstoneRL 全程只读。没有访问 NUS、提交作业、打开游戏或操作游戏输入。

## 先读这些

| 资料 | 内容 |
|---|---|
| `AGENTS.md` | 不可忽略的契约与工作规则 |
| `docs/nus/ENVIRONMENT.md` | 服务器限制、部署与安全恢复 |
| `docs/nus/INCIDENTS.md` | 历史故障、原因和已采用修复 |
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

## 后续取回服务器资料

不需要停止当前训练。等稳定提交点或作业正常结束后，由用户独立归档并下载：当前完整 checkpoint 的 model/trainer及同目录来源元数据，起点4159完整状态，6157历史状态，当前校准库三分割与原始Bot源数据，plan/source-fork/config/protocol/manifest、history、评测与诊断、Slurm日志、STOP恢复说明。下载后先核 SHA 与引擎/协议，不覆盖既有目录。**不要只取 model.pt 就宣称可恢复训练。**
