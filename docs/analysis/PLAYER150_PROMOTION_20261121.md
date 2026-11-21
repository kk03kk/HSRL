# Player150 末代 3286 上线准备与停训交接

2026-11-21 · D-198

## 当前模型

| 项目 | 值 |
| --- | --- |
| 模型 | `player150-round05-3286`，第五轮正式封存 |
| 权重 | `artifacts/models/player150-round05-3286/model.pt`，39,385,089 字节 |
| SHA-256 | `778cc6f4ca9ba6175ec6f54c2241a030a70ce4e35b7ac7120492666360ba86df` |
| 冻结训练源码 | `cb9587ac0cfa4b1875415277652392043a43e082` |
| 引擎和源码锁 SHA-256 | `8bd1f4785018c474` / `03031a660c1461696b5007bf528257ab9054baba3b5bc1c709b7ec3f3739b4d5` |
| 观测 SHA-256 | `ad4c812c2446b1c768064ea0cf5ed8eeb4cad0625744ec4c68e2dbb27cbeb485` |
| 评测协议 | 固定场 `5361375f96441316237a792205b5ff76587bff27a9fd80d7cad25307075927d9` · 外部场 `24d387ec449479425bd356b61c639b35f6095b4a32cab700519a0a62a04581f4` |
| 实机观察与动作 | 4,967 浮点数、124 卡牌位、245 原生动作。每回合 150 步、3 次场上换位、冻结及解冻共 2 次 |
| 旧模型 3236 | 保留为同一 Player150 配置下的手动回退。发动输入仍需正常版本与状态门禁 |

训练在 NUS `artifacts/runs/nus-player150-five-20261120-01/round-05/train/fit/train/snapshots/final_003286/` 封存。Windows 权重复核了封存回执 1173164、固定场 1173165 和真人战团场 1173166。可在权重同目录读取 `manifest.json`、`checkpoint.json`、`MODEL_CARD.md` 和下载回执。Trainer 可在 NUS 原封存目录接续。

## Windows 入口

- 双击项目根目录下的 `client/start-ai-control.cmd`。预选 3286，可在下拉菜单选 3236 手动回退。启动控制台与开始匹配分两步。
- 离线环境检查用 `client/start.ps1 -GameDir D:\Hearthstone`。对公开历史快照推理用 `client/start.ps1 -Mode Decide -Snapshot <Player150 快照 JSON>`。
- 离线回放用 `client/start.ps1 -Mode Rehearsal -Trace <decisions-full.jsonl.gz>`。原始决策轨迹须记录当时可用动作、公开观测和回合剩余步数。
- 在已安装实机工作树的 Windows 本机，项目根目录的 `client/live_245.py` 会加载 Player150 引擎和默认权重。用 `--legacy` 可显式调用旧环境。PowerShell 旧入口用 `-Legacy`。
- 本机当前生效的 pin 和桌面配置位于 `artifacts/local/r1-player150-live/config/live/`，桌面默认读取同一 pin。运行前校验权重、卡牌顺序、引擎与客户端构建。

## 验收和归档

- 离线 Doctor 读取已校准的炉石可执行文件，核对客户端构建 255790、本机引擎指纹、模型维度及有限推理输出。历史实机存档的 237 个决策点全部返回该帧合法动作，未创建输入后端。
- 3236 在同一份轨迹上逐项复现 237 个原动作。R1 原权重、R1 迁移版、2583、2981、3186 各通过 13 项抽样决策。旧环境的 14858 通过独立 Doctor 推理。
- 本地回归 116 项通过、5 项因当前工作树缺少旧权重或存档而跳过。覆盖 7–10 张手牌布局、第 9 回合误抓、第 8 回合抉择提交、模型选择、回执与控制器。桌面初始化验证了预选 3286、手动选项 3236 以及打开控制台时不会匹配。
- 9 份不同摘要的旧权重归档在 `artifacts/models/historical/player150-promotion-20261121/`。索引 `index.json` 列出原来的 11 处入口、归档位置、SHA-256 及元数据。NTFS 硬链接保留历史工具可复现的原路径，未额外复制权重字节。这是同盘归档。原有历史模型总表仍在 `artifacts/models/registry.json`。
- 3286 的真实游戏表现待确认，正式评测在[上一轮评测报告](PLAYER150_FINAL_REVIEW_20261121.md)。

## 服务器

1173164、1173165、1173166 正常结束。复查 NUS 本账户作业队列为空，无需取消作业。暂停提交训练和新评测，等待用户安排。正式配方、评测结果和恢复文件均保留。
