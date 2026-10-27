# 脚本对局的自动对手采集

入口在主项目 `tools/collect_opponents.py`。现有脚本继续照常运行；本工具只读炉石和HDT已有日志，不启动游戏、不发送输入、不加载模型，也不修改旧自动化工程。无需额外手填自己的阵容。

默认读取 `D:\Hearthstone\Logs\**\Power.log` 和当前用户的 `AppData\Roaming\HearthstoneDeckTracker\Logs\**\hdt_log*.txt`。默认输出 `D:\HearthstoneRL\AutoTavern\runs\opponent-capture`。纯Python标准库，不需要重装环境、Torch或GPU。

## 使用

在主项目双击 `启动对手战团采集.cmd`，或执行：

```powershell
cd D:\HearthstoneRL\AutoTavern
python tools/collect_opponents.py watch
```

本轮交付会启动一个隐藏的采集进程；它运行时不必再次启动。电脑重启或进程退出后才需重新启动。没有修改系统启动项、注册服务或旧脚本启动器。

查看统计：

```powershell
python tools/collect_opponents.py report
Get-Content runs\opponent-capture\status.json
```

停止采集，不影响游戏或旧脚本：

```powershell
python tools/collect_opponents.py stop
```

主动停止会留下STOP文件。再次启动前明确清除这个控制文件：

```powershell
Remove-Item -LiteralPath .\runs\opponent-capture\STOP
python tools/collect_opponents.py watch
```

只导入历史日志一次：

```powershell
python tools/collect_opponents.py scan --out runs/opponent-import-new
```

更换游戏安装目录用 `--game-logs`。同一输出目录有进程锁，避免两个采集器重复工作；`scan`与`watch`也不应同时写同一输出目录。解析器指纹改变后会要求新输出目录，保留旧数据，不把不同解析语义混在一起。

## 输出

| 文件 | 含义 |
|---|---|
| encounters.sqlite3 | 主数据库，事务保存字节游标、解析状态和快照修订 |
| encounters.jsonl | 全部解析快照：对手及最小己方战斗上下文、事件、结果来源和缺失字段 |
| opponents.jsonl | 明确单人酒馆模式、已记录对手战团的候选；仍需规则/状态适配 |
| quarantine.jsonl | 队友上下文、模式/版本不明、关键字段不完整等其他候选 |
| summary.json | 数量、模式、版本、回合和缺失信息统计 |
| status.json | 进程号、最新轮询时间和running/stopped状态 |

JSONL是数据库的可重建导出，临时文件原子替换；不要在里面手改数据。进程重启继续读取已提交游标；行尾未完整写出的UTF-8记录留待下次，不提前消费。新会话自动发现，文件缩短、替换或被覆写时分新代保存，旧记录不覆盖。

## 数据口径

- Power.log只取GameState事件通道，排除PowerTaskList重复事件；只在明确 `GT_BATTLEGROUNDS` 的会话提取战斗，构筑不入库。
- 通过本机商店控制实体确定己方controller，不把DebugPrintGame列出的第二名玩家误当自己。历史实体引用不能覆盖已处理的移除/区域变化，避免旧商店随从混入对手战团。
- 原生快照时点是开战阶段第一次记录的攻击之前；**不保证所有开战效果已经结束**，也不代表招募末状态。保留时点标签，不自动重放开战加成。
- 保存站位、卡牌ID、攻血、金色、关键词、可见附件、数值标签、可见手牌随从与公开战斗对象。隐藏或未收到的信息保持未知，省略手牌不代表空手。
- 己方只作为诊断战斗上下文，不当专家示范。采集的对手强度标为未测量，不把脚本初期匹配到的对手自动标成顶尖水平。
- HDT输入属于第二实现的输入快照；预测概率标记为非官方。它报告的客户端结果保留来源，按相邻模拟输入限定关联，不能当成服务端逐事件ID已精确对齐。
- HDT非空队友上下文与未证实的单人模式隔离；CardDefs数据build不冒充客户端build或热修版本。未知卡名不强行映射到近似牌。
- `encounters`计数是**来源快照数量，不是独立战斗场数**；HDT同一场可能多次模拟，Power与HDT也可能重复记录同一场。
- `split=unassigned`、`training_import_allowed=false`。先关联重连和跨源重复，再按整局划分训练/验证/测试；当前不能直接把这些资料喂给PPO。
- 数据不作为策略输入；模型仍只使用玩家允许获得的观测。日志采集不等于完整全知游戏记录。

解析状态和导出不保留玩家名、BattleTag、账户ID、原始日志正文或原文件名；输出留在本机主项目。规则不明确、缺牌、缺手牌等只登记，不要求用户补截图、链接或逐张复核。

## 与模型接入的关系

无需等第一个模型就能采集。模型回来后先做匹配环境下的评测/回放，再接真实客户端影子推理和语义动作核验。本工具本身没有控制权限，不会因为模型文件出现就自动开始实机操作。

战团库训练环境的状态注入、结算与课程采样仍是后续开发项；本次交付是自动采集、候选整理与诊断上下文，不声称已完成论文库席位训练。
