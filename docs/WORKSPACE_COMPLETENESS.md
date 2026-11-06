# 工作区完整性与清理判定（2026-11-06）

活动代码、6039选择配置、完整模型/Trainer、历史数据/回放、NUS经验均在HSAI。新建本地`.venv`，Torch等通用依赖复用外部Anaconda，NumPy/SciPy及OCR包在本环境。Python导入路径、核心模块来源和已加载DLL均不依赖三个旧项目目录，也不依赖D:/tmp/hsai-migration。当前引擎wheel已安装至新环境。

本轮215项Python测试通过、7项跳过；源码锁/卡表检查通过。6039在55条历史公开快照上完成合法绑定、推理、几何和Controller dry-run；Windows PowerShell5.1启动器通过。OCR在本地生成的中文图上完成检测/识别。没有打开游戏、截图、获取窗口或发送输入，没有改变NUS。证据在artifacts/local；不把离线结果当当前游戏验收。

## 除24h收尾外的缺口

| 优先级 | 缺口 | 影响/处理 |
|---|---|---|
| P1 | 当前游戏UI与公开日志→状态语义验收 | 旧255790 profile不适配6039的256589要求；隐藏字段仍是显式public-approx |
| P1 | 注入前物种账本偏差 | 35计数/5采样批次，注入不改变它；基础规则或统计根因待定位 |
| P1 | 无进展play/cancel循环 | 少量首回合82动作强制收尾，保护有效但动作浪费未解 |
| P1 | HSAI新布局的NUS迁移/GPU恢复 | 路径与来源摘要改变，须做保留Adam/RNG的显式分叉，不能原目录盲目resume |
| P2 | 原版4159、6157/6160、B100k/B10M历史状态 | 当前已归档迁移4159及两阶段状态，不等于全部历史已齐 |
| P2 | 高本经营、评测泛化 | 固定Dev改善不能证明真人棋力；后期均值有幸存者偏差 |
| P2 | 实机与训练预算不同 | 旧practical控制器保留200动作/8换位，训练80阈值/5换位；实机验收时明确决定，本轮不改训练行为 |
| 可选 | C#插件SDK/部署验收 | BepInEx/Harmony等未验证齐全，当前245日志桥不要求启动此插件 |

与收尾相关的缺项另有：最终Trainer、全history/终评/Slurm日志、当前库frames和raw；metadata_only目录不可冒充训练库。

## 旧目录清理

重新核对43532个已入库源文件，大小/修改时间无变化；相同排除范围下无新增源文件。逐文件和ZIP成员哈希来自上一轮完整迁移。Git历史、旧环境、构建缓存、合成测试权重有意未迁移。没有删除旧文件。

| 来源 | 判定 |
|---|---|
| HSBG/personal/reports、docs、FirestoneData | 已逐项迁移，可手动清理这些子目录 |
| HSBG/personal/local/models、upstream-artifacts、firestone-20261102、firestone-255790-refresh | 已完整备份，可手动清理这些子目录；RAR仅校验原包字节 |
| HSBG/personal/artifacts | 全部200文件迁移校验，可手动清理该本地子目录 |
| HearthstoneAI/runtime、hsdata、reversed | 已完整归档，新运行无依赖，可手动清理这些子目录 |
| HearthstoneRL/FirestoneData、results | 已完整迁移，可手动清理这些子目录 |
| D:/tmp/hsai-migration | 新环境不再使用此扩展，wheel和关键结果已在HSAI，生成验证/构建产物可清理；不包括整个D:/tmp |
| D:/HSBG整目录 | 不无条件删除：含非AI课程内容、未迁移Git和旧环境，先分开核实 |
| D:/HearthstoneAI整目录 | 项目资源已迁移；确认放弃未备份.git和旧环境后，才考虑整目录清理 |
| D:/HearthstoneRL整目录 | 保留：.audit_tools含Rust/rustup/链接工具等外部构建环境，不能连同源码一起删 |
| D:/Anaconda、系统工具、游戏本体、驱动、SSH配置 | 外部环境，保留 |

上表目录均指D盘本地，范围之外不自动标记可删。不得将本地清理建议用于NUS正在运行的~/projects/HSRL。

## 后续顺序

24h结束后先归档和分析，再做必要优化，随后建立新Git并按授权覆盖个人HSRL main；现在没有执行Git、部署或停训。切换时保留旧NUS目录、模型和库；先做一次实际完整状态迁移、保存恢复和严格注入工程验收。
