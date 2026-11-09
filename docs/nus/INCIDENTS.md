# 故障与防复发记录

> 状态：历史故障与防复发参考；当前命令以 [ENVIRONMENT.md](ENVIRONMENT.md) 和 [EVAL_DEPLOYMENT.md](EVAL_DEPLOYMENT.md) 为准。

| 现象 | 已查明原因 | 已采用措施/不可重复的做法 |
|---|---|---|
| 登录节点卡顿/内存错误 | 1GiB虚拟内存下安装Rust或pytest | 重操作只在Slurm计算节点，不能靠重登录当修复 |
| 首轮Rust600秒超时 | 冷编译及全测不适合短统一超时 | 计算节点独立合理时限；复用匹配构建缓存 |
| unknown_ref拒绝且状态改变 | 快照引用和失败原子性问题 | 严格验证、拒绝状态身份测试；不能把拒绝当正常回退 |
| collect-probe输出非空 | benchmark先写config违反探针契约 | 配置与证据目录分离，探针目录不得预填 |
| 63.3%注入 | not_alive与pending_choice阻挡大量尝试 | 待选合法处理后再严格注入；活席应注入100%，不回退Bot |
| 1069298文件描述符耗尽 | SHM视图引用使close失败并积累句柄 | 父进程独立数组、finally释放视图并及时close，Linux检查fd不增长 |
| 饰品必须选却买不起 | 嵌套菜单退出后露出不可负担菜单，状态机遗漏 | Rust所有选择出口/回合收尾处理、未知待选不静默清除，Kiri禁用和排除列表修正 |
| 质量报告零值不充分 | truncated_combats当函数调用且吞异常 | 常数时间原生诊断贯穿采集/建库/评测，异常状态拒绝入PPO |
| 1079510Rust保护测试读不到现场 | 相对GUARD_DIR在Cargo crate cwd下变了位置 | ROOT锚定绝对目录，覆盖不同cwd与中文路径回归 |
| 1067566/1079746 GRES2但实际1设备 | 作业声明与Driver实际不一致；服务端根因未证实 | 原始Driver/UUID与Torch计算预检，不取消Slurm隔离，不声称改好了集群 |
| 1079842/1079902 batch2而step1 | srun步分配收缩；增加GRES参数仍复现 | 移除srun包装，直接使用已分配batch；不以nvidia-smi列卡代替可计算证明 |
| 1080016启动9秒缺plan.json | 长训要求工程父目录没有产出的文件 | 显式source-fork.json契约，真实产物布局连通测试，提交前标准库预检 |
| PPO采样多但只学6.8% | lag1旧策略批次KL超阈值，甚至零optimizer.step | 降低独立分叉LR、零步同步回退，保留KL保护；训练iteration版本不等于Adam步数 |
| 1092598 CPU作业15秒失败 | PowerShell 改写把 UTF-8 BOM(U+FEFF) 写进 `workflows/run_forced_tier_long.py`（另有 `tests/test_parallel_collect.py`），而 `tools/verify_workspace.py` 用 `read_text(encoding='utf-8')` + `ast.parse`，把 BOM 当非法字符报 SyntaxError；这两个文件此前都不在检查器的语法清单里 | 两个文件去 BOM；检查器改 `utf-8-sig` 解析 + **代码目录一律拒绝 BOM** 的独立门禁，排在所有构建/测试之前；`tools/strip_source_bom.py --check/--fix`；`tests/test_source_hygiene.py` 11 项回归（含"清单必须覆盖 tests/ 与 workflows/"）。`records/`、`artifacts/` 的历史证据不动 |
| GPU验收forced_actions虚高512倍 | lock-step 里一个 policy 服务 N 局，`game_stats()` 汇总**全部局**，却被写进每一局的 meta，顶层再求和 ⇒ 计数×N（比率仍然正确） | `game_stats(game_id, seat)` 按 `(game_id, seat)` 过滤；lockstep 传 `st["gid"]`；`tests/test_parallel_collect.py` 断言「每目标结局计数 == 局数」 |

旧目录完整失败现场、Slurm文本和修复报告已归档。请通过 `tools/resource_catalog.py` 搜索原始路径/作业号；归档里的旧命令不是当前指令。不能凭过去941项Rust测试通过，就承诺没有任何未覆盖游戏机制；公开信息重建、近似机制与新补丁仍需独立验证。
